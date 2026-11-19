#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/models/hy4_flash/generate.h>
#include <nlohmann/json.hpp>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace {
using namespace shelf::hy4;
using namespace shelf;
template <class T> T take(Result<T> r) {
    if (!r.ok())
        throw std::runtime_error(r.status().message());
    return r.release();
}
void compare(const std::vector<uint16_t> &output, const uint16_t *expected, int chunk) {
    double ss = 0, diff = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        float ref = std::bit_cast<float>(uint32_t(expected[i]) << 16);
        float val = std::bit_cast<float>(uint32_t(output[i]) << 16);
        ss += double(ref) * ref;
        diff += double(val - ref) * (val - ref);
    }
    double error = std::sqrt(diff / ss);
    printf("chunk=%d relative_l2=%.8f\n", chunk, error);
    if (!(error < 0.04))
        throw std::runtime_error("CPU の層出力との誤差が大きすぎます");
}
void check_rejection(Ops &op) {
    std::vector<int> proposals{11, 22, 33, 44, 55};
    for (int agree = 0; agree <= 5; ++agree) {
        op.arena.cursor = 0;
        std::vector<float> q(5 * 120832), p(5 * 120832);
        for (int i = 0; i < 5; ++i) {
            q[i * 120832 + proposals[i]] = 1;
            p[i * 120832 + (i < agree ? proposals[i] : 900 + i)] = 1;
        }
        auto copy = [&](const std::vector<float> &values) {
            auto ptr = static_cast<float *>(op.arena.next(values.size() * 4));
            op.copy(ptr, values.data(), values.size() * 4, hipMemcpyHostToDevice);
            return ptr;
        };
        auto draft = copy(q), target = copy(p);
        std::vector<int> actual{90, 90, 90, 90, 90, 91};
        int keep = verify_speculation(op, target, draft, 2047, 42, proposals, actual);
        if (keep != agree + 1 || !std::equal(proposals.begin(), proposals.begin() + agree, actual.begin()) ||
            actual[agree] != (agree < 5 ? 900 + agree : 91))
            throw std::runtime_error("DFly 確率的棄却の結果が一致しません");
    }
    op.arena.cursor = 0;
}
int run(const Path &dir, int repeats) {
    Checkpoint model;
    std::ifstream in(dir / "fixture.json");
    auto fixture = nlohmann::json::parse(in);
    int layer = fixture.at("layer");
    int start = fixture.value("start", 0);
    auto ids = fixture.at("ids").get<std::vector<int>>();
    if (layer < 0 || layer >= 42 || ids.empty() || ids.size() > 128)
        throw std::runtime_error("fixture の範囲が不正です");
    model.shape = fixture.at("shapes").get<decltype(model.shape)>();
    for (int g = 0; g < 4; ++g) {
        auto sizes = fixture.at("moduli").at(std::to_string(2 + g * 10));
        for (int i = 0; i < 16; ++i)
            model.moduli[g][i] = sizes.at(i / 8).at(i % 8);
    }
    for (const auto &[name, keys] : fixture.at("files").items()) {
        auto reader = take(io::SafetensorsReader::open(name));
        for (const auto &[source, target] : keys.items()) {
            auto spec = take(reader.tensor_spec(source));
            size_t bytes;
            const auto ptr = take(reader.tensor_data(source, bytes));
            std::string key = target;
            if (key.ends_with(".trellis") && spec.shape.size() == 3 && spec.shape[2] == 48)
                model.mixed = true;
            int rank = (key.find("multi_head_embedding.") != std::string::npos ||
                        key.ends_with(".lookup_table") || key.ends_with(".mul1"))
                           ? -1
                           : 0;
            size_t offset = (model.bytes[0] + 255) & ~size_t(255);
            if (rank == 0)
                model.bytes[0] = offset + bytes;
            model.weights.emplace(key, Weight{spec, ptr, bytes, rank, offset});
        }
        model.files.push_back(std::move(reader));
    }
    auto rows = take(io::SafetensorsReader::open((dir / "rows.safetensors").string()));
    size_t input_size, expected_size;
    auto input = static_cast<const uint16_t *>(take(rows.tensor_data("input", input_size)));
    auto expected = static_cast<const uint16_t *>(take(rows.tensor_data("expected", expected_size)));
    if (input_size != ids.size() * 12288 * 2 || expected_size != input_size)
        throw std::runtime_error("fixture の形状が不正です");
    size_t capacity_tokens = start + ids.size();
    size_t slots = (layer == 0 || layer % 4 == 1) ? capacity_tokens : std::min(size_t(518), capacity_tokens);
    Residency gpu(model, capacity_tokens);
    gpu.ranks = 1;
    constexpr size_t work = 512ull << 20;
    size_t capacity = model.bytes[0] + work + 64ull * 1024 * 1024;
    gpu.arenas.push_back(take(gpu::GpuArena::create(0, capacity)));
    try {
        auto alloc = take(gpu.arenas[0].allocate_aligned(capacity, 256));
        Region memory{static_cast<uint8_t *>(alloc.data()), capacity};
        gpu.weight_base[0] = static_cast<uint8_t *>(memory.next(model.bytes[0]));
        for (const auto &[key, w] : model.weights)
            if (w.rank == 0)
                checked(hipMemcpy(gpu.weight_base[0] + w.offset, w.host, w.bytes, hipMemcpyHostToDevice));
        gpu.cache[layer] = static_cast<uint8_t *>(memory.next(slots * 294));
        gpu.index_keys[layer] = static_cast<uint16_t *>(memory.next((capacity_tokens / 4 + 12) * 256));
        for (auto &[key, dest] : gpu.folded)
            dest = memory.next(size_t(model.shape.at(key)[1]) * 512 * 2);
        gpu.matrices[0] = static_cast<Hy4ExpertProjection *>(memory.next(gpu.matrix_bytes));
        if (layer)
            gpu.upload_experts(0, layer);
        gpu.workspace[0] = {static_cast<uint8_t *>(memory.next(work)), work};
        if (!gpu.folded.empty())
            fold_kv_weights(gpu);
        Ops op(gpu);
        if (!layer)
            check_rejection(op);
        for (int chunk : {int(ids.size()), 1}) {
            std::vector<double> samples;
            for (int pass = 0; pass < repeats; ++pass) {
                std::vector<uint16_t> actual(input_size / 2);
                double seconds = 0;
                std::vector<int> history(start, 120000);
                if (start) {
                    auto restore = [&](const char *key, void *dest, size_t length) {
                        size_t size;
                        auto ptr = take(rows.tensor_data(key, size));
                        if (size != length)
                            throw std::runtime_error("保存 KV のサイズが不正です");
                        op.copy(dest, ptr, size, hipMemcpyHostToDevice);
                    };
                    restore("kv", gpu.cache[layer], slots * 294);
                    if (layer == 0 || layer % 4 == 1)
                        restore("index", gpu.index_keys[layer], (capacity_tokens / 4 + 12) * 256);
                }
                for (int i = 0; i < int(ids.size()); i += chunk) {
                    int count = std::min(chunk, int(ids.size()) - i);
                    op.arena.cursor = 0;
                    Mat x = op.mat(count, 12288);
                    op.copy(x.p, input + size_t(i) * 12288, x.bytes(), hipMemcpyHostToDevice);
                    Selection selected;
                    std::vector<int> batch(ids.begin() + i, ids.begin() + i + count);
                    auto begin = std::chrono::steady_clock::now();
                    decoder_layer(op, layer, start + i, x, history, batch, selected);
                    for (int j = 0; j < count; ++j) {
                        auto positions = selected.at(j);
                        std::sort(positions.begin(), positions.end());
                        if (positions.empty() || positions.back() != start + i + j || positions.size() > 2048 ||
                            std::adjacent_find(positions.begin(), positions.end()) != positions.end())
                            throw std::runtime_error("選択した位置が causal 条件に違反しています");
                    }
                    checked(hipDeviceSynchronize());
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                    op.copy(actual.data() + size_t(i) * 12288, x.p, x.bytes(), hipMemcpyDeviceToHost);
                    history.insert(history.end(), batch.begin(), batch.end());
                }
                if (pass == 0)
                    compare(actual, expected, chunk);
                else
                    samples.push_back(1000 * seconds * chunk / ids.size());
            }
            if (!samples.empty()) {
                std::sort(samples.begin(), samples.end());
                printf("chunk=%d median_ms=%.6f\n", chunk, samples[samples.size() / 2]);
            }
        }
    } catch (...) {
        gpu.shutdown();
        throw;
    }
    gpu.shutdown();
    return 0;
}
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 3)
        return 2;
    try {
        int repeats = argc == 2 ? 1 : std::stoi(argv[2]);
        if (repeats < 1 || repeats > 128)
            return 2;
        return run(argv[1], repeats);
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
