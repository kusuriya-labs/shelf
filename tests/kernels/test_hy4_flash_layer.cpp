#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/decoder.h>
#include <nlohmann/json.hpp>
#include <bit>
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
int run(const Path &dir) {
    Checkpoint model;
    std::ifstream in(dir / "fixture.json");
    auto fixture = nlohmann::json::parse(in);
    int layer = fixture.at("layer");
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
    Residency gpu(model, ids.size());
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
        gpu.cache[layer] = static_cast<uint8_t *>(memory.next(ids.size() * 294));
        gpu.index_keys[layer] = static_cast<uint16_t *>(memory.next((ids.size() / 4 + 12) * 256));
        for (auto &[key, dest] : gpu.folded)
            dest = memory.next(size_t(model.shape.at(key)[1]) * 512 * 2);
        gpu.workspace[0] = {static_cast<uint8_t *>(memory.next(work)), work};
        if (!gpu.folded.empty())
            fold_kv_weights(gpu);
        Ops op(gpu);
        for (int chunk : {int(ids.size()), 1}) {
            std::vector<uint16_t> actual(input_size / 2);
            std::vector<int> history;
            for (int i = 0; i < int(ids.size()); i += chunk) {
                int count = std::min(chunk, int(ids.size()) - i);
                op.arena.cursor = 0;
                Mat x = op.mat(count, 12288);
                op.copy(x.p, input + size_t(i) * 12288, x.bytes(), hipMemcpyHostToDevice);
                Selection selected;
                std::vector<int> batch(ids.begin() + i, ids.begin() + i + count);
                decoder_layer(op, layer, i, x, history, batch, selected);
                op.copy(actual.data() + size_t(i) * 12288, x.p, x.bytes(), hipMemcpyDeviceToHost);
                history.insert(history.end(), batch.begin(), batch.end());
            }
            compare(actual, expected, chunk);
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
    if (argc != 2)
        return 2;
    try {
        return run(argv[1]);
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
