#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/models/hy4_flash/generate.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
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
void check_index(Ops &op, const uint16_t *input, int start, int rows) {
    op.arena.cursor = 0;
    Mat x = op.mat(rows, 12288);
    op.copy(x.p, input, x.bytes(), hipMemcpyHostToDevice);
    std::string layer = "model.layers.0.", attn = layer + "self_attn.", index = attn + "compressed_indexer.";
    float *post;
    Mat hidden = op.collapse_norm(x, layer + "hc_attn_layer.hc_pre.hc_", post, layer + "input_layernorm.weight");
    Mat qres = op.norm(op.linear(attn + "q_a_proj", hidden), attn + "q_a_layernorm.weight");
    Mat q = op.rope(op.linear(index + "query_projection", qres), 16, start, 1e7f);
    Mat w = op.linear(index + "head_weight_projection", hidden);
    int first = std::max(0, 2048 - start), selected_rows = rows - first;
    int stride = (start + rows - 1) / 4;
    auto scores = static_cast<float *>(op.arena.next(index_workspace_floats(selected_rows, stride) * sizeof(float)));
    auto top = static_cast<int *>(op.arena.next(size_t(selected_rows) * 511 * sizeof(int)));
    std::array<std::vector<int>, 2> groups;
    std::array<std::vector<float>, 2> values;
    for (int mode = 0; mode < 2; ++mode) {
        index_select(q, w, op.gpu.index_keys[0][0], first, selected_rows, start, stride, scores, top, nullptr, mode, op.stream);
        groups[mode].resize(size_t(selected_rows) * 511);
        values[mode].resize(size_t(selected_rows) * stride);
        op.copy(groups[mode].data(), top, groups[mode].size() * sizeof(int), hipMemcpyDeviceToHost);
        op.copy(values[mode].data(), scores, values[mode].size() * sizeof(float), hipMemcpyDeviceToHost);
        for (int r = 0; r < selected_rows; ++r) {
            auto begin = groups[mode].begin() + r * 511;
            std::sort(begin, begin + 511);
        }
    }
    double error = 0, peak = 0;
    int tokens = 0;
    for (int r = 0; r < selected_rows; ++r) {
        int pos = start + first + r;
        tokens += 2045 + pos % 4;
        for (int i = 0; i < pos / 4; ++i) {
            size_t j = size_t(r) * stride + i;
            double ref = values[0][j], val = values[1][j];
            if (!std::isfinite(ref) || !std::isfinite(val)) throw std::runtime_error("Indexer のスコアが非有限です");
            peak = std::max(peak, std::abs(ref));
            error = std::max(error, std::abs(ref - val));
        }
    }
    if (groups[0] != groups[1] || error > 2e-5 * std::max(1.0, peak))
        throw std::runtime_error("Indexer の選択集合またはスコアが基準と異なります");
    printf("index_start=%d selected_tokens=%d/%d max_score_error=%.10g\n", start + first, tokens, tokens, error);
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
    if (layer < 0 || layer >= 42 || ids.empty() || ids.size() > max_prefill_chunk)
        throw std::runtime_error("fixture の範囲が不正です");
    model.shape = fixture.at("shapes").get<decltype(model.shape)>();
    model.local_shape[0] = model.shape;
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
            (rank < 0 ? model.source : model.local[0]).emplace(key, Weight{spec, ptr, bytes, rank, offset});
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
    const size_t work = ids.size() > 128 ? 2ull << 30 : 512ull << 20;
    size_t capacity = model.bytes[0] + work + 64ull * 1024 * 1024;
    gpu.arenas.push_back(take(gpu::GpuArena::create(0, capacity)));
    try {
        auto alloc = take(gpu.arenas[0].allocate_aligned(capacity, 256));
        Region memory{static_cast<uint8_t *>(alloc.data()), capacity};
        gpu.weight_base[0] = static_cast<uint8_t *>(memory.next(model.bytes[0]));
        for (const auto &[key, w] : model.local[0])
            if (w.rank == 0)
                checked(hipMemcpy(gpu.weight_base[0] + w.offset, w.host, w.bytes, hipMemcpyHostToDevice));
        gpu.cache[0][layer] = static_cast<uint8_t *>(memory.next(slots * 294));
        gpu.index_keys[0][layer] = static_cast<uint16_t *>(memory.next((capacity_tokens / 4 + 12) * 256));
        for (auto &[key, dest] : gpu.folded[0])
            dest = memory.next(size_t(model.shape.at(key)[1]) * 512 * 2);
        gpu.reserve_experts(0, memory);
        if (layer)
            gpu.upload_experts(0, layer);
        gpu.workspace[0] = {static_cast<uint8_t *>(memory.next(work)), work};
        if (!gpu.folded[0].empty())
            fold_kv_weights(gpu, 0);
        checked(hipStreamCreateWithFlags(&gpu.compute[0], hipStreamNonBlocking));
        Ops op(gpu);
        if (!layer)
            check_rejection(op);
        auto chunks = ids.size() > 1024 ? std::vector<int>{1024, 2048, 4096}
                                        : std::vector<int>{int(ids.size()), 1};
        std::vector<std::vector<int>> expected_positions(ids.size());
        for (int chunk : chunks) {
            std::vector<double> samples;
            for (int pass = 0; pass < repeats; ++pass) {
                std::vector<uint16_t> actual(input_size / 2);
                double seconds = 0;
                uint64_t initial_routes = gpu.routed_rows[0];
                std::vector<int> history(start, 120000);
                if (start) {
                    auto restore = [&](const char *key, void *dest, size_t length) {
                        size_t size;
                        auto ptr = take(rows.tensor_data(key, size));
                        if (size != length)
                            throw std::runtime_error("保存 KV のサイズが不正です");
                        op.copy(dest, ptr, size, hipMemcpyHostToDevice);
                    };
                    restore("kv", gpu.cache[0][layer], slots * 294);
                    if (layer == 0 || layer % 4 == 1)
                        restore("index", gpu.index_keys[0][layer], (capacity_tokens / 4 + 12) * 256);
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
                        if (positions.empty() && gpu.sparse_top[0][layer]) {
                            std::array<int, 511> groups;
                            op.copy(groups.data(), gpu.sparse_top[0][layer] + j * groups.size(),
                                    sizeof(groups), hipMemcpyDeviceToHost);
                            int end = start + i + j;
                            for (int group : groups)
                                for (int t = 0; t < 4; ++t)
                                    positions.push_back(group * 4 + t);
                            for (int t = end & ~3; t <= end; ++t)
                                positions.push_back(t);
                        }
                        std::sort(positions.begin(), positions.end());
                        if (positions.empty() || positions.back() != start + i + j || positions.size() > 2048 ||
                            std::adjacent_find(positions.begin(), positions.end()) != positions.end())
                            throw std::runtime_error("選択した位置が causal 条件に違反しています");
                        if (pass == 0) {
                            if (chunk == chunks.front())
                                expected_positions[i + j] = std::move(positions);
                            else if (expected_positions[i + j] != positions)
                                throw std::runtime_error("チャンク長によって選択位置が変化しました");
                        }
                    }
                    checked(hipDeviceSynchronize());
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                    op.copy(actual.data() + size_t(i) * 12288, x.p, x.bytes(), hipMemcpyDeviceToHost);
                    if (!pass && !layer && count > 1 && count <= 16 && start + i + count > 2048)
                        check_index(op, input + size_t(i) * 12288, start + i, count);
                    history.insert(history.end(), batch.begin(), batch.end());
                }
                if (layer) {
                    gpu.collect_route_counts(0);
                    if (gpu.routed_rows[0] - initial_routes != ids.size() * Hy4MoePlan::choices)
                        throw std::runtime_error("expert の経路数が入力行数と一致しません");
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
