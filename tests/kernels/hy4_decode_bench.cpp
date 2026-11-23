#include <shelf/models/hy4_flash/generate.h>
#include <shelf/models/hy4_flash/target.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace shelf::hy4 {
namespace {
template<class T>
T *upload(Ops &op, const std::vector<T> &values) {
    auto p = static_cast<T *>(op.arena.next(values.size() * sizeof(T)));
    op.copy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice);
    return p;
}

Mat input(Ops &op, int rows, int cols) {
    Mat x = op.mat(rows, cols);
    std::vector<uint16_t> values(size_t(rows) * cols);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::bit_cast<uint32_t>(float(int(i * 37 % 101) - 50) / 64) >> 16;
    op.copy(x.p, values.data(), x.bytes(), hipMemcpyHostToDevice);
    return x;
}

void print_hash(Ops &op, Mat output) {
    std::vector<uint16_t> words(size_t(output.rows) * output.cols);
    op.copy(words.data(), output.p, output.bytes(), hipMemcpyDeviceToHost);
    uint64_t hash = 14695981039346656037ull;
    for (uint16_t word : words) hash = (hash ^ word) * 1099511628211ull;
    printf("output_hash=%016llx\n", static_cast<unsigned long long>(hash));
}

template<class Step>
void measure(const char *name, int rows, size_t bytes, int warmup, Step step) {
    using Clock = std::chrono::steady_clock;
    std::vector<double> times;
    for (int trial = 0; trial < warmup + 5; ++trial) {
        auto start = Clock::now();
        for (int i = 0; i < 41; ++i) step(i);
        checked(hipDeviceSynchronize());
        auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - start);
        if (trial >= warmup) times.push_back(elapsed.count() / 41);
    }
    std::sort(times.begin(), times.end());
    double us = times[times.size() / 2];
    printf("%s rows=%d us=%.3f", name, rows, us);
    if (bytes) printf(" weight_bytes=%zu GB_s=%.3f", bytes, bytes / us / 1000);
    putchar('\n');
    fflush(stdout);
}

void expert_projections(Ops &op, int rows) {
    constexpr int width = 1536;
    int capacity = rows * 16, experts = std::min(capacity, 512);
    Mat x = input(op, rows, width);
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(capacity) * 2 * width));
    auto scales = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * (width / 128) * sizeof(float)));
    auto partial = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * width * sizeof(float)));
    Mat output = op.mat(capacity, width);
    std::array<Hy4Route *, 41> routes;
    std::vector<Hy4ExpertBatch> batches;
    for (int e = 0, first = 0; e < experts; ++e) {
        int count = 1 + (capacity - 1 - e) / 512;
        batches.push_back({first, count});
        first += count;
    }
    for (int i = 0; i < int(routes.size()); ++i) {
        std::vector<Hy4Route> picks;
        for (int e = 0; e < experts; ++e)
            for (int j = 0; j < batches[e].count; ++j)
                picks.push_back({(e + 512 * j) / 16, (e * 37 + i * 19) % 512, 1.f / 16});
        routes[i] = upload(op, picks);
    }
    bool grouped = rows != 1 && rows != 6;
    auto groups = upload(op, batches);
    auto count = upload(op, std::vector<int>{capacity, experts});
    constexpr size_t projection_bytes = width * width / 2 + 2 * width * sizeof(uint16_t);
    measure("experts", rows, experts * 3 * projection_bytes, 5, [&](int i) {
        checked(shelf_hy4_expert_decode(x.p, op.gpu.matrices[0], routes[i], count, capacity, width, codes,
                                        scales, partial, output.p, op.stream, grouped ? groups : nullptr,
                                        grouped ? count + 1 : nullptr));
    });
    print_hash(op, output);
}
}

void benchmark_hy4_index(Residency &gpu) {
    constexpr int rows = 16;
    Ops op(gpu);
    unsigned seed = 97;
    auto data = [&](int count, int cols) {
        std::vector<uint16_t> values(size_t(count) * cols);
        for (auto &value : values) {
            seed = seed * 1664525u + 1013904223u;
            value = std::bit_cast<uint32_t>(float(int(seed >> 16) - 32768) / 32768.f) >> 16;
        }
        return Mat{upload(op, values), count, cols};
    };
    Mat q = data(rows, 2048), w = data(rows, 16), keys = data(32772, 128);
    auto mark = op.arena.cursor;
    for (const auto [start, cancellation] : std::array<std::pair<int, bool>, 6>{
             {{2045, false}, {4096, false}, {8192, false}, {32768, false}, {131072, false}, {8192, true}}}) {
        if (cancellation) {
            std::vector<uint16_t> weights(size_t(rows) * 16);
            for (int row = 0; row < rows; ++row) {
                auto query = q.p + size_t(row) * 2048;
                checked(hipMemcpyAsync(query + 1024, query, 1024 * sizeof(uint16_t), hipMemcpyDeviceToDevice, op.stream));
                for (int h = 0; h < 16; ++h)
                    weights[row * 16 + h] = h < 8 ? 0x3f80u : 0xbf80u;
            }
            op.copy(w.p, weights.data(), w.bytes(), hipMemcpyHostToDevice);
        }
        int stride = (start + rows - 1) / 4 + 1;
        op.arena.cursor = mark;
        auto scores = static_cast<float *>(op.arena.next(index_workspace_floats(rows, stride) * sizeof(float)));
        auto top = static_cast<int *>(op.arena.next(rows * 511 * sizeof(int)));
        std::array<std::vector<double>, 2> times;
        std::array<std::vector<float>, 2> values;
        std::array<std::vector<int>, 2> ids;
        for (int trial = 0; trial < 5; ++trial)
            for (int order = 0; order < 2; ++order) {
                int mode = (trial + order) % 2;
                auto run = [&] {
                    index_select(q, w, keys.p, 0, rows, start, stride, scores, top, nullptr, mode, op.stream);
                };
                checked(hipMemsetAsync(scores, 0, size_t(rows) * stride * sizeof(float), op.stream));
                run();
                checked(hipStreamSynchronize(op.stream));
                auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < 12; ++i) run();
                checked(hipStreamSynchronize(op.stream));
                times[mode].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / 12);
                if (!trial) {
                    values[mode].resize(rows * stride);
                    ids[mode].resize(rows * 511);
                    op.copy(values[mode].data(), scores, values[mode].size() * sizeof(float), hipMemcpyDeviceToHost);
                    op.copy(ids[mode].data(), top, ids[mode].size() * sizeof(int), hipMemcpyDeviceToHost);
                }
            }
        double error = 0;
        int matching = 0;
        for (int row = 0; row < rows; ++row) {
            int count = (start + row) / 4;
            for (int mode = 0; mode < 2; ++mode) {
                auto first = ids[mode].begin() + row * 511;
                std::sort(first, first + 511);
                if (*first < 0 || first[510] >= count || std::adjacent_find(first, first + 511) != first + 511)
                    throw std::runtime_error("Indexer が選択したグループが範囲外または重複しています");
            }
            auto first = ids[1].begin() + row * 511;
            for (int j = 0; j < 511; ++j)
                matching += std::binary_search(first, first + 511, ids[0][row * 511 + j]);
            for (int j = 0; j < stride; ++j) {
                auto i = row * stride + j;
                if (!std::isfinite(values[1][i]) || (j >= count && values[1][i] != 0))
                    throw std::runtime_error("Indexer のスコアが不正です");
                error = std::max(error, std::abs(double(values[1][i]) - values[0][i]));
            }
        }
        for (auto &samples : times) std::sort(samples.begin(), samples.end());
        printf("index start=%d rows=%d cancellation=%d scalar_ms=%.6f wmma_ms=%.6f match=%d/%d max_error=%.9g\n",
               start, rows, cancellation, times[0][2], times[1][2], matching, rows * 511, error);
        if (error > 0.001 || matching != rows * 511)
            throw std::runtime_error("Indexer のスコアまたは選択集合が許容誤差を超えました");
    }
}

void check_hy4_prefill_capacity(Residency &gpu, int rows) {
    Ops op(gpu);
    Mat x = input(op, rows, 3072);
    auto mark = op.arena.cursor;
    std::vector<uint16_t> reference;
    for (int owner : {0, gpu.ranks - 1}) {
        gpu.model.owners[0].fill(owner);
        op.copy(gpu.moe_owners[0], gpu.model.owners[0].data(), 512 * sizeof(int), hipMemcpyHostToDevice);
        op.arena.cursor = mark;
        Mat out = moe(op, 1, x);
        std::vector<uint16_t> values(out.bytes() / sizeof(uint16_t));
        op.copy(values.data(), out.p, out.bytes(), hipMemcpyDeviceToHost);
        gpu.collect_route_counts(0);
        for (int r = 0; r < gpu.ranks; ++r) {
            if (gpu.routed_rows[r] != uint64_t(r == owner ? rows * 16 : 0))
                throw std::runtime_error("経路が集中した rank の件数が不正です");
            gpu.routed_rows[r] = 0;
        }
        if (reference.empty())
            reference = std::move(values);
        else if (reference != values)
            throw std::runtime_error("全 expert の配置先によって prefill 出力が変化しました");
    }
    printf("ep=%d rows=%d choices=%d concentrated=pass\n", gpu.ranks, rows, rows * 16);
}

void benchmark_hy4_prefill(Residency &gpu, int rows) {
    Ops op(gpu);
    Mat x = input(op, rows, 3072);
    auto mark = op.arena.cursor;
    for (bool routed : {false, true}) {
        std::vector<double> times;
        Mat output{};
        for (int i = 0; i < 7; ++i) {
            op.arena.cursor = mark;
            auto start = std::chrono::steady_clock::now();
            output = routed ? moe(op, 1, x) : feedforward(op, "model.layers.1.mlp.shared_experts", x);
            checked(hipDeviceSynchronize());
            if (i >= 2)
                times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(times.begin(), times.end());
        printf("%s rows=%d ms=%.5f ", routed ? "moe" : "fixed_ffn", rows, times[2]);
        print_hash(op, output);
        if (routed) {
            gpu.collect_route_counts(0);
            for (int r = 1; r < gpu.ranks; ++r)
                printf("peer=%d send_bytes=%llu return_bytes=%llu\n", r,
                       static_cast<unsigned long long>(gpu.transfer_bytes[0][r] / 7),
                       static_cast<unsigned long long>(gpu.transfer_bytes[r][0] / 7));
        }
    }
}

void benchmark_hy4_decode(Residency &gpu) {
    Ops op(gpu);
    for (int rows : {1, 6}) {
        op.arena.cursor = 0;
        Mat x = input(op, rows, 3072);
        auto base = op.arena.cursor;
        Mat output{};
        measure("linear", rows, 3072 * 1536 / 2 + (3072 + 1536) * sizeof(uint16_t), 64, [&](int i) {
            op.arena.cursor = base;
            output = op.linear("model.layers." + std::to_string(i + 1) + ".mlp.shared_experts.gate_proj", x);
        });
        print_hash(op, output);
        measure("fixed_ffn", rows, 3 * (3072 * 1536 / 2 + (3072 + 1536) * sizeof(uint16_t)), 32, [&](int i) {
            op.arena.cursor = base;
            output = feedforward(op, "model.layers." + std::to_string(i + 1) + ".mlp.shared_experts", x);
        });
        print_hash(op, output);
    }
    for (int rows : {1, 4, 8, 16, 6, 24, 48, 96}) {
        op.arena.cursor = 0;
        expert_projections(op, rows);
    }
}

void benchmark_hy4_moe(Residency &gpu) {
    Ops op(gpu);
    for (int rows : {1, 4, 8, 16, 6, 24, 48, 96}) {
        op.arena.cursor = 0;
        op.batch_decode = rows != 1 && rows != 6;
        Mat x = input(op, rows, 3072), output{};
        auto base = op.arena.cursor;
        measure("moe", rows, 0, 5, [&](int i) {
            op.arena.cursor = base;
            output = moe(op, i + 1, x);
        });
        print_hash(op, output);
    }
}

void benchmark_hy4_steps(Residency &gpu) {
    using Clock = std::chrono::steady_clock;
    const int batch = gpu.max_slots;
    const std::vector<int> prompt{802, 8778, 299, 9480, 341, 39144, 13, 360,
                                  46544, 52315, 533, 10044, 565, 1823, 269, 7370};
    std::vector<int> expected;
    for (int trial = 0; trial < 5; ++trial) {
        for (int side = 0; side < 2; ++side) {
            bool graph = (trial + side) % 2;
            setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
            setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
            std::vector<int> generated;
            std::vector<double> times;
            auto measure_steps = [&](auto step) {
                for (int i = 0; i < 10; ++i) {
                    auto start = Clock::now();
                    step();
                    auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
                    if (i >= 3) times.push_back(elapsed.count());
                }
            };
            auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
            if (batch == 1) {
                Target target(gpu, 0, 0, 0, 1);
                int anchor = target.forward(prompt, false).front();
                measure_steps([&] {
                    std::vector<int> ids{anchor};
                    if (gpu.model.has_draft) {
                        auto candidates = target.propose(anchor);
                        ids.insert(ids.end(), candidates.begin(), candidates.end());
                    }
                    auto actual = target.forward(ids, gpu.model.has_draft);
                    int keep = 1;
                    while (keep < int(ids.size()) && actual[keep - 1] == ids[keep]) ++keep;
                    if (gpu.model.has_draft) target.accept(keep, ids.size());
                    generated.insert(generated.end(), actual.begin(), actual.begin() + keep);
                    anchor = actual[keep - 1];
                });
            } else {
                Hy4BatchSession session(gpu, [&](int, int id) { generated.push_back(id); },
                                        [](int, const auto &, const char *) {});
                for (int i = 0; i < batch; ++i) session.admit({i, prompt, 64, {}});
                for (int i = 0; i < batch; ++i) session.step();
                measure_steps([&] {
                    if (session.active() != size_t(batch))
                        throw std::runtime_error("計測途中でバッチの件数が変わりました");
                    session.step();
                });
            }
            if (expected.empty()) expected = generated;
            if (generated != expected || (gpu.target_replays > target_before) != graph ||
                (gpu.draft_replays > draft_before) != (graph && gpu.model.has_draft))
                throw std::runtime_error("計測時の生成 ID または Graph の起動条件が異なります");
            std::sort(times.begin(), times.end());
            printf("B=%d draft=%d graph=%d trial=%d step_ms=%.5f\n", batch, gpu.model.has_draft, graph,
                   trial, times[times.size() / 2]);
            fflush(stdout);
        }
    }
}
}
