#include <shelf/models/hy4_flash/generate.h>
#include <shelf/models/hy4_flash/target.h>

#include <algorithm>
#include <bit>
#include <chrono>
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
