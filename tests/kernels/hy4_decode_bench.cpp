#include <shelf/models/hy4_flash/generate.h>
#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/decoder.h>

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

void print_f32_hash(Ops &op, const float *p, int rows, int cols) {
    std::vector<uint32_t> words(size_t(rows) * cols);
    op.copy(words.data(), p, words.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
    uint64_t hash = 14695981039346656037ull;
    for (uint32_t word : words) hash = (hash ^ word) * 1099511628211ull;
    printf("output_hash=%016llx\n", static_cast<unsigned long long>(hash));
}

void benchmark_hy4_decode(Residency &gpu) {
    Ops op(gpu);
    for (int rows : {1, 3, 4, 5, 6, 16, 80, 96}) {
        op.arena.cursor = 0;
        Mat x = input(op, rows, 3072);
        Mat qx = input(op, rows, 1024);
        Mat ox = input(op, rows, 16384);
        auto base = op.arena.cursor;
        Mat output{};
        int warmup = rows <= 16 ? 16 : 5;
        measure("linear", rows, 3072 * 1536 / 2 + (3072 + 1536) * sizeof(uint16_t), warmup, [&](int i) {
            op.arena.cursor = base;
            output = op.linear("model.layers." + std::to_string(i + 1) + ".mlp.shared_experts.gate_proj", x);
        });
        print_hash(op, output);
        measure("fixed_ffn", rows, 3 * (3072 * 1536 / 2 + (3072 + 1536) * sizeof(uint16_t)), warmup, [&](int i) {
            op.arena.cursor = base;
            output = feedforward(op, "model.layers." + std::to_string(i + 1) + ".mlp.shared_experts", x);
        });
        print_hash(op, output);
        measure("mla_qb", rows, 1024 * 16384 / 2 + (1024 + 16384) * sizeof(uint16_t), warmup, [&](int i) {
            op.arena.cursor = base;
            int layer = 2 + (i % 10) * 4;
            if (layer % 4 == 1) ++layer;
            output = op.linear("model.layers." + std::to_string(layer) + ".self_attn.q_b_proj", qx);
        });
        print_hash(op, output);
        measure("mla_o", rows, 16384 * 3072 / 2 + (16384 + 3072) * sizeof(uint16_t), warmup, [&](int i) {
            op.arena.cursor = base;
            int layer = 2 + (i % 10) * 4;
            if (layer % 4 == 1) ++layer;
            output = op.linear("model.layers." + std::to_string(layer) + ".self_attn.o_proj", ox);
        });
        print_hash(op, output);
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(rows) * 120832 * 4));
        auto logits = static_cast<float *>(op.arena.next(size_t(rows) * 120832 * 4));
        auto suh = static_cast<const uint16_t *>(op.param("lm_head.suh").p);
        auto svh = static_cast<const uint16_t *>(op.param("lm_head.svh").p);
        auto trellis = static_cast<const uint32_t *>(op.param("lm_head.trellis").p);
        auto tile_vals = static_cast<float *>(op.arena.next(size_t(rows) * 944 * 4));
        auto tile_ids = static_cast<int *>(op.arena.next(size_t(rows) * 944 * 4));
        auto tokens = static_cast<int *>(op.arena.next(size_t(rows) * sizeof(int)));
        measure("lm_head", rows, 3072ull * 120832 * 6 / 8 + (3072 + 120832) * sizeof(uint16_t), 3, [&](int) {
            checked(shelf_hy4_head(x.p, suh, svh, trellis, codes, scales, partial, logits, rows, op.stream));
        });
        print_f32_hash(op, logits, rows, 120832);
        measure("lm_head_argmax", rows, 3072ull * 120832 * 6 / 8 + (3072 + 120832) * sizeof(uint16_t), 3, [&](int) {
            checked(shelf_hy4_head_argmax(x.p, suh, svh, trellis, codes, scales, partial, tile_vals, tile_ids, tokens,
                                          rows, op.stream));
        });
        std::vector<float> host_logits(size_t(rows) * 120832);
        std::vector<int> host_tokens(rows);
        op.copy(host_logits.data(), logits, host_logits.size() * sizeof(float), hipMemcpyDeviceToHost);
        op.copy(host_tokens.data(), tokens, host_tokens.size() * sizeof(int), hipMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) {
            float best = -INFINITY;
            int ref_id = -1;
            const float *row = host_logits.data() + size_t(r) * 120832;
            for (int c = 0; c < 120832; ++c) {
                if (row[c] > best) {
                    best = row[c];
                    ref_id = c;
                }
            }
            if (host_tokens[r] != ref_id)
                throw std::runtime_error("lm_head_argmax が不一致です");
        }
        printf("argmax_match=%d/%d\n", rows, rows);
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

void benchmark_hy4_router(Residency &gpu) {
    Ops op(gpu);
    auto w = static_cast<const uint16_t *>(op.param("model.layers.1.mlp.gate.weight").p);
    auto b = static_cast<const float *>(op.param("model.layers.1.mlp.gate.e_score_correction_bias").p);
    for (int rows : {32, 96, 1024}) {
        op.arena.cursor = 0;
        Mat hidden = input(op, rows, 3072);
        Mat x = op.norm(hidden, "model.layers.1.post_attention_layernorm.weight");
        auto ref = static_cast<Hy4Route *>(op.arena.next(size_t(rows) * 16 * sizeof(Hy4Route)));
        auto got = static_cast<Hy4Route *>(op.arena.next(size_t(rows) * 16 * sizeof(Hy4Route)));
        auto base = op.arena.cursor;
        for (bool exact : {true, false}) {
            measure(exact ? "router_exact" : "router_wmma", rows, 0, 5, [&](int) {
                op.arena.cursor = base;
                route_select(op, x, w, b, gpu.gate_expert_norms[0], exact ? ref : got, exact);
            });
        }
        std::vector<Hy4Route> h_ref(size_t(rows) * 16), h_got(size_t(rows) * 16);
        op.copy(h_ref.data(), ref, h_ref.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
        op.copy(h_got.data(), got, h_got.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
        if (std::memcmp(h_ref.data(), h_got.data(), h_ref.size() * sizeof(Hy4Route)) != 0)
            throw std::runtime_error("router 出力が一致しません");
        printf("router_match rows=%d\n", rows);
    }
    {
        constexpr int rows = 40;
        op.arena.cursor = 0;
        std::vector<uint16_t> hx(size_t(rows) * 3072, 0x3c00u);
        std::vector<uint16_t> hw(size_t(512) * 3072, 0x3c00u);
        std::vector<float> hb(512, 0.f);
        std::vector<float> hwn(512, std::sqrt(3072.f));
        for (int c = 0; c < 3072; ++c) {
            hx[1 * 3072 + c] = uint16_t(std::bit_cast<uint32_t>(float((c % 7) - 3) * 0.1f) >> 16);
            hx[2 * 3072 + c] = c == 0 ? 0x7fc0u : 0x3c00u;
        }
        for (int e = 0; e < 512; ++e) {
            hb[e] = e < 24 ? 0.5f + float(24 - e) * 1e-6f : 0.f;
            if (e % 3 == 0) hw[size_t(e) * 3072] = 0xbc00u;
        }
        Mat ax{upload(op, hx), rows, 3072};
        auto aw = upload(op, hw);
        auto ab = upload(op, hb);
        auto awn = upload(op, hwn);
        auto ref = static_cast<Hy4Route *>(op.arena.next(size_t(rows) * 16 * sizeof(Hy4Route)));
        auto got = static_cast<Hy4Route *>(op.arena.next(size_t(rows) * 16 * sizeof(Hy4Route)));
        route_select(op, ax, aw, ab, awn, ref, true);
        route_select(op, ax, aw, ab, awn, got, false);
        std::vector<Hy4Route> h_ref(size_t(rows) * 16), h_got(size_t(rows) * 16);
        op.copy(h_ref.data(), ref, h_ref.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
        op.copy(h_got.data(), got, h_got.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
        if (std::memcmp(h_ref.data(), h_got.data(), h_ref.size() * sizeof(Hy4Route)) != 0)
            throw std::runtime_error("人工同点・僅差・非有限入力で router 出力が一致しません");
        printf("router_edge_cases=pass\n");
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
