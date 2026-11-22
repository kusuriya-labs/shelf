#include <shelf/models/hy4_flash/routes.h>
#include <shelf/models/hy4_flash/residency.h>
#include <cstdio>
#include <limits>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

using namespace shelf::hy4;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char *message) {
        if (!ok) {
            std::fprintf(stderr, "%s\n", message);
            ++failures;
        }
    };
    check(!Hy4MoePlan::supports(0) && !Hy4MoePlan::supports(7), "decode の行数境界が不正です");
    for (int ranks = 1; ranks <= 4; ++ranks) {
        std::array<std::array<bool, 4>, 4> peer{};
        for (auto &row : peer)
            row.fill(true);
        check(graph_topology(ranks, 1, true, true, peer), "EP の Graph が無効です");
        check(!graph_topology(ranks, 2, true, true, peer), "TP の Graph が有効です");
        check(!graph_topology(ranks, 1, false, true, peer), "push 以外の Graph が有効です");
        check(!graph_topology(ranks, 1, true, false, peer), "VRAM 不足時の Graph が有効です");
        if (ranks > 1) {
            peer[ranks - 1][ranks - 2] = false;
            check(!graph_topology(ranks, 1, true, true, peer), "peer access 不足時の Graph が有効です");
        }
        for (int rows : {1, 6, 12, 16, 32, 48, 64, 96}) {
            std::array<int, 512> owners;
            for (int e = 0; e < 512; ++e) owners[e] = e % ranks;
            std::vector<Hy4Route> picks;
            unsigned masks[3][512]{};
            for (int row = 0; row < rows; ++row)
                for (int j = 15; j >= 0; --j) {
                    int e = (row % 5 + j) % 512;
                    picks.push_back({row, e, float(row + j)});
                    masks[row / 32][e] |= 1u << (row % 32);
                }
            auto ordered = picks;
            std::array<std::vector<Hy4Route>, 4> routes;
            std::array<std::vector<int>, 4> offsets;
            partition_routes(ordered, rows, owners, ranks, routes, offsets);
            for (int r = 0; r < ranks; ++r) {
                Hy4GroupedPlan plan{};
                std::copy(offsets[r].begin(), offsets[r].end(), plan.offsets);
                std::array<int, 512> starts{};
                int cursor = 0;
                for (int e = 0; e < 512; ++e) {
                    starts[e] = cursor;
                    if (owners[e] == r)
                        for (auto &word : masks) cursor += __builtin_popcount(word[e]);
                }
                for (auto pick : picks)
                    if (owners[pick.expert] == r)
                        plan.place(pick, picks.data() + pick.row * 16, owners.data(), masks, starts[pick.expert]);
                std::vector<int> inverse;
                std::vector<Hy4ExpertBatch> batches;
                group_routes(routes[r], inverse, batches);
                check(std::equal(inverse.begin(), inverse.end(), plan.inverse), "重み付き和の順序が異なります");
                for (size_t i = 0; i < routes[r].size(); ++i) {
                    auto actual = plan.routes[i], expected = routes[r][i];
                    check(actual.expert == expected.expert && actual.row == expected.row && actual.weight == expected.weight,
                          "expert のグループ化が CPU 計画と異なります");
                }
            }
        }
        for (int rows = 1; rows <= Hy4MoePlan::max_rows; ++rows) {
            check(Hy4MoePlan::supports(rows), "有効な行数が拒否されました");
            std::vector<Hy4Route> picks;
            for (int row = 0; row < rows; ++row)
                for (int j = 15; j >= 0; --j)
                    picks.push_back({row, (row * 7 + j) % 512, float(j + 1) / 16});
            for (bool concentrated : {false, true}) {
                std::array<int, 512> owners{};
                for (int e = 0; e < 512; ++e)
                    owners[e] = concentrated ? ranks - 1 : e % ranks;
                auto sorted = picks;
                std::array<std::vector<Hy4Route>, 4> routes;
                std::array<std::vector<int>, 4> offsets;
                partition_routes(sorted, rows, owners, ranks, routes, offsets);
                int total = 0;
                for (int group = 0; group < ranks; ++group) {
                    Hy4MoePlan plan{};
                    plan.partition(picks.data(), rows, owners.data(), group);
                    offsets[group].push_back(routes[group].size());
                    check(std::equal(offsets[group].begin(), offsets[group].end(), plan.offsets),
                          "GPU 計画の行境界が CPU の経路と一致しません");
                    check(plan.count == int(routes[group].size()), "GPU 計画の経路数が一致しません");
                    check(std::equal(routes[group].begin(), routes[group].end(), plan.routes,
                                     [](auto a, auto b) {
                                         return a.row == b.row && a.expert == b.expert && a.weight == b.weight;
                                     }), "GPU 計画の expert 順または重みが一致しません");
                    total += plan.count;
                }
                check(total == Hy4MoePlan::capacity(rows), "rank 間で経路が欠落しています");
            }
        }
    }
    for (bool push : {false, true})
        for (bool source : {false, true})
            for (bool destination : {false, true}) {
                auto expected = push ? (source ? ReduceTransport::push : ReduceTransport::host)
                                     : (destination ? ReduceTransport::peer : ReduceTransport::host);
                check(select_reduce_transport(push, source, destination) == expected,
                      "転送方式の選択が不正です");
            }
    {
        SyncBarrier barrier(2);
        std::atomic<int> progress{0};
        std::thread worker([&] {
            for (int i = 0; i < 100; ++i) {
                ++progress;
                barrier.arrive_and_wait();
                barrier.arrive_and_wait();
            }
        });
        for (int i = 0; i < 100; ++i) {
            barrier.arrive_and_wait();
            check(progress == i + 1, "barrier の世代が一致しません");
            barrier.arrive_and_wait();
        }
        worker.join();
        std::promise<void> started;
        auto released = std::async(std::launch::async, [&] {
            started.set_value();
            try { barrier.arrive_and_wait(); }
            catch (const CollectiveAborted &) { return true; }
            return false;
        });
        started.get_future().wait();
        barrier.abort();
        check(released.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
              "中断時に待機が解除されません");
        check(released.get(), "中断例外が送出されません");
        bool aborted = false;
        try { barrier.arrive_and_wait(); }
        catch (const CollectiveAborted &) { aborted = true; }
        check(aborted, "中断後の参加が拒否されません");
        barrier.reset();
        std::thread again([&] { barrier.arrive_and_wait(); });
        barrier.arrive_and_wait();
        again.join();
    }
    check(multi_gpu_vmm_allowed(1, "1"), "単一 GPU の VMM が拒否されました");
    check(multi_gpu_vmm_allowed(4, "0"), "通常の複数 GPU 実行が拒否されました");
    check(!multi_gpu_vmm_allowed(2, "1") && !multi_gpu_vmm_allowed(4, "1"),
          "複数 GPU の VMM が拒否されません");
    for (int rows : {1, 8, 24, 1024}) {
        for (int ranks : {1, 3, 4}) {
            std::array<int, 512> owners{};
            for (int e = 0; e < 512; ++e)
                owners[e] = e % ranks;
            std::vector<Hy4Route> host;
            for (int row = 0; row < rows; ++row)
                for (int j = 0; j < 16; ++j)
                    host.push_back({row, (row + 15 - j) % 512, float(j + 1)});
            std::array<std::vector<Hy4Route>, 4> routes;
            std::array<std::vector<int>, 4> offsets;
            partition_routes(host, rows, owners, ranks, routes, offsets);
            size_t total = 0;
            for (int r = 0; r < ranks; ++r) {
                total += routes[r].size();
                check(offsets[r].size() == size_t(rows), "offset の行数が違います");
                for (int row = 0; row < rows; ++row) {
                    int end = row + 1 == rows ? int(routes[r].size()) : offsets[r][row + 1];
                    for (int i = offsets[r][row]; i < end; ++i)
                        check(routes[r][i].row == row && owners[routes[r][i].expert] == r,
                              "route と配置が一致しません");
                }
            }
            check(total == size_t(rows) * 16, "route の数が違います");
            for (int r = 0; r < ranks; ++r) {
                auto original = routes[r];
                std::vector<int> undo;
                std::vector<Hy4ExpertBatch> groups;
                group_routes(routes[r], undo, groups);
                int next = 0;
                for (const auto &group : groups) {
                    check(group.first == next && group.count >= 1 && group.count <= 16,
                          "expert batch の範囲が不正です");
                    for (int i = group.first; i < group.first + group.count; ++i)
                        check(routes[r][i].expert == routes[r][group.first].expert,
                              "expert batch 内の expert が一致しません");
                    next += group.count;
                }
                check(next == int(original.size()), "expert batch が全 route を覆っていません");
                for (size_t i = 0; i < original.size(); ++i)
                    check(routes[r][undo[i]].row == original[i].row &&
                              routes[r][undo[i]].expert == original[i].expert &&
                              routes[r][undo[i]].weight == original[i].weight,
                          "逆写像が一致しません");
            }

        }
    }
    {
        const int rows = 1024;
        std::array<int, 512> owners{};
        std::vector<Hy4Route> host;
        for (int row = 0; row < rows; ++row)
            for (int expert = 0; expert < 16; ++expert)
                host.push_back({row, expert, 1.f});
        std::array<std::vector<Hy4Route>, 4> routes;
        std::array<std::vector<int>, 4> offsets;
        partition_routes(host, rows, owners, 3, routes, offsets);
        std::vector<int> undo;
        std::vector<Hy4ExpertBatch> groups;
        group_routes(routes[0], undo, groups);
        check(routes[0].size() == 16384 && groups.size() == 1024,
              "全 route が 1 rank に集中した場合の分割が不正です");
        for (const auto &group : groups)
            check(group.count == 16, "16 行を超える expert が分割されません");
    }
    std::array<int, 512> owners{};
    std::vector<Hy4Route> valid;
    for (int i = 0; i < 16; ++i)
        valid.push_back({0, i, 1.f});
    auto reject = [&](std::vector<Hy4Route> host, std::array<int, 512> placement) {
        std::array<std::vector<Hy4Route>, 4> routes;
        std::array<std::vector<int>, 4> offsets;
        try {
            partition_routes(host, 1, placement, 3, routes, offsets);
            return false;
        } catch (const std::runtime_error &) {
            return true;
        }
    };
    for (int invalid : {-1, 512}) {
        auto host = valid;
        host[0].expert = invalid;
        check(reject(host, owners), "範囲外 expert が拒否されません");
    }
    auto host = valid;
    host[0].row = 1;
    check(reject(host, owners), "不正な行が拒否されません");
    host = valid;
    host[0].expert = host[1].expert;
    check(reject(host, owners), "重複 expert が拒否されません");
    host = valid;
    host[0].weight = std::numeric_limits<float>::quiet_NaN();
    check(reject(host, owners), "非有限の重みが拒否されません");
    for (int invalid : {-1, 3}) {
        auto placement = owners;
        placement[0] = invalid;
        check(reject(valid, placement), "範囲外 rank が拒否されません");
    }
    {
        std::array<int, 512> placement{};
        for (int e = 0; e < 512; ++e)
            placement[e] = e % 3;
        std::vector<Hy4Route> rows;
        for (int row = 0; row < 24; ++row)
            for (int j = 0; j < 16; ++j)
                rows.push_back({row, (row * 16 + j) % 512, 1.f});
        for (int rank = 0; rank < 3; ++rank) {
            std::vector<Hy4ExpertProjection> table(512 * 3);
            for (int e = 0; e < 512; ++e)
                if (placement[e] == rank)
                    for (int kind = 0; kind < 3; ++kind)
                        table[size_t(e) * 3 + kind] = {reinterpret_cast<const uint16_t *>(0x10),
                                                        reinterpret_cast<const uint16_t *>(0x20),
                                                        reinterpret_cast<const uint32_t *>(0x30), 4};
            validate_expert_table(table, placement, rank);
            std::array<std::vector<Hy4Route>, 4> routes;
            std::array<std::vector<int>, 4> offsets;
            partition_routes(rows, 24, placement, 3, routes, offsets);
            for (const auto &route : routes[rank])
                for (int kind = 0; kind < 3; ++kind) {
                    const auto &entry = table[size_t(route.expert) * 3 + kind];
                    check(entry.suh && entry.svh && entry.trellis,
                          "rank に振られた経路の射影表エントリが空です");
                }
            auto broken = table;
            for (int e = 0; e < 512; ++e)
                if (placement[e] == rank) {
                    broken[size_t(e) * 3 + 1].svh = nullptr;
                    break;
                }
            bool threw = false;
            try {
                validate_expert_table(broken, placement, rank);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw, "所有 expert の svh 欠落が検出されません");
            auto stranger = table;
            for (int e = 0; e < 512; ++e)
                if (placement[e] != rank) {
                    stranger[size_t(e) * 3] = {reinterpret_cast<const uint16_t *>(0x10),
                                               reinterpret_cast<const uint16_t *>(0x20),
                                               reinterpret_cast<const uint32_t *>(0x30), 4};
                    break;
                }
            threw = false;
            try {
                validate_expert_table(stranger, placement, rank);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw, "他 rank の expert エントリが検出されません");
        }
        std::vector<Hy4ExpertProjection> short_table(512 * 3 - 1);
        bool threw = false;
        try {
            validate_expert_table(short_table, placement, 0);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "射影表の長さ違反が検出されません");
    }
    return failures ? 1 : 0;
}
