#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/routes.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace shelf::hy4 {
namespace {
size_t align(size_t n) { return (n + 255) & ~size_t(255); }
void check(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
}
}
void *Region::next(size_t length) {
    size_t start = align(cursor);
    if (start > size || length > size - start)
        throw std::runtime_error("作業域の容量が不足しています");
    cursor = start + length;
    return base + start;
}
Residency::Residency(Checkpoint &model, size_t tokens) : model(model), tokens(tokens) {
    for (const auto &[name, dims] : model.shape)
        if (name.ends_with(".self_attn.kv_b_proj"))
            folded[name] = nullptr;
    if (!tokens || tokens > 2097152)
        throw std::runtime_error("KV の予約長が範囲外です");
}
Residency::~Residency() noexcept {
    try {
        shutdown();
    } catch (const std::exception &e) {
        fprintf(stderr, "Hy4 の解放エラー: %s\n", e.what());
    }
}
Status Residency::reserve() {
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count < 1 || count > 4)
        throw std::runtime_error("R9700 を 1〜4 台指定してください");
    ranks = count;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, device));
        if (!std::string_view(props.gcnArchName).starts_with("gfx1201"))
            throw std::runtime_error("gfx1201 の R9700 のみ指定してください");
        physical[device] = device;
    }
    std::array<size_t, 4> available{}, overhead{};
    std::array<int, 4> used{};
    std::array<int64_t, 4> remaining{};
    for (const auto &row : model.owners)
        for (int r : row) {
            if (r < 0 || r >= ranks)
                throw std::runtime_error("配置表の rank が指定 GPU 数を超えています");
            ++used[r];
        }
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        size_t total = 0;
        check(hipMemGetInfo(&available[r], &total));
        // 空きを使い切ると arena の hipMalloc が丸めや他プロセスの揺れで std::bad_alloc になるため、余裕を残す。
        // 実測では起動後の追加確保は無く、必要なのは数 MiB（余裕 0.29 MiB で 21 回中 1 回失敗、1.7 MiB 以上は成功）。
        // 失敗が起きるたびに、この値を 2 倍にすること（8 → 16 → 32 → ... MiB）。
        available[r] -= std::min(available[r], size_t(8) << 20);
        overhead[r] = (r ? size_t(512) << 20 : size_t(2) << 30) + matrix_bytes;
        if (!r)
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                overhead[r] += align(294 * (sparse ? tokens : std::min(tokens, size_t(518))));
                if (sparse)
                    overhead[r] += align((tokens / 4 + 12) * 256);
            }
        if (!r)
            for (const auto &[name, ignored] : folded)
                overhead[0] += size_t(model.shape.at(name)[1]) * 512 * 2;
        remaining[r] = int64_t(available[r]) - int64_t(align(model.bytes[r]) + overhead[r]);
    }
    if (std::accumulate(remaining.begin(), remaining.begin() + ranks, int64_t{}) < 0)
        return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    auto plan = model.owners;
    int moved = 0;
    for (int expert = 511; expert >= 0; --expert)
        for (int layer = 0; layer < 41; ++layer) {
            int from = plan[layer][expert];
            if (remaining[from] >= 0)
                continue;
            int64_t bytes = 3 * (1536ll * 1536 * model.expert_bits[layer][expert] / 8 + 1536 * 4);
            int dest = std::max_element(remaining.begin(), remaining.begin() + ranks) - remaining.begin();
            if (remaining[dest] < bytes)
                continue;
            plan[layer][expert] = dest;
            remaining[from] += bytes;
            remaining[dest] -= bytes;
            --used[from];
            ++used[dest];
            ++moved;
        }
    if (*std::min_element(remaining.begin(), remaining.begin() + ranks) < 0)
        return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    if (moved)
        model.relocate(plan);
    fprintf(stderr, "relocated_experts=%d rank_counts=%d,%d,%d,%d\n", moved, used[0], used[1], used[2], used[3]);
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        size_t free, total;
        check(hipMemGetInfo(&free, &total));
        size_t work = r ? size_t(512) << 20 : size_t(2) << 30;
        size_t bytes = align(model.bytes[r]) + overhead[r];
        fprintf(stderr, "rank=%d weight=%zu workspace=%zu request=%zu free=%zu total=%zu\n", r, model.bytes[r],
                work, bytes, free, total);
        auto result =
            gpu::GpuArena::create(physical[r], bytes);
        if (!result.ok()) {
            if (result.status().code() == Status::Code::insufficient_memory)
                return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
            throw std::runtime_error(result.status().message());
        }
        arenas.push_back(result.release());
        auto allocation = arenas.back().allocate_aligned(bytes, 256);
        if (!allocation.ok())
            throw std::runtime_error(allocation.status().message());
        Region buffer{static_cast<uint8_t *>(allocation.value().data()), bytes};
        weight_base[r] = static_cast<uint8_t *>(buffer.next(model.bytes[r]));
        if (!r)
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                cache[i] = static_cast<uint8_t *>(
                    buffer.next(294 * (sparse ? tokens : std::min(tokens, size_t(518)))));
                if (sparse)
                    index_keys[i] = static_cast<uint16_t *>(buffer.next((tokens / 4 + 12) * 256));
            }
        if (!r)
            for (auto &[name, ptr] : folded)
                ptr = buffer.next(size_t(model.shape.at(name)[1]) * 512 * 2);
        matrices[r] = static_cast<Hy4ExpertProjection *>(buffer.next(matrix_bytes));
        workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        check(hipMemGetInfo(&free, &total));
        fprintf(stderr, "rank=%d committed=%zu free_after=%zu\n", r, arenas.back().committed(), free);
    }
    check(hipSetDevice(physical[0]));
    return Status::make_ok();
}
void Residency::upload_experts(int r, int layer) {
    std::vector<Hy4ExpertProjection> table(512 * 3);
    for (int e = 0; e < 512; ++e)
        if (model.owners[layer - 1][e] == r) {
            std::string key =
                "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
            int kind = 0;
            for (auto suffix : {"gate_proj", "up_proj", "down_proj"}) {
                auto name = key + suffix;
                table[e * 3 + kind++] = {static_cast<const uint16_t *>(device(model.at(name + ".suh"))),
                                         static_cast<const uint16_t *>(device(model.at(name + ".svh"))),
                                         static_cast<const uint32_t *>(device(model.at(name + ".trellis"))),
                                         model.at(name + ".trellis").spec.shape[2] / 16};
            }
        }
    validate_expert_table(table, model.owners[layer - 1], r);
    check(hipMemcpy(matrices[r] + (layer - 1) * 1536, table.data(), table.size() * sizeof(table[0]),
                    hipMemcpyHostToDevice));
}
void Residency::upload() {
    auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
    // 診断用: SHELF_HY4_EP_TRANSPORT=host で rank 間の転送をすべて host 経由にする（peer 転送の切り分け）。
    auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const bool force_host = transport && std::string_view(transport) == "host";
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src) {
            if (src == dst || (dst && src))
                continue;
            int accessible = 0;
            check(hipSetDevice(physical[dst]));
            check(hipDeviceCanAccessPeer(&accessible, physical[dst], physical[src]));
            if (accessible && !force_host) {
                auto status = hipDeviceEnablePeerAccess(physical[src], 0);
                peer[dst][src] = status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled;
                owns_peer[dst][src] = status == hipSuccess;
                if (status != hipSuccess)
                    (void)hipGetLastError();
            }
            if (peer[dst][src]) {
                check(hipSetDevice(physical[src]));
                check(
                    hipEventCreateWithFlags(&ready[src][dst], hipEventDisableTiming | hipEventReleaseToSystem));
            }
            if (!peer[dst][src] && !staging)
                check(hipHostMalloc(&staging, max_prefill_chunk * 1536u * 2u, hipHostMallocPortable));
            fprintf(stderr, "transfer=%d->%d method=%s\n", src, dst, peer[dst][src] ? "peer" : "host");
        }
    const auto begin = std::chrono::steady_clock::now();
    constexpr size_t chunk = 16ull << 20;
    void *buffer = nullptr;
    check(hipHostMalloc(&buffer, ranks * chunk, hipHostMallocPortable));
    try {
        std::memset(buffer, 0, ranks * chunk);
        std::array<size_t, 4> base{}, used{};
        auto flush = [&](int r) {
            if (!used[r])
                return;
            check(hipSetDevice(physical[r]));
            check(hipMemcpy(weight_base[r] + base[r], static_cast<uint8_t *>(buffer) + r * chunk, used[r],
                            hipMemcpyHostToDevice));
            used[r] = 0;
        };
        for (const auto *w : model.upload_order) {
            const int r = w->rank;
            for (size_t n = 0; n < w->bytes;) {
                size_t pos = w->offset + n, first = pos / chunk * chunk;
                if (first != base[r]) {
                    flush(r);
                    base[r] = first;
                }
                size_t local = pos - first, bytes = std::min(chunk - local, w->bytes - n);
                std::memcpy(static_cast<uint8_t *>(buffer) + r * chunk + local,
                            static_cast<const uint8_t *>(w->host) + n, bytes);
                used[r] = local + bytes;
                n += bytes;
            }
        }
        for (int r = 0; r < ranks; ++r)
            flush(r);
    } catch (...) {
        check(hipHostFree(buffer));
        throw;
    }
    check(hipHostFree(buffer));
    fprintf(stderr, "weight_upload_s=%.3f\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        if (profile && std::string_view(profile) == "1") {
            check(hipEventCreate(&expert_begin[r]));
            check(hipEventCreate(&expert_end[r]));
        }
        for (int layer = 1; layer < 42; ++layer)
            upload_experts(r, layer);
    }
    check(hipSetDevice(physical[0]));
    if (!folded.empty())
        fold_kv_weights(*this);
}
void Residency::transfer(void *dst, int destination, const void *src, int source, size_t bytes) {
    transfer_bytes[source][destination] += bytes;
    std::chrono::steady_clock::time_point begin;
    if (expert_begin[0])
        begin = std::chrono::steady_clock::now();
    if (peer[destination][source]) {
        check(hipSetDevice(physical[source]));
        check(hipEventRecord(ready[source][destination], nullptr));
        check(hipSetDevice(physical[destination]));
        check(hipStreamWaitEvent(nullptr, ready[source][destination], 0));
        check(hipMemcpyPeerAsync(dst, physical[destination], src, physical[source], bytes, nullptr));
    } else {
        if (!staging || bytes > max_prefill_chunk * 1536u * 2u)
            throw std::runtime_error("転送作業域の範囲外です");
        check(hipSetDevice(physical[source]));
        check(hipMemcpy(staging, src, bytes, hipMemcpyDeviceToHost));
        check(hipSetDevice(physical[destination]));
        check(hipMemcpy(dst, staging, bytes, hipMemcpyHostToDevice));
    }
    if (expert_begin[0])
        transfer_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
void *Residency::route_buffer(int rank, size_t bytes) {
    constexpr size_t capacity =
        max_prefill_chunk * (16 * (sizeof(Hy4Route) + sizeof(Hy4ExpertBatch) + sizeof(int)) + sizeof(int)) +
        sizeof(int);
    if (bytes > capacity)
        throw std::runtime_error("routing 情報が作業域の上限を超えました");
    if (!route_host[rank])
        check(hipHostMalloc(&route_host[rank], capacity, hipHostMallocPortable));
    return route_host[rank];
}
void Residency::shutdown() {
    std::string errors;
    for (auto it = arenas.rbegin(); it != arenas.rend(); ++it) {
        auto status = it->shutdown();
        if (!status.ok())
            errors += status.message() + "\n";
    }
    arenas.clear();
    auto collect = [&](hipError_t e) {
        if (e != hipSuccess)
            errors += std::string(hipGetErrorString(e)) + "\n";
    };
    for (int r = 0; r < ranks; ++r)
        if (expert_begin[r] || expert_end[r]) {
            auto status = hipSetDevice(physical[r]);
            collect(status);
            if (status == hipSuccess) {
                if (expert_begin[r])
                    collect(hipEventDestroy(expert_begin[r]));
                if (expert_end[r])
                    collect(hipEventDestroy(expert_end[r]));
            }
            expert_begin[r] = expert_end[r] = nullptr;
        }
    for (int r = 0; r < ranks; ++r) {
        if (route_host[r]) {
            collect(hipHostFree(route_host[r]));
            route_host[r] = nullptr;
        }
        for (auto &event : ready[r])
            if (event) {
                auto status = hipSetDevice(physical[r]);
                collect(status);
                if (status == hipSuccess)
                    collect(hipEventDestroy(event));
                event = nullptr;
            }
    }
    if (staging) {
        collect(hipHostFree(staging));
        staging = nullptr;
    }
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src)
            if (owns_peer[dst][src]) {
                auto e = hipSetDevice(physical[dst]);
                collect(e);
                if (e == hipSuccess)
                    collect(hipDeviceDisablePeerAccess(physical[src]));
                owns_peer[dst][src] = false;
            }
    if (!errors.empty())
        throw std::runtime_error(errors);
}
const void *Residency::device(const Weight &w) const {
    if (w.rank < 0)
        throw std::runtime_error("ホスト常駐 tensor です");
    return weight_base.at(w.rank) + w.offset;
}
}
