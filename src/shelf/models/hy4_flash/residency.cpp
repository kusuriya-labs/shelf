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
    ep_size = model.ep;
    tp_size = model.tp;
    for (int r = 0; r < model.ranks; ++r)
        for (const auto &[name, dims] : model.shape)
            if (name.ends_with(".self_attn.kv_b_proj"))
                folded[r][name] = nullptr;
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
    if (count != model.ranks)
        throw std::runtime_error("EP × TP と GPU 数が一致しません");
    ranks = count;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, device));
        if (!std::string_view(props.gcnArchName).starts_with("gfx1201"))
            throw std::runtime_error("gfx1201 の R9700 のみ指定してください");
        physical[device] = device;
    }
    std::array<size_t, 4> available{}, overhead{};
    std::array<int64_t, 4> remaining{};
    std::array<int, 4> used{};
    for (const auto &row : model.owners)
        for (int r : row) {
            if (r < 0 || r >= ep_size)
                throw std::runtime_error("配置表の group が EP 数を超えています");
            ++used[r];
        }
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        size_t total = 0;
        check(hipMemGetInfo(&available[r], &total));
        available[r] -= std::min(available[r], size_t(8) << 20);
        const bool attention_rank = tp_size > 1 || r == 0;
        overhead[r] = (tp_size > 1 ? size_t(1) << 30 : (r == 0 ? size_t(2) << 30 : size_t(512) << 20)) + matrix_bytes;
        if (attention_rank) {
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                overhead[r] += align(294 * (sparse ? tokens : std::min(tokens, size_t(518))));
                if (sparse)
                    overhead[r] += align((tokens / 4 + 12) * 256);
            }
            for (const auto &[name, ignored] : folded[r])
                overhead[r] += size_t(model.shape_of(name, r)->at(1)) * 512 * 2;
        }
        if (tp_size > 1)
            overhead[r] += max_prefill_chunk * 3072u * 2u;
        if (tp_size > 1 && ep_size > 1)
            overhead[r] += max_prefill_chunk * 1536u * 2u;
        remaining[r] = int64_t(available[r]) - int64_t(align(model.bytes[r]) + overhead[r]);
    }
    if (*std::min_element(remaining.begin(), remaining.begin() + ranks) < 0)
        return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    fprintf(stderr, "rank_counts=%d,%d,%d,%d\n", used[0], used[1], used[2], used[3]);
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        size_t free, total;
        check(hipMemGetInfo(&free, &total));
        const bool attention_rank = tp_size > 1 || r == 0;
        size_t work = tp_size > 1 ? size_t(1) << 30 : (r == 0 ? size_t(2) << 30 : size_t(512) << 20);
        size_t bytes = align(model.bytes[r]) + overhead[r];
        fprintf(stderr, "rank=%d weight=%zu workspace=%zu request=%zu free=%zu total=%zu\n", r, model.bytes[r],
                work, bytes, free, total);
        auto result = gpu::GpuArena::create(physical[r], bytes);
        if (!result.ok()) {
            if (result.status().code() == Status::Code::insufficient_memory)
                return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
            throw std::runtime_error(result.status().message());
        }
        arenas.push_back(result.release());
        auto allocation = arenas.back().allocate_aligned(bytes, 256);
        if (!allocation.ok())
            throw std::runtime_error(allocation.status().message());
        check(hipMemset(allocation.value().data(), 0, bytes));
        Region buffer{static_cast<uint8_t *>(allocation.value().data()), bytes};
        weight_base[r] = static_cast<uint8_t *>(buffer.next(model.bytes[r]));
        if (attention_rank) {
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                cache[r][i] = static_cast<uint8_t *>(
                    buffer.next(294 * (sparse ? tokens : std::min(tokens, size_t(518)))));
                if (sparse)
                    index_keys[r][i] = static_cast<uint16_t *>(buffer.next((tokens / 4 + 12) * 256));
            }
            for (auto &[name, ptr] : folded[r])
                ptr = buffer.next(size_t(model.shape_of(name, r)->at(1)) * 512 * 2);
        }
        if (tp_size > 1)
            tp_recv[r] = buffer.next(max_prefill_chunk * 3072u * 2u);
        if (tp_size > 1 && ep_size > 1)
            ep_recv[r] = buffer.next(max_prefill_chunk * 1536u * 2u);
        matrices[r] = static_cast<Hy4ExpertProjection *>(buffer.next(matrix_bytes));
        workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        check(hipMemGetInfo(&free, &total));
        fprintf(stderr, "rank=%d committed=%zu free_after=%zu\n", r, arenas.back().committed(), free);
    }
    check(hipSetDevice(physical[0]));
    if (tp_size > 1)
        for (int g = 0; g < ep_size; ++g)
            tp_barrier[g] = std::make_unique<std::barrier<>>(tp_size);
    if (tp_size > 1 && ep_size > 1)
        for (int t = 0; t < tp_size; ++t)
            ep_barrier[t] = std::make_unique<std::barrier<>>(2);
    return Status::make_ok();
}
void Residency::upload_experts(int r, int layer) {
    std::vector<Hy4ExpertProjection> table(512 * 3);
    const int group = r / tp_size;
    for (int e = 0; e < 512; ++e)
        if (model.owners[layer - 1][e] == group) {
            std::string key =
                "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
            int kind = 0;
            for (auto suffix : {"gate_proj", "up_proj", "down_proj"}) {
                auto name = key + suffix;
                const auto *shape = model.shape_of(name, r);
                table[e * 3 + kind++] = {static_cast<const uint16_t *>(device(model.at(name + ".suh", r))),
                                         static_cast<const uint16_t *>(device(model.at(name + ".svh", r))),
                                         static_cast<const uint32_t *>(device(model.at(name + ".trellis", r))),
                                         model.at(name + ".trellis", r).spec.shape[2] / 16,
                                         shape ? shape->at(0) : 1536, shape ? shape->at(1) : 1536};
            }
        }
    validate_expert_table(table, model.owners[layer - 1], group);
    check(hipMemcpy(matrices[r] + (layer - 1) * 1536, table.data(), table.size() * sizeof(table[0]),
                    hipMemcpyHostToDevice));
}
void Residency::upload() {
    auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
    auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const std::string_view mode = transport ? transport : "";
    const bool force_host = mode == "host";
    push = mode != "host" && mode != "peer";
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src) {
            if (src == dst)
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
                for (int q = 0; q < 2; ++q) {
                    check(hipSetDevice(physical[src]));
                    check(hipEventCreateWithFlags(&ready[src][dst][q],
                                                  hipEventDisableTiming | hipEventReleaseToSystem));
                    check(hipSetDevice(physical[dst]));
                    check(hipEventCreateWithFlags(&ack[dst][src][q], hipEventDisableTiming));
                }
            }
            if (!peer[dst][src] && !staging)
                check(hipHostMalloc(&staging, max_prefill_chunk * 3072u * 2u, hipHostMallocPortable));
            fprintf(stderr, "transfer=%d->%d method=%s\n", src, dst,
                    peer[dst][src] ? (push ? "push" : "peer") : "host");
        }
    const auto begin = std::chrono::steady_clock::now();
    constexpr size_t chunk = 16ull << 20;
    void *buffer = nullptr;
    check(hipHostMalloc(&buffer, ranks * chunk, hipHostMallocPortable));
    try {
        for (int r = 0; r < ranks; ++r) {
            check(hipSetDevice(physical[r]));
            std::memset(buffer, 0, chunk);
            size_t base = 0, used = 0;
            auto flush = [&] {
                if (used) {
                    check(hipMemcpy(weight_base[r] + base, buffer, used, hipMemcpyHostToDevice));
                    used = 0;
                }
            };
            for (const auto *w : model.upload_order[r]) {
                for (size_t n = 0; n < w->bytes;) {
                    size_t pos = w->offset + n, first = pos / chunk * chunk;
                    if (first != base) {
                        flush();
                        base = first;
                    }
                    size_t local = pos - first, bytes = std::min(chunk - local, w->bytes - n);
                    std::memcpy(static_cast<uint8_t *>(buffer) + local,
                                static_cast<const uint8_t *>(w->host) + n, bytes);
                    used = local + bytes;
                    n += bytes;
                }
            }
            flush();
        }
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
        if (tp_size > 1) {
            check(hipEventCreateWithFlags(&lane_ready[r], hipEventDisableTiming));
            check(hipEventCreateWithFlags(&lane_done[r], hipEventDisableTiming));
            check(hipEventCreateWithFlags(&tp_consumed[r], hipEventDisableTiming | hipEventReleaseToSystem));
            if (ep_size > 1)
                check(hipEventCreateWithFlags(&ep_consumed[r], hipEventDisableTiming | hipEventReleaseToSystem));
        }
        for (int layer = 1; layer < 42; ++layer)
            upload_experts(r, layer);
        fold_kv_weights(*this, r);
    }
    check(hipSetDevice(physical[0]));
}
void Residency::transfer(void *dst, int destination, const void *src, int source, size_t bytes) {
    transfer_bytes[source][destination] += bytes;
    std::chrono::steady_clock::time_point begin;
    if (expert_begin[0])
        begin = std::chrono::steady_clock::now();
    if (peer[destination][source]) {
        const int p = parity[source][destination];
        hipEvent_t ready_event = ready[source][destination][p];
        hipEvent_t ack_event = ack[destination][source][p];
        parity[source][destination] = p ^ 1;
        if (push) {
            check(hipSetDevice(physical[destination]));
            check(hipEventRecord(ack_event, nullptr));
            check(hipSetDevice(physical[source]));
            check(hipStreamWaitEvent(nullptr, ack_event, 0));
            peer_copy(dst, src, bytes);
            check(hipEventRecord(ready_event, nullptr));
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(nullptr, ready_event, 0));
        } else {
            check(hipSetDevice(physical[source]));
            check(hipEventRecord(ready_event, nullptr));
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(nullptr, ready_event, 0));
            check(hipMemcpyPeerAsync(dst, physical[destination], src, physical[source], bytes, nullptr));
        }
    } else {
        if (!staging || bytes > max_prefill_chunk * 3072u * 2u)
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
void Residency::tp_sum(void *data, int rank, size_t bytes) {
    if (tp_size == 1)
        return;
    const int group = rank / tp_size;
    tp_ptr[rank] = data;
    tp_bytes[rank] = bytes;
    check(hipSetDevice(physical[rank]));
    check(hipEventRecord(lane_ready[rank], nullptr));
    tp_barrier[group]->arrive_and_wait();
    if (rank == group * tp_size) {
        const int a = group * tp_size;
        for (int t = 1; t < tp_size; ++t) {
            const int b = a + t;
            check(hipSetDevice(physical[b]));
            check(hipStreamWaitEvent(nullptr, lane_ready[b], 0));
            check(hipStreamWaitEvent(nullptr, tp_consumed[a], 0));
            peer_copy(tp_recv[a], tp_ptr[b], bytes);
            check(hipEventRecord(lane_done[b], nullptr));
            check(hipSetDevice(physical[a]));
            check(hipStreamWaitEvent(nullptr, lane_ready[a], 0));
            check(hipStreamWaitEvent(nullptr, tp_consumed[b], 0));
            peer_copy(tp_recv[b], tp_ptr[a], bytes);
            check(hipEventRecord(lane_done[a], nullptr));
            check(hipSetDevice(physical[a]));
            check(hipStreamWaitEvent(nullptr, lane_done[b], 0));
            tp_add_bf16(tp_ptr[a], tp_recv[a], bytes);
            check(hipEventRecord(tp_consumed[a], nullptr));
            check(hipSetDevice(physical[b]));
            check(hipStreamWaitEvent(nullptr, lane_done[a], 0));
            tp_add_bf16(tp_ptr[b], tp_recv[b], bytes);
            check(hipEventRecord(tp_consumed[b], nullptr));
        }
    }
    tp_barrier[group]->arrive_and_wait();
    check(hipSetDevice(physical[rank]));
}
void Residency::ep_reduce(void *data, int rank, size_t bytes) {
    if (ep_size == 1)
        return;
    const int lane = rank % tp_size;
    const int a = lane, b = lane + tp_size;
    tp_ptr[rank] = data;
    check(hipSetDevice(physical[rank]));
    check(hipEventRecord(lane_ready[rank], nullptr));
    ep_barrier[lane]->arrive_and_wait();
    if (rank == a) {
        check(hipSetDevice(physical[b]));
        check(hipStreamWaitEvent(nullptr, lane_ready[b], 0));
        check(hipStreamWaitEvent(nullptr, ep_consumed[a], 0));
        peer_copy(ep_recv[a], tp_ptr[b], bytes);
        check(hipEventRecord(lane_done[b], nullptr));
        check(hipSetDevice(physical[a]));
        check(hipStreamWaitEvent(nullptr, lane_ready[a], 0));
        check(hipStreamWaitEvent(nullptr, ep_consumed[b], 0));
        peer_copy(ep_recv[b], tp_ptr[a], bytes);
        check(hipEventRecord(lane_done[a], nullptr));
        check(hipSetDevice(physical[a]));
        check(hipStreamWaitEvent(nullptr, lane_done[b], 0));
        tp_add_bf16(tp_ptr[a], ep_recv[a], bytes);
        check(hipEventRecord(ep_consumed[a], nullptr));
        check(hipSetDevice(physical[b]));
        check(hipStreamWaitEvent(nullptr, lane_done[a], 0));
        tp_add_bf16(tp_ptr[b], ep_recv[b], bytes);
        check(hipEventRecord(ep_consumed[b], nullptr));
    }
    ep_barrier[lane]->arrive_and_wait();
    check(hipSetDevice(physical[rank]));
}
void *Residency::route_buffer(int rank, size_t bytes) {
    constexpr size_t capacity =
        max_prefill_chunk * (16 * (sizeof(Hy4Route) + sizeof(Hy4ExpertBatch) + sizeof(int)) + sizeof(int)) +
        sizeof(int);
    if (bytes > capacity)
        throw std::runtime_error("routing 情報が作業域の上限を超えました");
    if (!route_host[rank]) {
        check(hipHostMalloc(&route_host[rank], capacity, hipHostMallocPortable));
        std::memset(route_host[rank], 0, capacity);
    }
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
        if (lane_ready[r] || lane_done[r] || tp_consumed[r] || ep_consumed[r]) {
            auto status = hipSetDevice(physical[r]);
            collect(status);
            if (status == hipSuccess) {
                if (lane_ready[r])
                    collect(hipEventDestroy(lane_ready[r]));
                if (lane_done[r])
                    collect(hipEventDestroy(lane_done[r]));
                if (tp_consumed[r])
                    collect(hipEventDestroy(tp_consumed[r]));
                if (ep_consumed[r])
                    collect(hipEventDestroy(ep_consumed[r]));
            }
            lane_ready[r] = lane_done[r] = tp_consumed[r] = ep_consumed[r] = nullptr;
        }
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
        for (auto &row : ready[r])
            for (auto &event : row)
                if (event) {
                    auto status = hipSetDevice(physical[r]);
                    collect(status);
                    if (status == hipSuccess)
                        collect(hipEventDestroy(event));
                    event = nullptr;
                }
        for (auto &row : ack[r])
            for (auto &event : row)
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
