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
Residency::Residency(Checkpoint &model, size_t tokens, int slots)
    : model(model), tokens(tokens), max_slots(slots), batch_draft(false) {
    if (slots < 1 || slots > 16)
        throw std::invalid_argument("スロット数は 1〜16 で指定してください");
    const char *draft = std::getenv("SHELF_HY4_BATCH_DFLY");
    batch_draft = slots > 1 && model.has_draft && (!draft || std::string_view(draft) != "0");
    if (batch_draft)
        sparse_last.resize(slots);
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
Status Residency::check_batch_topology(int tp_size, int slots) {
    if (tp_size > 1 && slots > 1)
        return Status::unsupported("TP の同時リクエスト数は 1 です", __FILE__, __LINE__);
    return Status::make_ok();
}
Status Residency::reserve() {
    auto topology = check_batch_topology(tp_size, max_slots);
    if (!topology.ok())
        return topology;
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count < 1 || count > 4)
        throw std::runtime_error("R9700 を 1〜4 台指定してください");
    if (count != model.ranks)
        throw std::runtime_error("EP × TP と GPU 数が一致しません");
    ranks = count;
    const auto vmm = std::getenv("SHELF_ARENA_VMM");
    if (!multi_gpu_vmm_allowed(ranks, vmm ? vmm : ""))
        throw std::runtime_error("Hy4 の複数 GPU 実行では SHELF_ARENA_VMM=1 を使用できません");
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
            overhead[r] += align(41 * 512 * sizeof(int)) + 256;
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                overhead[r] += align(max_slots * 294 * (sparse ? tokens : std::min(tokens, size_t(518))));
                if (sparse)
                    overhead[r] += align((tokens / 4 + 12) * 256 * max_slots);
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
    graph_room = max_slots == 1 && tp_size == 1 && std::all_of(remaining.begin(), remaining.begin() + ranks,
                                            [](int64_t bytes) { return bytes >= int64_t(graph_headroom); });
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
        check(hipStreamCreateWithFlags(&compute[r], hipStreamNonBlocking));
        weight_base[r] = static_cast<uint8_t *>(buffer.next(model.bytes[r]));
        if (attention_rank) {
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                cache[r][i] = static_cast<uint8_t *>(
                    buffer.next(max_slots * 294 * (sparse ? tokens : std::min(tokens, size_t(518)))));
                if (sparse)
                    index_keys[r][i] = static_cast<uint16_t *>(buffer.next((tokens / 4 + 12) * 256 * max_slots));
            }
            for (auto &[name, ptr] : folded[r])
                ptr = buffer.next(size_t(model.shape_of(name, r)->at(1)) * 512 * 2);
        }
        if (tp_size > 1)
            tp_recv[r] = buffer.next(max_prefill_chunk * 3072u * 2u);
        if (tp_size > 1 && ep_size > 1)
            ep_recv[r] = buffer.next(max_prefill_chunk * 1536u * 2u);
        reserve_experts(r, buffer);
        workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        check(hipMemGetInfo(&free, &total));
        fprintf(stderr, "rank=%d committed=%zu free_after=%zu\n", r, arenas.back().committed(), free);
    }
    check(hipSetDevice(physical[0]));
    if (tp_size > 1)
        for (int g = 0; g < ep_size; ++g)
            tp_barrier[g] = std::make_unique<SyncBarrier>(tp_size);
    if (tp_size > 1 && ep_size > 1)
        for (int t = 0; t < tp_size; ++t)
            ep_barrier[t] = std::make_unique<SyncBarrier>(2);
    if (tp_size > 1)
        all_barrier = std::make_unique<SyncBarrier>(ranks);
    return Status::make_ok();
}
void Residency::reserve_experts(int r, Region &buffer) {
    matrices[r] = static_cast<Hy4ExpertProjection *>(buffer.next(matrix_bytes));
    if (tp_size == 1 && r != 0)
        return;
    moe_owners[r] = static_cast<int *>(buffer.next(41 * 512 * sizeof(int)));
    moe_counts[r] = static_cast<uint64_t *>(buffer.next(4 * sizeof(uint64_t)));
    check(hipMemset(moe_counts[r], 0, 4 * sizeof(uint64_t)));
    check(hipStreamCreateWithFlags(&local_expert[r], hipStreamNonBlocking));
    check(hipEventCreateWithFlags(&local_ready[r], hipEventDisableTiming));
    check(hipEventCreateWithFlags(&local_done[r], hipEventDisableTiming));
}

void Residency::collect_route_counts(int rank) {
    std::array<uint64_t, 4> totals{};
    check(hipMemcpyAsync(totals.data(), moe_counts[rank], sizeof(totals), hipMemcpyDeviceToHost, compute[rank]));
    check(hipMemsetAsync(moe_counts[rank], 0, sizeof(totals), compute[rank]));
    check(hipStreamSynchronize(compute[rank]));
    if (tp_size > 1)
        routed_rows[rank] += totals[rank / tp_size];
    else
        for (int r = 0; r < ep_size; ++r)
            routed_rows[r] += totals[r];
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
    if (moe_owners[r])
        check(hipMemcpy(moe_owners[r] + (layer - 1) * 512, model.owners[layer - 1].data(),
                        512 * sizeof(int), hipMemcpyHostToDevice));
    validate_expert_table(table, model.owners[layer - 1], group);
    check(hipMemcpy(matrices[r] + (layer - 1) * 1536, table.data(), table.size() * sizeof(table[0]),
                    hipMemcpyHostToDevice));
}
void Residency::upload() {
    const auto vmm = std::getenv("SHELF_ARENA_VMM");
    if (!multi_gpu_vmm_allowed(ranks, vmm ? vmm : ""))
        throw std::runtime_error("Hy4 の複数 GPU 実行では SHELF_ARENA_VMM=1 を使用できません");
    auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
    auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const std::string_view mode = transport ? transport : "";
    const bool force_host = mode == "host";
    push = mode.empty() || mode == "push";
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
        }
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src) {
            if (src == dst)
                continue;
            const bool available = push ? peer[src][dst] : peer[dst][src];
            if (!available && tp_size == 1)
                check(hipHostMalloc(&small_staging[src][dst], transfer_packet_bytes(),
                                    hipHostMallocPortable));
            if (available || small_staging[src][dst]) {
                for (int q = 0; q < 2; ++q) {
                    check(hipSetDevice(physical[src]));
                    check(hipEventCreateWithFlags(&ready[src][dst][q],
                                                  hipEventDisableTiming | hipEventReleaseToSystem));
                    check(hipSetDevice(physical[dst]));
                    check(hipEventCreateWithFlags(&ack[dst][src][q], hipEventDisableTiming | hipEventReleaseToSystem));
                }
            }
            if (!available && !staging)
                check(hipHostMalloc(&staging, max_prefill_chunk * 3072u * 2u, hipHostMallocPortable));
            fprintf(stderr, "transfer=%d->%d method=%s\n", src, dst,
                    available ? (push ? "push" : "peer") : "host");
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
            for (int p = 0; p < 2; ++p) {
                for (auto events : {&tp_ready, &tp_done, &tp_arrived, &tp_consumed})
                    check(hipEventCreateWithFlags(&(*events)[r][p], hipEventDisableTiming | hipEventReleaseToSystem));
                if (ep_size > 1)
                    for (auto events : {&ep_ready, &ep_done, &ep_arrived, &ep_consumed})
                        check(hipEventCreateWithFlags(&(*events)[r][p], hipEventDisableTiming | hipEventReleaseToSystem));
            }
            if (staging)
                check(hipHostMalloc(&reduce_staging[r], max_prefill_chunk * 3072u * 2u, hipHostMallocPortable));
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
    void *host = bytes <= transfer_packet_bytes() ? small_staging[source][destination] : nullptr;
    if (host || (push ? peer[source][destination] : peer[destination][source])) {
        const int p = parity[source][destination];
        hipEvent_t ready_event = ready[source][destination][p];
        hipEvent_t ack_event = ack[destination][source][p];
        const bool source_writes = push && !host;
        check(hipEventSynchronize(source_writes ? ready_event : ack_event));
        parity[source][destination] = p ^ 1;
        if (source_writes) {
            check(hipSetDevice(physical[destination]));
            check(hipEventRecord(ack_event, compute[destination]));
            check(hipSetDevice(physical[source]));
            check(hipStreamWaitEvent(compute[source], ack_event, 0));
            peer_copy(dst, src, bytes, compute[source]);
            check(hipEventRecord(ready_event, compute[source]));
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(compute[destination], ready_event, 0));
        } else {
            check(hipSetDevice(physical[source]));
            if (host)
                check(hipMemcpyAsync(host, src, bytes, hipMemcpyDeviceToHost, compute[source]));
            check(hipEventRecord(ready_event, compute[source]));
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(compute[destination], ready_event, 0));
            if (host)
                check(hipMemcpyAsync(dst, host, bytes, hipMemcpyHostToDevice, compute[destination]));
            else
                peer_copy(dst, src, bytes, compute[destination]);
            check(hipEventRecord(ack_event, compute[destination]));
            check(hipSetDevice(physical[source]));
            check(hipStreamWaitEvent(compute[source], ack_event, 0));
            check(hipSetDevice(physical[destination]));
        }
    } else {
        if (!staging || bytes > max_prefill_chunk * 3072u * 2u)
            throw std::runtime_error("転送作業域の範囲外です");
        check(hipSetDevice(physical[source]));
        check(hipMemcpyAsync(staging, src, bytes, hipMemcpyDeviceToHost, compute[source]));
        check(hipStreamSynchronize(compute[source]));
        check(hipSetDevice(physical[destination]));
        check(hipMemcpyAsync(dst, staging, bytes, hipMemcpyHostToDevice, compute[destination]));
        check(hipStreamSynchronize(compute[destination]));
    }
    if (expert_begin[0])
        transfer_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
void Residency::abort_collectives() {
    if (all_barrier) all_barrier->abort();
    for (auto &barrier : tp_barrier) if (barrier) barrier->abort();
    for (auto &barrier : ep_barrier) if (barrier) barrier->abort();
}
void Residency::reset_collectives() {
    if (all_barrier) all_barrier->reset();
    for (auto &barrier : tp_barrier) if (barrier) barrier->reset();
    for (auto &barrier : ep_barrier) if (barrier) barrier->reset();
}
void Residency::tp_sum(void *data, int rank, size_t bytes) {
    if (tp_size > 1) reduce_pair(data, rank, bytes, false);
}
void Residency::ep_reduce(void *data, int rank, size_t bytes) {
    if (ep_size > 1) reduce_pair(data, rank, bytes, true);
}
void Residency::reduce_pair(void *data, int rank, size_t bytes, bool ep) {
    const int index = ep ? rank % tp_size : rank / tp_size;
    const int a = ep ? index : index * tp_size;
    const int b = ep ? a + tp_size : a + 1;
    auto &barrier = ep ? ep_barrier[index] : tp_barrier[index];
    auto &ready_events = ep ? ep_ready : tp_ready;
    auto &done_events = ep ? ep_done : tp_done;
    auto &arrived_events = ep ? ep_arrived : tp_arrived;
    auto &consumed_events = ep ? ep_consumed : tp_consumed;
    auto &recorded = ep ? ep_recorded : tp_recorded;
    auto &parity = ep ? ep_parity[index] : tp_parity[index];
    auto &recv = ep ? ep_recv : tp_recv;
    const int p = parity;
    check(hipSetDevice(physical[rank]));
    if (recorded[rank][p])
        check(hipEventSynchronize(consumed_events[rank][p]));
    tp_ptr[rank] = data;
    tp_bytes[rank] = bytes;
    check(hipEventRecord(ready_events[rank][p], compute[rank]));
    barrier->arrive_and_wait();
    if (rank == a) {
        if (tp_bytes[a] != tp_bytes[b] || bytes > max_prefill_chunk * (ep ? 1536u : 3072u) * 2u)
            throw std::runtime_error("collective の転送サイズが一致しません");
        for (int source : {a, b}) {
            const int destination = source == a ? b : a;
            auto mode = select_reduce_transport(push, peer[source][destination], peer[destination][source]);
            if (mode == ReduceTransport::push) {
                check(hipSetDevice(physical[source]));
                check(hipStreamWaitEvent(compute[source], ready_events[source][p], 0));
                if (recorded[destination][p ^ 1])
                    check(hipStreamWaitEvent(compute[source], consumed_events[destination][p ^ 1], 0));
                peer_copy(recv[destination], tp_ptr[source], bytes, compute[source]);
                check(hipEventRecord(done_events[source][p], compute[source]));
                check(hipSetDevice(physical[destination]));
                check(hipStreamWaitEvent(compute[destination], done_events[source][p], 0));
            } else if (mode == ReduceTransport::peer) {
                check(hipSetDevice(physical[destination]));
                check(hipStreamWaitEvent(compute[destination], ready_events[source][p], 0));
                peer_copy(recv[destination], tp_ptr[source], bytes, compute[destination]);
            } else {
                check(hipSetDevice(physical[source]));
                check(hipMemcpyAsync(reduce_staging[source], tp_ptr[source], bytes, hipMemcpyDeviceToHost,
                                     compute[source]));
                check(hipEventRecord(done_events[source][p], compute[source]));
                check(hipSetDevice(physical[destination]));
                check(hipStreamWaitEvent(compute[destination], done_events[source][p], 0));
                check(hipMemcpyAsync(recv[destination], reduce_staging[source], bytes, hipMemcpyHostToDevice,
                                     compute[destination]));
            }
            check(hipEventRecord(arrived_events[destination][p], compute[destination]));
        }
        for (int destination : {a, b}) {
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(compute[destination], arrived_events[a][p], 0));
            check(hipStreamWaitEvent(compute[destination], arrived_events[b][p], 0));
            tp_add_bf16(tp_ptr[destination], recv[destination], bytes, compute[destination]);
            check(hipEventRecord(consumed_events[destination][p], compute[destination]));
        }
        recorded[a][p] = recorded[b][p] = true;
        parity ^= 1;
    }
    barrier->arrive_and_wait();
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
    auto collect = [&](hipError_t e) {
        if (e != hipSuccess)
            errors += std::string(hipGetErrorString(e)) + "\n";
    };
    for (auto &plan : target_graphs)
        if (plan.tail)
            plan.release(*this, errors);
    for (auto &plan : draft_graphs)
        if (plan.tail)
            plan.release(*this, errors);
    for (int r = 0; r < ranks; ++r) {
        if (local_expert[r]) {
            collect(hipSetDevice(physical[r]));
            collect(hipStreamSynchronize(local_expert[r]));
            collect(hipStreamDestroy(local_expert[r]));
            local_expert[r] = nullptr;
        }
        for (auto event : {&local_ready[r], &local_done[r]})
            if (*event) {
                collect(hipSetDevice(physical[r]));
                collect(hipEventDestroy(*event));
                *event = nullptr;
            }
        if (compute[r]) {
            collect(hipSetDevice(physical[r]));
            collect(hipStreamSynchronize(compute[r]));
            collect(hipStreamDestroy(compute[r]));
            compute[r] = nullptr;
        }
    }
    for (auto it = arenas.rbegin(); it != arenas.rend(); ++it) {
        auto status = it->shutdown();
        if (!status.ok())
            errors += status.message() + "\n";
    }
    arenas.clear();
    for (int r = 0; r < ranks; ++r) {
        collect(hipSetDevice(physical[r]));
        for (auto events : {&tp_ready, &tp_done, &tp_arrived, &tp_consumed, &ep_ready, &ep_done, &ep_arrived, &ep_consumed})
            for (auto &event : (*events)[r])
                if (event) {
                    collect(hipEventDestroy(event));
                    event = nullptr;
                }
        if (reduce_staging[r]) {
            collect(hipHostFree(reduce_staging[r]));
            reduce_staging[r] = nullptr;
        }
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
        for (void *&buffer : small_staging[r])
            if (buffer) {
                collect(hipHostFree(buffer));
                buffer = nullptr;
            }
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
