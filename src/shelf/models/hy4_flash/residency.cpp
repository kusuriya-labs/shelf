#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/routes.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
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
int Residency::parse_draft_tokens(std::string_view value) {
    int drafts = 0;
    auto [p, err] = std::from_chars(value.data(), value.data() + value.size(), drafts);
    if (value.empty() || p != value.data() + value.size() || err != std::errc{} ||
        drafts < min_draft_tokens || drafts > max_draft_tokens)
        throw std::invalid_argument("DFly2 の候補数は 2〜5 で指定してください");
    return drafts;
}
int Residency::resolve_draft_tokens(int requested) {
    if (requested != 0) {
        if (requested < min_draft_tokens || requested > max_draft_tokens)
            throw std::invalid_argument("DFly2 の候補数は 2〜5 で指定してください");
        return requested;
    }
    const char *env = std::getenv("SHELF_HY4_DFLY_DRAFTS");
    if (env && *env)
        return parse_draft_tokens(env);
    return default_draft_tokens;
}
Residency::Residency(Checkpoint &model, size_t tokens, int slots, size_t kv_pool_tokens, int drafts)
    : model(model), tokens(tokens), max_slots(slots), draft_tokens(resolve_draft_tokens(drafts)),
      kv_pool_tokens(kv_pool_tokens), batch_draft(false) {
    if (slots < 1 || slots > 32)
        throw std::invalid_argument("スロット数は 1〜32 で指定してください");
    if (this->kv_pool_tokens == 0) {
        if (const char *env = std::getenv("SHELF_HY4_KV_POOL_TOKENS"); env && *env) {
            size_t parsed = 0;
            auto [p, err] = std::from_chars(env, env + std::strlen(env), parsed);
            if (err != std::errc{} || p != env + std::strlen(env) || !parsed)
                throw std::runtime_error("SHELF_HY4_KV_POOL_TOKENS が不正です");
            this->kv_pool_tokens = parsed;
        }
    }
    const char *wmma = std::getenv("SHELF_HY4_INDEX_WMMA");
    index_wmma = !wmma || std::string_view(wmma) != "0";
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
    lm_head_tp = model.split_lm_head;
    lm_host_fallback = false;
    lm_parity = 0;
    lm_recorded = {};
    lm_ready_rec = {};
    lm_done_rec = {};
    lm_arrived_rec = {};
    lm_consumed_rec = {};
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
    std::array<size_t, 4> available{}, base_overhead{}, overhead{};
    std::array<int64_t, 4> remaining{};
    std::array<int, 4> used{};
    for (const auto &row : model.owners)
        for (int r : row) {
            if (r < 0 || r >= ep_size)
                throw std::runtime_error("配置表の group が EP 数を超えています");
            ++used[r];
        }
    kv_page_stride = pages_for_tokens(tokens);
    constexpr size_t dsa_layers = 12;
    const size_t per_page_bytes = dsa_layers * (align(kv_page_tokens * 294) + align(kv_page_groups * 256));
    int64_t min_attention_budget = std::numeric_limits<int64_t>::max();
    for (int r = 0; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        size_t total = 0;
        check(hipMemGetInfo(&available[r], &total));
        available[r] -= std::min(available[r], size_t(8) << 20);
        const bool attention_rank = tp_size > 1 || r == 0;
        base_overhead[r] =
            (tp_size > 1 ? size_t(1) << 30 : (r == 0 ? size_t(2) << 30 : size_t(512) << 20)) + matrix_bytes;
        if (attention_rank) {
            base_overhead[r] += align(41 * 512 * sizeof(int)) + align(41 * 512 * sizeof(float)) + 256;
            for (int i = 0; i < 42; ++i) {
                if (!is_sparse_layer(i))
                    base_overhead[r] += align(max_slots * 294 * std::min(tokens, size_t(518)));
                else
                    base_overhead[r] += align(size_t(max_slots) * 12 * 256);
            }
            base_overhead[r] += align(size_t(max_slots) * kv_page_stride * sizeof(int32_t));
            for (const auto &[name, ignored] : folded[r])
                base_overhead[r] += size_t(model.shape_of(name, r)->at(1)) * 512 * 2;
        }
        if (tp_size > 1)
            base_overhead[r] += tp_prefill_rows * 3072u * 2u;
        if (tp_size > 1 && ep_size > 1)
            base_overhead[r] += tp_prefill_rows * 1536u * 2u;
        if (lm_head_tp) {
            base_overhead[r] += 2 * lm_hidden_chunk_bytes + 2 * 4 * lm_argmax_stride;
            if (r == 0 || tp_size > 1)
                base_overhead[r] += 2 * lm_logits_chunk_bytes;
        }
        const int64_t budget = int64_t(available[r]) - int64_t(align(model.bytes[r]) + base_overhead[r]);
        if (budget < 0)
            return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
        if (attention_rank)
            min_attention_budget = std::min(min_attention_budget, budget);
    }
    if (kv_pool_tokens > 0) {
        kv_pages_total = std::max<size_t>(1, pages_for_tokens(kv_pool_tokens));
        if (min_attention_budget < int64_t(kv_pages_total * per_page_bytes))
            return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    } else {
        const size_t max_useful = size_t(max_slots) * kv_page_stride;
        const size_t fit_with_graph = (tp_size == 1 && min_attention_budget > int64_t(graph_headroom))
                                          ? size_t(min_attention_budget - int64_t(graph_headroom)) / per_page_bytes
                                          : 0;
        const size_t fit_no_graph = size_t(min_attention_budget) / per_page_bytes;
        const size_t min_required = max_slots == 1 ? kv_page_stride : std::min(kv_page_stride, max_useful);
        if (fit_with_graph >= min_required)
            kv_pages_total = std::min(max_useful, fit_with_graph);
        else if (fit_no_graph >= min_required)
            kv_pages_total = std::min(max_useful, fit_no_graph);
        else
            return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    }
    for (int r = 0; r < ranks; ++r) {
        const bool attention_rank = tp_size > 1 || r == 0;
        overhead[r] = base_overhead[r] + (attention_rank ? kv_pages_total * per_page_bytes : 0);
        remaining[r] = int64_t(available[r]) - int64_t(align(model.bytes[r]) + overhead[r]);
    }
    if (*std::min_element(remaining.begin(), remaining.begin() + ranks) < 0)
        return Status::unsupported("要求された配置と予約長は GPU の空き容量に収まりません", __FILE__, __LINE__);
    graph_room = tp_size == 1 && std::all_of(remaining.begin(), remaining.begin() + ranks,
                                            [](int64_t bytes) { return bytes >= int64_t(graph_headroom); });
    fprintf(stderr, "rank_counts=%d,%d,%d,%d kv_pages=%zu kv_page_tokens=%zu\n", used[0], used[1], used[2], used[3],
            kv_pages_total, kv_page_tokens);
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
            page_table[r] = static_cast<int32_t *>(buffer.next(size_t(max_slots) * kv_page_stride * sizeof(int32_t)));
            check(hipMemset(page_table[r], 0xff, size_t(max_slots) * kv_page_stride * sizeof(int32_t)));
            for (int i = 0; i < 42; ++i) {
                if (!is_sparse_layer(i)) {
                    cache[r][i] = static_cast<uint8_t *>(
                        buffer.next(max_slots * 294 * std::min(tokens, size_t(518))));
                } else {
                    cache[r][i] = static_cast<uint8_t *>(buffer.next(kv_pages_total * kv_page_tokens * 294));
                    index_keys[r][i] = static_cast<uint16_t *>(
                        buffer.next(kv_pages_total * kv_page_groups * 128 * sizeof(uint16_t)));
                    index_pending[r][i] = static_cast<uint16_t *>(
                        buffer.next(size_t(max_slots) * 12 * 128 * sizeof(uint16_t)));
                }
            }
            for (auto &[name, ptr] : folded[r])
                ptr = buffer.next(size_t(model.shape_of(name, r)->at(1)) * 512 * 2);
        }
        if (tp_size > 1)
            tp_recv[r] = buffer.next(tp_prefill_rows * 3072u * 2u);
        if (tp_size > 1 && ep_size > 1)
            ep_recv[r] = buffer.next(tp_prefill_rows * 1536u * 2u);
        if (lm_head_tp) {
            lm_hidden_buf[r] = buffer.next(2 * lm_hidden_chunk_bytes);
            lm_head_recv[r] = buffer.next(2 * 4 * lm_argmax_stride);
            if (r == 0 || tp_size > 1)
                lm_logits_recv[r] = buffer.next(2 * lm_logits_chunk_bytes);
        }
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
    kv_page_refs.assign(kv_pages_total, 0);
    kv_free_pages.resize(kv_pages_total);
    for (size_t i = 0; i < kv_pages_total; ++i)
        kv_free_pages[i] = int32_t(kv_pages_total - 1 - i);
    slot_page_tables.assign(max_slots, {});
    return Status::make_ok();
}
void Residency::sync_slot_page_table(int slot) {
    if (slot < 0 || size_t(slot) >= slot_page_tables.size() || !kv_page_stride)
        return;
    const auto &pages = slot_page_tables[slot];
    const size_t count = std::min(pages.size(), kv_page_stride);
    for (int r = 0; r < ranks; ++r) {
        if (!page_table[r])
            continue;
        check(hipSetDevice(physical[r]));
        if (count)
            check(hipMemcpyAsync(page_table[r] + size_t(slot) * kv_page_stride, pages.data(),
                                 count * sizeof(int32_t), hipMemcpyHostToDevice, compute[r]));
        if (count < kv_page_stride)
            check(hipMemsetAsync(page_table[r] + size_t(slot) * kv_page_stride + count, 0xff,
                                 (kv_page_stride - count) * sizeof(int32_t), compute[r]));
        check(hipStreamSynchronize(compute[r]));
    }
    if (ranks > 0)
        check(hipSetDevice(physical[0]));
}
bool Residency::allocate_slot_pages(int slot, size_t needed_tokens) {
    if (!kv_pages_total)
        return true;
    if (slot < 0 || size_t(slot) >= slot_page_tables.size())
        return false;
    const size_t need = pages_for_tokens(needed_tokens);
    if (need > kv_page_stride)
        return false;
    auto &table = slot_page_tables[slot];
    if (table.size() >= need)
        return true;
    if (need - table.size() > kv_free_pages.size())
        return false;
    table.reserve(need);
    while (table.size() < need)
        table.push_back(alloc_single_page());
    sync_slot_page_table(slot);
    return true;
}
void Residency::release_slot_pages(int slot) {
    if (slot < 0 || size_t(slot) >= slot_page_tables.size())
        return;
    if (!slot_page_tables[slot].empty()) {
        release_pages(slot_page_tables[slot]);
        sync_slot_page_table(slot);
    }
}
void Residency::retain_pages(const std::vector<int32_t> &pages) {
    for (int32_t p : pages) {
        if (!kv_page_refs.at(p))
            throw std::logic_error("未使用の KV ページは共有できません");
        ++kv_page_refs[p];
    }
}
void Residency::release_pages(std::vector<int32_t> &pages) {
    if (std::any_of(pages.begin(), pages.end(), [&](int32_t p) { return kv_page_refs.at(p) == 1; })) {
        for (int r = 0; r < ranks; ++r) {
            if (page_table[r]) {
                check(hipSetDevice(physical[r]));
                check(hipStreamSynchronize(compute[r]));
            }
        }
        check(hipSetDevice(physical[0]));
    }
    for (int32_t p : pages) {
        if (!kv_page_refs.at(p))
            throw std::logic_error("未使用の KV ページを解放しようとしました");
        if (--kv_page_refs[p] == 0)
            kv_free_pages.push_back(p);
    }
    pages.clear();
}
int32_t Residency::alloc_single_page() {
    if (kv_free_pages.empty())
        return -1;
    int32_t page = kv_free_pages.back();
    kv_free_pages.pop_back();
    kv_page_refs[page] = 1;
    return page;
}
void Residency::reserve_experts(int r, Region &buffer) {
    matrices[r] = static_cast<Hy4ExpertProjection *>(buffer.next(matrix_bytes));
    if (tp_size == 1 && r != 0)
        return;
    moe_owners[r] = static_cast<int *>(buffer.next(41 * 512 * sizeof(int)));
    gate_expert_norms[r] = static_cast<float *>(buffer.next(41 * 512 * sizeof(float)));
    moe_counts[r] = static_cast<uint64_t *>(buffer.next(12 * sizeof(uint64_t)));
    check(hipMemset(moe_counts[r], 0, 12 * sizeof(uint64_t)));
    check(hipStreamCreateWithFlags(&local_expert[r], hipStreamNonBlocking));
    check(hipEventCreateWithFlags(&local_ready[r], hipEventDisableTiming));
    check(hipEventCreateWithFlags(&local_done[r], hipEventDisableTiming));
}

void Residency::collect_route_counts(int rank) {
    std::array<uint64_t, 12> totals{};
    check(hipMemcpyAsync(totals.data(), moe_counts[rank], sizeof(totals), hipMemcpyDeviceToHost, compute[rank]));
    check(hipMemsetAsync(moe_counts[rank], 0, sizeof(totals), compute[rank]));
    check(hipStreamSynchronize(compute[rank]));
    if (tp_size > 1)
        routed_rows[rank] += totals[rank / tp_size];
    else
        for (int r = 0; r < ep_size; ++r) {
            routed_rows[r] += totals[r];
            if (push ? peer[0][r] : peer[r][0]) transfer_bytes[0][r] -= totals[4 + r];
            if (push ? peer[r][0] : peer[0][r]) transfer_bytes[r][0] -= totals[8 + r];
        }
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
namespace {
bool probe_peer_direction(const Residency &gpu, int src, int dst, Residency::PeerProbe probe) {
    constexpr size_t probe_bytes = 4096;
    constexpr size_t words = probe_bytes / sizeof(uint32_t);
    if (gpu.workspace[src].size < probe_bytes || gpu.workspace[dst].size < probe_bytes || !gpu.compute[src] ||
        !gpu.compute[dst])
        return false;
    void *src_buf = gpu.workspace[src].base;
    void *dst_buf = gpu.workspace[dst].base;
    std::array<uint32_t, words> expected{}, actual{};
    for (size_t i = 0; i < words; ++i)
        expected[i] = uint32_t(i + 1) * 0x9e3779b9u ^ uint32_t(src * 17 + dst * 31 + 1);
    const int initiator = gpu.push ? src : dst;
    try {
        check(hipSetDevice(gpu.physical[src]));
        check(hipMemcpyAsync(src_buf, expected.data(), probe_bytes, hipMemcpyHostToDevice, gpu.compute[src]));
        check(hipStreamSynchronize(gpu.compute[src]));
        check(hipSetDevice(gpu.physical[dst]));
        check(hipMemsetAsync(dst_buf, 0, probe_bytes, gpu.compute[dst]));
        check(hipStreamSynchronize(gpu.compute[dst]));
        check(hipSetDevice(gpu.physical[initiator]));
        bool ok = true;
        if (probe)
            ok = probe(dst_buf, dst, src_buf, src, probe_bytes, gpu.compute[initiator], gpu.push);
        else
            peer_copy(dst_buf, src_buf, probe_bytes, gpu.compute[initiator]);
        check(hipStreamSynchronize(gpu.compute[initiator]));
        if (!ok)
            return false;
        check(hipSetDevice(gpu.physical[dst]));
        check(hipMemcpyAsync(actual.data(), dst_buf, probe_bytes, hipMemcpyDeviceToHost, gpu.compute[dst]));
        check(hipStreamSynchronize(gpu.compute[dst]));
        return actual == expected;
    } catch (...) {
        (void)hipGetLastError();
        return false;
    }
}
}
bool Residency::configure_transports(PeerProbe probe) {
    auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const std::string_view mode = transport ? transport : "";
    const bool force_host = mode == "host";
    bool host_reduce = false;
    lm_host_fallback = false;
    push = mode.empty() || mode == "push";
    peer = {};
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src) {
            if (src == dst)
                continue;
            if (physical[dst] == physical[src]) {
                peer[dst][src] = !force_host;
                continue;
            }
            int accessible = 0;
            check(hipSetDevice(physical[dst]));
            check(hipDeviceCanAccessPeer(&accessible, physical[dst], physical[src]));
            if (accessible && !force_host) {
                auto status = hipDeviceEnablePeerAccess(physical[src], 0);
                peer[dst][src] = status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled;
                if (status == hipSuccess)
                    owns_peer[dst][src] = true;
                else if (status != hipErrorPeerAccessAlreadyEnabled)
                    (void)hipGetLastError();
            }
        }
    for (int dst = 0; dst < ranks; ++dst)
        for (int src = 0; src < ranks; ++src) {
            if (src == dst)
                continue;
            bool &direction = push ? peer[src][dst] : peer[dst][src];
            if (direction && !probe_peer_direction(*this, src, dst, probe)) {
                (void)hipGetLastError();
                fprintf(stderr, "P2P プローブ失敗のため host 転送へフォールバックします: %d->%d (%s)\n",
                        src, dst, push ? "push" : "peer");
                direction = false;
            }
            const bool available = direction;
            if (!available && tp_size == 1 && !small_staging[src][dst])
                check(hipHostMalloc(&small_staging[src][dst], transfer_packet_bytes(),
                                    hipHostMallocPortable));
            if (available || small_staging[src][dst]) {
                for (int q = 0; q < 2; ++q) {
                    if (!ready[src][dst][q]) {
                        create_event(ready[src][dst][q], src);
                    }
                    if (!ack[dst][src][q]) {
                        create_event(ack[dst][src][q], dst);
                    }
                }
            }
            host_reduce |= !available && tp_size > 1;
            if (!available && lm_head_tp && (src == 0 || dst == 0))
                lm_host_fallback = true;
            fprintf(stderr, "transfer=%d->%d method=%s\n", src, dst,
                    available ? (push ? "push" : "peer") : "host");
        }
    if (lm_head_tp && lm_host_fallback) {
        for (int r = 0; r < ranks; ++r) {
            if (!lm_staging[r]) {
                check(hipSetDevice(physical[r]));
                check(hipHostMalloc(&lm_staging[r], lm_staging_bytes(), hipHostMallocPortable));
            }
        }
    }
    check(hipSetDevice(physical[0]));
    return host_reduce;
}
void Residency::upload() {
    const auto vmm = std::getenv("SHELF_ARENA_VMM");
    if (!multi_gpu_vmm_allowed(ranks, vmm ? vmm : ""))
        throw std::runtime_error("Hy4 の複数 GPU 実行では SHELF_ARENA_VMM=1 を使用できません");
    auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
    const bool host_reduce = configure_transports();
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
                    create_event((*events)[r][p], r);
                if (ep_size > 1)
                    for (auto events : {&ep_ready, &ep_done, &ep_arrived, &ep_consumed})
                        create_event((*events)[r][p], r);
            }
            if (host_reduce)
                check(hipHostMalloc(&reduce_staging[r], tp_prefill_rows * 3072u * 2u, hipHostMallocPortable));
        }
        if (lm_head_tp) {
            for (int p = 0; p < 2; ++p)
                for (auto events : {&lm_ready, &lm_done, &lm_arrived, &lm_consumed})
                    create_event((*events)[r][p], r);
        }
        for (int layer = 1; layer < 42; ++layer)
            upload_experts(r, layer);
        fold_kv_weights(*this, r);
    }
    for (int layer = 1; layer < 42; ++layer) {
        if (const auto *w = model.find("model.layers." + std::to_string(layer) + ".mlp.gate.weight", 0)) {
            const auto *p = static_cast<const uint16_t *>(w->host);
            std::array<float, 512> norms{};
            for (int e = 0; e < 512; ++e) {
                double sq = 0.0;
                bool finite = true;
                const uint16_t *row = p + size_t(e) * 3072;
                for (int c = 0; c < 3072; ++c) {
                    uint32_t bits = uint32_t(row[c]) << 16;
                    float v;
                    std::memcpy(&v, &bits, 4);
                    if (!std::isfinite(v))
                        finite = false;
                    sq += double(v) * double(v);
                }
                float norm = float(std::sqrt(sq));
                norms[e] = (finite && std::isfinite(norm)) ? norm : std::numeric_limits<float>::infinity();
            }
            for (int r = 0; r < ranks; ++r)
                if (gate_expert_norms[r]) {
                    check(hipSetDevice(physical[r]));
                    check(hipMemcpy(gate_expert_norms[r] + (layer - 1) * 512, norms.data(),
                                    512 * sizeof(float), hipMemcpyHostToDevice));
                }
        }
    }
    check(hipSetDevice(physical[0]));
}
void Residency::create_event(RankEvent &event, int rank) {
    check(hipSetDevice(physical[rank]));
    check(hipEventCreateWithFlags(&event.handle, hipEventDisableTiming | hipEventReleaseToSystem));
    event.owner = rank;
}
void Residency::record_event(const RankEvent &event, int rank) const {
    if (event.owner != rank)
        throw std::logic_error("event と記録先 stream の所有 rank が一致しません");
    check(hipEventRecord(event.handle, compute[rank]));
}
void Residency::sync_record(const RankEvent &event, bool &recorded, int rank) const {
    if (recorded)
        check(hipEventSynchronize(event));
    record_event(event, rank);
    recorded = true;
}
void Residency::transfer(void *dst, int destination, const void *src, int source, size_t bytes, int prefill_rows) {
    transfer_bytes[source][destination] += bytes;
    std::chrono::steady_clock::time_point begin;
    if (expert_begin[0])
        begin = std::chrono::steady_clock::now();
    void *host = bytes <= transfer_packet_bytes() ? small_staging[source][destination] : nullptr;
    auto copy = [&](hipStream_t stream) {
        if (prefill_rows)
            peer_prefill_packet(dst, src, prefill_rows, destination, stream);
        else
            peer_copy(dst, src, bytes, stream);
    };
    if (!host && !(push ? peer[source][destination] : peer[destination][source]))
        throw std::runtime_error("転送作業域の範囲外です");
    const int p = parity[source][destination];
    const auto &ready_event = ready[source][destination][p];
    const auto &ack_event = ack[destination][source][p];
    check(hipEventSynchronize(push && !host ? ready_event : ack_event));
    parity[source][destination] = p ^ 1;
    if (!host && push) {
        check(hipSetDevice(physical[destination]));
        record_event(ack_event, destination);
        check(hipSetDevice(physical[source]));
        check(hipStreamWaitEvent(compute[source], ack_event, 0));
        copy(compute[source]);
        record_event(ready_event, source);
        check(hipSetDevice(physical[destination]));
        check(hipStreamWaitEvent(compute[destination], ready_event, 0));
    } else {
        check(hipSetDevice(physical[source]));
        if (host)
            check(hipMemcpyAsync(host, src, bytes, hipMemcpyDeviceToHost, compute[source]));
        record_event(ready_event, source);
        check(hipSetDevice(physical[destination]));
        check(hipStreamWaitEvent(compute[destination], ready_event, 0));
        if (host)
            check(hipMemcpyAsync(dst, host, bytes, hipMemcpyHostToDevice, compute[destination]));
        else
            copy(compute[destination]);
        record_event(ack_event, destination);
        check(hipSetDevice(physical[source]));
        check(hipStreamWaitEvent(compute[source], ack_event, 0));
        check(hipSetDevice(physical[destination]));
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
    record_event(ready_events[rank][p], rank);
    barrier->arrive_and_wait();
    if (rank == a) {
        if (tp_bytes[a] != tp_bytes[b] || bytes > tp_prefill_rows * (ep ? 1536u : 3072u) * 2u)
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
                record_event(done_events[source][p], source);
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
                record_event(done_events[source][p], source);
                check(hipSetDevice(physical[destination]));
                check(hipStreamWaitEvent(compute[destination], done_events[source][p], 0));
                check(hipMemcpyAsync(recv[destination], reduce_staging[source], bytes, hipMemcpyHostToDevice,
                                     compute[destination]));
            }
            record_event(arrived_events[destination][p], destination);
        }
        for (int destination : {a, b}) {
            check(hipSetDevice(physical[destination]));
            check(hipStreamWaitEvent(compute[destination], arrived_events[a][p], 0));
            check(hipStreamWaitEvent(compute[destination], arrived_events[b][p], 0));
            tp_add_bf16(tp_ptr[destination], recv[destination], bytes, compute[destination]);
            record_event(consumed_events[destination][p], destination);
        }
        recorded[a][p] = recorded[b][p] = true;
        parity ^= 1;
    }
    barrier->arrive_and_wait();
    check(hipSetDevice(physical[rank]));
}
void Residency::lm_scatter_hidden(const void *src, size_t bytes, std::array<void *, 4> &dst_out, bool capture) {
    dst_out[0] = const_cast<void *>(src);
    if (!lm_head_tp || tp_size > 1)
        return;
    if (bytes > lm_hidden_chunk_bytes)
        throw std::runtime_error("lm_head hidden チャンク超過です");
    if (capture) {
        for (int r = 1; r < ranks; ++r) {
            dst_out[r] = lm_hidden_buf[r];
            peer_copy(dst_out[r], src, bytes, compute[0]);
        }
        return;
    }
    const int p = lm_parity;
    check(hipSetDevice(physical[0]));
    bool any_host = false;
    for (int r = 1; r < ranks; ++r) {
        if (lm_done_rec[r][p])
            check(hipStreamWaitEvent(compute[0], lm_done[r][p], 0));
        if (!(push ? peer[0][r] : peer[r][0]))
            any_host = true;
    }
    if (any_host)
        check(hipMemcpyAsync(lm_staging[0], src, bytes, hipMemcpyDeviceToHost, compute[0]));
    for (int r = 1; r < ranks; ++r) {
        dst_out[r] = static_cast<char *>(lm_hidden_buf[r]) + size_t(p) * lm_hidden_chunk_bytes;
        if (push && peer[0][r])
            peer_copy(dst_out[r], src, bytes, compute[0]);
    }
    sync_record(lm_ready[0][p], lm_ready_rec[0][p], 0);
    if (any_host)
        check(hipEventSynchronize(lm_ready[0][p]));
    for (int r = 1; r < ranks; ++r) {
        check(hipSetDevice(physical[r]));
        if (push && peer[0][r]) {
            check(hipStreamWaitEvent(compute[r], lm_ready[0][p], 0));
        } else if (!push && peer[r][0]) {
            check(hipStreamWaitEvent(compute[r], lm_ready[0][p], 0));
            peer_copy(dst_out[r], src, bytes, compute[r]);
        } else {
            check(hipMemcpyAsync(dst_out[r], lm_staging[0], bytes, hipMemcpyHostToDevice, compute[r]));
            check(hipStreamSynchronize(compute[r]));
        }
    }
    check(hipSetDevice(physical[0]));
}
void Residency::lm_gather_argmax(const std::array<const float *, 4> &vals, const std::array<const int *, 4> &ids,
                                 int rows, int *out_ids) {
    if (!lm_head_tp || tp_size > 1 || rows <= 0)
        return;
    for (int first = 0; first < rows; first += lm_argmax_max_rows) {
        const int count = std::min(lm_argmax_max_rows, rows - first);
        const size_t bytes = size_t(count) * 8;
        const int p = lm_parity;
        char *base_recv0 = static_cast<char *>(lm_head_recv[0]) + size_t(p) * 4 * lm_argmax_stride;
        check(hipSetDevice(physical[0]));
        tp_argmax_pack(vals[0] + first, ids[0] + first, base_recv0, count, compute[0]);
        for (int r = 1; r < ranks; ++r) {
            void *local_pairs = static_cast<char *>(lm_head_recv[r]) + size_t(p) * lm_argmax_stride;
            void *dst_r = base_recv0 + size_t(r) * lm_argmax_stride;
            check(hipSetDevice(physical[r]));
            tp_argmax_pack(vals[r] + first, ids[r] + first, local_pairs, count, compute[r]);
            auto mode = select_reduce_transport(push, peer[r][0], peer[0][r]);
            if (mode == ReduceTransport::push) {
                if (lm_consumed_rec[0][p])
                    check(hipStreamWaitEvent(compute[r], lm_consumed[0][p], 0));
                peer_copy(dst_r, local_pairs, bytes, compute[r]);
                sync_record(lm_done[r][p], lm_done_rec[r][p], r);
                check(hipSetDevice(physical[0]));
                check(hipStreamWaitEvent(compute[0], lm_done[r][p], 0));
            } else if (mode == ReduceTransport::peer) {
                sync_record(lm_done[r][p], lm_done_rec[r][p], r);
                check(hipSetDevice(physical[0]));
                check(hipStreamWaitEvent(compute[0], lm_done[r][p], 0));
                peer_copy(dst_r, local_pairs, bytes, compute[0]);
            } else {
                check(hipMemcpyAsync(lm_staging[r], local_pairs, bytes, hipMemcpyDeviceToHost, compute[r]));
                sync_record(lm_done[r][p], lm_done_rec[r][p], r);
                check(hipEventSynchronize(lm_done[r][p]));
                check(hipSetDevice(physical[0]));
                check(hipMemcpyAsync(dst_r, lm_staging[r], bytes, hipMemcpyHostToDevice, compute[0]));
            }
            lm_recorded[r][p] = true;
        }
        check(hipSetDevice(physical[0]));
        tp_argmax_reduce(base_recv0, lm_argmax_max_rows, ranks, count, out_ids + first, compute[0]);
        sync_record(lm_consumed[0][p], lm_consumed_rec[0][p], 0);
        lm_recorded[0][p] = true;
        lm_parity ^= 1;
    }
}
void Residency::lm_gather_logits(const std::array<const float *, 4> &local_logits, int rows, int local_vocab,
                                 int full_vocab, float *out_logits) {
    if (!lm_head_tp || tp_size > 1 || rows <= 0)
        return;
    check(hipSetDevice(physical[0]));
    tp_scatter_logits(local_logits[0], out_logits, rows, local_vocab, full_vocab, 0, compute[0]);
    const int chunk_rows = std::max(1, int(lm_logits_chunk_bytes / (size_t(local_vocab) * sizeof(float))));
    for (int r = 1; r < ranks; ++r) {
        auto mode = select_reduce_transport(push, peer[r][0], peer[0][r]);
        if (mode == ReduceTransport::peer) {
            check(hipSetDevice(physical[r]));
            sync_record(lm_ready[r][0], lm_ready_rec[r][0], r);
            check(hipSetDevice(physical[0]));
            check(hipStreamWaitEvent(compute[0], lm_ready[r][0], 0));
        }
        for (int first = 0; first < rows; first += chunk_rows) {
            const int count = std::min(chunk_rows, rows - first);
            const size_t chunk_bytes = size_t(count) * size_t(local_vocab) * sizeof(float);
            const int p = lm_parity;
            const float *src_r = local_logits[r] + size_t(first) * local_vocab;
            float *recv_r = reinterpret_cast<float *>(
                static_cast<char *>(lm_logits_recv[0]) + size_t(p) * lm_logits_chunk_bytes);
            if (mode == ReduceTransport::push) {
                check(hipSetDevice(physical[r]));
                if (lm_consumed_rec[0][p])
                    check(hipStreamWaitEvent(compute[r], lm_consumed[0][p], 0));
                peer_copy(recv_r, src_r, chunk_bytes, compute[r]);
                sync_record(lm_done[r][p], lm_done_rec[r][p], r);
                check(hipSetDevice(physical[0]));
                check(hipStreamWaitEvent(compute[0], lm_done[r][p], 0));
            } else if (mode == ReduceTransport::peer) {
                check(hipSetDevice(physical[0]));
                peer_copy(recv_r, src_r, chunk_bytes, compute[0]);
            } else {
                check(hipSetDevice(physical[r]));
                check(hipMemcpyAsync(lm_staging[r], src_r, chunk_bytes, hipMemcpyDeviceToHost, compute[r]));
                sync_record(lm_done[r][p], lm_done_rec[r][p], r);
                check(hipEventSynchronize(lm_done[r][p]));
                check(hipSetDevice(physical[0]));
                check(hipMemcpyAsync(recv_r, lm_staging[r], chunk_bytes, hipMemcpyHostToDevice, compute[0]));
                check(hipStreamSynchronize(compute[0]));
            }
            check(hipSetDevice(physical[0]));
            tp_scatter_logits(recv_r, out_logits + size_t(first) * full_vocab, count, local_vocab, full_vocab,
                              r * local_vocab, compute[0]);
            sync_record(lm_consumed[0][p], lm_consumed_rec[0][p], 0);
            lm_recorded[0][p] = true;
            lm_parity ^= 1;
        }
        if (mode == ReduceTransport::peer) {
            check(hipSetDevice(physical[0]));
            sync_record(lm_arrived[0][0], lm_arrived_rec[0][0], 0);
            check(hipSetDevice(physical[r]));
            check(hipStreamWaitEvent(compute[r], lm_arrived[0][0], 0));
        }
    }
    check(hipSetDevice(physical[0]));
}
void Residency::tp_argmax(const float *vals, const int *ids, int *out_ids, int rank, int rows) {
    if (!lm_head_tp || tp_size <= 1 || rows <= 0)
        return;
    for (int first = 0; first < rows; first += lm_argmax_max_rows) {
        const int count = std::min(lm_argmax_max_rows, rows - first);
        const size_t bytes = size_t(count) * 8;
        const size_t id_bytes = size_t(count) * sizeof(int);
        const int p = lm_parity;
        check(hipSetDevice(physical[rank]));
        if (lm_recorded[rank][p])
            check(hipEventSynchronize(lm_arrived[rank][p]));
        void *local_buf = rank == 0
            ? static_cast<char *>(lm_head_recv[0]) + size_t(p) * 4 * lm_argmax_stride
            : static_cast<char *>(lm_head_recv[rank]) + size_t(p) * lm_argmax_stride;
        tp_argmax_pack(vals + first, ids + first, local_buf, count, compute[rank]);
        lm_src_ptr[rank] = local_buf;
        lm_dst_ptr[rank] = out_ids + first;
        lm_xfer_bytes[rank] = bytes;
        if (rank != 0)
            record_event(lm_ready[rank][p], rank);
        all_barrier->arrive_and_wait();
        if (rank == 0) {
            char *base_recv0 = static_cast<char *>(lm_head_recv[0]) + size_t(p) * 4 * lm_argmax_stride;
            for (int src = 1; src < ranks; ++src) {
                void *dst_r = base_recv0 + size_t(src) * lm_argmax_stride;
                auto mode = select_reduce_transport(push, peer[src][0], peer[0][src]);
                if (mode == ReduceTransport::push) {
                    check(hipSetDevice(physical[src]));
                    check(hipStreamWaitEvent(compute[src], lm_ready[src][p], 0));
                    peer_copy(dst_r, lm_src_ptr[src], bytes, compute[src]);
                    record_event(lm_done[src][p], src);
                    check(hipSetDevice(physical[0]));
                    check(hipStreamWaitEvent(compute[0], lm_done[src][p], 0));
                } else if (mode == ReduceTransport::peer) {
                    check(hipSetDevice(physical[0]));
                    check(hipStreamWaitEvent(compute[0], lm_ready[src][p], 0));
                    peer_copy(dst_r, lm_src_ptr[src], bytes, compute[0]);
                } else {
                    check(hipSetDevice(physical[src]));
                    check(hipStreamWaitEvent(compute[src], lm_ready[src][p], 0));
                    check(hipMemcpyAsync(lm_staging[src], lm_src_ptr[src], bytes, hipMemcpyDeviceToHost, compute[src]));
                    record_event(lm_done[src][p], src);
                    check(hipEventSynchronize(lm_done[src][p]));
                    check(hipSetDevice(physical[0]));
                    check(hipMemcpyAsync(dst_r, lm_staging[src], bytes, hipMemcpyHostToDevice, compute[0]));
                }
            }
            check(hipSetDevice(physical[0]));
            tp_argmax_reduce(base_recv0, lm_argmax_max_rows, ranks, count, static_cast<int *>(lm_dst_ptr[0]),
                             compute[0]);
            broadcast_lm_ids(id_bytes, p);
        }
        all_barrier->arrive_and_wait();
        check(hipSetDevice(physical[rank]));
    }
}
void Residency::tp_gather_logits(const float *local_logits, float *out_logits, int rank, int rows, int local_vocab,
                                 int full_vocab) {
    if (!lm_head_tp || tp_size <= 1 || rows <= 0)
        return;
    check(hipSetDevice(physical[rank]));
    if (rank == 0)
        tp_scatter_logits(local_logits, out_logits, rows, local_vocab, full_vocab, 0, compute[0]);
    lm_src_ptr[rank] = local_logits;
    record_event(lm_ready[rank][0], rank);
    all_barrier->arrive_and_wait();
    if (rank == 0) {
        const int chunk_rows = std::max(1, int(lm_logits_chunk_bytes / (size_t(local_vocab) * sizeof(float))));
        for (int src = 1; src < ranks; ++src) {
            auto mode = select_reduce_transport(push, peer[src][0], peer[0][src]);
            if (mode == ReduceTransport::push) {
                check(hipSetDevice(physical[src]));
                check(hipStreamWaitEvent(compute[src], lm_ready[src][0], 0));
            } else if (mode == ReduceTransport::peer) {
                check(hipSetDevice(physical[0]));
                check(hipStreamWaitEvent(compute[0], lm_ready[src][0], 0));
            } else {
                check(hipSetDevice(physical[src]));
                check(hipStreamWaitEvent(compute[src], lm_ready[src][0], 0));
            }
            for (int first = 0; first < rows; first += chunk_rows) {
                const int count = std::min(chunk_rows, rows - first);
                const size_t chunk_bytes = size_t(count) * size_t(local_vocab) * sizeof(float);
                const int p = lm_parity;
                const float *src_r = static_cast<const float *>(lm_src_ptr[src]) + size_t(first) * local_vocab;
                float *recv_r = reinterpret_cast<float *>(
                    static_cast<char *>(lm_logits_recv[0]) + size_t(p) * lm_logits_chunk_bytes);
                if (mode == ReduceTransport::push) {
                    check(hipSetDevice(physical[src]));
                    if (lm_recorded[0][p])
                        check(hipStreamWaitEvent(compute[src], lm_consumed[0][p], 0));
                    peer_copy(recv_r, src_r, chunk_bytes, compute[src]);
                    record_event(lm_done[src][p], src);
                    check(hipSetDevice(physical[0]));
                    check(hipStreamWaitEvent(compute[0], lm_done[src][p], 0));
                } else if (mode == ReduceTransport::peer) {
                    check(hipSetDevice(physical[0]));
                    peer_copy(recv_r, src_r, chunk_bytes, compute[0]);
                } else {
                    check(hipSetDevice(physical[src]));
                    check(hipMemcpyAsync(lm_staging[src], src_r, chunk_bytes, hipMemcpyDeviceToHost, compute[src]));
                    record_event(lm_done[src][p], src);
                    check(hipEventSynchronize(lm_done[src][p]));
                    check(hipSetDevice(physical[0]));
                    check(hipMemcpyAsync(recv_r, lm_staging[src], chunk_bytes, hipMemcpyHostToDevice, compute[0]));
                    check(hipStreamSynchronize(compute[0]));
                }
                check(hipSetDevice(physical[0]));
                tp_scatter_logits(recv_r, out_logits + size_t(first) * full_vocab, count, local_vocab, full_vocab,
                                  src * local_vocab, compute[0]);
                record_event(lm_consumed[0][p], 0);
                lm_recorded[0][p] = true;
                lm_parity ^= 1;
            }
        }
        check(hipSetDevice(physical[0]));
        record_event(lm_done[0][0], 0);
    }
    all_barrier->arrive_and_wait();
    check(hipSetDevice(physical[rank]));
    if (rank > 0)
        check(hipStreamWaitEvent(compute[rank], lm_done[0][0], 0));
}
void Residency::broadcast_lm_ids(size_t bytes, int p) {
    check(hipSetDevice(physical[0]));
    bool host = false;
    for (int dst = 1; dst < ranks; ++dst) {
        auto mode = select_reduce_transport(push, peer[0][dst], peer[dst][0]);
        if (mode == ReduceTransport::push) {
            check(hipStreamWaitEvent(compute[0], lm_ready[dst][p], 0));
            peer_copy(lm_dst_ptr[dst], lm_dst_ptr[0], bytes, compute[0]);
        } else if (mode == ReduceTransport::host) {
            host = true;
        }
    }
    if (host)
        check(hipMemcpyAsync(lm_staging[0], lm_dst_ptr[0], bytes, hipMemcpyDeviceToHost, compute[0]));
    record_event(lm_done[0][p], 0);
    if (host)
        check(hipEventSynchronize(lm_done[0][p]));
    for (int dst = 1; dst < ranks; ++dst) {
        auto mode = select_reduce_transport(push, peer[0][dst], peer[dst][0]);
        check(hipSetDevice(physical[dst]));
        check(hipStreamWaitEvent(compute[dst], lm_done[0][p], 0));
        if (mode == ReduceTransport::peer)
            peer_copy(lm_dst_ptr[dst], lm_dst_ptr[0], bytes, compute[dst]);
        else if (mode == ReduceTransport::host) {
            check(hipMemcpyAsync(lm_dst_ptr[dst], lm_staging[0], bytes, hipMemcpyHostToDevice, compute[dst]));
            check(hipStreamSynchronize(compute[dst]));
        }
        record_event(lm_arrived[dst][p], dst);
        lm_recorded[dst][p] = true;
        if (mode == ReduceTransport::peer) {
            check(hipSetDevice(physical[0]));
            check(hipStreamWaitEvent(compute[0], lm_arrived[dst][p], 0));
        }
    }
    check(hipSetDevice(physical[0]));
    record_event(lm_arrived[0][p], 0);
    lm_recorded[0][p] = true;
    lm_parity ^= 1;
}
void Residency::tp_broadcast_ids(int *ids, int rank, int rows) {
    if (!lm_head_tp || tp_size <= 1 || rows <= 0)
        return;
    const int p = lm_parity;
    check(hipSetDevice(physical[rank]));
    if (lm_recorded[rank][p])
        check(hipEventSynchronize(lm_arrived[rank][p]));
    lm_dst_ptr[rank] = ids;
    if (rank != 0)
        record_event(lm_ready[rank][p], rank);
    all_barrier->arrive_and_wait();
    if (rank == 0)
        broadcast_lm_ids(size_t(rows) * sizeof(int), p);
    all_barrier->arrive_and_wait();
    check(hipSetDevice(physical[rank]));
}
void Residency::shutdown() {
    std::string errors;
    auto collect = [&](hipError_t e) {
        if (e != hipSuccess)
            errors += std::string(hipGetErrorString(e)) + "\n";
    };
    for (auto &plan : target_graphs)
        if (plan.tail || !plan.head_steps.empty())
            plan.release(*this, errors);
    for (auto &plan : draft_graphs)
        if (plan.tail || !plan.head_steps.empty())
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
    kv_free_pages.clear();
    kv_page_refs.clear();
    slot_page_tables.clear();
    kv_pages_total = 0;
    kv_page_stride = 0;
    page_table.fill(nullptr);
    index_pending = {};
    cache = {};
    index_keys = {};
    for (int r = 0; r < ranks; ++r) {
        collect(hipSetDevice(physical[r]));
        for (auto events : {&tp_ready, &tp_done, &tp_arrived, &tp_consumed, &ep_ready, &ep_done, &ep_arrived,
                            &ep_consumed, &lm_ready, &lm_done, &lm_arrived, &lm_consumed})
            for (auto &event : (*events)[r])
                if (event) {
                    collect(hipEventDestroy(event));
                    event = {};
                }
        if (reduce_staging[r]) {
            collect(hipHostFree(reduce_staging[r]));
            reduce_staging[r] = nullptr;
        }
        if (lm_staging[r]) {
            collect(hipHostFree(lm_staging[r]));
            lm_staging[r] = nullptr;
        }
    }
    lm_hidden_buf.fill(nullptr);
    lm_head_recv.fill(nullptr);
    lm_logits_recv.fill(nullptr);
    lm_src_ptr.fill(nullptr);
    lm_dst_ptr.fill(nullptr);
    lm_xfer_bytes.fill(0);
    lm_recorded = {};
    lm_ready_rec = {};
    lm_done_rec = {};
    lm_arrived_rec = {};
    lm_consumed_rec = {};
    lm_parity = 0;
    lm_head_tp = false;
    lm_host_fallback = false;
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
        for (auto &row : ready[r])
            for (auto &event : row)
                if (event) {
                    auto status = hipSetDevice(physical[r]);
                    collect(status);
                    if (status == hipSuccess)
                        collect(hipEventDestroy(event));
                    event = {};
                }
        for (auto &row : ack[r])
            for (auto &event : row)
                if (event) {
                    auto status = hipSetDevice(physical[r]);
                    collect(status);
                    if (status == hipSuccess)
                        collect(hipEventDestroy(event));
                    event = {};
                }
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
