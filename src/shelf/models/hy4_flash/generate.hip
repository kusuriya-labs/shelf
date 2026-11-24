#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/fused/detail/radix_select.h>
#include <shelf/models/hy4_flash/generate.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <numeric>
#include <shelf/models/hy4_flash/target.h>
#include <cstring>
#include <exception>
#include <deque>
#include <list>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
constexpr std::array<int, 5> sources{25, 29, 33, 37, 41};
__device__ uint64_t mix(uint64_t v) {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
    return v ^ (v >> 31);
}
__global__ void best_ids(const float *logits, int *tokens, int position, uint64_t seed, float temperature,
                         const float *subtract = nullptr, const Hy4StepParams *params = nullptr) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int tid = threadIdx.x, row = blockIdx.x, id = -1;
    if (params) {
        position += params->position;
        seed ^= params->seed;
    }
    float best = -INFINITY;
    for (int c = tid; c < 120832; c += 256) {
        float v = logits[size_t(row) * 120832 + c];
        if (subtract) {
            float mass = v - subtract[c];
            if (mass <= 0)
                continue;
            v = logf(mass);
        }
        if (temperature > 0) {
            uint64_t bits = mix((uint64_t(position + row) << 32 | uint32_t(c)) ^ seed);
            float uniform = (float(bits >> 41) + 0.5f) * 0x1p-23f;
            v -= temperature * logf(-logf(uniform));
        }
        if (v > best) {
            best = v;
            id = c;
        }
    }
    vals[tid] = best;
    ids[tid] = id;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d && (vals[tid + d] > vals[tid] || (vals[tid + d] == vals[tid] && ids[tid + d] < ids[tid]))) {
            vals[tid] = vals[tid + d];
            ids[tid] = ids[tid + d];
        }
        __syncthreads();
    }
    if (!tid)
        tokens[row] = ids[0];
}
__global__ void truncate_logits(float *logits, int k, float p, float temperature) {
    __shared__ uint32_t histogram[256], state[4];
    __shared__ int ids[64];
    __shared__ float values[64], masses[64];
    float *row = logits + size_t(blockIdx.x) * 120832;
    ::shelf::kernel::detail::radix_select<3>(row, 120832, k, histogram, state, ids, values, 0);
    const int t = threadIdx.x;
    if (t < k) {
        float peak = values[0];
        for (int j = 1; j < k; ++j)
            peak = fmaxf(peak, values[j]);
        masses[t] = expf((values[t] - peak) / temperature);
    }
    __syncthreads();
    for (int i = t; i < 120832; i += blockDim.x)
        row[i] = -INFINITY;
    __syncthreads();
    if (t < k) {
        double before = 0, total = 0;
        for (int j = 0; j < k; ++j) {
            total += masses[j];
            if (values[j] > values[t] || (values[j] == values[t] && ids[j] < ids[t]))
                before += masses[j];
        }
        if (p == 1 || before < p * total)
            row[ids[t]] = values[t];
    }
}
__global__ void probabilities(const float *logits, float *out, float temp) {
    __shared__ float scratch[256];
    int t = threadIdx.x;
    logits += size_t(blockIdx.x) * 120832;
    out += size_t(blockIdx.x) * 120832;
    float peak = -INFINITY;
    for (int c = t; c < 120832; c += 256)
        peak = fmaxf(peak, logits[c]);
    scratch[t] = peak;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n)
            scratch[t] = fmaxf(scratch[t], scratch[t + n]);
        __syncthreads();
    }
    peak = scratch[0];
    __syncthreads();
    float sum = 0;
    for (int c = t; c < 120832; c += 256) {
        out[c] = expf((logits[c] - peak) / temp);
        sum += out[c];
    }
    scratch[t] = sum;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n)
            scratch[t] += scratch[t + n];
        __syncthreads();
    }
    for (int c = t; c < 120832; c += 256)
        out[c] /= scratch[0];
}
__global__ void test_drafts(const float *target, const float *draft, const int *ids, int position,
                            uint64_t seed, int *accept) {
    int i = threadIdx.x, token = ids[i];
    size_t offset = size_t(i) * 120832 + token;
    uint64_t bits = mix((uint64_t(position + i) << 32) ^ seed ^ 0xc343d97fb3efull);
    double u = (double(bits >> 12) + 0.5) * 0x1p-52;
    accept[i] = u * double(draft[offset]) <= double(target[offset]);
}
__global__ void fuse_anchor(Mat anchor, Mat fusion, const uint16_t *gate) {
    int i = blockIdx.x * 256 + threadIdx.x;
    anchor.p[i] = f32_to_bf16_rne_device(bf16_bits_to_f32(anchor.p[i]) +
                                         bf16_bits_to_f32(fusion.p[i]) * bf16_bits_to_f32(gate[i]));
}

template <size_t N, typename Body>
GraphPlan *capture_cached(Ops &op, std::array<GraphPlan, N> &slots, size_t &victim, bool &failed,
                          const GraphKey &key, Body body) {
    for (auto &slot : slots)
        if (slot.tail && slot.key == key)
            return &slot;
    auto &slot = slots[victim++ % N];
    std::string errors;
    if (slot.tail)
        slot.release(op.gpu, errors);
    if (!errors.empty())
        throw std::runtime_error(errors);
    GraphCapture capture(op.gpu, slot);
    auto last = std::to_array(op.sparse_last_layer);
    op.params = key.batch ? nullptr : &op.gpu.graph_input->step;
    op.max_kv_len = key.bucket;
    op.capture = &capture;
    try {
        capture.begin(0);
        body(slot);
        capture.finish(slot.tail);
        slot.key = key;
    } catch (const std::exception &e) {
        capture.cancel(errors);
        slot.release(op.gpu, errors);
        failed = true;
        (void)hipGetLastError();
        fprintf(stderr, "Hy4 Graph の capture に失敗したため eager に切り替えます: %s\n", e.what());
    }
    if (key.batch) op.slot = 0;
    op.params = nullptr;
    op.capture = nullptr;
    op.max_kv_len = 0;
    std::copy(last.begin(), last.end(), op.sparse_last_layer);
    if (!errors.empty())
        throw std::runtime_error(errors);
    return failed ? nullptr : &slot;
}

bool graph_enabled(const char *name) {
    auto value = std::getenv(name);
    return !value || std::string_view(value) != "0";
}

size_t swa_window_slots(const Residency &gpu) {
    return std::min(gpu.tokens, size_t(518));
}

size_t swa_layer_count() {
    size_t count = 0;
    for (int layer = 0; layer < 42; ++layer)
        if (layer != 0 && layer % 4 != 1) ++count;
    return count;
}

bool swa_range_intact(size_t pos, size_t max_written, size_t swa_slots) {
    size_t min_swa = pos > 511 ? pos - 511 : 0;
    return max_written <= min_swa + swa_slots;
}

size_t align_hy4_prefix(size_t common, size_t input_size) {
    size_t capped = std::min(common, input_size > 1 ? input_size - 1 : size_t(0));
    return (capped / 4) * 4;
}

size_t match_hy4_prefix(const Hy4PrefixSlot &entry, const std::vector<int> &input,
                        const Residency &gpu, bool &use_snap) {
    use_snap = false;
    if (!entry.valid || entry.tokens.empty() || input.size() <= 1) return 0;
    size_t common = 0;
    size_t limit = std::min(entry.tokens.size(), input.size() - 1);
    while (common < limit && entry.tokens[common] == input[common]) ++common;
    const size_t swa_slots = swa_window_slots(gpu);
    size_t live_pos = (common / 4) * 4;
    if (live_pos > 0 && swa_range_intact(live_pos, entry.max_written, swa_slots))
        return live_pos;
    if (entry.snap_valid) {
        size_t snap_limit = std::min(common, entry.snap_pos);
        size_t snap_pos = (snap_limit / 4) * 4;
        if (snap_pos > 0 && swa_range_intact(snap_pos, entry.snap_max_written, swa_slots)) {
            use_snap = true;
            return snap_pos;
        }
    }
    return 0;
}

void save_swa_snapshot(Residency &gpu, int slot, int rank, hipStream_t stream,
                       std::vector<uint8_t> &dst) {
    if (!gpu.cache[rank][2]) return;
    const size_t slot_bytes = swa_window_slots(gpu) * 294;
    dst.resize(swa_layer_count() * slot_bytes);
    size_t idx = 0;
    for (int layer = 0; layer < 42; ++layer) {
        if (layer == 0 || layer % 4 == 1) continue;
        checked(hipMemcpyAsync(dst.data() + idx * slot_bytes,
                               gpu.cache[rank][layer] + size_t(slot) * slot_bytes,
                               slot_bytes, hipMemcpyDeviceToHost, stream));
        ++idx;
    }
    checked(hipStreamSynchronize(stream));
}

void restore_swa_snapshot(Residency &gpu, int slot, int rank, hipStream_t stream,
                          const std::vector<uint8_t> &src) {
    if (!gpu.cache[rank][2] || src.empty()) return;
    const size_t slot_bytes = swa_window_slots(gpu) * 294;
    if (src.size() != swa_layer_count() * slot_bytes) return;
    size_t idx = 0;
    for (int layer = 0; layer < 42; ++layer) {
        if (layer == 0 || layer % 4 == 1) continue;
        checked(hipMemcpyAsync(gpu.cache[rank][layer] + size_t(slot) * slot_bytes,
                               src.data() + idx * slot_bytes,
                               slot_bytes, hipMemcpyHostToDevice, stream));
        ++idx;
    }
}
}

Target::Target(Residency &gpu, float temp, uint64_t rng, int k, float p, int rank, int drafts)
    : op(gpu, rank), temperature(temp), top_k(k), top_p(p), seed(rng),
      draft_tokens(Residency::resolve_draft_tokens(drafts ? drafts : gpu.draft_tokens)) {
    op.arena.cursor = 0;
    if (gpu.model.has_draft && temperature > 0) {
        target_probs = static_cast<float *>(op.arena.next(max_verify_rows * 120832 * 4));
        draft_probs = static_cast<float *>(op.arena.next(max_draft_tokens * 120832 * 4));
    }
    checked(hipMemsetAsync(gpu.moe_counts[rank], 0, 12 * sizeof(uint64_t), op.stream));
    if (gpu.tp_size == 1)
        gpu.routed_rows.fill(0);
    else
        gpu.routed_rows[rank] = 0;
    gpu.expert_ms[rank] = 0;
    if (rank == 0) {
        gpu.attention_ms = gpu.ffn_ms = gpu.transfer_ms = 0;
        gpu.transfer_bytes = {};
    }
    state = op.mat(gpu.batch_draft ? max_verify_rows * gpu.max_slots : max_verify_rows, 3072);
    base = op.arena.cursor;
    bool room = gpu.graph_room && !gpu.expert_begin[rank];
    bool topology = graph_topology(gpu.ep_size, gpu.tp_size, gpu.push, room, gpu.peer);
    target_graph = topology && graph_enabled("SHELF_HY4_GRAPH");
    bool draft_available = gpu.max_slots == 1 ? gpu.model.has_draft : gpu.batch_draft;
    draft_graph = draft_available && graph_enabled("SHELF_HY4_DRAFT_GRAPH") &&
        ((gpu.max_slots == 1 && !gpu.lm_head_tp) ? (gpu.tp_size == 1 && room) : topology);
    if ((target_graph || draft_graph) && !gpu.graph_input && !gpu.batch_input) {
        if (gpu.max_slots > 1)
            gpu.batch_input = static_cast<BatchGraphInput *>(op.reserve_persistent(sizeof(BatchGraphInput)));
        else
            gpu.graph_input = static_cast<GraphInput *>(op.reserve_persistent(sizeof(GraphInput)));
        if (gpu.tokens > 2048)
            for (int layer = 0; layer < 42; ++layer)
                if (gpu.index_keys[0][layer] && !gpu.sparse_top[0][layer])
                    gpu.sparse_top[0][layer] = static_cast<int *>(op.reserve_persistent(
                        size_t(gpu.prefill_rows() + (gpu.batch_draft ? gpu.max_slots * max_verify_rows : 0)) * 511 * sizeof(int)));
    }
}
float *Target::head_logits(Mat hidden) {
    if (!op.gpu.lm_head_tp) {
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
        auto logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
        checked(shelf_hy4_head(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                               static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                               static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                               partial, logits, hidden.rows, op.stream));
        return logits;
    }
    const int n = op.gpu.ranks;
    const int local_vocab = hy4_vocab_size / n;
    if (op.gpu.tp_size > 1) {
        float *logits = op.rank == 0
            ? static_cast<float *>(op.arena.next(size_t(hidden.rows) * hy4_vocab_size * 4))
            : nullptr;
        const size_t work_mark = op.arena.cursor;
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        auto local_logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        checked(shelf_hy4_head(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                               static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                               static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                               partial, local_logits, hidden.rows, op.stream, local_vocab));
        op.gpu.tp_gather_logits(local_logits, logits, op.rank, hidden.rows, local_vocab, hy4_vocab_size);
        op.arena.cursor = work_mark;
        return logits;
    }
    float *logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * hy4_vocab_size * 4));
    if (op.capture) {
        const size_t work_mark = op.arena.cursor;
        std::array<void *, 4> hidden_ptrs{};
        op.gpu.lm_scatter_hidden(hidden.p, size_t(hidden.rows) * lm_hidden_row_bytes, hidden_ptrs, true);
        std::array<float *, 4> recv_logits{};
        for (int r = 1; r < n; ++r)
            recv_logits[r] = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        op.capture->plan.head_steps.emplace_back();
        auto &hs = op.capture->plan.head_steps.back();
        op.capture->finish(hs.scatter);
        for (int r = 1; r < n; ++r) {
            op.capture->begin(r);
            auto &arena = op.gpu.workspace[r];
            arena.cursor = 0;
            auto codes = static_cast<int8_t *>(arena.next(size_t(hidden.rows) * 3072));
            auto scales = static_cast<float *>(arena.next(size_t(hidden.rows) * 24 * 4));
            auto partial = static_cast<float *>(arena.next(size_t(hidden.rows) * local_vocab * 4));
            auto local_logits = static_cast<float *>(arena.next(size_t(hidden.rows) * local_vocab * 4));
            checked(shelf_hy4_head(static_cast<const uint16_t *>(hidden_ptrs[r]),
                                   static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", r))),
                                   static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", r))),
                                   static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", r))),
                                   codes, scales, partial, local_logits, hidden.rows, op.gpu.compute[r], local_vocab));
            peer_copy(recv_logits[r], local_logits, size_t(hidden.rows) * local_vocab * 4, op.gpu.compute[r]);
            op.capture->finish(hs.rank_exec[r]);
        }
        op.capture->begin(0);
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        auto local_logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        checked(shelf_hy4_head(static_cast<const uint16_t *>(hidden_ptrs[0]),
                               static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", 0))),
                               static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", 0))),
                               static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", 0))),
                               codes, scales, partial, local_logits, hidden.rows, op.stream, local_vocab));
        tp_scatter_logits(local_logits, logits, hidden.rows, local_vocab, hy4_vocab_size, 0, op.stream);
        op.capture->finish(hs.rank_exec[0]);
        op.capture->begin(0);
        for (int r = 1; r < n; ++r)
            tp_scatter_logits(recv_logits[r], logits, hidden.rows, local_vocab, hy4_vocab_size, r * local_vocab,
                              op.stream);
        op.arena.cursor = work_mark;
        return logits;
    }
    const int chunk_rows = 192;
    for (int first = 0; first < hidden.rows; first += chunk_rows) {
        const int count = std::min(chunk_rows, hidden.rows - first);
        const size_t work_mark = op.arena.cursor;
        std::array<void *, 4> hidden_ptrs{};
        op.gpu.lm_scatter_hidden(hidden.row(first, count).p, size_t(count) * lm_hidden_row_bytes, hidden_ptrs, false);
        std::array<const float *, 4> local_logits{};
        for (int r = 0; r < n; ++r) {
            checked(hipSetDevice(op.gpu.physical[r]));
            Region &arena = r == 0 ? op.arena : op.gpu.workspace[r];
            if (r > 0)
                arena.cursor = 0;
            auto codes = static_cast<int8_t *>(arena.next(size_t(count) * 3072));
            auto scales = static_cast<float *>(arena.next(size_t(count) * 24 * 4));
            auto partial = static_cast<float *>(arena.next(size_t(count) * local_vocab * 4));
            auto local = static_cast<float *>(arena.next(size_t(count) * local_vocab * 4));
            local_logits[r] = local;
            checked(shelf_hy4_head(static_cast<const uint16_t *>(hidden_ptrs[r]),
                                   static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", r))),
                                   static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", r))),
                                   static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", r))),
                                   codes, scales, partial, local, count, op.gpu.compute[r], local_vocab));
        }
        checked(hipSetDevice(op.gpu.physical[0]));
        op.gpu.lm_gather_logits(local_logits, count, local_vocab, hy4_vocab_size,
                                logits + size_t(first) * hy4_vocab_size);
        op.arena.cursor = work_mark;
        for (int r = 1; r < n; ++r)
            op.gpu.workspace[r].cursor = 0;
    }
    return logits;
}
void Target::head_argmax(Mat hidden, int *tokens) {
    if (!op.gpu.lm_head_tp) {
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
        auto tile_vals = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 944 * 4));
        auto tile_ids = static_cast<int *>(op.arena.next(size_t(hidden.rows) * 944 * 4));
        checked(shelf_hy4_head_argmax(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                                      static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                                      static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                                      partial, tile_vals, tile_ids, tokens, hidden.rows, op.stream));
        return;
    }
    const int n = op.gpu.ranks;
    const int local_vocab = hy4_vocab_size / n;
    const int tiles = local_vocab / hy4_lm_head_tile;
    if (op.gpu.tp_size > 1) {
        const size_t mark = op.arena.cursor;
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        auto tile_vals = static_cast<float *>(op.arena.next(size_t(hidden.rows) * tiles * 4));
        auto tile_ids = static_cast<int *>(op.arena.next(size_t(hidden.rows) * tiles * 4));
        auto best_vals = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 4));
        auto best_ids = static_cast<int *>(op.arena.next(size_t(hidden.rows) * 4));
        checked(shelf_hy4_head_argmax(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                                      static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                                      static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                                      partial, tile_vals, tile_ids, nullptr, hidden.rows, op.stream, local_vocab,
                                      op.rank * local_vocab, best_vals, best_ids));
        op.gpu.tp_argmax(best_vals, best_ids, tokens, op.rank, hidden.rows);
        op.arena.cursor = mark;
        return;
    }
    if (op.capture) {
        const size_t mark = op.arena.cursor;
        std::array<void *, 4> hidden_ptrs{};
        op.gpu.lm_scatter_hidden(hidden.p, size_t(hidden.rows) * lm_hidden_row_bytes, hidden_ptrs, true);
        op.capture->plan.head_steps.emplace_back();
        auto &hs = op.capture->plan.head_steps.back();
        op.capture->finish(hs.scatter);
        for (int r = 1; r < n; ++r) {
            op.capture->begin(r);
            auto &arena = op.gpu.workspace[r];
            arena.cursor = 0;
            auto codes = static_cast<int8_t *>(arena.next(size_t(hidden.rows) * 3072));
            auto scales = static_cast<float *>(arena.next(size_t(hidden.rows) * 24 * 4));
            auto partial = static_cast<float *>(arena.next(size_t(hidden.rows) * local_vocab * 4));
            auto tile_vals = static_cast<float *>(arena.next(size_t(hidden.rows) * tiles * 4));
            auto tile_ids = static_cast<int *>(arena.next(size_t(hidden.rows) * tiles * 4));
            auto best_vals = static_cast<float *>(arena.next(size_t(hidden.rows) * 4));
            auto best_ids = static_cast<int *>(arena.next(size_t(hidden.rows) * 4));
            auto local_pairs = arena.next(size_t(hidden.rows) * 8);
            checked(shelf_hy4_head_argmax(
                static_cast<const uint16_t *>(hidden_ptrs[r]),
                static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", r))),
                static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", r))),
                static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", r))), codes, scales,
                partial, tile_vals, tile_ids, nullptr, hidden.rows, op.gpu.compute[r], local_vocab,
                r * local_vocab, best_vals, best_ids));
            tp_argmax_pack(best_vals, best_ids, local_pairs, hidden.rows, op.gpu.compute[r]);
            peer_copy(static_cast<char *>(op.gpu.lm_head_recv[0]) + size_t(r) * lm_argmax_stride, local_pairs,
                      size_t(hidden.rows) * 8, op.gpu.compute[r]);
            op.capture->finish(hs.rank_exec[r]);
        }
        op.capture->begin(0);
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * local_vocab * 4));
        auto tile_vals = static_cast<float *>(op.arena.next(size_t(hidden.rows) * tiles * 4));
        auto tile_ids = static_cast<int *>(op.arena.next(size_t(hidden.rows) * tiles * 4));
        auto best_vals = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 4));
        auto best_ids = static_cast<int *>(op.arena.next(size_t(hidden.rows) * 4));
        checked(shelf_hy4_head_argmax(
            static_cast<const uint16_t *>(hidden_ptrs[0]),
            static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", 0))),
            static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", 0))),
            static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", 0))), codes, scales,
            partial, tile_vals, tile_ids, nullptr, hidden.rows, op.stream, local_vocab, 0, best_vals, best_ids));
        tp_argmax_pack(best_vals, best_ids, op.gpu.lm_head_recv[0], hidden.rows, op.stream);
        op.capture->finish(hs.rank_exec[0]);
        op.capture->begin(0);
        tp_argmax_reduce(op.gpu.lm_head_recv[0], lm_argmax_max_rows, n, hidden.rows, tokens, op.stream);
        op.arena.cursor = mark;
        return;
    }
    const int chunk_rows = 192;
    for (int first = 0; first < hidden.rows; first += chunk_rows) {
        const int count = std::min(chunk_rows, hidden.rows - first);
        const size_t mark = op.arena.cursor;
        std::array<void *, 4> hidden_ptrs{};
        op.gpu.lm_scatter_hidden(hidden.row(first, count).p, size_t(count) * lm_hidden_row_bytes, hidden_ptrs, false);
        std::array<const float *, 4> best_vals{};
        std::array<const int *, 4> best_ids{};
        for (int r = 0; r < n; ++r) {
            checked(hipSetDevice(op.gpu.physical[r]));
            Region &arena = r == 0 ? op.arena : op.gpu.workspace[r];
            if (r > 0)
                arena.cursor = 0;
            auto codes = static_cast<int8_t *>(arena.next(size_t(count) * 3072));
            auto scales = static_cast<float *>(arena.next(size_t(count) * 24 * 4));
            auto partial = static_cast<float *>(arena.next(size_t(count) * local_vocab * 4));
            auto tile_vals = static_cast<float *>(arena.next(size_t(count) * tiles * 4));
            auto tile_ids = static_cast<int *>(arena.next(size_t(count) * tiles * 4));
            auto vals_r = static_cast<float *>(arena.next(size_t(count) * 4));
            auto ids_r = static_cast<int *>(arena.next(size_t(count) * 4));
            best_vals[r] = vals_r;
            best_ids[r] = ids_r;
            checked(shelf_hy4_head_argmax(
                static_cast<const uint16_t *>(hidden_ptrs[r]),
                static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.suh", r))),
                static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at("lm_head.svh", r))),
                static_cast<const uint32_t *>(op.gpu.device(op.gpu.model.at("lm_head.trellis", r))), codes, scales,
                partial, tile_vals, tile_ids, nullptr, count, op.gpu.compute[r], local_vocab, r * local_vocab, vals_r,
                ids_r));
        }
        checked(hipSetDevice(op.gpu.physical[0]));
        op.gpu.lm_gather_argmax(best_vals, best_ids, count, tokens + first);
        op.arena.cursor = mark;
        for (int r = 1; r < n; ++r)
            op.gpu.workspace[r].cursor = 0;
    }
}
std::vector<int> Target::predict(Mat hidden, int position, bool draft, const Path &dump,
                                 float *distribution, float *logits, int *tokens) {
    std::chrono::steady_clock::time_point begin;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0])
        begin = std::chrono::steady_clock::now();
    size_t mark = op.arena.cursor;
    int *device_ids = tokens;
    if (!device_ids) {
        device_ids = static_cast<int *>(op.arena.next(hidden.rows * sizeof(int)));
        if (!logits && temperature == 0 && !distribution && dump.empty()) {
            head_argmax(hidden, device_ids);
        } else {
            if (!logits)
                logits = head_logits(hidden);
            if (op.rank == 0 || !op.gpu.lm_head_tp || op.gpu.tp_size == 1) {
                if (!dump.empty()) {
                    std::vector<float> host(120832);
                    op.copy(host.data(), logits + size_t(hidden.rows - 1) * 120832, host.size() * 4,
                            hipMemcpyDeviceToHost);
                    std::ofstream file(dump, std::ios::binary);
                    if (!file.write(reinterpret_cast<const char *>(host.data()), host.size() * 4))
                        throw std::runtime_error("logits を保存できません");
                }
                sample(logits, hidden.rows, position, seed ^ (draft ? 0x735a7ec51b09ull : 0ull), device_ids,
                       distribution);
            }
            if (op.gpu.tp_size > 1 && op.gpu.lm_head_tp)
                op.gpu.tp_broadcast_ids(device_ids, op.rank, hidden.rows);
        }
    }
    std::vector<int> result(hidden.rows);
    op.copy(result.data(), device_ids, result.size() * 4, hipMemcpyDeviceToHost);
    op.arena.cursor = mark;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0])
        head_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (std::find(result.begin(), result.end(), -1) != result.end())
        throw std::runtime_error("有効な出力 logit がありません");
    return result;
}
void Target::sample(float *logits, int rows, int position, uint64_t rng, int *ids, float *distribution,
                    const Hy4StepParams *params) {
    if (temperature > 0 && top_k > 0) {
        truncate_logits<<<rows, 256, 0, op.stream>>>(logits, top_k, top_p, temperature);
        checked(hipGetLastError());
    }
    if (distribution) {
        probabilities<<<rows, 256, 0, op.stream>>>(logits, distribution, temperature);
        checked(hipGetLastError());
    }
    best_ids<<<rows, 256, 0, op.stream>>>(logits, ids, position, rng, temperature, nullptr, params);
    checked(hipGetLastError());
}
Mat Target::layers(const std::vector<int> &ids, bool verify, bool need_head) {
    int start = history.size();
    op.prefill_ids = nullptr;
    if (!op.capture) {
        for (int id : ids)
            if (id < 0 || id >= 120832)
                throw std::runtime_error("token ID が範囲外です");
        op.prefill_ids = op.indices(ids);
    }
    Mat x = op.embed_device(op.capture ? op.params->ids : op.prefill_ids, ids.size(), true);
    for (int layer = 0; layer < 42; ++layer) {
        Selection selection;
        decoder_layer(op, layer, start, x, history, ids, selection);
        if (!layer && op.capture) {
            op.capture->finish(op.capture->plan.entry);
            op.capture->begin(0);
        }
        auto it = std::find(sources.begin(), sources.end(), layer);
        if (!op.capture && it != sources.end()) {
            if (op.slot_draft && need_head && start + ids.size() > 2048) {
                checked(hipMemcpyAsync(op.gpu.sparse_top_slot(layer, op.slot),
                                       op.gpu.sparse_top[op.rank][layer] + (ids.size() - 1) * 511,
                                       511 * sizeof(int), hipMemcpyDeviceToDevice, op.stream));
                op.gpu.sparse_last[op.slot][layer] = 0;
            }
            if (!verify)
                selection = Selection{std::move(selection.back())};
            tapped[it - sources.begin()] = std::move(selection);
        }
    }
    op.prefill_ids = nullptr;
    if (!need_head)
        return {};
    if (!verify)
        x = x.row(x.rows - 1);
    float *head_post;
    Mat h = op.collapse_norm(x, "model.hc_head.hc_head_", head_post, "model.norm.weight");
    checked(hipMemcpyAsync(state.row(op.gpu.batch_draft ? max_verify_rows * op.slot : 0).p, h.p, h.bytes(), hipMemcpyDeviceToDevice, op.stream));
    return h;
}
std::vector<int> Target::forward(const std::vector<int> &ids, bool verify, bool need_head, const Path &dump) {
    const int rows = int(ids.size()), start = int(history.size());
    if (ids.empty() || history.size() + ids.size() > op.gpu.tokens || (verify && rows > max_verify_rows))
        throw std::runtime_error("入力列が KV の範囲外です");
    if (std::any_of(ids.begin(), ids.end(), [](int id) { return id < 0 || id >= 120832; }))
        throw std::runtime_error("token ID が範囲外です");
    auto &gpu = op.gpu;
    if (gpu.kv_pages_total > 0 && !gpu.allocate_slot_pages(op.slot, size_t(start) + size_t(rows)))
        throw std::runtime_error("KV ページが不足しています");
    op.arena.cursor = base;
    if (gpu.graph_input && target_graph && !gpu.target_capture_failed && need_head && dump.empty() &&
        Hy4MoePlan::supports(rows) && (verify || rows == 1)) {
        const int dense = std::clamp(2048 - start, 0, rows);
        int bucket = std::max(2048, (start + rows + 1023) / 1024 * 1024);
        GraphKey key{rows, dense, bucket, verify, base, temperature == 0 ? 0.0f : 1.0f, 0, 0, 0, verify ? rows - 1 : 0};
        auto plan = capture_cached(op, gpu.target_graphs, gpu.target_victim, gpu.target_capture_failed, key,
            [&](GraphPlan &slot) {
                Mat hidden = layers(ids, verify, true);
                if (temperature == 0) {
                    slot.tokens = static_cast<int *>(op.arena.next(size_t(rows) * sizeof(int)));
                    slot.logits = nullptr;
                    head_argmax(hidden, slot.tokens);
                } else {
                    slot.logits = head_logits(hidden);
                    slot.tokens = nullptr;
                }
                slot.cursor = op.arena.cursor;
            });
        if (plan) {
            GraphInput input{};
            input.step.position = start;
            input.step.rows = rows;
            std::copy(ids.begin(), ids.end(), input.step.ids);
            checked(hipMemcpyAsync(&gpu.graph_input->step, &input.step, sizeof(input.step),
                                   hipMemcpyHostToDevice, op.stream));
            checked(hipGraphLaunch(plan->entry, op.stream));
            for (int g = 0; g < 4; ++g) {
                auto packed = gather_engram(gpu.model, 1 + g * 10, rows, ids, history);
                std::copy(packed.begin(), packed.end(), input.engram[g]);
            }
            checked(hipMemcpyAsync(gpu.graph_input->engram, input.engram, sizeof(input.engram),
                                   hipMemcpyHostToDevice, op.stream));
            plan->launch(gpu);
            ++gpu.target_replays;
            for (size_t k = 0; k < sources.size(); ++k) {
                op.sparse_last_layer[sources[k]] = rows - 1;
                tapped[k].assign(rows, {});
                for (int i = 0; i < dense; ++i) {
                    tapped[k][i].resize(start + i + 1);
                    std::iota(tapped[k][i].begin(), tapped[k][i].end(), 0);
                }
            }
            history.insert(history.end(), ids.begin(), ids.end());
            op.arena.cursor = plan->cursor;
            return predict(state.row(0, rows), start + 1, false, dump, verify ? target_probs : nullptr, plan->logits,
                           plan->tokens);
        }
        op.arena.cursor = base;
    }
    std::vector<int> prior;
    if (history.size() > 3)
        prior.assign(history.end() - 3, history.end());
    else
        prior = history;
    auto prefetch = std::getenv("SHELF_HY4_ENGRAM_PREFETCH");
    if (!prefetch || prefetch[0] != '0')
        op.prefetch_engram(ids, prior);
    Mat h = layers(ids, verify, need_head);
    history.insert(history.end(), ids.begin(), ids.end());
    return need_head ? predict(h, history.size() - h.rows + 1, false, dump, verify ? target_probs : nullptr)
                     : std::vector<int>{};
}
std::vector<int> Target::draft_steps(int anchor, int *device_ids) {
    op.arena.cursor = base;
    const int drafts = draft_tokens;
    const int block_rows = drafts + 1;
    Mat x;
    if (device_ids)
        x = op.embed_device(device_ids, block_rows, false);
    else {
        std::vector<int> block(block_rows, 120009);
        block[0] = anchor;
        x = op.embed(block, false);
    }
    std::string p = "model.dfly_layers.";
    auto e = op.norm(x.row(0), p + "mtp_fusion.enorm.weight");
    auto h = op.norm(state.row(0), p + "mtp_fusion.hnorm.weight");
    auto fused = op.linear(p + "mtp_fusion.eh_proj", op.join(e, h));
    fuse_anchor<<<12, 256, 0, op.stream>>>(x.row(0), fused,
                             static_cast<const uint16_t *>(op.param(p + "mtp_fusion.gate").p));
    checked(hipGetLastError());
    int start = history.size();
    size_t save = op.arena.cursor;
    for (int i = 0; i < 5; ++i) {
        std::string prefix = p + std::to_string(i) + ".";
        Selection select;
        if (!device_ids) {
            select.assign(block_rows, tapped[i][0]);
            for (int row = 0; row < block_rows; ++row)
                for (int j = 0; j <= row; ++j)
                    select[row].push_back(start + j);
        }
        auto a = op.norm(x, prefix + "input_layernorm.weight");
        auto y = op.add(x, attention(op, prefix, sources[i], start, a, select, true));
        auto f = feedforward(op, prefix + "mlp", op.norm(y, prefix + "post_attention_layernorm.weight"));
        op.add_into(y, f, x);
        op.arena.cursor = save;
    }
    x = op.norm(x, p + "final_norm.weight");
    std::string correction = p + "hidden_correction.";
    int previous = anchor;
    std::vector<int> draft;
    for (int i = 1; i <= drafts; ++i) {
        auto q = op.norm(x.row(i), correction + "hidden_norm.weight");
        Mat embedding;
        if (device_ids) {
            embedding = op.embed_device(device_ids + i - 1, 1, false);
        } else {
            embedding = op.embed({previous}, false);
        }
        auto k = op.norm(embedding, correction + "embed_norm.weight");
        auto corrected = op.add(x.row(i), feedforward(op, p + "hidden_correction", op.join(q, k)));
        float *distribution = draft_probs ? draft_probs + (i - 1) * 120832 : nullptr;
        if (device_ids) {
            size_t mark = op.arena.cursor;
            if (temperature == 0 && !distribution)
                head_argmax(corrected, device_ids + i);
            else
                sample(head_logits(corrected), 1, i, 0x735a7ec51b09ull, device_ids + i, distribution, op.params);
            op.arena.cursor = mark;
        } else {
            previous = predict(corrected, start + i, true, {}, distribution)[0];
            draft.push_back(previous);
        }
    }
    return draft;
}
std::vector<int> Target::propose(int anchor) {
    if (anchor < 0 || anchor >= 120832)
        throw std::runtime_error("token ID が範囲外です");
    auto &gpu = op.gpu;
    if (!gpu.graph_input || gpu.draft_capture_failed || !draft_graph)
        return draft_steps(anchor, nullptr);
    const int drafts = draft_tokens;
    const int block_rows = drafts + 1;
    int start = history.size();
    bool dense = start <= 2048;
    int bucket = dense ? 2048 : (start + block_rows + 1023) / 1024 * 1024;
    GraphKey key{block_rows, int(dense), bucket, false, base, temperature, top_p, top_k, 0, drafts};
    auto plan = capture_cached(op, gpu.draft_graphs, gpu.draft_victim, gpu.draft_capture_failed, key,
        [&](GraphPlan &) { draft_steps(anchor, gpu.graph_input->step.ids); });
    if (!plan)
        return draft_steps(anchor, nullptr);
    Hy4StepParams params{};
    params.position = start;
    params.rows = block_rows;
    params.seed = seed;
    params.draft_last = op.sparse_last_layer[sources[0]];
    params.ids[0] = anchor;
    std::fill_n(params.ids + 1, drafts, 120009);
    checked(hipMemcpyAsync(&gpu.graph_input->step, &params, sizeof(params), hipMemcpyHostToDevice, op.stream));
    plan->launch(gpu);
    ++gpu.draft_replays;
    std::vector<int> result(drafts);
    op.copy(result.data(), gpu.graph_input->step.ids + 1, drafts * sizeof(int), hipMemcpyDeviceToHost);
    if (std::find(result.begin(), result.end(), -1) != result.end())
        throw std::runtime_error("有効な出力 logit がありません");
    return result;
}
void Target::accept(int keep, int submitted) {
    history.resize(history.size() - submitted + keep);
    if (keep > 1)
        checked(hipMemcpyAsync(state.p, state.row(keep - 1).p, 3072 * 2, hipMemcpyDeviceToDevice, op.stream));
    for (auto layer : sources)
        op.sparse_last_layer[layer] = keep - 1;
    for (auto &s : tapped)
        s = Selection{std::move(s[keep - 1])};
}

int verify_speculation(Ops &op, const float *target_probs, const float *draft_probs, int position,
                       uint64_t seed, const std::vector<int> &draft, std::vector<int> &actual) {
    const int drafts = int(draft.size());
    if (!target_probs) {
        int keep = 1;
        while (keep <= drafts && actual[keep - 1] == draft[keep - 1])
            ++keep;
        return keep;
    }
    auto ids = op.indices(draft);
    auto out = static_cast<int *>(op.arena.next(drafts * sizeof(int)));
    test_drafts<<<1, drafts, 0, op.stream>>>(target_probs, draft_probs, ids, position, seed, out);
    checked(hipGetLastError());
    int accepted[max_draft_tokens]{};
    op.copy(accepted, out, drafts * sizeof(int), hipMemcpyDeviceToHost);
    for (int i = 0; i < drafts; ++i) {
        if (accepted[i]) {
            actual[i] = draft[i];
            continue;
        }
        best_ids<<<1, 256, 0, op.stream>>>(target_probs + i * 120832, out, position + i, seed ^ 0xc1f8efb6703ull, 1.f,
                             draft_probs + i * 120832);
        checked(hipGetLastError());
        int replacement;
        op.copy(&replacement, out, sizeof(int), hipMemcpyDeviceToHost);
        if (replacement >= 0)
            actual[i] = replacement;
        return i + 1;
    }
    return drafts + 1;
}


void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selection) {
    auto now = std::chrono::steady_clock::now;
    std::chrono::steady_clock::time_point start_time;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        start_time = now();
    }
    size_t save = op.arena.cursor;
    auto p = "model.layers." + std::to_string(layer) + ".";
    if (layer % 10 == 1 && layer <= 31) {
        std::vector<int> prior;
        if (history.size() > 3)
            prior.assign(history.end() - 3, history.end());
        else
            prior = history;
        engram(op, layer, x, prior, ids);
        if (x.rows > Hy4GroupedPlan::max_rows) op.arena.cursor = save;
    }
    float *post;
    Mat a = op.collapse_norm(x, p + "hc_attn_layer.hc_pre.hc_", post, p + "input_layernorm.weight");
    a = attention(op, p, layer, start, a, selection, false);
    op.expand(x, a, post);
    op.arena.cursor = save;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        auto end = now();
        op.gpu.attention_ms += std::chrono::duration<double, std::milli>(end - start_time).count();
        start_time = end;
    }
    Mat f = op.collapse_norm(x, p + "hc_mlp_layer.hc_pre.hc_", post, p + "post_attention_layernorm.weight");
    if (layer == 0)
        f = op.norm(feedforward(op, p + "mlp", f), p + "mlp.post_fc2_norm.weight");
    else
        f = moe(op, layer, f);
    op.expand(x, f, post);
    op.arena.cursor = save;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        op.gpu.ffn_ms += std::chrono::duration<double, std::milli>(now() - start_time).count();
    }
}
Hy4DflyAdaptiveController::Hy4DflyAdaptiveController(int initial_k, int switch_interval, float alpha)
    : current_k(Residency::resolve_draft_tokens(initial_k)), min_switch_interval(switch_interval), ema_alpha(alpha) {
    if (switch_interval < 1 || !std::isfinite(alpha) || alpha <= 0.f || alpha > 1.f)
        throw std::invalid_argument("Adaptive K の更新間隔または EMA 係数が不正です");
}
void Hy4DflyAdaptiveController::reset(int initial_k) {
    current_k = Residency::resolve_draft_tokens(initial_k);
    acc_ema = 0.0;
    has_sample = false;
    rounds = 0;
    rounds_since_switch = 0;
}
int Hy4DflyAdaptiveController::select_k(int current_k, double acc_ema, int batch_size) {
    if (batch_size > 16) {
        const double thr23 = current_k <= 2 ? 1.95 : 1.65;
        const double thr34 = current_k <= 3 ? 3.35 : 3.05;
        if (acc_ema < thr23)
            return 2;
        if (acc_ema < thr34)
            return 3;
        return 4;
    }
    const double thr23 = current_k <= 2 ? 1.25 : 1.05;
    const double thr34 = current_k <= 3 ? 1.95 : 1.65;
    const double thr45 = current_k <= 4 ? 3.35 : 3.05;
    if (acc_ema < thr23)
        return 2;
    if (acc_ema < thr34)
        return 3;
    if (acc_ema < thr45)
        return 4;
    return 5;
}
int Hy4DflyAdaptiveController::observe(double accepted_per_lane, int batch_size) {
    if (batch_size < 1 || !std::isfinite(accepted_per_lane) || accepted_per_lane < 0.0)
        throw std::invalid_argument("Adaptive K の観測値が不正です");
    if (!has_sample) {
        acc_ema = accepted_per_lane;
        has_sample = true;
    } else {
        acc_ema = (1.0 - double(ema_alpha)) * acc_ema + double(ema_alpha) * accepted_per_lane;
    }
    ++rounds;
    ++rounds_since_switch;
    if (rounds_since_switch < size_t(min_switch_interval))
        return current_k;
    int next_k = select_k(current_k, acc_ema, batch_size);
    if (next_k != current_k) {
        current_k = next_k;
        ++switches;
        rounds_since_switch = 0;
    }
    return current_k;
}

std::vector<int> generate(Residency &gpu, const std::vector<int> &ids, int count, int chunk, float temperature,
              uint64_t seed, const Path &logits_file, int top_k, float top_p, const std::function<void(int)> &emit,
              bool ignore_eos, Hy4PrefixSlot *prefix_slot, Hy4PrefixStats *prefix_stats, size_t *restored_out,
              int draft_tokens, bool adaptive, Hy4DflyStats *stats_out) {
    if (ids.empty() || !count || ids.size() + count > gpu.tokens || gpu.tokens > 1048576 || chunk < 1 ||
        chunk > max_prefill_chunk)
        throw std::runtime_error("入力・出力長または prefill chunk が範囲外です");
    if (gpu.tp_size > 1)
        chunk = std::min(chunk, tp_prefill_rows);
    if (top_k < 0 || top_k > 64 || !(top_p > 0 && top_p <= 1) || (temperature > 0 && top_p < 1 && !top_k))
        throw std::invalid_argument("top_k は 0〜64、top_p を使う場合は top_k が必要です");
    const int resolved_drafts = Residency::resolve_draft_tokens(draft_tokens ? draft_tokens : gpu.draft_tokens);
    const bool use_adaptive = adaptive || gpu.adaptive_draft;
    size_t restored = 0;
    bool use_snap = false;
    std::vector<std::vector<uint8_t>> pending_restore;
    if (gpu.kv_pages_total > 0 && gpu.pages_for_tokens(ids.size() + size_t(count)) > gpu.kv_pages_total)
        throw std::runtime_error("要求長が KV プール容量を超えています");
    struct SlotPageGuard {
        Residency &gpu;
        bool release = true;
        ~SlotPageGuard() {
            if (release && gpu.kv_pages_total > 0)
                gpu.release_slot_pages(0);
        }
    } page_guard{gpu, true};
    if (prefix_slot) {
        if (prefix_stats && prefix_stats->enabled) {
            restored = match_hy4_prefix(*prefix_slot, ids, gpu, use_snap);
            if (gpu.kv_pages_total > 0 && restored > 0 &&
                prefix_slot->pages.size() < gpu.pages_for_tokens(restored)) {
                restored = 0;
                use_snap = false;
            }
            if (restored > 0) {
                ++prefix_stats->hits;
                ++prefix_stats->restores;
                if (use_snap)
                    pending_restore = std::move(prefix_slot->swa_snap);
            } else {
                ++prefix_stats->misses;
            }
            prefix_stats->reuse_slots = gpu.max_slots;
            prefix_stats->entries = prefix_slot->valid ? 1 : 0;
        }
        gpu.release_slot_pages(0);
        if (gpu.kv_pages_total > 0 && restored > 0) {
            const size_t keep = gpu.pages_for_tokens(restored);
            for (size_t pi = 0; pi < keep; ++pi)
                gpu.slot_page_tables[0].push_back(prefix_slot->pages[pi]);
            gpu.retain_pages(gpu.slot_page_tables[0]);
            gpu.sync_slot_page_table(0);
        }
        gpu.release_pages(prefix_slot->pages);
        prefix_slot->valid = false;
        prefix_slot->snap_valid = false;
        prefix_slot->tokens.clear();
        prefix_slot->max_written = 0;
        prefix_slot->snap_pos = 0;
        prefix_slot->snap_max_written = 0;
        if (prefix_stats && prefix_stats->enabled && gpu.cache[0][2])
            prefix_slot->swa_snap.resize(gpu.ranks);
        else
            prefix_slot->swa_snap.clear();
        if (prefix_stats)
            prefix_stats->entries = 0;
    } else if (gpu.kv_pages_total > 0) {
        gpu.release_slot_pages(0);
    }
    if (gpu.kv_pages_total > 0) {
        const size_t reserve_toks = std::min(gpu.tokens, ids.size() + size_t(count) + (gpu.model.has_draft ? 5 : 0));
        const size_t target_toks = gpu.pages_for_tokens(reserve_toks) <= gpu.kv_pages_total
            ? reserve_toks
            : ids.size() + size_t(count);
        if (!gpu.allocate_slot_pages(0, target_toks))
            throw std::runtime_error("KV ページ割当に失敗しました");
    }
    if (restored_out) *restored_out = restored;
    struct Lane {
        std::vector<int> generated;
        std::vector<int> history;
        size_t max_written = 0;
        double prefill = 0, decode = 0, draft = 0, head = 0;
        double pp_attn = 0, pp_ffn = 0, pp_transfer = 0, pp_head = 0;
        uint64_t pp_routes = 0;
        double pp_expert = 0;
        std::array<uint64_t, 4> pp_routes_all{};
        std::array<double, 4> pp_expert_all{};
        int rounds = 0, accepted = 0, final_drafts = default_draft_tokens;
        size_t k_switches = 0;
        std::array<int, max_verify_rows> histogram{};
    };
    std::array<Lane, 4> lane;
    std::exception_ptr failure;
    bool failure_is_abort = false;
    std::mutex failure_mutex;
    std::vector<int> shared_tokens;
    int shared_keep = 0;
    gpu.reset_collectives();
    auto run = [&](int rank) {
        try {
            checked(hipSetDevice(gpu.physical[rank]));
            auto now = std::chrono::steady_clock::now;
            Target target(gpu, temperature, seed, top_k, top_p, rank, resolved_drafts);
            if (use_snap && restored > 0 && size_t(rank) < pending_restore.size())
                restore_swa_snapshot(gpu, 0, rank, target.op.stream, pending_restore[rank]);
            if (restored > 0)
                target.history.assign(ids.begin(), ids.begin() + restored);
            Hy4DflyAdaptiveController controller(resolved_drafts);
            auto begin = now();
            int next = 0;
            for (size_t first = restored; first < ids.size(); first += chunk) {
                auto end = std::min(first + chunk, ids.size());
                auto output = target.forward({ids.begin() + first, ids.begin() + end}, false,
                                             end == ids.size(), rank == 0 ? logits_file : Path{});
                if (!output.empty())
                    next = output.back();
            }
            size_t max_written = ids.size();
            if (prefix_slot && prefix_stats && prefix_stats->enabled &&
                size_t(rank) < prefix_slot->swa_snap.size() && align_hy4_prefix(ids.size(), ids.size() + 1) > 0)
                save_swa_snapshot(gpu, 0, rank, target.op.stream, prefix_slot->swa_snap[rank]);
            gpu.collect_route_counts(rank);
            auto prefill = now();
            lane[rank].pp_attn = gpu.attention_ms;
            lane[rank].pp_ffn = gpu.ffn_ms;
            lane[rank].pp_transfer = gpu.transfer_ms;
            lane[rank].pp_head = target.head_ms;
            lane[rank].pp_routes = gpu.routed_rows[rank];
            lane[rank].pp_expert = gpu.expert_ms[rank];
            if (rank == 0 && gpu.tp_size == 1) {
                lane[0].pp_routes_all = gpu.routed_rows;
                lane[0].pp_expert_all = gpu.expert_ms;
            }
            auto broadcast = [&](std::vector<int> &tokens, int &keep) {
                if (gpu.tp_size == 1) return;
                if (rank == 0) {
                    shared_tokens = tokens;
                    shared_keep = keep;
                }
                gpu.all_barrier->arrive_and_wait();
                tokens = shared_tokens;
                keep = shared_keep;
                gpu.all_barrier->arrive_and_wait();
            };
            std::vector<int> initial{next};
            int shared_value = 0;
            if (rank == 0 && emit) emit(next);
            broadcast(initial, shared_value);
            std::vector<int> generated{initial[0]};
            auto append = [&](int id) {
                generated.push_back(id);
                if (rank == 0 && emit) emit(id);
            };
            auto &stats = lane[rank];
            while (int(generated.size()) < count && (ignore_eos || generated.back() != 120025)) {
                int anchor = generated.back();
                const int drafts = target.draft_tokens;
                const int block_rows = drafts + 1;
                const size_t draft_end = target.history.size() + size_t(block_rows);
                const bool draft_fits_kv = gpu.kv_pages_total == 0 ||
                    gpu.pages_for_tokens(draft_end) <= gpu.slot_page_tables[0].size();
                if (!gpu.model.has_draft || draft_end > gpu.tokens || !draft_fits_kv ||
                    int(generated.size()) + 1 == count) {
                    max_written = std::max(max_written, target.history.size() + 1);
                    auto output = target.forward({anchor}, false);
                    if (rank == 0 && emit) emit(output[0]);
                    broadcast(output, shared_value);
                    generated.push_back(output[0]);
                    continue;
                }
                std::chrono::steady_clock::time_point draft_begin;
                if (gpu.tp_size == 1 && target.op.rank == 0 && gpu.expert_begin[0])
                    draft_begin = now();
                auto draft = target.propose(anchor);
                if (gpu.tp_size == 1 && target.op.rank == 0 && gpu.expert_begin[0])
                    stats.draft += std::chrono::duration<double, std::milli>(now() - draft_begin).count();
                broadcast(draft, shared_value);
                std::vector<int> block{anchor};
                block.insert(block.end(), draft.begin(), draft.end());
                max_written = std::max(max_written, target.history.size() + size_t(block_rows));
                auto actual = target.forward(block, true);
                int keep = verify_speculation(target.op, target.target_probs, target.draft_probs,
                                              target.history.size() - drafts, seed, draft, actual);
                broadcast(actual, keep);
                ++stats.rounds;
                stats.accepted += keep - 1;
                ++stats.histogram[keep - 1];
                target.accept(keep, block_rows);
                if (use_adaptive)
                    target.draft_tokens = controller.observe(double(keep - 1), 1);
                for (int i = 0; i < keep && int(generated.size()) < count; ++i) {
                    append(actual[i]);
                    if (!ignore_eos && actual[i] == 120025)
                        break;
                }
            }
            gpu.collect_route_counts(rank);
            stats.generated = std::move(generated);
            stats.history = std::move(target.history);
            stats.max_written = max_written;
            stats.head = target.head_ms;
            stats.final_drafts = target.draft_tokens;
            stats.k_switches = controller.switches;
            stats.prefill = std::chrono::duration<double>(prefill - begin).count();
            stats.decode = std::chrono::duration<double>(now() - prefill).count();
        } catch (...) {
            auto error = std::current_exception();
            bool aborted = false;
            try { std::rethrow_exception(error); }
            catch (const CollectiveAborted &) { aborted = true; }
            catch (...) {}
            gpu.abort_collectives();
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure || (failure_is_abort && !aborted)) {
                failure = error;
                failure_is_abort = aborted;
            }
        }
    };
    if (gpu.tp_size == 1) {
        run(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(gpu.ranks);
        try {
            for (int r = 0; r < gpu.ranks; ++r)
                threads.emplace_back(run, r);
        } catch (...) {
            gpu.abort_collectives();
            for (auto &thread : threads)
                thread.join();
            throw;
        }
        for (auto &thread : threads)
            thread.join();
    }
    if (failure)
        std::rethrow_exception(failure);
    if (prefix_slot && prefix_stats && prefix_stats->enabled) {
        prefix_slot->tokens = lane[0].history;
        prefix_slot->max_written = lane[0].max_written;
        prefix_slot->valid = !prefix_slot->tokens.empty();
        prefix_slot->snap_pos = align_hy4_prefix(ids.size(), ids.size() + 1);
        prefix_slot->snap_max_written = ids.size();
        prefix_slot->snap_valid = prefix_slot->snap_pos > 0 && !prefix_slot->swa_snap.empty() &&
                                  !prefix_slot->swa_snap[0].empty();
        if (gpu.kv_pages_total > 0 && prefix_slot->valid) {
            const size_t keep = gpu.pages_for_tokens(prefix_slot->tokens.size());
            prefix_slot->pages.assign(gpu.slot_page_tables[0].begin(),
                                      gpu.slot_page_tables[0].begin() +
                                          ptrdiff_t(std::min(keep, gpu.slot_page_tables[0].size())));
            gpu.retain_pages(prefix_slot->pages);
        }
        prefix_stats->entries = prefix_slot->valid ? 1 : 0;
        if (prefix_slot->valid) ++prefix_stats->saves;
    }
    const auto &stats = lane[0];
    if (stats_out)
        *stats_out = {stats.final_drafts, size_t(stats.rounds), size_t(stats.accepted), stats.k_switches};
    fprintf(stderr,
            "prefill_tokens=%zu prefill_s=%.6f decode_tokens=%zu decode_s=%.6f draft_rounds=%d accepted=%d "
            "draft_tokens=%d k_switches=%zu\n",
            ids.size(), stats.prefill, stats.generated.size() - 1, stats.decode, stats.rounds, stats.accepted,
            stats.final_drafts, stats.k_switches);
    fprintf(stderr, "accepted_histogram=");
    const int max_hist = use_adaptive ? max_draft_tokens : resolved_drafts;
    for (int i = 0; i <= max_hist; ++i)
        fprintf(stderr, "%s%d", i ? "," : "", stats.histogram[i]);
    fputc('\n', stderr);
    for (int r = 0; r < gpu.ranks; ++r) {
        const uint64_t prefill_routes = gpu.tp_size == 1 ? stats.pp_routes_all[r] : lane[r].pp_routes;
        const double prefill_expert = gpu.tp_size == 1 ? stats.pp_expert_all[r] : lane[r].pp_expert;
        fprintf(stderr, "rank=%d prefill_choices=%llu decode_choices=%llu", r,
                static_cast<unsigned long long>(prefill_routes),
                static_cast<unsigned long long>(gpu.routed_rows[r] - prefill_routes));
        if (gpu.expert_begin[r])
            fprintf(stderr, " prefill_expert_ms=%.4f decode_expert_ms=%.4f", prefill_expert,
                    gpu.expert_ms[r] - prefill_expert);
        fputc('\n', stderr);
    }
    if (gpu.expert_begin[0])
        fprintf(stderr,
                "profile_pp_ms attention=%.3f ffn=%.3f transfer=%.3f head=%.3f\n"
                "profile_tg_ms attention=%.3f ffn=%.3f transfer=%.3f head=%.3f draft=%.3f\n",
                stats.pp_attn, stats.pp_ffn, stats.pp_transfer, stats.pp_head,
                gpu.attention_ms - stats.pp_attn, gpu.ffn_ms - stats.pp_ffn,
                gpu.transfer_ms - stats.pp_transfer, stats.head - stats.pp_head, stats.draft);
    return lane[0].generated;
}
namespace {
struct BatchLane {
    int slot, id, limit;
    std::vector<int> &history;
    std::vector<int> &generated;
    const Hy4BatchOptions &options;
    size_t &max_written;
    bool ignore_eos = false;
    float *target_probs = nullptr;
    float *draft_probs = nullptr;
    size_t token_cap = 0;
};
struct BatchStats {
    size_t rounds = 0, accepted = 0, tokens = 0;
};
struct BatchLayout {
    std::vector<int> slots, positions;
    std::vector<const std::vector<int> *> priors;
    int *device_slots, *device_positions;
    BatchLayout(Ops &op, const std::vector<BatchLane> &lanes, int rows) {
        for (auto &lane : lanes) {
            priors.push_back(&lane.history);
            for (int i = 0; i < rows; ++i) {
                slots.push_back(lane.slot);
                positions.push_back(int(lane.history.size()) + i);
            }
        }
        device_slots = static_cast<int *>(op.arena.next(2 * slots.size() * sizeof(int)));
        device_positions = device_slots + slots.size();
        checked(hipMemcpyAsync(device_slots, slots.data(), slots.size() * sizeof(int), hipMemcpyHostToDevice, op.stream));
        checked(hipMemcpyAsync(device_positions, positions.data(), positions.size() * sizeof(int), hipMemcpyHostToDevice, op.stream));
    }
};
__global__ void gather_block_rows(Mat src, Mat out, const int *slots, int src_stride, int slot_stride, int offset) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    int block = slots ? slots[r * slot_stride] : r;
    out.p[r * 3072 + c] = src.p[(block * src_stride + offset) * 3072 + c];
}
__global__ void fuse_block_anchors(Mat x, Mat fusion, const uint16_t *gate, int block_rows) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    auto &value = x.p[r * block_rows * 3072 + c];
    value = f32_to_bf16_rne_device(bf16_bits_to_f32(value) +
                                   bf16_bits_to_f32(fusion.p[r * 3072 + c]) * bf16_bits_to_f32(gate[c]));
}
void sample_batch(Ops &op, const std::vector<BatchLane> &lanes, float *logits, int *ids, int rows, int stage) {
    if (std::all_of(lanes.begin(), lanes.end(), [](const BatchLane &lane) { return lane.options.temperature == 0; })) {
        best_ids<<<lanes.size() * rows, 256, 0, op.stream>>>(logits, ids, 0, 0, 0);
        checked(hipGetLastError());
        return;
    }
    for (size_t r = 0; r < lanes.size(); ++r) {
        const auto &lane = lanes[r];
        const auto &o = lane.options;
        float *row = logits + r * rows * 120832;
        if (o.temperature > 0 && o.top_k) {
            truncate_logits<<<rows, 256, 0, op.stream>>>(row, o.top_k, o.top_p, o.temperature);
            checked(hipGetLastError());
        }
        if (lane.target_probs) {
            auto probs = stage ? lane.draft_probs + (stage - 1) * 120832 : lane.target_probs;
            probabilities<<<rows, 256, 0, op.stream>>>(row, probs, o.temperature);
            checked(hipGetLastError());
        }
        auto params = op.capture ? &op.gpu.batch_input->step.lanes[r] : nullptr;
        int position = (params ? 0 : int(lane.history.size())) + (stage ? stage : 1);
        uint64_t seed = (params ? 0 : o.seed) ^ (stage ? 0x735a7ec51b09ull : 0);
        best_ids<<<rows, 256, 0, op.stream>>>(row, ids + r * rows, position, seed, o.temperature, nullptr, params);
        checked(hipGetLastError());
    }
}
GraphKey batch_graph_key(Target &target, const std::vector<BatchLane> &lanes, int rows, bool draft) {
    GraphKey key{};
    key.batch = lanes.size();
    key.rows = rows;
    key.draft_tokens = rows > 1 ? rows - 1 : 0;
    key.workspace = target.op.arena.cursor;
    key.state = target.state.p;
    int end = 0;
    for (int i = 0; i < key.batch; ++i) {
        const auto &lane = lanes[i];
        int start = lane.history.size();
        end = std::max(end, start + rows);
        if (draft ? start > 2048 : start + rows > 2048) {
            key.lanes[i].sparse_slot = lane.slot;
            key.lanes[i].dense_rows = std::clamp(2048 - start, 0, rows);
        }
        if (draft) {
            key.lanes[i].temperature = lane.options.temperature;
            key.lanes[i].top_p = lane.options.top_p;
            key.lanes[i].top_k = lane.options.top_k;
        }
    }
    if (!draft)
        key.temperature = std::all_of(lanes.begin(), lanes.end(), [](const BatchLane &lane) {
            return lane.options.temperature == 0;
        }) ? 0.0f : 1.0f;
    key.bucket = end <= 2048 ? int(std::bit_ceil(unsigned(end))) : (end + 1023) / 1024 * 1024;
    return key;
}
BatchGraphParams batch_step(Ops &op, const std::vector<BatchLane> &lanes, const std::vector<int> &ids, bool draft) {
    BatchGraphParams step{};
    std::copy(ids.begin(), ids.end(), step.ids);
    for (size_t i = 0; i < lanes.size(); ++i) {
        step.anchors[i] = lanes[i].generated.back();
        auto &params = step.lanes[i];
        params.position = lanes[i].history.size();
        params.rows = ids.size() / lanes.size();
        params.seed = lanes[i].options.seed;
        if (draft) params.draft_last = op.gpu.sparse_last[lanes[i].slot][sources[0]];
    }
    return step;
}
int *draft_batch_steps(Target &target, const std::vector<BatchLane> &lanes, const BatchLayout &layout,
                       const std::vector<int> &ids, const std::vector<int> &anchors) {
    Ops &op = target.op;
    const int drafts = target.draft_tokens;
    const int block_rows = drafts + 1;
    int batch = lanes.size();
    Mat x = op.capture ? op.embed_device(op.gpu.batch_input->step.ids, ids.size(), false) : op.embed(ids, false);
    const std::string p = "model.dfly_layers.";
    auto gather = [&](Mat src, const int *slots, int src_stride, int slot_stride, int offset) {
        Mat out = op.mat(batch, 3072);
        gather_block_rows<<<dim3(12, batch), 256, 0, op.stream>>>(src, out, slots, src_stride, slot_stride, offset);
        checked(hipGetLastError());
        return out;
    };
    size_t mark = op.arena.cursor;
    Mat e = op.norm(gather(x, nullptr, block_rows, 0, 0), p + "mtp_fusion.enorm.weight");
    Mat h = op.norm(gather(target.state, layout.device_slots, max_verify_rows, block_rows, 0), p + "mtp_fusion.hnorm.weight");
    Mat fused = op.linear(p + "mtp_fusion.eh_proj", op.join(e, h));
    fuse_block_anchors<<<dim3(12, batch), 256, 0, op.stream>>>(
        x, fused, static_cast<const uint16_t *>(op.param(p + "mtp_fusion.gate").p), block_rows);
    checked(hipGetLastError());
    op.arena.cursor = mark;
    for (int i = 0; i < 5; ++i) {
        std::string prefix = p + std::to_string(i) + ".";
        Mat a = op.norm(x, prefix + "input_layernorm.weight");
        Mat y = op.add(x, attention_batch(op, prefix, sources[i], layout.slots, layout.positions,
                                         layout.device_slots, layout.device_positions, a, block_rows, true));
        Mat f = feedforward(op, prefix + "mlp", op.norm(y, prefix + "post_attention_layernorm.weight"));
        op.add_into(y, f, x);
        op.arena.cursor = mark;
    }
    x = op.norm(x, p + "final_norm.weight");
    const std::string correction = p + "hidden_correction";
    auto device = op.capture ? op.gpu.batch_input->candidates : static_cast<int *>(op.arena.next(drafts * batch * sizeof(int)));
    int *previous = op.capture ? op.gpu.batch_input->step.anchors : op.indices(anchors);
    mark = op.arena.cursor;
    const bool all_greedy =
        std::all_of(lanes.begin(), lanes.end(), [](const BatchLane &lane) { return lane.options.temperature == 0; });
    for (int i = 1; i <= drafts; ++i) {
        Mat hidden = gather(x, nullptr, block_rows, 0, i);
        Mat q = op.norm(hidden, correction + ".hidden_norm.weight");
        Mat embedding = op.embed_device(previous, batch, false);
        Mat k = op.norm(embedding, correction + ".embed_norm.weight");
        Mat corrected = op.add(hidden, feedforward(op, correction, op.join(q, k)));
        previous = device + (i - 1) * batch;
        if (all_greedy)
            target.head_argmax(corrected, previous);
        else
            sample_batch(op, lanes, target.head_logits(corrected), previous, 1, i);
        op.arena.cursor = mark;
    }
    return device;
}
std::vector<int> propose_batch(Target &target, const std::vector<BatchLane> &lanes, const BatchLayout &layout) {
    Ops &op = target.op;
    auto &gpu = op.gpu;
    const int drafts = target.draft_tokens;
    const int block_rows = drafts + 1;
    const int batch = lanes.size();
    std::vector<int> ids(batch * block_rows, 120009), anchors;
    for (int r = 0; r < batch; ++r) {
        ids[r * block_rows] = lanes[r].generated.back();
        anchors.push_back(ids[r * block_rows]);
    }
    BatchGraphParams step;
    int *device = nullptr;
    if (gpu.batch_input && target.draft_graph && !gpu.draft_capture_failed) {
        auto key = batch_graph_key(target, lanes, block_rows, true);
        auto plan = capture_cached(op, gpu.draft_graphs, gpu.draft_victim, gpu.draft_capture_failed, key,
            [&](GraphPlan &) { draft_batch_steps(target, lanes, layout, ids, anchors); });
        op.arena.cursor = key.workspace;
        if (plan) {
            step = batch_step(op, lanes, ids, true);
            checked(hipMemcpyAsync(&gpu.batch_input->step, &step, sizeof(step), hipMemcpyHostToDevice, op.stream));
            plan->launch(gpu);
            ++gpu.draft_replays;
            device = gpu.batch_input->candidates;
        }
    }
    if (!device) device = draft_batch_steps(target, lanes, layout, ids, anchors);
    std::vector<int> candidates(drafts * batch);
    op.copy(candidates.data(), device, candidates.size() * sizeof(int), hipMemcpyDeviceToHost);
    if (std::find(candidates.begin(), candidates.end(), -1) != candidates.end())
        throw std::runtime_error("DFly2 に有効な出力 logit がありません");
    for (int r = 0; r < batch; ++r)
        for (int i = 0; i < drafts; ++i)
            ids[r * block_rows + i + 1] = candidates[i * batch + r];
    return ids;
}
Mat verify_batch(Target &target, const BatchLayout &layout,
                 const std::vector<int> &ids, int rows) {
    Ops &op = target.op;
    auto &gpu = op.gpu;
    auto now = std::chrono::steady_clock::now;
    const auto &[slots, positions, priors, device_slots, device_positions] = layout;
    Mat x = op.capture ? op.embed_device(gpu.batch_input->step.ids, ids.size(), true) : op.embed(ids, true);
    size_t mark = op.arena.cursor;
    for (int layer = 0; layer < 42; ++layer) {
        auto start = now();
        if (gpu.expert_begin[0]) checked(hipStreamSynchronize(op.stream));
        auto p = "model.layers." + std::to_string(layer) + ".";
        if (layer % 10 == 1 && layer <= 31) engram_batch(op, layer, x, priors, ids);
        float *post;
        Mat a = op.collapse_norm(x, p + "hc_attn_layer.hc_pre.hc_", post, p + "input_layernorm.weight");
        a = attention_batch(op, p, layer, slots, positions, device_slots, device_positions, a, rows);
        op.expand(x, a, post);
        op.arena.cursor = mark;
        if (gpu.expert_begin[0]) {
            checked(hipStreamSynchronize(op.stream));
            auto end = now();
            gpu.attention_ms += std::chrono::duration<double, std::milli>(end - start).count();
            start = end;
        }
        Mat f = op.collapse_norm(x, p + "hc_mlp_layer.hc_pre.hc_", post, p + "post_attention_layernorm.weight");
        f = layer == 0 ? op.norm(feedforward(op, p + "mlp", f), p + "mlp.post_fc2_norm.weight")
                       : moe(op, layer, f);
        op.expand(x, f, post);
        op.arena.cursor = mark;
        if (!layer && op.capture) {
            op.capture->finish(op.capture->plan.entry);
            op.capture->begin(0);
        }
        if (gpu.expert_begin[0]) {
            checked(hipStreamSynchronize(op.stream));
            gpu.ffn_ms += std::chrono::duration<double, std::milli>(now() - start).count();
        }
    }
    float *post;
    return op.collapse_norm(x, "model.hc_head.hc_head_", post, "model.norm.weight");
}
void decode_batch(Target &target, std::vector<BatchLane> &lanes, BatchStats &stats,
                  const std::function<void(int, int)> &emit) {
    Ops &op = target.op;
    auto &gpu = op.gpu;
    op.batch_decode = true;
    op.arena.cursor = target.base;
    const int drafts = target.draft_tokens;
    const int block_rows = drafts + 1;
    bool speculative = gpu.batch_draft && lanes.size() <= Hy4GroupedPlan::max_rows / block_rows &&
        std::all_of(lanes.begin(), lanes.end(), [&](const BatchLane &lane) {
            size_t cap = lane.token_cap ? lane.token_cap : gpu.tokens;
            return lane.history.size() + size_t(block_rows) <= cap && lane.generated.size() + 1 < size_t(lane.limit);
        });
    int rows = speculative ? block_rows : 1;
    if (gpu.kv_pages_total > 0) {
        for (const auto &lane : lanes) {
            if (!gpu.allocate_slot_pages(lane.slot, lane.history.size() + size_t(rows)))
                throw std::runtime_error("decode_batch の KV ページ割当に失敗しました");
        }
    }
    BatchLayout layout(op, lanes, rows);
    std::vector<int> ids;
    if (speculative) {
        for (auto &lane : lanes)
            if (lane.options.temperature > 0) {
                lane.target_probs = static_cast<float *>(op.arena.next(block_rows * 120832 * sizeof(float)));
                lane.draft_probs = static_cast<float *>(op.arena.next(drafts * 120832 * sizeof(float)));
            }
        size_t mark = op.arena.cursor;
        ids = propose_batch(target, lanes, layout);
        op.arena.cursor = mark;
    } else {
        for (auto &lane : lanes) ids.push_back(lane.generated.back());
    }
    BatchGraphParams step;
    std::array<std::vector<uint8_t>, 4> engrams;
    const bool all_greedy =
        std::all_of(lanes.begin(), lanes.end(), [](const BatchLane &lane) { return lane.options.temperature == 0; });
    Mat h{};
    float *logits = nullptr;
    int *tokens = nullptr;
    if (gpu.batch_input && target.target_graph && !gpu.target_capture_failed) {
        auto key = batch_graph_key(target, lanes, rows, false);
        auto plan = capture_cached(op, gpu.target_graphs, gpu.target_victim, gpu.target_capture_failed, key,
            [&](GraphPlan &slot) {
                Mat hidden = verify_batch(target, layout, ids, rows);
                slot.hidden = hidden.p;
                if (all_greedy) {
                    slot.tokens = static_cast<int *>(op.arena.next(ids.size() * sizeof(int)));
                    slot.logits = nullptr;
                    target.head_argmax(hidden, slot.tokens);
                } else {
                    slot.logits = target.head_logits(hidden);
                    slot.tokens = nullptr;
                }
                slot.cursor = op.arena.cursor;
            });
        op.arena.cursor = key.workspace;
        if (plan) {
            step = batch_step(op, lanes, ids, false);
            checked(hipMemcpyAsync(&gpu.batch_input->step, &step, sizeof(step), hipMemcpyHostToDevice, op.stream));
            checked(hipGraphLaunch(plan->entry, op.stream));
            for (int g = 0; g < 4; ++g) {
                auto &packed = engrams[g];
                packed = gather_batch_engram(gpu.model, 1 + 10 * g, layout.priors, ids);
                checked(hipMemcpyAsync(gpu.batch_input->engram + g * packed.size(), packed.data(), packed.size(),
                                       hipMemcpyHostToDevice, op.stream));
            }
            plan->launch(gpu);
            ++gpu.target_replays;
            op.arena.cursor = plan->cursor;
            h = {plan->hidden, int(ids.size()), 3072};
            logits = plan->logits;
            tokens = plan->tokens;
        }
    }
    if (!logits && !tokens) {
        h = verify_batch(target, layout, ids, rows);
        if (all_greedy) {
            tokens = static_cast<int *>(op.arena.next(ids.size() * sizeof(int)));
            target.head_argmax(h, tokens);
        } else {
            logits = target.head_logits(h);
        }
    }
    int *device = tokens;
    if (!device) {
        device = static_cast<int *>(op.arena.next(ids.size() * sizeof(int)));
        sample_batch(op, lanes, logits, device, rows, 0);
    }
    std::vector<int> next(ids.size());
    op.copy(next.data(), device, next.size() * sizeof(int), hipMemcpyDeviceToHost);
    if (std::find(next.begin(), next.end(), -1) != next.end())
        throw std::runtime_error("有効な出力 logit がありません");
    for (size_t r = 0; r < lanes.size(); ++r) {
        auto &lane = lanes[r];
        lane.max_written = std::max(lane.max_written, lane.history.size() + size_t(rows));
        int first = r * rows, keep = 1;
        std::vector<int> actual(next.begin() + first, next.begin() + first + rows);
        if (speculative) {
            keep = verify_speculation(op, lane.target_probs, lane.draft_probs, int(lane.history.size()) + 1,
                                      lane.options.seed, {ids.begin() + first + 1, ids.begin() + first + rows}, actual);
            ++stats.rounds;
            stats.accepted += keep - 1;
        }
        keep = std::min(keep, lane.limit - int(lane.generated.size()));
        if (!lane.ignore_eos) {
            auto stop = std::find(actual.begin(), actual.begin() + keep, 120025);
            if (stop != actual.begin() + keep) keep = int(stop - actual.begin()) + 1;
        }
        if (gpu.batch_draft) {
            Mat state = target.state.row(lane.slot * max_verify_rows, rows);
            checked(hipMemcpyAsync(state.p, h.row(first).p, state.bytes(), hipMemcpyDeviceToDevice, op.stream));
            if (keep > 1)
                checked(hipMemcpyAsync(state.p, state.row(keep - 1).p, state.row(0).bytes(),
                                       hipMemcpyDeviceToDevice, op.stream));
            for (int layer : sources) gpu.sparse_last[lane.slot][layer] = keep - 1;
        }
        lane.history.insert(lane.history.end(), ids.begin() + first, ids.begin() + first + keep);
        lane.generated.insert(lane.generated.end(), actual.begin(), actual.begin() + keep);
        stats.tokens += keep;
        if (emit)
            for (int i = 0; i < keep; ++i) emit(lane.id, actual[i]);
    }
}
}
std::vector<std::vector<int>> generate_batch(Residency &gpu, const std::vector<std::vector<int>> &prompts,
                                              const std::vector<int> &counts, const Hy4BatchOptions &options,
                                              const std::function<void(int, int)> &emit) {
    const size_t batch = prompts.size();
    if (batch < 1 || batch > size_t(gpu.max_slots) || counts.size() != batch || options.chunk < 1 ||
        options.chunk > max_prefill_chunk || !std::isfinite(options.temperature) || options.temperature < 0 ||
        options.top_k < 0 || options.top_k > 64 || !(options.top_p > 0 && options.top_p <= 1) ||
        (options.temperature > 0 && options.top_p < 1 && !options.top_k))
        throw std::invalid_argument("生成バッチの指定が不正です");
    for (size_t s = 0; s < batch; ++s)
        if (prompts[s].empty() || counts[s] <= 0 || prompts[s].size() + counts[s] > gpu.tokens ||
            std::any_of(prompts[s].begin(), prompts[s].end(), [](int id) { return id < 0 || id >= 120832; }))
            throw std::invalid_argument("入力列または生成長が不正です");
    const int resolved_drafts = Residency::resolve_draft_tokens(options.draft_tokens ? options.draft_tokens : gpu.draft_tokens);
    const bool use_adaptive = options.adaptive || gpu.adaptive_draft;
    if (batch == 1) {
        auto output = generate(gpu, prompts.front(), counts.front(), options.chunk, options.temperature,
                               options.seed, {}, options.top_k, options.top_p, [&](int id) { if (emit) emit(0, id); },
                               options.ignore_eos, nullptr, nullptr, nullptr, resolved_drafts, use_adaptive);
        return {std::move(output)};
    }
    auto now = std::chrono::steady_clock::now;
    auto begin = now();
    struct BatchPageGuard {
        Residency &gpu;
        size_t count;
        ~BatchPageGuard() {
            if (gpu.kv_pages_total > 0)
                for (size_t s = 0; s < count; ++s)
                    gpu.release_slot_pages(int(s));
        }
    } batch_guard{gpu, batch};
    if (gpu.kv_pages_total > 0) {
        for (size_t s = 0; s < batch; ++s) {
            gpu.release_slot_pages(int(s));
            if (!gpu.allocate_slot_pages(int(s), prompts[s].size() + size_t(counts[s])))
                throw std::runtime_error("バッチの KV ページが不足しています");
        }
    }
    Target target(gpu, options.temperature, options.seed, options.top_k, options.top_p, 0, resolved_drafts);
    Hy4DflyAdaptiveController controller(resolved_drafts);
    target.op.slot_draft = gpu.batch_draft;
    std::vector<std::vector<int>> histories(batch), generated(batch);
    std::vector<size_t> written(batch, 0);
    for (size_t s = 0; s < batch; ++s) {
        target.history.clear();
        target.op.slot = int(s);
        int next = 0;
        for (size_t first = 0; first < prompts[s].size(); first += options.chunk) {
            auto end = std::min(first + size_t(options.chunk), prompts[s].size());
            auto result = target.forward({prompts[s].begin() + first, prompts[s].begin() + end}, false,
                                         end == prompts[s].size());
            if (!result.empty()) next = result.back();
        }
        histories[s] = target.history;
        written[s] = prompts[s].size();
        generated[s].push_back(next);
        if (emit) emit(int(s), next);
    }
    target.op.slot = 0;
    auto prefill = now();
    BatchStats stats;
    for (;;) {
        std::vector<BatchLane> lanes;
        for (size_t s = 0; s < batch; ++s)
            if (generated[s].size() < size_t(counts[s]) && (options.ignore_eos || generated[s].back() != 120025))
                lanes.push_back({int(s), int(s), counts[s], histories[s], generated[s], options, written[s], options.ignore_eos});
        if (lanes.empty()) break;
        const size_t rounds_before = stats.rounds, accepted_before = stats.accepted;
        decode_batch(target, lanes, stats, emit);
        if (use_adaptive && stats.rounds > rounds_before) {
            const size_t round_lanes = stats.rounds - rounds_before;
            const size_t round_accepted = stats.accepted - accepted_before;
            target.draft_tokens = controller.observe(double(round_accepted) / double(round_lanes), int(round_lanes));
        }
    }
    size_t decode_tokens = stats.tokens;
    fprintf(stderr, "batch_draft_rounds=%zu batch_accepted=%zu draft_tokens=%d k_switches=%zu\n",
            stats.rounds, stats.accepted, target.draft_tokens, controller.switches);
    auto end = now();
    double pp = std::chrono::duration<double>(prefill - begin).count();
    double tg = std::chrono::duration<double>(end - prefill).count();
    fprintf(stderr, "batch_prefill_s=%.6f batch_decode_s=%.6f batch_decode_tokens=%zu batch_agg_tok_s=%.6f batch_slot_tok_s=%.6f\n",
            pp, tg, decode_tokens, tg ? decode_tokens / tg : 0, tg ? decode_tokens / tg / batch : 0);
    return generated;
}

struct Hy4BatchSession::Impl {
    struct State {
        Hy4Request request;
        std::vector<int> history, generated;
        size_t prefetched = 0;
        size_t restored_tokens = 0;
        size_t max_written = 0;
        size_t snap_pos = 0;
        size_t snap_max_written = 0;
        size_t reserved_pages = 0;
        uint64_t admit_seq = 0;
        std::vector<uint8_t> swa_snap_buf;
        std::vector<uint8_t> swa_pending;
    };
    Residency &gpu;
    int draft_tokens;
    bool adaptive;
    Hy4DflyAdaptiveController controller;
    std::optional<Target> target;
    Emit emit;
    Done done;
    std::vector<std::optional<State>> states;
    std::list<Hy4PrefixSlot> prefix_entries;
    std::vector<uint64_t> slot_swa_gen;
    std::vector<size_t> slot_reserved_pages;
    std::deque<Hy4Request> waiting;
    BatchStats stats;
    Hy4PrefixStats prefix;
    uint64_t use_clock = 0;
    uint64_t admit_clock = 0;
    size_t last_prompt = 0;
    size_t last_restored = 0;
    int prefill_decode_interval = 1;
    size_t prefill_chunks_since_decode = 0;
    size_t prefill_tokens_since_decode = 0;
    bool last_prefill_incomplete = false;
    size_t max_prefix_entries = 64;
    Impl(Residency &gpu, Emit emit, Done done, int drafts, bool adaptive)
        : gpu(gpu), draft_tokens(Residency::resolve_draft_tokens(drafts ? drafts : gpu.draft_tokens)),
          adaptive(adaptive || gpu.adaptive_draft), controller(draft_tokens),
          emit(std::move(emit)), done(std::move(done)), states(gpu.max_slots),
          slot_swa_gen(gpu.max_slots, 0), slot_reserved_pages(gpu.max_slots, 0),
          max_prefix_entries(std::max<size_t>(size_t(gpu.max_slots) * 4, 64)) {
        if (gpu.tp_size > 1)
            throw std::invalid_argument("バッチセッションは TP=1 で使用してください");
        prefix.enabled = true;
        prefix.reuse_slots = gpu.max_slots;
        if (const char *env = std::getenv("SHELF_HY4_PREFILL_DECODE_INTERVAL"); env && *env) {
            int val = 0;
            auto [p, err] = std::from_chars(env, env + std::strlen(env), val);
            if (err != std::errc{} || p != env + std::strlen(env) || val < 0)
                throw std::runtime_error("SHELF_HY4_PREFILL_DECODE_INTERVAL が不正です");
            prefill_decode_interval = val;
        }
    }
    ~Impl() {
        if (!gpu.kv_pages_total) return;
        for (size_t s = 0; s < states.size(); ++s) {
            if (states[s])
                gpu.release_slot_pages(int(s));
        }
        for (auto &entry : prefix_entries)
            gpu.release_pages(entry.pages);
    }
    size_t unmapped_reserved_pages() const {
        size_t res = 0;
        for (size_t s = 0; s < states.size(); ++s) {
            if (!states[s]) continue;
            size_t mapped = gpu.slot_page_tables[s].size();
            if (slot_reserved_pages[s] > mapped)
                res += slot_reserved_pages[s] - mapped;
        }
        return res;
    }
    size_t avail_for_new() const {
        size_t free_p = gpu.free_kv_pages();
        size_t res_p = unmapped_reserved_pages();
        return free_p > res_p ? free_p - res_p : 0;
    }
    bool evict_one_prefix(const Hy4PrefixSlot *protect = nullptr) {
        auto best_it = prefix_entries.end();
        for (auto it = prefix_entries.begin(); it != prefix_entries.end(); ++it) {
            if (&(*it) == protect) continue;
            if (best_it == prefix_entries.end() || it->last_used < best_it->last_used)
                best_it = it;
        }
        if (best_it == prefix_entries.end())
            return false;
        gpu.release_pages(best_it->pages);
        prefix_entries.erase(best_it);
        ++prefix.evictions;
        return true;
    }
    size_t match_prefix_entry(const Hy4PrefixSlot &entry, const std::vector<int> &input, size_t swa_slots,
                              bool &use_snap, int &preferred_slot) const {
        use_snap = false;
        preferred_slot = -1;
        if (!entry.valid || entry.tokens.empty() || input.size() <= 1)
            return 0;
        size_t common = 0;
        size_t limit = std::min(entry.tokens.size(), input.size() - 1);
        while (common < limit && entry.tokens[common] == input[common])
            ++common;
        if (gpu.kv_pages_total > 0)
            common = std::min(common, entry.pages.size() * kv_page_tokens);
        const bool live_ok = entry.live_swa_slot >= 0 && size_t(entry.live_swa_slot) < states.size() &&
                             !states[entry.live_swa_slot] &&
                             slot_swa_gen[entry.live_swa_slot] == entry.live_swa_gen;
        if (live_ok) {
            size_t live_pos = (common / 4) * 4;
            if (live_pos > 0 && swa_range_intact(live_pos, entry.max_written, swa_slots)) {
                preferred_slot = entry.live_swa_slot;
                return live_pos;
            }
        }
        if (entry.snap_valid) {
            size_t snap_limit = std::min(common, entry.snap_pos);
            size_t snap_pos = (snap_limit / 4) * 4;
            if (snap_pos > 0 && swa_range_intact(snap_pos, entry.snap_max_written, swa_slots)) {
                use_snap = true;
                return snap_pos;
            }
        }
        return 0;
    }
    void fill() {
        while (!waiting.empty()) {
            const auto &req = waiting.front();
            int free_slot = -1;
            for (size_t s = 0; s < states.size(); ++s) {
                if (!states[s]) {
                    free_slot = int(s);
                    break;
                }
            }
            if (free_slot < 0)
                break;
            auto best = prefix_entries.end();
            size_t best_match = 0;
            bool best_snap = false;
            int best_pref_slot = -1;
            uint64_t best_clock = 0;
            if (prefix.enabled) {
                const size_t swa_slots = swa_window_slots(gpu);
                for (auto it = prefix_entries.begin(); it != prefix_entries.end(); ++it) {
                    bool snap = false;
                    int pref_slot = -1;
                    size_t m = match_prefix_entry(*it, req.input_ids, swa_slots, snap, pref_slot);
                    if (m > 0 && (best == prefix_entries.end() || m > best_match ||
                                  (m == best_match && it->last_used > best_clock))) {
                        best = it;
                        best_match = m;
                        best_snap = snap;
                        best_pref_slot = pref_slot;
                        best_clock = it->last_used;
                    }
                }
            }
            int chosen = best_pref_slot >= 0 ? best_pref_slot : free_slot;
            size_t total_pages = gpu.kv_pages_total ? gpu.pages_for_tokens(req.input_ids.size() + size_t(req.count)) : 0;
            std::vector<uint8_t> snap_copy;
            if (gpu.kv_pages_total > 0) {
                size_t shared_full = best_match / kv_page_tokens;
                size_t rem = best_match % kv_page_tokens;
                size_t need_new = total_pages > shared_full ? total_pages - shared_full : 0;
                while (avail_for_new() < need_new) {
                    if (!evict_one_prefix(best == prefix_entries.end() ? nullptr : &*best)) break;
                }
                bool steal_tail = false;
                if (avail_for_new() < need_new && best != prefix_entries.end()) {
                    const size_t used = shared_full + (rem != 0);
                    steal_tail = rem && gpu.kv_page_refs[best->pages[shared_full]] == 1;
                    const size_t reclaimable = std::count_if(best->pages.begin() + used, best->pages.end(),
                        [&](int32_t p) { return gpu.kv_page_refs[p] == 1; });
                    if (avail_for_new() + reclaimable + steal_tail >= need_new) {
                        std::vector<int32_t> trailing(best->pages.begin() + used, best->pages.end());
                        best->pages.resize(used);
                        gpu.release_pages(trailing);
                        best->tokens.resize(best_match);
                    } else {
                        gpu.release_pages(best->pages);
                        prefix_entries.erase(best);
                        ++prefix.evictions;
                        best = prefix_entries.end();
                        best_match = 0;
                        chosen = free_slot;
                        need_new = total_pages;
                        steal_tail = false;
                    }
                }
                if (avail_for_new() + steal_tail < need_new)
                    break;
                gpu.release_slot_pages(chosen);
                if (best != prefix_entries.end() && best_match > 0) {
                    auto &entry = *best;
                    if (best_snap && !entry.swa_snap.empty())
                        snap_copy = entry.swa_snap[0];
                    std::vector<int32_t> restored_pages;
                    restored_pages.reserve(shared_full + (rem > 0 ? 1 : 0));
                    for (size_t pi = 0; pi < shared_full; ++pi)
                        restored_pages.push_back(entry.pages[pi]);
                    gpu.retain_pages(restored_pages);
                    int32_t tail_src = -1;
                    if (rem > 0) {
                        if (steal_tail) {
                            restored_pages.push_back(entry.pages.back());
                            entry.pages.pop_back();
                        } else {
                            tail_src = entry.pages[shared_full];
                            int32_t p = gpu.alloc_single_page();
                            if (p < 0)
                                throw std::runtime_error("KV ページ割当に失敗しました");
                            restored_pages.push_back(p);
                        }
                    }
                    gpu.slot_page_tables[chosen] = std::move(restored_pages);
                    gpu.sync_slot_page_table(chosen);
                    if (tail_src >= 0)
                        copy_dsa_page_prefix(gpu, 0, gpu.slot_page_tables[chosen].back(), tail_src, int(rem), gpu.compute[0]);
                    if (steal_tail) {
                        gpu.release_pages(entry.pages);
                        prefix_entries.erase(best);
                        ++prefix.evictions;
                    } else {
                        entry.last_used = ++use_clock;
                    }
                }
                if (!gpu.allocate_slot_pages(chosen, req.input_ids.size()))
                    throw std::runtime_error("KV ページ割当に失敗しました");
            } else if (best != prefix_entries.end() && best_match > 0 && best_snap && !best->swa_snap.empty()) {
                snap_copy = best->swa_snap[0];
            }
            ++slot_swa_gen[chosen];
            slot_reserved_pages[chosen] = total_pages;
            auto &state = states[chosen].emplace();
            state.request = std::move(waiting.front());
            waiting.pop_front();
            state.reserved_pages = total_pages;
            state.admit_seq = ++admit_clock;
            if (prefix.enabled) {
                if (best_match > 0) {
                    ++prefix.hits;
                    ++prefix.restores;
                    if (best_snap && !snap_copy.empty())
                        state.swa_pending = std::move(snap_copy);
                    state.prefetched = best_match;
                    state.restored_tokens = best_match;
                    state.max_written = best_match;
                    state.history.assign(state.request.input_ids.begin(), state.request.input_ids.begin() + best_match);
                } else {
                    ++prefix.misses;
                }
            }
        }
    }
    void finish(int slot, const char *reason) {
        auto state = std::move(*states[slot]);
        states[slot].reset();
        slot_reserved_pages[slot] = 0;
        if (prefix.enabled && reason &&
            (std::string_view(reason) == "stop" || std::string_view(reason) == "length") &&
            !state.history.empty()) {
            Hy4PrefixSlot cache;
            cache.tokens = std::move(state.history);
            cache.max_written = state.max_written;
            cache.last_used = ++use_clock;
            cache.valid = true;
            cache.snap_pos = state.snap_pos;
            cache.snap_max_written = state.snap_max_written;
            cache.snap_valid = !state.swa_snap_buf.empty() && cache.snap_pos > 0;
            if (cache.snap_valid) {
                cache.swa_snap.resize(1);
                cache.swa_snap[0] = std::move(state.swa_snap_buf);
            }
            cache.live_swa_slot = slot;
            cache.live_swa_gen = slot_swa_gen[slot];
            if (gpu.kv_pages_total > 0) {
                const size_t keep = gpu.pages_for_tokens(cache.tokens.size());
                auto &table = gpu.slot_page_tables[slot];
                if (table.size() > keep) {
                    std::vector<int32_t> extra(table.begin() + ptrdiff_t(keep), table.end());
                    table.resize(keep);
                    gpu.release_pages(extra);
                }
                cache.pages.swap(table);
                gpu.sync_slot_page_table(slot);
            }
            while (prefix_entries.size() >= max_prefix_entries) {
                if (!evict_one_prefix())
                    break;
            }
            prefix_entries.push_back(std::move(cache));
            ++prefix.saves;
        } else {
            ++slot_swa_gen[slot];
            if (gpu.kv_pages_total > 0)
                gpu.release_slot_pages(slot);
        }
        last_prompt = state.request.input_ids.size();
        last_restored = state.restored_tokens;
        done(state.request.id, state.generated, reason);
    }
    void step() {
        fill();
        if (!target) {
            if (std::none_of(states.begin(), states.end(), [](const auto &state) { return bool(state); }))
                return;
            target.emplace(gpu, 0, 0, 0, 1, 0, draft_tokens);
            target->op.slot_draft = gpu.batch_draft;
        }
        target->draft_tokens = draft_tokens;
        int prefill_slot = -1;
        uint64_t best_admit = UINT64_MAX;
        bool has_decode = false;
        for (size_t s = 0; s < states.size(); ++s) {
            if (!states[s]) continue;
            if (states[s]->prefetched < states[s]->request.input_ids.size()) {
                if (states[s]->admit_seq < best_admit) {
                    best_admit = states[s]->admit_seq;
                    prefill_slot = int(s);
                }
            } else {
                has_decode = true;
            }
        }
        const bool should_decode_interleave =
            prefill_slot >= 0 && has_decode && prefill_decode_interval > 0 &&
            prefill_chunks_since_decode >= size_t(prefill_decode_interval) &&
            (last_prefill_incomplete ||
             prefill_tokens_since_decode >= size_t(states[prefill_slot]->request.options.chunk));
        if (prefill_slot >= 0 && !should_decode_interleave) {
            const int s = prefill_slot;
            auto &state = *states[s];
            auto &request = state.request;
            if (!state.swa_pending.empty()) {
                restore_swa_snapshot(gpu, s, 0, target->op.stream, state.swa_pending);
                checked(hipStreamSynchronize(target->op.stream));
                state.swa_pending.clear();
            }
            auto end = std::min(state.prefetched + size_t(request.options.chunk), request.input_ids.size());
            const size_t chunk_len = end - state.prefetched;
            const bool incomplete = end < request.input_ids.size();
            target->op.batch_decode = false;
            target->op.slot = s;
            target->history = std::move(state.history);
            target->temperature = request.options.temperature;
            target->seed = request.options.seed;
            target->top_k = request.options.top_k;
            target->top_p = request.options.top_p;
            auto result = target->forward({request.input_ids.begin() + state.prefetched, request.input_ids.begin() + end},
                                         false, !incomplete);
            state.history = std::move(target->history);
            state.max_written = std::max(state.max_written, end);
            state.prefetched = end;
            ++prefill_chunks_since_decode;
            prefill_tokens_since_decode += chunk_len;
            last_prefill_incomplete = incomplete;
            if (!incomplete && prefix.enabled && gpu.cache[0][2]) {
                state.snap_pos = align_hy4_prefix(end, end + 1);
                state.snap_max_written = end;
                if (state.snap_pos > 0)
                    save_swa_snapshot(gpu, s, 0, target->op.stream, state.swa_snap_buf);
            }
            if (!result.empty()) {
                state.generated.push_back(result.back());
                if (emit) emit(request.id, result.back());
                const bool hit_eos = !request.options.ignore_eos && result.back() == 120025;
                if (hit_eos || state.generated.size() == size_t(request.count)) {
                    finish(s, hit_eos ? "stop" : "length");
                    fill();
                }
            }
            return;
        }
        std::vector<BatchLane> lanes;
        for (size_t s = 0; s < states.size(); ++s) if (states[s] && states[s]->prefetched == states[s]->request.input_ids.size()) {
            auto &state = *states[s];
            size_t cap = gpu.kv_pages_total ? std::min(gpu.tokens, state.reserved_pages * kv_page_tokens) : gpu.tokens;
            lanes.push_back({int(s), state.request.id, state.request.count, state.history, state.generated,
                             state.request.options, state.max_written, state.request.options.ignore_eos,
                             nullptr, nullptr, cap});
        }
        if (lanes.empty()) return;
        target->draft_tokens = draft_tokens;
        prefill_chunks_since_decode = 0;
        prefill_tokens_since_decode = 0;
        last_prefill_incomplete = false;
        const size_t rounds_before = stats.rounds, accepted_before = stats.accepted;
        decode_batch(*target, lanes, stats, emit);
        if (adaptive && stats.rounds > rounds_before) {
            const size_t round_lanes = stats.rounds - rounds_before;
            const size_t round_accepted = stats.accepted - accepted_before;
            draft_tokens = controller.observe(double(round_accepted) / double(round_lanes), int(round_lanes));
            target->draft_tokens = draft_tokens;
        }
        bool any_finished = false;
        for (auto &lane : lanes) {
            int slot = lane.slot;
            auto &state = *states[slot];
            const bool hit_eos = !state.request.options.ignore_eos && state.generated.back() == 120025;
            if (hit_eos || state.generated.size() == size_t(state.request.count)) {
                finish(slot, hit_eos ? "stop" : "length");
                any_finished = true;
            }
        }
        if (any_finished)
            fill();
    }
};
Hy4BatchSession::Hy4BatchSession(Residency &gpu, Emit emit, Done done, int draft_tokens, bool adaptive)
    : impl(std::make_unique<Impl>(gpu, std::move(emit), std::move(done), draft_tokens, adaptive)) {}
Hy4BatchSession::~Hy4BatchSession() = default;
void Hy4BatchSession::set_draft_tokens(int draft_tokens) {
    impl->draft_tokens = Residency::resolve_draft_tokens(draft_tokens);
    impl->controller.reset(impl->draft_tokens);
}
int Hy4BatchSession::draft_tokens() const { return impl->draft_tokens; }
bool Hy4BatchSession::adaptive() const { return impl->adaptive; }
size_t Hy4BatchSession::k_switches() const { return impl->controller.switches; }
double Hy4BatchSession::acceptance_ema() const { return impl->controller.acc_ema; }
void Hy4BatchSession::admit(Hy4Request request) {
    auto &o = request.options;
    if (request.input_ids.empty() || request.count <= 0 || request.input_ids.size() + request.count > impl->gpu.tokens ||
        o.chunk < 1 || o.chunk > max_prefill_chunk || !std::isfinite(o.temperature) || o.temperature < 0 ||
        o.top_k < 0 || o.top_k > 64 || !(o.top_p > 0 && o.top_p <= 1) ||
        (o.temperature > 0 && o.top_p < 1 && !o.top_k) ||
        (impl->gpu.kv_pages_total > 0 &&
         impl->gpu.pages_for_tokens(request.input_ids.size() + size_t(request.count)) > impl->gpu.kv_pages_total) ||
        std::any_of(request.input_ids.begin(), request.input_ids.end(), [](int id) { return id < 0 || id >= 120832; }))
        throw std::invalid_argument("入力列または生成パラメータが不正です");
    for (const auto &state : impl->states)
        if (state && state->request.id == request.id) throw std::invalid_argument("request_id が重複しています");
    for (const auto &queued : impl->waiting)
        if (queued.id == request.id) throw std::invalid_argument("request_id が重複しています");
    impl->waiting.push_back(std::move(request));
    impl->fill();
}
void Hy4BatchSession::cancel(int id) {
    for (size_t s = 0; s < impl->states.size(); ++s)
        if (impl->states[s] && impl->states[s]->request.id == id) {
            impl->finish(int(s), "cancelled");
            impl->fill();
            return;
        }
    for (auto it = impl->waiting.begin(); it != impl->waiting.end(); ++it)
        if (it->id == id) {
            impl->last_prompt = it->input_ids.size();
            impl->last_restored = 0;
            impl->waiting.erase(it);
            impl->done(id, {}, "cancelled");
            return;
        }
}
void Hy4BatchSession::step() { impl->step(); }
void Hy4BatchSession::set_prefix_cache(bool enabled) {
    impl->prefix.enabled = enabled;
    if (!enabled) {
        for (auto &entry : impl->prefix_entries)
            impl->gpu.release_pages(entry.pages);
        impl->prefix_entries.clear();
    }
}
void Hy4BatchSession::set_prefill_decode_interval(int interval) {
    if (interval < 0)
        throw std::invalid_argument("prefill_decode_interval は 0 以上で指定してください");
    impl->prefill_decode_interval = interval;
}
int Hy4BatchSession::prefill_decode_interval() const {
    return impl->prefill_decode_interval;
}
Hy4PrefixStats Hy4BatchSession::prefix_stats() const {
    Hy4PrefixStats copy = impl->prefix;
    copy.entries = impl->prefix_entries.size();
    return copy;
}
size_t Hy4BatchSession::last_prompt_tokens() const { return impl->last_prompt; }
size_t Hy4BatchSession::last_restored_tokens() const { return impl->last_restored; }
size_t Hy4BatchSession::active() const {
    return std::count_if(impl->states.begin(), impl->states.end(), [](const auto &s) { return bool(s); });
}
size_t Hy4BatchSession::draft_rounds() const { return impl->stats.rounds; }
size_t Hy4BatchSession::queued() const { return impl->waiting.size(); }

}
