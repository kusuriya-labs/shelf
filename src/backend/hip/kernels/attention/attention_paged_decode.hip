// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/e4m3_codec.h>
#include <shelf/backend/hip/kernels/attention/kv_types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cfloat>

namespace shelf::hipk {

// in PhaseShift this alias came from detail/attention_paged_common.h
// (included but otherwise unused there; not vendored — alias kept verbatim).
using bf16 = __hip_bfloat16;

namespace { namespace v_decode_bf16 {

using bf16 = __hip_bfloat16;

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadLimit = kPagedAttentionHeadLimit;
static_assert(kThreads == kPagedAttentionBlock);
constexpr uint32_t kElemsPerLane = kHeadLimit / 32u;

__device__ __forceinline__ float warp_sum_all(float x) {
    x += __shfl_xor(x, 16, 32);
    x += __shfl_xor(x, 8, 32);
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    return x;
}

__device__ __forceinline__ float warp_max_all(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

// ---------------------------------------------------------------------------
// Algorithm layer (Phase 5, contract §4 / guide §7.1). BF16 decode stages:
//   task map -> visible range -> query load -> page walk + online softmax
//   (+ tree ancestors) -> stage wave partials -> [barrier] -> merge -> store.
// Barriers and block-uniform returns stay in the body; helper bodies are the
// original code moved verbatim (same expressions, same order).
// ---------------------------------------------------------------------------

struct DecodeTaskMap {
    uint32_t row;
    uint32_t q_head;
    uint32_t kv_head;
    bool active;   // row < rows; block-uniform (task = blockIdx.x)
};

__device__ __forceinline__ DecodeTaskMap attention_map_decode_task(
    const PagedAttentionCommonArgs& a) {
    const uint32_t task = blockIdx.x;
    const uint32_t q_per_kv = a.q_heads / a.kv_heads;
    DecodeTaskMap t{};
    t.row = task / a.q_heads;
    t.q_head = task % a.q_heads;
    t.kv_head = t.q_head / q_per_kv;
    t.active = t.row < a.rows;
    return t;
}

// tid0 metadata pass: sequence slot validity, visible range incl. tree
// ancestor contiguity fold-in, block-table pointer (contract §4 formulas).
__device__ __forceinline__ void attention_initialize_visible_range(
    const PagedAttentionCommonArgs& a, uint32_t row,
    uint32_t& row_valid_shared, uint32_t& visible_shared, uint32_t& invalid_shared,
    const uint32_t*& block_table_shared, uint32_t& anc_base_shared, uint32_t& anc_count_shared) {
    const uint32_t page_tokens = a.page_tokens;
    const uint32_t slot = a.row_sequence_slots[row];
    uint32_t valid = 1u;
    uint32_t visible = 0u;
    anc_base_shared = 0u;
    anc_count_shared = 0u;
    if (slot >= a.max_sequences) {
        valid = 0u;
    } else if (a.tree_anc_offsets != nullptr) {
        anc_base_shared = a.tree_anc_offsets[row];
        uint32_t anc_count = a.tree_anc_offsets[row + 1u] - anc_base_shared;
        const uint32_t committed_end = a.tree_committed_end_rows != nullptr
                                           ? a.tree_committed_end_rows[row]
                                           : a.tree_committed_end;
        bool contiguous = true;
        for (uint32_t i = 0u; i < anc_count; ++i) {
            if (a.tree_anc_positions[anc_base_shared + i] != committed_end + i) {
                contiguous = false;
                break;
            }
        }
        const uint64_t v64 = contiguous
                                 ? static_cast<uint64_t>(committed_end) + anc_count
                                 : static_cast<uint64_t>(committed_end);
        const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
        if (v64 > cap) valid = 0u;
        else {
            visible = static_cast<uint32_t>(v64);
            anc_count_shared = contiguous ? 0u : anc_count;
        }
    } else {
        const uint64_t v64 = static_cast<uint64_t>(a.row_positions[row]) + 1u;
        const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
        if (v64 > cap) valid = 0u;
        else visible = static_cast<uint32_t>(v64);
    }
    row_valid_shared = valid;
    visible_shared = visible;
    invalid_shared = 0u;
    block_table_shared = a.block_tables
        + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
}

// Query tile into registers: lane holds d = lane + 32*j (contract §4).
template <bool kFullHead>
__device__ __forceinline__ void attention_load_query_tile(
    const PagedAttentionCommonArgs& a, const DecodeTaskMap& task, uint32_t lane,
    float (&q_reg)[kElemsPerLane]) {
    const uint32_t head_dim = a.head_dim;
    const bf16* q_row = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(task.row) * a.q_row_stride
        + static_cast<size_t>(task.q_head) * head_dim;
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        q_reg[j] = (kFullHead || d < head_dim) ? __bfloat162float(q_row[d]) : 0.0f;
    }
}

// Shared addressing state for the token walk.
struct DecodeWalkState {
    uint64_t head_off;
    uint64_t layer_base;
    const uint32_t* block_table;
    uint32_t page_tokens;
    uint32_t visible;
    uint64_t elems_per_page;
    uint64_t elems_per_token;
    uint32_t num_pages;
    uint32_t wave;
    uint32_t lane;
};

// One token: masked K/V load, wave-reduced dot, online softmax update.
// Order frozen: j-ascending partial -> warp_sum_all -> *scale -> max/alpha/
// weight -> run_sum -> acc[j] (contract §4).
template <bool kFullHead>
__device__ __forceinline__ void attention_attend_token_bf16(
    const float (&q_reg)[kElemsPerLane],
    const bf16* k_head, const bf16* v_head,
    uint32_t head_dim, uint32_t lane, float scale,
    float& run_max, float& run_sum, float (&acc)[kElemsPerLane]) {
    bf16 k_reg[kElemsPerLane];
    bf16 v_reg[kElemsPerLane];
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        if (kFullHead || d < head_dim) {
            k_reg[j] = k_head[d];
            v_reg[j] = v_head[d];
        } else {
            k_reg[j] = __float2bfloat16(0.0f);
            v_reg[j] = __float2bfloat16(0.0f);
        }
    }
    float partial = 0.0f;
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j)
        partial += q_reg[j] * __bfloat162float(k_reg[j]);
    const float score = warp_sum_all(partial) * scale;
    const float next_max = fmaxf(run_max, score);
    const float alpha = __expf(run_max - next_max);
    const float weight = __expf(score - next_max);
    run_sum = run_sum * alpha + weight;
    run_max = next_max;
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j)
        acc[j] = acc[j] * alpha + weight * __bfloat162float(v_reg[j]);
}

template <bool kFullHead>
__device__ __forceinline__ void attention_visit_pages_bf16(
    const DecodeWalkState& st, const bf16* k_pool, const bf16* v_pool,
    const float (&q_reg)[kElemsPerLane], uint32_t head_dim, float scale,
    float& run_max, float& run_sum, float (&acc)[kElemsPerLane], uint32_t& bad) {
    const uint32_t page_count = (st.visible + st.page_tokens - 1u) / st.page_tokens;
    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = st.block_table[page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page;
        const uint32_t page_start = page_index * st.page_tokens;
        const uint32_t tokens_here =
            (st.visible - page_start < st.page_tokens) ? (st.visible - page_start) : st.page_tokens;
        for (uint32_t off = st.wave; off < tokens_here; off += kWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * st.elems_per_token;
            attention_attend_token_bf16<kFullHead>(
                q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
                head_dim, st.lane, scale, run_max, run_sum, acc);
        }
    }
}

template <bool kFullHead>
__device__ __forceinline__ void attention_visit_ancestors_bf16(
    const DecodeWalkState& st, uint32_t anc_base, uint32_t anc_count,
    const uint32_t* tree_anc_positions, const bf16* k_pool, const bf16* v_pool,
    const float (&q_reg)[kElemsPerLane], uint32_t head_dim, float scale,
    float& run_max, float& run_sum, float (&acc)[kElemsPerLane], uint32_t& bad) {
    for (uint32_t ai = st.wave; ai < anc_count; ai += kWaves) {
        const uint32_t pos = tree_anc_positions[anc_base + ai];
        const uint32_t anc_page_index = pos / st.page_tokens;
        const uint32_t anc_off = pos % st.page_tokens;
        const uint32_t page = st.block_table[anc_page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t elem =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page +
            static_cast<uint64_t>(anc_off) * st.elems_per_token;
        attention_attend_token_bf16<kFullHead>(
            q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
            head_dim, st.lane, scale, run_max, run_sum, acc);
    }
}

__device__ __forceinline__ void attention_stage_wave_partials(
    uint32_t tid, uint32_t lane, uint32_t wave, uint32_t head_dim,
    float run_max, float run_sum, uint32_t bad, const float (&acc)[kElemsPerLane],
    float (&merge_max_shared)[kWaves], float (&merge_sum_shared)[kWaves],
    uint32_t& invalid_shared, float (&acc_shared)[kWaves * kHeadLimit]) {
    if (lane == 0u) {
        merge_max_shared[wave] = run_max;
        merge_sum_shared[wave] = run_sum;
        if (bad != 0u) invalid_shared = 1u;
    }
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        if (d < head_dim) acc_shared[wave * kHeadLimit + d] = acc[j];
    }
    (void)tid;
}

// Wave merge with the frozen reduction order (contract §4).
__device__ __forceinline__ void attention_merge_wave_partials(
    uint32_t lane, const float (&merge_max_shared)[kWaves],
    const float (&merge_sum_shared)[kWaves],
    float (&weight_shared)[kWaves], float& sum_total) {
    const float global_max =
        warp_max_all((lane < kWaves) ? merge_max_shared[lane] : -FLT_MAX);
    sum_total = 0.0f;
    #pragma unroll
    for (uint32_t w = 0u; w < kWaves; ++w) {
        weight_shared[w] = __expf(merge_max_shared[w] - global_max);
        sum_total += merge_sum_shared[w] * weight_shared[w];
    }
}

__device__ __forceinline__ void attention_store_output_bf16(
    const PagedAttentionCommonArgs& a, const DecodeTaskMap& task, void* out_base,
    uint32_t tid, uint32_t head_dim,
    const float (&acc_shared)[kWaves * kHeadLimit], const float (&weight_shared)[kWaves],
    float sum_total) {
    if (tid < head_dim && sum_total > 0.0f) {
        float total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0u; w < kWaves; ++w)
            total += acc_shared[w * kHeadLimit + tid] * weight_shared[w];
        const float result = total / sum_total;
        if (a.output_dtype == PagedAttentionOutputDType::F32) {
            reinterpret_cast<float*>(out_base)
                [static_cast<size_t>(task.row) * a.output_row_stride + static_cast<size_t>(task.q_head) * head_dim + tid] = result;
        } else {
            reinterpret_cast<bf16*>(out_base)
                [static_cast<size_t>(task.row) * a.output_row_stride + static_cast<size_t>(task.q_head) * head_dim + tid] = __float2bfloat16(result);
        }
    }
}

template <bool kFullHead>
__device__ __forceinline__ void attention_paged_bf16_body(
    const PagedAttentionCommonArgs& a,
    const bf16* k_pool,
    const bf16* v_pool,
    void* out_base) {
    const DecodeTaskMap task = attention_map_decode_task(a);
    if (!task.active) return;

    const uint32_t head_dim = a.head_dim;
    const float scale = a.scale;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;

    __shared__ float merge_max_shared[kWaves];
    __shared__ float merge_sum_shared[kWaves];
    __shared__ float acc_shared[kWaves * kHeadLimit];
    __shared__ uint32_t row_valid_shared;
    __shared__ uint32_t visible_shared;
    __shared__ uint32_t invalid_shared;
    __shared__ const uint32_t* block_table_shared;
    __shared__ uint32_t anc_base_shared;
    __shared__ uint32_t anc_count_shared;

    if (tid == 0u) {
        attention_initialize_visible_range(a, task.row, row_valid_shared, visible_shared,
                                           invalid_shared, block_table_shared,
                                           anc_base_shared, anc_count_shared);
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    float q_reg[kElemsPerLane];
    attention_load_query_tile<kFullHead>(a, task, lane, q_reg);

    DecodeWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = block_table_shared;
    st.page_tokens = a.page_tokens;
    st.visible = visible_shared;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float acc[kElemsPerLane] = {};
    uint32_t bad = 0u;

    attention_visit_pages_bf16<kFullHead>(st, k_pool, v_pool, q_reg, head_dim, scale,
                                          run_max, run_sum, acc, bad);

    if (a.tree_anc_offsets != nullptr) {
        attention_visit_ancestors_bf16<kFullHead>(
            st, anc_base_shared, anc_count_shared, a.tree_anc_positions,
            k_pool, v_pool, q_reg, head_dim, scale, run_max, run_sum, acc, bad);
    }

    attention_stage_wave_partials(tid, lane, wave, head_dim, run_max, run_sum, bad, acc,
                                  merge_max_shared, merge_sum_shared, invalid_shared,
                                  acc_shared);
    __syncthreads();
    if (invalid_shared != 0u) return;

    float weight_shared[kWaves];
    float sum_total = 0.0f;
    attention_merge_wave_partials(lane, merge_max_shared, merge_sum_shared,
                                  weight_shared, sum_total);
    attention_store_output_bf16(a, task, out_base, tid, head_dim, acc_shared,
                                weight_shared, sum_total);
}

template <bool kFullHead>
__global__ __launch_bounds__(kThreads)
void attention_paged_bf16_impl(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
    void* out_base) {
    attention_paged_bf16_body<kFullHead>(a, k_pool, v_pool, out_base);
}

} }  // namespace v_decode_bf16

extern "C" __global__ __launch_bounds__(kPagedAttentionBlock)
void phaseshift_attention_paged_bf16(
    ::shelf::hipk::PagedAttentionBf16Args args) {
    v_decode_bf16::attention_paged_bf16_body<true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.common.output);
}

hipError_t launch_attention_paged_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    hipStream_t stream) {
    using namespace v_decode_bf16;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (common.head_dim == kHeadLimit) {
        PagedAttentionBf16Args args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        phaseshift_attention_paged_bf16<<<grid, kThreads, 0, stream>>>(args);
    } else {
        attention_paged_bf16_impl<false><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const bf16*>(k_pool), reinterpret_cast<const bf16*>(v_pool), common.output);
    }
    return hipGetLastError();
}

namespace { namespace v_decode_bf16_split {
using namespace v_decode_bf16;

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
static_assert(kSplitThreads == kPagedAttentionSplitBlock);
constexpr uint32_t kSplitMaxSplits = 16u;

// ---------------------------------------------------------------------------
// Algorithm layer (Phase 5, contract §4): BF16 split staging. Same stage
// names as decode where the responsibilities match; split owns a page window
// [page_begin, page_end) and writes per-split partials (workspace layout of
// the two-launch scheme is unchanged).
// ---------------------------------------------------------------------------

struct SplitTaskMap {
    uint32_t split;
    uint32_t row;
    uint32_t kv_head;
    bool active;   // row < rows; block-uniform
};

__device__ __forceinline__ SplitTaskMap attention_map_split_task(
    const PagedAttentionCommonArgs& a, uint32_t splits) {
    const uint32_t task = blockIdx.x;
    SplitTaskMap t{};
    t.split = task % splits;
    const uint32_t rest = task / splits;
    t.kv_head = rest % a.kv_heads;
    t.row = rest / a.kv_heads;
    t.active = t.row < a.rows;
    return t;
}

__device__ __forceinline__ void attention_split_init_page_range(
    const PagedAttentionCommonArgs& a, const SplitTaskMap& task, uint32_t splits,
    uint32_t& row_valid_shared, uint32_t& visible_shared, uint32_t& page_begin_shared,
    uint32_t& page_end_shared, const uint32_t*& block_table_shared) {
    const uint32_t page_tokens = a.page_tokens;
    const uint32_t slot = a.row_sequence_slots[task.row];
    uint32_t valid = 1u;
    uint32_t visible = 0u;
    if (slot >= a.max_sequences) {
        valid = 0u;
    } else {
        const uint64_t v64 = static_cast<uint64_t>(a.row_positions[task.row]) + 1u;
        const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
        if (v64 > cap) valid = 0u;
        else visible = static_cast<uint32_t>(v64);
    }
    const uint32_t page_count = (visible + page_tokens - 1u) / page_tokens;
    const uint32_t per = (page_count + splits - 1u) / splits;
    row_valid_shared = valid;
    visible_shared = visible;
    page_begin_shared = task.split * per;
    page_end_shared = (page_count < page_begin_shared + per) ? page_count : page_begin_shared + per;
    block_table_shared = a.block_tables
        + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
}

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_load_q_tile_split(
    const PagedAttentionCommonArgs& a, const SplitTaskMap& task, uint32_t tid,
    float (&q_lds)[kQPerKV * kHeadLimit]) {
    const uint32_t head_dim = a.head_dim;
    const bf16* q_base = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(task.row) * a.q_row_stride
        + static_cast<size_t>(task.kv_head * kQPerKV) * head_dim;
    for (uint32_t e = tid; e < kQPerKV * kHeadLimit; e += kSplitThreads) {
        const uint32_t qh = e / kHeadLimit;
        const uint32_t d = e % kHeadLimit;
        q_lds[e] = (kFullHead || d < head_dim)
            ? __bfloat162float(q_base[static_cast<size_t>(qh) * head_dim + d])
            : 0.0f;
    }
}

struct SplitWalkState {
    uint64_t head_off;
    uint64_t layer_base;
    const uint32_t* block_table;
    uint32_t page_tokens;
    uint32_t visible;
    uint32_t page_begin;
    uint32_t page_end;
    uint64_t elems_per_page;
    uint64_t elems_per_token;
    uint32_t num_pages;
    uint32_t wave;
    uint32_t lane;
};

// Page walk over this split's window; inner q-head loop order and the float
// K/V staging (converted at load) are the frozen split semantics (contract §4).
template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_visit_pages_split_bf16(
    const SplitWalkState& st, const bf16* k_pool, const bf16* v_pool,
    const float (&q_lds)[kQPerKV * kHeadLimit], uint32_t head_dim, float scale,
    float (&run_max)[kQPerKV], float (&run_sum)[kQPerKV],
    float (&acc)[kQPerKV][kElemsPerLane], uint32_t& bad) {
    const uint32_t lane = st.lane;
    for (uint32_t page_index = st.page_begin; page_index < st.page_end; ++page_index) {
        const uint32_t page = st.block_table[page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page;
        const uint32_t page_start = page_index * st.page_tokens;
        const uint32_t tokens_here =
            (st.visible - page_start < st.page_tokens) ? (st.visible - page_start) : st.page_tokens;
        for (uint32_t off = st.wave; off < tokens_here; off += kSplitWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * st.elems_per_token;
            const bf16* k_head = k_pool + elem + st.head_off;
            const bf16* v_head = v_pool + elem + st.head_off;
            float kf[kElemsPerLane];
            float vf[kElemsPerLane];
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
                const uint32_t d = lane + 32u * j;
                if (kFullHead || d < head_dim) {
                    kf[j] = __bfloat162float(k_head[d]);
                    vf[j] = __bfloat162float(v_head[d]);
                } else {
                    kf[j] = 0.0f;
                    vf[j] = 0.0f;
                }
            }
            #pragma unroll
            for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
                float partial = 0.0f;
                #pragma unroll
                for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                    partial += q_lds[qh * kHeadLimit + lane + 32u * j] * kf[j];
                const float score = warp_sum_all(partial) * scale;
                const float next_max = fmaxf(run_max[qh], score);
                const float alpha = __expf(run_max[qh] - next_max);
                const float weight = __expf(score - next_max);
                run_sum[qh] = run_sum[qh] * alpha + weight;
                run_max[qh] = next_max;
                #pragma unroll
                for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                    acc[qh][j] = acc[qh][j] * alpha + weight * vf[j];
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_stage_split_wave_partials(
    uint32_t lane, uint32_t wave, uint32_t head_dim,
    const float (&run_max)[kQPerKV], const float (&run_sum)[kQPerKV],
    const float (&acc)[kQPerKV][kElemsPerLane],
    float (&m_lds)[kSplitWaves * kQPerKV], float (&s_lds)[kSplitWaves * kQPerKV],
    float (&acc_lds)[kSplitWaves * kQPerKV * kHeadLimit]) {
    if (lane == 0u) {
        #pragma unroll
        for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
            m_lds[wave * kQPerKV + qh] = run_max[qh];
            s_lds[wave * kQPerKV + qh] = run_sum[qh];
        }
    }
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        #pragma unroll
        for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
            const uint32_t d = lane + 32u * j;
            if (kFullHead || d < head_dim)
                acc_lds[(wave * kQPerKV + qh) * kHeadLimit + d] = acc[qh][j];
        }
    }
}

// In-block wave merge (fmaxf tree m0..m3 exactly as before) + partial write.
template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_merge_store_split_partials(
    uint32_t tid, const float (&m_lds)[kSplitWaves * kQPerKV],
    const float (&s_lds)[kSplitWaves * kQPerKV],
    const float (&acc_lds)[kSplitWaves * kQPerKV * kHeadLimit],
    float* partial_base, uint32_t head_dim) {
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        float m[kSplitWaves];
        #pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w)
            m[w] = m_lds[w * kQPerKV + qh];
        const float g = fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3]));
        float wgt[kSplitWaves];
        float p_total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w) {
            wgt[w] = __expf(m[w] - g);
            p_total += s_lds[w * kQPerKV + qh] * wgt[w];
        }
        if (tid == 0u) {
            partial_base[qh] = g;
            partial_base[kQPerKV + qh] = p_total;
        }
        #pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t d = tid + r * 128u;
            if (kFullHead || d < head_dim) {
                float total = 0.0f;
                #pragma unroll
                for (uint32_t w = 0u; w < kSplitWaves; ++w)
                    total += acc_lds[(w * kQPerKV + qh) * kHeadLimit + d] * wgt[w];
                partial_base[2u * kQPerKV + qh * kHeadLimit + d] = total;
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_paged_split_bf16_body(
    const PagedAttentionCommonArgs& a,
    const bf16* k_pool,
    const bf16* v_pool,
    float* partials,
    uint32_t splits) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;
    const float scale = a.scale;

    const SplitTaskMap task = attention_map_split_task(a, splits);
    if (!task.active) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;

    __shared__ float q_lds[kQPerKV * kHeadLimit];
    __shared__ float acc_lds[kSplitWaves * kQPerKV * kHeadLimit];
    __shared__ float m_lds[kSplitWaves * kQPerKV];
    __shared__ float s_lds[kSplitWaves * kQPerKV];
    __shared__ uint32_t row_valid_shared;
    __shared__ uint32_t visible_shared;
    __shared__ uint32_t page_begin_shared;
    __shared__ uint32_t page_end_shared;
    __shared__ const uint32_t* block_table_shared;

    if (tid == 0u) {
        attention_split_init_page_range(a, task, splits, row_valid_shared, visible_shared,
                                        page_begin_shared, page_end_shared,
                                        block_table_shared);
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    float* partial_base = partials
        + static_cast<size_t>((static_cast<size_t>(task.row) * kv_heads + task.kv_head) * splits + task.split)
          * (kQPerKV * (2u + kHeadLimit));

    attention_load_q_tile_split<kQPerKV, kFullHead>(a, task, tid, q_lds);
    __syncthreads();

    SplitWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = block_table_shared;
    st.page_tokens = a.page_tokens;
    st.visible = visible_shared;
    st.page_begin = page_begin_shared;
    st.page_end = page_end_shared;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;

    float run_max[kQPerKV];
    float run_sum[kQPerKV];
    float acc[kQPerKV][kElemsPerLane];
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        run_max[qh] = -FLT_MAX;
        run_sum[qh] = 0.0f;
        #pragma unroll
        for (uint32_t j = 0u; j < kElemsPerLane; ++j)
            acc[qh][j] = 0.0f;
    }
    uint32_t bad = 0u;

    attention_visit_pages_split_bf16<kQPerKV, kFullHead>(
        st, k_pool, v_pool, q_lds, head_dim, scale, run_max, run_sum, acc, bad);

    if (bad != 0u) {
        #pragma unroll
        for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
            run_max[qh] = -FLT_MAX;
            run_sum[qh] = 0.0f;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                acc[qh][j] = 0.0f;
        }
    }

    attention_stage_split_wave_partials<kQPerKV, kFullHead>(lane, wave, head_dim,
                                                            run_max, run_sum, acc,
                                                            m_lds, s_lds, acc_lds);
    __syncthreads();

    attention_merge_store_split_partials<kQPerKV, kFullHead>(
        tid, m_lds, s_lds, acc_lds, partial_base, head_dim);
}


// ---------------------------------------------------------------------------
// Algorithm layer (Phase 5): second launch of the split scheme. Reads the
// partials produced above; three-barrier LDS publish order is contract §5.
// ---------------------------------------------------------------------------

template <uint32_t kQPerKV>
__device__ __forceinline__ void attention_reduce_max_pass(
    const float* base, uint32_t splits, uint32_t tid, float (&g_lds)[kQPerKV]) {
    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        if (tid == qh) {
            float g = -FLT_MAX;
            for (uint32_t s = 0u; s < splits; ++s)
                g = fmaxf(g, base[static_cast<uint64_t>(s) * a_stride + qh]);
            g_lds[qh] = g;
        }
    }
}

template <uint32_t kQPerKV>
__device__ __forceinline__ void attention_reduce_weight_sum_pass(
    const float* base, uint32_t splits, uint32_t tid, const float (&g_lds)[kQPerKV],
    float (&wgt_lds)[kQPerKV * kSplitMaxSplits], float (&sum_lds)[kQPerKV]) {
    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        if (tid == qh) {
            const float g = g_lds[qh];
            float sum_total = 0.0f;
            for (uint32_t s = 0u; s < splits; ++s) {
                const float m_s = base[static_cast<uint64_t>(s) * a_stride + qh];
                const float w_s = __expf(m_s - g);
                wgt_lds[qh * kSplitMaxSplits + s] = w_s;
                sum_total += w_s * base[static_cast<uint64_t>(s) * a_stride + kQPerKV + qh];
            }
            sum_lds[qh] = sum_total;
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_reduce_store_bf16(
    const PagedAttentionCommonArgs& a, uint32_t row, uint32_t kv_head,
    const float* base, const float (&sum_lds)[kQPerKV],
    const float (&wgt_lds)[kQPerKV * kSplitMaxSplits],
    void* out_base, uint32_t tid, uint32_t head_dim, uint32_t splits) {
    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    const uint32_t q_head0 = kv_head * kQPerKV;
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        const float sum_total = sum_lds[qh];
        if (sum_total <= 0.0f) continue;
        #pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t d = tid + r * 128u;
            if (kFullHead || d < head_dim) {
                float total = 0.0f;
                for (uint32_t s = 0u; s < splits; ++s)
                    total += wgt_lds[qh * kSplitMaxSplits + s]
                        * base[static_cast<uint64_t>(s) * a_stride + 2u * kQPerKV + qh * kHeadLimit + d];
                const float result = total / sum_total;
                const uint64_t out_off =
                    static_cast<uint64_t>(row) * a.output_row_stride
                    + static_cast<uint64_t>(q_head0 + qh) * head_dim + d;
                if (a.output_dtype == PagedAttentionOutputDType::F32) {
                    reinterpret_cast<float*>(out_base)[out_off] = result;
                } else {
                    reinterpret_cast<bf16*>(out_base)[out_off] = __float2bfloat16(result);
                }
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_paged_split_reduce_bf16_body(
    const PagedAttentionCommonArgs& a,
    const float* partials,
    uint32_t splits,
    void* out_base) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;

    const uint32_t task = blockIdx.x;
    const uint32_t row = task / kv_heads;
    const uint32_t kv_head = task % kv_heads;
    const uint32_t tid = threadIdx.x;

    if (row >= a.rows) return;

    __shared__ float g_lds[kQPerKV];
    __shared__ float sum_lds[kQPerKV];
    __shared__ float wgt_lds[kQPerKV * kSplitMaxSplits];
    __shared__ uint32_t row_valid_shared;

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        row_valid_shared = (slot < a.max_sequences) ? 1u : 0u;
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    const float* base = partials
        + static_cast<size_t>((static_cast<size_t>(row) * kv_heads + kv_head) * splits)
          * kQPerKV * (2u + kHeadLimit);

    attention_reduce_max_pass<kQPerKV>(base, splits, tid, g_lds);
    __syncthreads();
    attention_reduce_weight_sum_pass<kQPerKV>(base, splits, tid, g_lds, wgt_lds, sum_lds);
    __syncthreads();
    attention_reduce_store_bf16<kQPerKV, kFullHead>(a, row, kv_head, base, sum_lds,
                                                    wgt_lds, out_base, tid, head_dim, splits);
}


template <uint32_t kQPerKV, bool kFullHead>
__global__ __launch_bounds__(kSplitThreads)
void attention_paged_split_bf16_impl(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
    float* partials,
    const uint32_t splits) {
    attention_paged_split_bf16_body<kQPerKV, kFullHead>(a, k_pool, v_pool,
                                                       partials, splits);
}

template <uint32_t kQPerKV, bool kFullHead>
__global__ __launch_bounds__(kSplitThreads)
void attention_paged_split_reduce_bf16_impl(
    const PagedAttentionCommonArgs a,
    const float* partials,
    const uint32_t splits,
    void* out_base) {
    attention_paged_split_reduce_bf16_body<kQPerKV, kFullHead>(a, partials,
                                                              splits, out_base);
}

} }  // namespace v_decode_bf16_split

namespace { namespace v_decode_bf16_split_qh {

using bf16 = __hip_bfloat16;

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
constexpr uint32_t kHeadLimit = kPagedAttentionHeadLimit;
constexpr uint32_t kSplitMaxSplits = 16u;

template <uint32_t kQPerKV, bool kFullHead, uint32_t kQhPerWg>
__device__ __forceinline__ void attention_paged_split_reduce_qh_bf16_body(
    const PagedAttentionCommonArgs& a,
    const float* partials,
    uint32_t splits,
    void* out_base) {
    constexpr uint32_t kTiles = kQPerKV / kQhPerWg;
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;

    const uint32_t task = blockIdx.x;
    const uint32_t kv_head = task % kv_heads;
    const uint32_t rest = task / kv_heads;
    const uint32_t row = rest / kTiles;
    const uint32_t qh0 = (rest % kTiles) * kQhPerWg;
    const uint32_t tid = threadIdx.x;

    if (row >= a.rows) return;

    __shared__ float g_lds[kQhPerWg];
    __shared__ float sum_lds[kQhPerWg];
    __shared__ float wgt_lds[kQhPerWg * kSplitMaxSplits];
    __shared__ uint32_t row_valid_shared;

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        row_valid_shared = (slot < a.max_sequences) ? 1u : 0u;
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    const float* base = partials
        + static_cast<size_t>((static_cast<size_t>(row) * kv_heads + kv_head) * splits)
          * kQPerKV * (2u + kHeadLimit);

    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t q = 0u; q < kQhPerWg; ++q) {
        if (tid == q) {
            float g = -FLT_MAX;
            for (uint32_t s = 0u; s < splits; ++s)
                g = fmaxf(g, base[static_cast<uint64_t>(s) * a_stride + qh0 + q]);
            g_lds[q] = g;
        }
    }
    __syncthreads();
    for (uint32_t q = 0u; q < kQhPerWg; ++q) {
        if (tid == q) {
            const float g = g_lds[q];
            float sum_total = 0.0f;
            for (uint32_t s = 0u; s < splits; ++s) {
                const float m_s = base[static_cast<uint64_t>(s) * a_stride + qh0 + q];
                const float w_s = __expf(m_s - g);
                wgt_lds[q * kSplitMaxSplits + s] = w_s;
                sum_total += w_s * base[static_cast<uint64_t>(s) * a_stride + kQPerKV + qh0 + q];
            }
            sum_lds[q] = sum_total;
        }
    }
    __syncthreads();

    const uint32_t q_head0 = kv_head * kQPerKV + qh0;
    #pragma unroll
    for (uint32_t q = 0u; q < kQhPerWg; ++q) {
        const float sum_total = sum_lds[q];
        if (sum_total <= 0.0f) continue;
        #pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t d = tid + r * 128u;
            if (kFullHead || d < head_dim) {
                float total = 0.0f;
                for (uint32_t s = 0u; s < splits; ++s)
                    total += wgt_lds[q * kSplitMaxSplits + s]
                        * base[static_cast<uint64_t>(s) * a_stride + 2u * kQPerKV
                               + (qh0 + q) * kHeadLimit + d];
                const float result = total / sum_total;
                const uint64_t out_off =
                    static_cast<uint64_t>(row) * a.output_row_stride
                    + static_cast<uint64_t>(q_head0 + q) * head_dim + d;
                if (a.output_dtype == PagedAttentionOutputDType::F32) {
                    reinterpret_cast<float*>(out_base)[out_off] = result;
                } else {
                    reinterpret_cast<bf16*>(out_base)[out_off] = __float2bfloat16(result);
                }
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead, uint32_t kQhPerWg>
__global__ __launch_bounds__(kSplitThreads)
void attention_paged_split_reduce_qh_bf16_impl(
    const PagedAttentionCommonArgs a,
    const float* partials,
    const uint32_t splits,
    void* out_base) {
    attention_paged_split_reduce_qh_bf16_body<kQPerKV, kFullHead, kQhPerWg>(
        a, partials, splits, out_base);
}

template <uint32_t kQPerKV, bool kFullHead>
hipError_t launch_split_reduce_qh(const PagedAttentionCommonArgs& a,
                                  const float* partials,
                                  uint32_t splits,
                                  void* out_base,
                                  uint32_t qh_per_wg,
                                  hipStream_t stream) {
    const uint32_t tiles = kQPerKV / qh_per_wg;
    if (tiles == 0u || kQPerKV % qh_per_wg != 0u) return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(a.rows) * a.kv_heads * tiles;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    switch (qh_per_wg) {
        case 1u:
            attention_paged_split_reduce_qh_bf16_impl<kQPerKV, kFullHead, 1u>
                <<<grid, kSplitThreads, 0, stream>>>(a, partials, splits, out_base);
            break;
        case 2u:
            attention_paged_split_reduce_qh_bf16_impl<kQPerKV, kFullHead, 2u>
                <<<grid, kSplitThreads, 0, stream>>>(a, partials, splits, out_base);
            break;
        case 3u:
            attention_paged_split_reduce_qh_bf16_impl<kQPerKV, kFullHead, 3u>
                <<<grid, kSplitThreads, 0, stream>>>(a, partials, splits, out_base);
            break;
        case 4u:
            attention_paged_split_reduce_qh_bf16_impl<kQPerKV, kFullHead, 4u>
                <<<grid, kSplitThreads, 0, stream>>>(a, partials, splits, out_base);
            break;
        default:
            return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

} }  // namespace v_decode_bf16_split_qh

namespace {

uint32_t attention_split_reduce_qh_per_wg() {
    const char* s = std::getenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH");
    if (s == nullptr || s[0] == '\0') return 1u;
    const long n = std::strtol(s, nullptr, 10);
    if (n == 1 || n == 2 || n == 3 || n == 4 || n == 6) return static_cast<uint32_t>(n);
    return 1u;
}

}  // namespace

uint32_t paged_attention_split_reduce_qh_per_wg() {
    return attention_split_reduce_qh_per_wg();
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_split_q4_full(
    ::shelf::hipk::PagedAttentionSplitArgs args) {
    v_decode_bf16_split::attention_paged_split_bf16_body<4u, true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_qh1(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split_qh::attention_paged_split_reduce_qh_bf16_body<4u, true, 1u>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_qh2(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split_qh::attention_paged_split_reduce_qh_bf16_body<4u, true, 2u>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_base(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split::attention_paged_split_reduce_bf16_body<4u, true>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_split_other(
    ::shelf::hipk::PagedAttentionSplitArgs args) {
    v_decode_bf16_split::attention_paged_split_bf16_body<6u, true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_other(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split::attention_paged_split_reduce_bf16_body<6u, true>(
        args.common, args.partials, args.splits, args.common.output);
}

hipError_t launch_attention_paged_bf16_split(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x =
        static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    const bool full_head = common.head_dim == kHeadLimit;
    if (q_per_kv == 4u && full_head) {
        PagedAttentionSplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_bf16_split_q4_full
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        if (full_head) {
            attention_paged_split_bf16_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, reinterpret_cast<const bf16*>(k_pool),
                reinterpret_cast<const bf16*>(v_pool), partials, splits);
        }
    } else {
        if (full_head) {
            PagedAttentionSplitArgs stable{};
            stable.common = a;
            stable.k_pool = k_pool;
            stable.v_pool = v_pool;
            stable.partials = partials;
            stable.splits = splits;
            phaseshift_attention_paged_bf16_split_other
                <<<grid, kSplitThreads, 0, stream>>>(stable);
        } else {
            attention_paged_split_bf16_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, reinterpret_cast<const bf16*>(k_pool),
                reinterpret_cast<const bf16*>(v_pool), partials, splits);
        }
    }
    return hipGetLastError();
}

hipError_t launch_attention_paged_bf16_split_reduce(
    const PagedAttentionCommonArgs& common,
    const float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.output == nullptr || common.row_sequence_slots == nullptr ||
        common.row_positions == nullptr)
        return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const bool full_head = common.head_dim == kHeadLimit;
    {
        using namespace v_decode_bf16_split_qh;
        const uint32_t qh_per_wg = attention_split_reduce_qh_per_wg();
        if (qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0u) {
            if (q_per_kv == 4u) {
                if (full_head && (qh_per_wg == 1u || qh_per_wg == 2u)) {
                    PagedAttentionSplitReduceArgs stable{};
                    stable.common = a;
                    stable.partials = partials;
                    stable.splits = splits;
                    const uint64_t qh_grid = static_cast<uint64_t>(a.rows) *
                                             a.kv_heads *
                                             (q_per_kv / qh_per_wg);
                    const dim3 qhg(static_cast<uint32_t>(qh_grid));
                    if (qh_per_wg == 1u) {
                        phaseshift_attention_paged_bf16_reduce_q4_full_qh1
                            <<<qhg, kPagedAttentionSplitBlock, 0, stream>>>(stable);
                    } else {
                        phaseshift_attention_paged_bf16_reduce_q4_full_qh2
                            <<<qhg, kPagedAttentionSplitBlock, 0, stream>>>(stable);
                    }
                    return hipGetLastError();
                }
                return full_head
                    ? launch_split_reduce_qh<4u, true>(a, partials, splits, common.output,
                                                       qh_per_wg, stream)
                    : launch_split_reduce_qh<4u, false>(a, partials, splits, common.output,
                                                        qh_per_wg, stream);
            }
            return full_head
                ? launch_split_reduce_qh<6u, true>(a, partials, splits, common.output,
                                                   qh_per_wg, stream)
                : launch_split_reduce_qh<6u, false>(a, partials, splits, common.output,
                                                    qh_per_wg, stream);
        }
    }
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (q_per_kv == 4u && full_head) {
        PagedAttentionSplitReduceArgs stable{};
        stable.common = a;
        stable.partials = partials;
        stable.splits = splits;
        phaseshift_attention_paged_bf16_reduce_q4_full_base
            <<<grid, kSplitThreads, 0, stream>>>(stable);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        if (full_head) {
            attention_paged_split_reduce_bf16_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, partials, splits, common.output);
        }
    } else {
        if (full_head) {
            PagedAttentionSplitReduceArgs stable{};
            stable.common = a;
            stable.partials = partials;
            stable.splits = splits;
            phaseshift_attention_paged_bf16_reduce_other
                <<<grid, kSplitThreads, 0, stream>>>(stable);
        } else {
            attention_paged_split_reduce_bf16_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, partials, splits, common.output);
        }
    }
    return hipGetLastError();
}

namespace { namespace v_decode_fp8 {

using bf16 = __hip_bfloat16;
using fp8 = ::shelf::hipk::fp8e4m3_storage_t;
using ::shelf::hipk::e4m3_decode_f32;

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadLimit = 256u;
constexpr uint32_t kElemsPerLane = kHeadLimit / 32u;

__device__ __forceinline__ float warp_sum_all(float x) {
    x += __shfl_xor(x, 16, 32);
    x += __shfl_xor(x, 8, 32);
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    return x;
}

__device__ __forceinline__ float warp_max_all(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

__device__ __forceinline__ unsigned short f2bf_p8(float f) {
    return static_cast<unsigned short>(
        (__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16u);
}

__device__ __forceinline__ float fp8_lut_value(
    const unsigned short* lut, fp8 code) {
    return __bfloat162float(__ushort_as_bfloat16(lut[code]));
}

template <bool kFullHead>
__global__ __launch_bounds__(kThreads)
void attention_paged_fp8_impl(
    const PagedAttentionCommonArgs a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    void* out_base) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t q_heads = a.q_heads;
    const uint32_t kv_heads = a.kv_heads;
    const uint32_t page_tokens = a.page_tokens;
    const uint32_t q_per_kv = q_heads / kv_heads;
    const float scale = a.scale;

    const uint32_t task = blockIdx.x;
    const uint32_t row = task / q_heads;
    const uint32_t q_head = task % q_heads;
    const uint32_t kv_head = q_head / q_per_kv;

    if (row >= a.rows) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;

    __shared__ float merge_max_shared[kWaves];
    __shared__ float merge_sum_shared[kWaves];
    __shared__ float acc_shared[kWaves * kHeadLimit];
    __shared__ uint32_t row_valid_shared;
    __shared__ uint32_t visible_shared;
    __shared__ uint32_t invalid_shared;
    __shared__ const uint32_t* block_table_shared;
    __shared__ unsigned short decode_lut[256];
    __shared__ uint32_t anc_base_shared;
    __shared__ uint32_t anc_count_shared;

    for (uint32_t i = tid; i < 256u; i += kThreads)
        decode_lut[i] = f2bf_p8(e4m3_decode_f32(static_cast<uint8_t>(i)));

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        uint32_t valid = 1u;
        uint32_t visible = 0u;
        anc_base_shared = 0u;
        anc_count_shared = 0u;
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else if (a.tree_anc_offsets != nullptr) {
            anc_base_shared = a.tree_anc_offsets[row];
            uint32_t anc_count = a.tree_anc_offsets[row + 1u] - anc_base_shared;
            const uint32_t committed_end = a.tree_committed_end_rows != nullptr
                                               ? a.tree_committed_end_rows[row]
                                               : a.tree_committed_end;
            bool contiguous = true;
            for (uint32_t i = 0u; i < anc_count; ++i) {
                if (a.tree_anc_positions[anc_base_shared + i] != committed_end + i) {
                    contiguous = false;
                    break;
                }
            }
            const uint64_t v64 = contiguous
                                     ? static_cast<uint64_t>(committed_end) + anc_count
                                     : static_cast<uint64_t>(committed_end);
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
            if (v64 > cap) valid = 0u;
            else {
                visible = static_cast<uint32_t>(v64);
                anc_count_shared = contiguous ? 0u : anc_count;
            }
        } else {
            const uint64_t v64 = static_cast<uint64_t>(a.row_positions[row]) + 1u;
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
            if (v64 > cap) valid = 0u;
            else visible = static_cast<uint32_t>(v64);
        }
        row_valid_shared = valid;
        visible_shared = visible;
        invalid_shared = 0u;
        block_table_shared = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    const uint32_t visible = visible_shared;
    const uint32_t anc_base = anc_base_shared;
    const uint32_t anc_count = anc_count_shared;
    const uint32_t* block_table = block_table_shared;

    const bf16* q_row = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(row) * a.q_row_stride
        + static_cast<size_t>(q_head) * head_dim;

    float q_reg[kElemsPerLane];
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        q_reg[j] = (kFullHead || d < head_dim) ? __bfloat162float(q_row[d]) : 0.0f;
    }

    const uint64_t head_off = static_cast<uint64_t>(kv_head) * head_dim;
    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const size_t scale_layer_base =
        static_cast<size_t>(a.layer) * a.num_pages * a.page_tokens * a.kv_heads;
    const uint32_t page_count = (visible + page_tokens - 1u) / page_tokens;

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float acc[kElemsPerLane] = {};
    uint32_t bad = 0u;

    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = block_table[page_index];
        if (page >= a.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const size_t page_scale_base =
            scale_layer_base + static_cast<size_t>(page) * page_tokens * a.kv_heads;
        const uint32_t page_start = page_index * page_tokens;
        const uint32_t tokens_here =
            (visible - page_start < page_tokens) ? (visible - page_start) : page_tokens;
        for (uint32_t off = wave; off < tokens_here; off += kWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * a.elems_per_token;
            const size_t si = page_scale_base + static_cast<size_t>(off) * a.kv_heads + kv_head;
            const float k_scale = k_scale_pool[si];
            const float v_scale = v_scale_pool[si];
            const fp8* k_head = k_pool + elem + head_off;
            const fp8* v_head = v_pool + elem + head_off;
            fp8 k_reg[kElemsPerLane];
            fp8 v_reg[kElemsPerLane];
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
                const uint32_t d = lane + 32u * j;
                if (kFullHead || d < head_dim) {
                    k_reg[j] = k_head[d];
                    v_reg[j] = v_head[d];
                } else {
                    k_reg[j] = static_cast<fp8>(0u);
                    v_reg[j] = static_cast<fp8>(0u);
                }
            }
            float partial = 0.0f;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                partial += q_reg[j] * fp8_lut_value(decode_lut, k_reg[j]);
            const float score = warp_sum_all(partial) * k_scale * scale;
            const float next_max = fmaxf(run_max, score);
            const float alpha = __expf(run_max - next_max);
            const float weight = __expf(score - next_max);
            const float v_weight = weight * v_scale;
            run_sum = run_sum * alpha + weight;
            run_max = next_max;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                acc[j] = acc[j] * alpha + v_weight * fp8_lut_value(decode_lut, v_reg[j]);
        }
    }

    if (a.tree_anc_offsets != nullptr) {
        for (uint32_t ai = wave; ai < anc_count; ai += kWaves) {
            const uint32_t pos = a.tree_anc_positions[anc_base + ai];
            const uint32_t anc_page_index = pos / page_tokens;
            const uint32_t anc_off = pos % page_tokens;
            const uint32_t page = block_table[anc_page_index];
            if (page >= a.num_pages) {
                bad = 1u;
                break;
            }
            const uint64_t elem =
                layer_base + static_cast<uint64_t>(page) * a.elems_per_page +
                static_cast<uint64_t>(anc_off) * a.elems_per_token;
            const size_t si =
                scale_layer_base +
                (static_cast<size_t>(page) * page_tokens + anc_off) * a.kv_heads + kv_head;
            const float k_scale = k_scale_pool[si];
            const float v_scale = v_scale_pool[si];
            const fp8* k_head = k_pool + elem + head_off;
            const fp8* v_head = v_pool + elem + head_off;
            fp8 k_reg[kElemsPerLane];
            fp8 v_reg[kElemsPerLane];
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
                const uint32_t d = lane + 32u * j;
                if (kFullHead || d < head_dim) {
                    k_reg[j] = k_head[d];
                    v_reg[j] = v_head[d];
                } else {
                    k_reg[j] = static_cast<fp8>(0u);
                    v_reg[j] = static_cast<fp8>(0u);
                }
            }
            float partial = 0.0f;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                partial += q_reg[j] * fp8_lut_value(decode_lut, k_reg[j]);
            const float score = warp_sum_all(partial) * k_scale * scale;
            const float next_max = fmaxf(run_max, score);
            const float alpha = __expf(run_max - next_max);
            const float weight = __expf(score - next_max);
            const float v_weight = weight * v_scale;
            run_sum = run_sum * alpha + weight;
            run_max = next_max;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                acc[j] = acc[j] * alpha + v_weight * fp8_lut_value(decode_lut, v_reg[j]);
        }
    }

    if (lane == 0u) {
        merge_max_shared[wave] = run_max;
        merge_sum_shared[wave] = run_sum;
        if (bad != 0u) invalid_shared = 1u;
    }
    #pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        if (d < head_dim) acc_shared[wave * kHeadLimit + d] = acc[j];
    }
    __syncthreads();
    if (invalid_shared != 0u) return;

    const float global_max =
        warp_max_all((lane < kWaves) ? merge_max_shared[lane] : -FLT_MAX);

    float weight_shared[kWaves];
    float sum_total = 0.0f;
    #pragma unroll
    for (uint32_t w = 0u; w < kWaves; ++w) {
        weight_shared[w] = __expf(merge_max_shared[w] - global_max);
        sum_total += merge_sum_shared[w] * weight_shared[w];
    }

    if (tid < head_dim && sum_total > 0.0f) {
        float total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0u; w < kWaves; ++w)
            total += acc_shared[w * kHeadLimit + tid] * weight_shared[w];
        const float result = total / sum_total;
        if (a.output_dtype == PagedAttentionOutputDType::F32) {
            reinterpret_cast<float*>(out_base)
                [static_cast<size_t>(row) * a.output_row_stride + static_cast<size_t>(q_head) * head_dim + tid] = result;
        } else {
            reinterpret_cast<bf16*>(out_base)
                [static_cast<size_t>(row) * a.output_row_stride + static_cast<size_t>(q_head) * head_dim + tid] = __float2bfloat16(result);
        }
    }
}

} }  // namespace v_decode_fp8

hipError_t launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,
    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    hipStream_t stream) {
    using namespace v_decode_fp8;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (common.head_dim == kHeadLimit) {
        attention_paged_fp8_impl<true><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, common.output);
    } else {
        attention_paged_fp8_impl<false><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, common.output);
    }
    return hipGetLastError();
}


namespace { namespace v_decode_fp8_split {

using namespace v_decode_fp8;

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
static_assert(kSplitThreads == kPagedAttentionSplitBlock);

template <uint32_t kQPerKV, bool kFullHead>
__device__ __forceinline__ void attention_paged_split_fp8_body(
    const PagedAttentionCommonArgs& a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    uint32_t splits) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;
    const uint32_t page_tokens = a.page_tokens;
    const float scale = a.scale;

    const uint32_t task = blockIdx.x;
    const uint32_t split = task % splits;
    const uint32_t rest = task / splits;
    const uint32_t kv_head = rest % kv_heads;
    const uint32_t row = rest / kv_heads;

    if (row >= a.rows) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;

    __shared__ float q_lds[kQPerKV * kHeadLimit];
    __shared__ float acc_lds[kSplitWaves * kQPerKV * kHeadLimit];
    __shared__ float m_lds[kSplitWaves * kQPerKV];
    __shared__ float s_lds[kSplitWaves * kQPerKV];
    __shared__ uint32_t row_valid_shared;
    __shared__ uint32_t visible_shared;
    __shared__ uint32_t page_begin_shared;
    __shared__ uint32_t page_end_shared;
    __shared__ const uint32_t* block_table_shared;
    __shared__ unsigned short decode_lut[256];

    for (uint32_t i = tid; i < 256u; i += kSplitThreads)
        decode_lut[i] = f2bf_p8(e4m3_decode_f32(static_cast<uint8_t>(i)));

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        uint32_t valid = 1u;
        uint32_t visible = 0u;
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint64_t v64 = static_cast<uint64_t>(a.row_positions[row]) + 1u;
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * page_tokens;
            if (v64 > cap) valid = 0u;
            else visible = static_cast<uint32_t>(v64);
        }
        const uint32_t page_count = (visible + page_tokens - 1u) / page_tokens;
        const uint32_t per = (page_count + splits - 1u) / splits;
        row_valid_shared = valid;
        visible_shared = visible;
        page_begin_shared = split * per;
        page_end_shared = (page_count < page_begin_shared + per) ? page_count : page_begin_shared + per;
        block_table_shared = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    __syncthreads();
    if (row_valid_shared == 0u) return;

    const uint32_t visible = visible_shared;
    const uint32_t* block_table = block_table_shared;
    const uint32_t page_begin = page_begin_shared;
    const uint32_t page_end = page_end_shared;

    float* partial_base = partials
        + static_cast<size_t>((static_cast<size_t>(row) * kv_heads + kv_head) * splits + split)
          * (kQPerKV * (2u + kHeadLimit));

    const bf16* q_base = reinterpret_cast<const bf16*>(a.q)
        + static_cast<size_t>(row) * a.q_row_stride
        + static_cast<size_t>(kv_head * kQPerKV) * head_dim;

    for (uint32_t e = tid; e < kQPerKV * kHeadLimit; e += kSplitThreads) {
        const uint32_t qh = e / kHeadLimit;
        const uint32_t d = e % kHeadLimit;
        q_lds[e] = (kFullHead || d < head_dim)
            ? __bfloat162float(q_base[static_cast<size_t>(qh) * head_dim + d])
            : 0.0f;
    }
    __syncthreads();

    const uint64_t head_off = static_cast<uint64_t>(kv_head) * head_dim;
    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const size_t scale_layer_base =
        static_cast<size_t>(a.layer) * a.num_pages * a.page_tokens * a.kv_heads;

    float run_max[kQPerKV];
    float run_sum[kQPerKV];
    float acc[kQPerKV][kElemsPerLane];
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        run_max[qh] = -FLT_MAX;
        run_sum[qh] = 0.0f;
        #pragma unroll
        for (uint32_t j = 0u; j < kElemsPerLane; ++j)
            acc[qh][j] = 0.0f;
    }
    uint32_t bad = 0u;

    for (uint32_t page_index = page_begin; page_index < page_end; ++page_index) {
        const uint32_t page = block_table[page_index];
        if (page >= a.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const size_t page_scale_base =
            scale_layer_base + static_cast<size_t>(page) * page_tokens * kv_heads;
        const uint32_t page_start = page_index * page_tokens;
        const uint32_t tokens_here =
            (visible - page_start < page_tokens) ? (visible - page_start) : page_tokens;
        for (uint32_t off = wave; off < tokens_here; off += kSplitWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * a.elems_per_token;
            const size_t si =
                page_scale_base + static_cast<size_t>(off) * kv_heads + kv_head;
            const float k_scale = k_scale_pool[si];
            const float v_scale = v_scale_pool[si];
            const fp8* k_head = k_pool + elem + head_off;
            const fp8* v_head = v_pool + elem + head_off;
            float kf[kElemsPerLane];
            float vf[kElemsPerLane];
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
                const uint32_t d = lane + 32u * j;
                if (kFullHead || d < head_dim) {
                    kf[j] = fp8_lut_value(decode_lut, k_head[d]);
                    vf[j] = fp8_lut_value(decode_lut, v_head[d]);
                } else {
                    kf[j] = 0.0f;
                    vf[j] = 0.0f;
                }
            }
            #pragma unroll
            for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
                float partial = 0.0f;
                #pragma unroll
                for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                    partial += q_lds[qh * kHeadLimit + lane + 32u * j] * kf[j];
                const float score = warp_sum_all(partial) * k_scale * scale;
                const float next_max = fmaxf(run_max[qh], score);
                const float alpha = __expf(run_max[qh] - next_max);
                const float weight = __expf(score - next_max);
                run_sum[qh] = run_sum[qh] * alpha + weight;
                run_max[qh] = next_max;
                #pragma unroll
                for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                    acc[qh][j] = acc[qh][j] * alpha + weight * v_scale * vf[j];
            }
        }
    }

    if (bad != 0u) {
        #pragma unroll
        for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
            run_max[qh] = -FLT_MAX;
            run_sum[qh] = 0.0f;
            #pragma unroll
            for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                acc[qh][j] = 0.0f;
        }
    }

    if (lane == 0u) {
        #pragma unroll
        for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
            m_lds[wave * kQPerKV + qh] = run_max[qh];
            s_lds[wave * kQPerKV + qh] = run_sum[qh];
        }
    }
    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        #pragma unroll
        for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
            const uint32_t d = lane + 32u * j;
            if (kFullHead || d < head_dim)
                acc_lds[(wave * kQPerKV + qh) * kHeadLimit + d] = acc[qh][j];
        }
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        float m[kSplitWaves];
        #pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w)
            m[w] = m_lds[w * kQPerKV + qh];
        const float g = fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3]));
        float wgt[kSplitWaves];
        float p_total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w) {
            wgt[w] = __expf(m[w] - g);
            p_total += s_lds[w * kQPerKV + qh] * wgt[w];
        }
        if (tid == 0u) {
            partial_base[qh] = g;
            partial_base[kQPerKV + qh] = p_total;
        }
        #pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t d = tid + r * 128u;
            if (kFullHead || d < head_dim) {
                float total = 0.0f;
                #pragma unroll
                for (uint32_t w = 0u; w < kSplitWaves; ++w)
                    total += acc_lds[(w * kQPerKV + qh) * kHeadLimit + d] * wgt[w];
                partial_base[2u * kQPerKV + qh * kHeadLimit + d] = total;
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
__global__ __launch_bounds__(kPagedAttentionSplitBlock)
void attention_paged_split_fp8_impl(
    const PagedAttentionCommonArgs a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    const uint32_t splits) {
    attention_paged_split_fp8_body<kQPerKV, kFullHead>(
        a, k_pool, v_pool, k_scale_pool, v_scale_pool, partials, splits);
}

} }  // namespace v_decode_fp8_split

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_fp8_split_q4_full(
    ::shelf::hipk::PagedAttentionFp8SplitArgs args) {
    v_decode_fp8_split::attention_paged_split_fp8_body<4u, true>(
        args.common,
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.k_pool),
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.v_pool),
        args.k_scale_pool, args.v_scale_pool, args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_fp8_split_other(
    ::shelf::hipk::PagedAttentionFp8SplitArgs args) {
    v_decode_fp8_split::attention_paged_split_fp8_body<6u, true>(
        args.common,
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.k_pool),
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.v_pool),
        args.k_scale_pool, args.v_scale_pool, args.partials, args.splits);
}

hipError_t launch_attention_paged_fp8_split(
    const PagedAttentionCommonArgs& common,
    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_fp8_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x =
        static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    const bool full_head = common.head_dim == kHeadLimit;
    if (q_per_kv == 4u && full_head) {
        PagedAttentionFp8SplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.k_scale_pool = k_scale_pool;
        args.v_scale_pool = v_scale_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_fp8_split_q4_full
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        attention_paged_split_fp8_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, partials, splits);
        return hipGetLastError();
    }
    if (full_head) {
        PagedAttentionFp8SplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.k_scale_pool = k_scale_pool;
        args.v_scale_pool = v_scale_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_fp8_split_other
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    attention_paged_split_fp8_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
        a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
        k_scale_pool, v_scale_pool, partials, splits);
    return hipGetLastError();
}


}  // namespace shelf::hipk
