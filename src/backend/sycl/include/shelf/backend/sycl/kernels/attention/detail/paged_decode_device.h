// shelf attention device layer (SYCL port of backend/hip
// kernels/attention/detail/paged_decode_device.h). Private header for
// src/backend/sycl/kernels/attention/attention_paged_decode.cpp: staged
// algorithm helpers per variant.
//
// Conversion contract vs the HIP original:
//   * stage names, task maps, page addressing, workspace layouts and the
//     FP8/BF16 numeric orders are preserved (same expressions, same order)
//     except where SYCL 2020 forces a different primitive:
//       - warp xor butterflies (warp_sum_all / warp_max_all) -> sycl
//         sub_group collectives (see kernels/detail/sycl_util.h note: wave
//         summation order may differ from the frozen HIP trees; results stay
//         within tolerance vs the CPU reference, NOT bitwise vs HIP).
//       - __shared__ -> sycl local_accessor memory handed in via a Locals
//         struct; barrier placement is identical to the HIP bodies.
//   * the tree's "__global__ + launcher" split becomes: bodies here, kernel
//     entries + launchers in the .cpp TU.
// Barriers and block-uniform returns stay inside each *_body.
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <shelf/backend/sycl/kernels/attention/kv_types.h>
#include <shelf/backend/sycl/kernels/attention/paged_attention.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/backend/sycl/kernels/e4m3_codec.h>
#include <shelf/foundation/numeric.hpp>

namespace shelf::syclk {
namespace attn_detail {

namespace v_decode_bf16 {

using bf16 = shelf::bf16_t;

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadLimit = kPagedAttentionHeadLimit;
static_assert(kThreads == kPagedAttentionBlock);
constexpr uint32_t kElemsPerLane = kHeadLimit / 32u;

// Local-memory staging for one decode block.
struct Locals {
    float* merge_max;      // [kWaves]
    float* merge_sum;      // [kWaves]
    float* acc;            // [kWaves * kHeadLimit]
    uint32_t* row_valid;
    uint32_t* visible;
    uint32_t* invalid;
    const uint32_t** block_table;
    uint32_t* anc_base;
    uint32_t* anc_count;
};

// ---------------------------------------------------------------------------
// Algorithm layer. BF16 decode stages:
//   task map -> visible range -> query load -> page walk + online softmax
//   (+ tree ancestors) -> stage wave partials -> [barrier] -> merge -> store.
// ---------------------------------------------------------------------------

struct DecodeTaskMap {
    uint32_t row;
    uint32_t q_head;
    uint32_t kv_head;
    bool active;   // row < rows; block-uniform (task = group id)
};

template <class IT>
inline DecodeTaskMap attention_map_decode_task(IT& it, const PagedAttentionCommonArgs& a) {
    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
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
inline void attention_initialize_visible_range(
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
template <bool kFullHead, class IT>
inline void attention_load_query_tile(
    IT& it, const PagedAttentionCommonArgs& a, const DecodeTaskMap& task,
    float (&q_reg)[kElemsPerLane]) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id()) & 31u;
    const bf16* q_row = a.q
        + static_cast<size_t>(task.row) * a.q_row_stride
        + static_cast<size_t>(task.q_head) * head_dim;
#pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        q_reg[j] = (kFullHead || d < head_dim) ? ::shelf::bf16_to_f32(q_row[d]) : 0.0f;
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
    uint32_t kv_head;   // used by FP8 scale indexing
    uint32_t kv_heads;
};

// One token: masked K/V load, wave-reduced dot, online softmax update.
// Order: j-ascending partial -> wave sum -> *scale -> max/alpha/weight ->
// run_sum -> acc[j] (summation order inside the wave sum is the one
// sycl's sub_group reduce picks; see file header note).
template <bool kFullHead, class IT>
inline void attention_attend_token_bf16(
    IT& it,
    const float (&q_reg)[kElemsPerLane],
    const bf16* k_head, const bf16* v_head,
    uint32_t head_dim, float scale,
    float& run_max, float& run_sum, float (&acc)[kElemsPerLane]) {
    const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id()) & 31u;
    bf16 k_reg[kElemsPerLane];
    bf16 v_reg[kElemsPerLane];
#pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        if (kFullHead || d < head_dim) {
            k_reg[j] = k_head[d];
            v_reg[j] = v_head[d];
        } else {
            k_reg[j] = ::shelf::f32_to_bf16(0.0f);
            v_reg[j] = ::shelf::f32_to_bf16(0.0f);
        }
    }
    float partial = 0.0f;
#pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j)
        partial += q_reg[j] * ::shelf::bf16_to_f32(k_reg[j]);
    const float score = wave_sum_all(it.get_sub_group(), partial) * scale;
    const float next_max = ::sycl::fmax(run_max, score);
    const float alpha = ::sycl::exp(run_max - next_max);
    const float weight = ::sycl::exp(score - next_max);
    run_sum = run_sum * alpha + weight;
    run_max = next_max;
#pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j)
        acc[j] = acc[j] * alpha + weight * ::shelf::bf16_to_f32(v_reg[j]);
}

template <bool kFullHead, class IT>
inline void attention_visit_pages_bf16(
    IT& it,
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
                it, q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
                head_dim, scale, run_max, run_sum, acc);
        }
    }
}

template <bool kFullHead, class IT>
inline void attention_visit_ancestors_bf16(
    IT& it,
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
            it, q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
            head_dim, scale, run_max, run_sum, acc);
    }
}

inline void attention_stage_wave_partials(
    uint32_t tid, uint32_t lane, uint32_t wave, uint32_t head_dim,
    float run_max, float run_sum, uint32_t bad, const float (&acc)[kElemsPerLane],
    Locals& L) {
    if (lane == 0u) {
        L.merge_max[wave] = run_max;
        L.merge_sum[wave] = run_sum;
        if (bad != 0u) *L.invalid = 1u;
    }
    float* accs = L.acc;
#pragma unroll
    for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
        const uint32_t d = lane + 32u * j;
        if (d < head_dim) accs[wave * kHeadLimit + d] = acc[j];
    }
    (void)tid;
}

// Wave merge with the merge order from the original (global max via wave
// reduce, then w-ascending weights/sum).
inline void attention_merge_wave_partials(
    const ::sycl::sub_group& sg, uint32_t lane, Locals& L,
    float (&weight_shared)[kWaves], float& sum_total) {
    const float global_max =
        wave_max_all(sg, (lane < kWaves) ? L.merge_max[lane] : -FLT_MAX);
    sum_total = 0.0f;
#pragma unroll
    for (uint32_t w = 0u; w < kWaves; ++w) {
        weight_shared[w] = ::sycl::exp(L.merge_max[w] - global_max);
        sum_total += L.merge_sum[w] * weight_shared[w];
    }
}

inline void attention_store_output_bf16(
    const PagedAttentionCommonArgs& a, const DecodeTaskMap& task, void* out_base,
    uint32_t tid, uint32_t head_dim, Locals& L,
    const float (&weight_shared)[kWaves], float sum_total) {
    if (tid < head_dim && sum_total > 0.0f) {
        float total = 0.0f;
#pragma unroll
        for (uint32_t w = 0u; w < kWaves; ++w)
            total += L.acc[w * kHeadLimit + tid] * weight_shared[w];
        const float result = total / sum_total;
        if (a.output_dtype == PagedAttentionOutputDType::F32) {
            reinterpret_cast<float*>(out_base)
                [static_cast<size_t>(task.row) * a.output_row_stride + static_cast<size_t>(task.q_head) * head_dim + tid] = result;
        } else {
            reinterpret_cast<bf16*>(out_base)
                [static_cast<size_t>(task.row) * a.output_row_stride + static_cast<size_t>(task.q_head) * head_dim + tid] = ::shelf::f32_to_bf16(result);
        }
    }
}

template <bool kFullHead, class IT>
inline void attention_paged_bf16_body(
    IT& it, Locals& L,
    const PagedAttentionCommonArgs& a,
    const bf16* k_pool,
    const bf16* v_pool,
    void* out_base) {
    const DecodeTaskMap task = attention_map_decode_task(it, a);
    if (!task.active) return;

    const uint32_t head_dim = a.head_dim;
    const float scale = a.scale;

    const Wave32 w = Wave32::current(it);
    const uint32_t tid = w.tid;
    const uint32_t lane = w.lane;
    const uint32_t wave = w.wave;

    if (tid == 0u) {
        attention_initialize_visible_range(a, task.row, *L.row_valid, *L.visible,
                                           *L.invalid, *L.block_table,
                                           *L.anc_base, *L.anc_count);
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.row_valid == 0u) return;

    float q_reg[kElemsPerLane];
    attention_load_query_tile<kFullHead>(it, a, task, q_reg);

    DecodeWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = *L.block_table;
    st.page_tokens = a.page_tokens;
    st.visible = *L.visible;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;
    st.kv_head = task.kv_head;
    st.kv_heads = a.kv_heads;

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float acc[kElemsPerLane] = {};
    uint32_t bad = 0u;

    attention_visit_pages_bf16<kFullHead>(it, st, k_pool, v_pool, q_reg, head_dim, scale,
                                          run_max, run_sum, acc, bad);

    if (a.tree_anc_offsets != nullptr) {
        attention_visit_ancestors_bf16<kFullHead>(
            it, st, *L.anc_base, *L.anc_count, a.tree_anc_positions,
            k_pool, v_pool, q_reg, head_dim, scale, run_max, run_sum, acc, bad);
    }

    attention_stage_wave_partials(tid, lane, wave, head_dim, run_max, run_sum, bad, acc,
                                  L);
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.invalid != 0u) return;

    float weight_shared[kWaves];
    float sum_total = 0.0f;
    attention_merge_wave_partials(it.get_sub_group(), lane, L, weight_shared, sum_total);
    attention_store_output_bf16(a, task, out_base, tid, head_dim, L,
                                weight_shared, sum_total);
}

} // namespace v_decode_bf16

namespace v_decode_bf16_split {
using namespace v_decode_bf16;

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
static_assert(kSplitThreads == kPagedAttentionSplitBlock);
constexpr uint32_t kSplitMaxSplits = 16u;

// Local staging for one split block (templated on the q-heads-per-kv width).
template <uint32_t kQPerKV>
struct SplitLocals {
    float* q_lds;        // [kQPerKV * kHeadLimit]
    float* acc_lds;      // [kSplitWaves * kQPerKV * kHeadLimit]
    float* m_lds;        // [kSplitWaves * kQPerKV]
    float* s_lds;        // [kSplitWaves * kQPerKV]
    uint32_t* row_valid;
    uint32_t* visible;
    uint32_t* page_begin;
    uint32_t* page_end;
    const uint32_t** block_table;
};

// ---------------------------------------------------------------------------
// BF16 split staging: same stage names as decode; split owns a page window
// [page_begin, page_end) and writes per-split partials.
// ---------------------------------------------------------------------------

struct SplitTaskMap {
    uint32_t split;
    uint32_t row;
    uint32_t kv_head;
    bool active;   // row < rows; block-uniform
};

template <class IT>
inline SplitTaskMap attention_map_split_task(IT& it, const PagedAttentionCommonArgs& a,
                                             uint32_t splits) {
    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
    SplitTaskMap t{};
    t.split = task % splits;
    const uint32_t rest = task / splits;
    t.kv_head = rest % a.kv_heads;
    t.row = rest / a.kv_heads;
    t.active = t.row < a.rows;
    return t;
}

inline void attention_split_init_page_range(
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

template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_load_q_tile_split(
    IT& it, const PagedAttentionCommonArgs& a, const SplitTaskMap& task,
    float (&q_lds)[kQPerKV * kHeadLimit]) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
    const bf16* q_base = a.q
        + static_cast<size_t>(task.row) * a.q_row_stride
        + static_cast<size_t>(task.kv_head * kQPerKV) * head_dim;
    for (uint32_t e = tid; e < kQPerKV * kHeadLimit; e += kSplitThreads) {
        const uint32_t qh = e / kHeadLimit;
        const uint32_t d = e % kHeadLimit;
        q_lds[e] = (kFullHead || d < head_dim)
            ? ::shelf::bf16_to_f32(q_base[static_cast<size_t>(qh) * head_dim + d])
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
    uint32_t kv_head;   // used by FP8 scale indexing
    uint32_t kv_heads;
};

// Page walk over this split's window; inner q-head loop order and the float
// K/V staging (converted at load) are the frozen split semantics (contract §4).
template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_visit_pages_split_bf16(
    IT& it,
    const SplitWalkState& st, const bf16* k_pool, const bf16* v_pool,
    const float (&q_lds)[kQPerKV * kHeadLimit], uint32_t head_dim, float scale,
    float (&run_max)[kQPerKV], float (&run_sum)[kQPerKV],
    float (&acc)[kQPerKV][kElemsPerLane], uint32_t& bad) {
    const uint32_t lane = st.lane;
    const auto sg = it.get_sub_group();
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
                    kf[j] = ::shelf::bf16_to_f32(k_head[d]);
                    vf[j] = ::shelf::bf16_to_f32(v_head[d]);
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
                const float score = wave_sum_all(sg, partial) * scale;
                const float next_max = ::sycl::fmax(run_max[qh], score);
                const float alpha = ::sycl::exp(run_max[qh] - next_max);
                const float weight = ::sycl::exp(score - next_max);
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
inline void attention_stage_split_wave_partials(
    uint32_t lane, uint32_t wave, uint32_t head_dim,
    const float (&run_max)[kQPerKV], const float (&run_sum)[kQPerKV],
    const float (&acc)[kQPerKV][kElemsPerLane], SplitLocals<kQPerKV>& L) {
    if (lane == 0u) {
#pragma unroll
        for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
            L.m_lds[wave * kQPerKV + qh] = run_max[qh];
            L.s_lds[wave * kQPerKV + qh] = run_sum[qh];
        }
    }
#pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
#pragma unroll
        for (uint32_t j = 0u; j < kElemsPerLane; ++j) {
            const uint32_t d = lane + 32u * j;
            if (kFullHead || d < head_dim)
                L.acc_lds[(wave * kQPerKV + qh) * kHeadLimit + d] = acc[qh][j];
        }
    }
}

// In-block wave merge (fmaxf tree m0..m3 exactly as before) + partial write.
template <uint32_t kQPerKV, bool kFullHead>
inline void attention_merge_store_split_partials(
    uint32_t tid, SplitLocals<kQPerKV>& L,
    float* partial_base, uint32_t head_dim) {
#pragma unroll
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        float m[kSplitWaves];
#pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w)
            m[w] = L.m_lds[w * kQPerKV + qh];
        const float g = ::sycl::fmax(::sycl::fmax(m[0], m[1]), ::sycl::fmax(m[2], m[3]));
        float wgt[kSplitWaves];
        float p_total = 0.0f;
#pragma unroll
        for (uint32_t w = 0u; w < kSplitWaves; ++w) {
            wgt[w] = ::sycl::exp(m[w] - g);
            p_total += L.s_lds[w * kQPerKV + qh] * wgt[w];
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
                    total += L.acc_lds[(w * kQPerKV + qh) * kHeadLimit + d] * wgt[w];
                partial_base[2u * kQPerKV + qh * kHeadLimit + d] = total;
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_paged_split_bf16_body(
    IT& it, SplitLocals<kQPerKV>& L,
    const PagedAttentionCommonArgs& a,
    const bf16* k_pool,
    const bf16* v_pool,
    float* partials,
    uint32_t splits) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;
    const float scale = a.scale;

    const SplitTaskMap task = attention_map_split_task(it, a, splits);
    if (!task.active) return;

    const Wave32 w = Wave32::current(it);
    const uint32_t tid = w.tid;
    const uint32_t lane = w.lane;
    const uint32_t wave = w.wave;

    if (tid == 0u) {
        attention_split_init_page_range(a, task, splits, *L.row_valid, *L.visible,
                                        *L.page_begin, *L.page_end, *L.block_table);
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.row_valid == 0u) return;

    float* partial_base = partials
        + static_cast<size_t>((static_cast<size_t>(task.row) * kv_heads + task.kv_head) * splits + task.split)
          * (kQPerKV * (2u + kHeadLimit));

    attention_load_q_tile_split<kQPerKV, kFullHead>(it, a, task,
        // reinterpret the local pointer as a fixed-size array ref
        *reinterpret_cast<float (*)[kQPerKV * kHeadLimit]>(L.q_lds));
    it.barrier(::sycl::access::fence_space::local_space);

    SplitWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = *L.block_table;
    st.page_tokens = a.page_tokens;
    st.visible = *L.visible;
    st.page_begin = *L.page_begin;
    st.page_end = *L.page_end;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;
    st.kv_head = task.kv_head;
    st.kv_heads = kv_heads;

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
        it, st, k_pool, v_pool,
        *reinterpret_cast<float (*)[kQPerKV * kHeadLimit]>(L.q_lds),
        head_dim, scale, run_max, run_sum, acc, bad);

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

    attention_stage_split_wave_partials<kQPerKV, kFullHead>(
        lane, wave, head_dim, run_max, run_sum, acc, L);
    it.barrier(::sycl::access::fence_space::local_space);

    attention_merge_store_split_partials<kQPerKV, kFullHead>(
        tid, L, partial_base, head_dim);
}

// ---------------------------------------------------------------------------
// Second launch of the split scheme. Reads the partials produced above;
// three-barrier LDS publish order is contract §5.
// ---------------------------------------------------------------------------

template <uint32_t kQPerKV>
inline void attention_reduce_max_pass(
    const float* base, uint32_t splits, uint32_t tid, float (&g_lds)[kQPerKV]) {
    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        if (tid == qh) {
            float g = -FLT_MAX;
            for (uint32_t s = 0u; s < splits; ++s)
                g = ::sycl::fmax(g, base[static_cast<uint64_t>(s) * a_stride + qh]);
            g_lds[qh] = g;
        }
    }
}

template <uint32_t kQPerKV>
inline void attention_reduce_weight_sum_pass(
    const float* base, uint32_t splits, uint32_t tid, const float (&g_lds)[kQPerKV],
    float (&wgt_lds)[kQPerKV * kSplitMaxSplits], float (&sum_lds)[kQPerKV]) {
    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t qh = 0u; qh < kQPerKV; ++qh) {
        if (tid == qh) {
            const float g = g_lds[qh];
            float sum_total = 0.0f;
            for (uint32_t s = 0u; s < splits; ++s) {
                const float m_s = base[static_cast<uint64_t>(s) * a_stride + qh];
                const float w_s = ::sycl::exp(m_s - g);
                wgt_lds[qh * kSplitMaxSplits + s] = w_s;
                sum_total += w_s * base[static_cast<uint64_t>(s) * a_stride + kQPerKV + qh];
            }
            sum_lds[qh] = sum_total;
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead>
inline void attention_reduce_store_bf16(
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
                    reinterpret_cast<bf16*>(out_base)[out_off] = ::shelf::f32_to_bf16(result);
                }
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_paged_split_reduce_bf16_body(
    IT& it,
    float* g_lds, float* sum_lds, float* wgt_lds, uint32_t* row_valid,
    const PagedAttentionCommonArgs& a,
    const float* partials,
    uint32_t splits,
    void* out_base) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;

    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
    const uint32_t row = task / kv_heads;
    const uint32_t kv_head = task % kv_heads;
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());

    if (row >= a.rows) return;

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        *row_valid = (slot < a.max_sequences) ? 1u : 0u;
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*row_valid == 0u) return;

    const float* base = partials
        + static_cast<size_t>((static_cast<size_t>(row) * kv_heads + kv_head) * splits)
          * kQPerKV * (2u + kHeadLimit);

    attention_reduce_max_pass<kQPerKV>(base, splits, tid,
        *reinterpret_cast<float (*)[kQPerKV]>(g_lds));
    it.barrier(::sycl::access::fence_space::local_space);
    attention_reduce_weight_sum_pass<kQPerKV>(
        base, splits, tid, *reinterpret_cast<float (*)[kQPerKV]>(g_lds),
        *reinterpret_cast<float (*)[kQPerKV * kSplitMaxSplits]>(wgt_lds),
        *reinterpret_cast<float (*)[kQPerKV]>(sum_lds));
    it.barrier(::sycl::access::fence_space::local_space);
    attention_reduce_store_bf16<kQPerKV, kFullHead>(
        a, row, kv_head, base, *reinterpret_cast<float (*)[kQPerKV]>(sum_lds),
        *reinterpret_cast<float (*)[kQPerKV * kSplitMaxSplits]>(wgt_lds),
        out_base, tid, head_dim, splits);
}

} // namespace v_decode_bf16_split

namespace v_decode_bf16_split_qh {

using bf16 = shelf::bf16_t;
using namespace v_decode_bf16;
using namespace v_decode_bf16_split;

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
constexpr uint32_t kHeadLimit = kPagedAttentionHeadLimit;
constexpr uint32_t kSplitMaxSplits = 16u;

template <uint32_t kQPerKV, bool kFullHead, uint32_t kQhPerWg, class IT>
inline void attention_paged_split_reduce_qh_bf16_body(
    IT& it,
    float* g_lds, float* sum_lds, float* wgt_lds, uint32_t* row_valid,
    const PagedAttentionCommonArgs& a,
    const float* partials,
    uint32_t splits,
    void* out_base) {
    constexpr uint32_t kTiles = kQPerKV / kQhPerWg;
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;

    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
    const uint32_t kv_head = task % kv_heads;
    const uint32_t rest = task / kv_heads;
    const uint32_t row = rest / kTiles;
    const uint32_t qh0 = (rest % kTiles) * kQhPerWg;
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());

    if (row >= a.rows) return;

    if (tid == 0u) {
        const uint32_t slot = a.row_sequence_slots[row];
        *row_valid = (slot < a.max_sequences) ? 1u : 0u;
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*row_valid == 0u) return;

    const float* base = partials
        + static_cast<size_t>((static_cast<size_t>(row) * kv_heads + kv_head) * splits)
          * kQPerKV * (2u + kHeadLimit);

    const uint64_t a_stride = kQPerKV * (2u + kHeadLimit);
    for (uint32_t q = 0u; q < kQhPerWg; ++q) {
        if (tid == q) {
            float g = -FLT_MAX;
            for (uint32_t s = 0u; s < splits; ++s)
                g = ::sycl::fmax(g, base[static_cast<uint64_t>(s) * a_stride + qh0 + q]);
            g_lds[q] = g;
        }
    }
    it.barrier(::sycl::access::fence_space::local_space);
    for (uint32_t q = 0u; q < kQhPerWg; ++q) {
        if (tid == q) {
            const float g = g_lds[q];
            float sum_total = 0.0f;
            for (uint32_t s = 0u; s < splits; ++s) {
                const float m_s = base[static_cast<uint64_t>(s) * a_stride + qh0 + q];
                const float w_s = ::sycl::exp(m_s - g);
                wgt_lds[q * kSplitMaxSplits + s] = w_s;
                sum_total += w_s * base[static_cast<uint64_t>(s) * a_stride + kQPerKV + qh0 + q];
            }
            sum_lds[q] = sum_total;
        }
    }
    it.barrier(::sycl::access::fence_space::local_space);

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
                    reinterpret_cast<bf16*>(out_base)[out_off] = ::shelf::f32_to_bf16(result);
                }
            }
        }
    }
}

} // namespace v_decode_bf16_split_qh

namespace v_decode_fp8 {

using bf16 = shelf::bf16_t;
using fp8 = ::shelf::syclk::fp8e4m3_storage_t;
using ::shelf::syclk::e4m3_decode_f32;

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadLimit = 256u;
constexpr uint32_t kElemsPerLane = kHeadLimit / 32u;

// f32 -> truncated-to-bf16 top bits (matches the HIP __builtin_bit_cast
// trick used for the LUT staging).
inline uint16_t f2bf_p8(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>((u + 0x8000u) >> 16u);
}

inline float fp8_lut_value(const uint16_t* lut, fp8 code) {
    return bf16_bits_to_f32(lut[code]);
}

// ---------------------------------------------------------------------------
// FP8 decode reuses the BF16 decode stage skeleton (task map / visible range
// / query tile / stage / merge / store shared with v_decode_bf16 helpers);
// only the token body differs: LUT decode + per-token k_scale / v_scale.
// ---------------------------------------------------------------------------

// stride must match the launch block (kThreads decode / kSplitThreads split).
inline void attention_build_decode_lut(
    uint32_t tid, uint32_t stride, uint16_t (&decode_lut)[256]) {
    for (uint32_t i = tid; i < 256u; i += stride)
        decode_lut[i] = f2bf_p8(e4m3_decode_f32(static_cast<uint8_t>(i)));
}

// Frozen FP8 token order: masked fp8 loads -> partial += q*lut(k) ascending
// -> (wave_sum * k_scale) * scale -> max/alpha/weight ->
// v_weight = weight * v_scale -> acc = acc*alpha + v_weight*lut(v).
template <bool kFullHead, class IT>
inline void attention_attend_token_fp8(
    IT& it,
    const float (&q_reg)[v_decode_bf16::kElemsPerLane],
    const fp8* k_head, const fp8* v_head, const uint16_t* decode_lut,
    float k_scale, float v_scale, uint32_t head_dim, float scale,
    float& run_max, float& run_sum, float (&acc)[v_decode_bf16::kElemsPerLane]) {
    const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id()) & 31u;
    fp8 k_reg[v_decode_bf16::kElemsPerLane];
    fp8 v_reg[v_decode_bf16::kElemsPerLane];
#pragma unroll
    for (uint32_t j = 0u; j < v_decode_bf16::kElemsPerLane; ++j) {
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
    for (uint32_t j = 0u; j < v_decode_bf16::kElemsPerLane; ++j)
        partial += q_reg[j] * fp8_lut_value(decode_lut, k_reg[j]);
    const float score = wave_sum_all(it.get_sub_group(), partial) * k_scale * scale;
    const float next_max = ::sycl::fmax(run_max, score);
    const float alpha = ::sycl::exp(run_max - next_max);
    const float weight = ::sycl::exp(score - next_max);
    const float v_weight = weight * v_scale;
    run_sum = run_sum * alpha + weight;
    run_max = next_max;
#pragma unroll
    for (uint32_t j = 0u; j < v_decode_bf16::kElemsPerLane; ++j)
        acc[j] = acc[j] * alpha + v_weight * fp8_lut_value(decode_lut, v_reg[j]);
}

struct Fp8ScaleWalk {
    const float* k_scale_pool;
    const float* v_scale_pool;
    size_t scale_layer_base;
};

template <bool kFullHead, class IT>
inline void attention_visit_pages_fp8(
    IT& it,
    const v_decode_bf16::DecodeWalkState& st, const Fp8ScaleWalk& sc,
    const fp8* k_pool, const fp8* v_pool,
    const float (&q_reg)[v_decode_bf16::kElemsPerLane], uint32_t head_dim, float scale,
    const uint16_t* decode_lut,
    float& run_max, float& run_sum, float (&acc)[v_decode_bf16::kElemsPerLane],
    uint32_t& bad) {
    const uint32_t page_tokens = st.page_tokens;
    const uint32_t kv_heads_n = st.kv_heads;
    const uint32_t page_count = (st.visible + page_tokens - 1u) / page_tokens;
    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const uint32_t page = st.block_table[page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page;
        const size_t page_scale_base =
            sc.scale_layer_base + static_cast<size_t>(page) * page_tokens * kv_heads_n;
        const uint32_t page_start = page_index * page_tokens;
        const uint32_t tokens_here =
            (st.visible - page_start < page_tokens) ? (st.visible - page_start) : page_tokens;
        for (uint32_t off = st.wave; off < tokens_here; off += v_decode_bf16::kWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * st.elems_per_token;
            const size_t si =
                page_scale_base + static_cast<size_t>(off) * kv_heads_n +
                st.kv_head;
            const float k_scale = sc.k_scale_pool[si];
            const float v_scale = sc.v_scale_pool[si];
            attention_attend_token_fp8<kFullHead>(
                it, q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
                decode_lut, k_scale, v_scale, head_dim, scale,
                run_max, run_sum, acc);
        }
    }
}

template <bool kFullHead, class IT>
inline void attention_visit_ancestors_fp8(
    IT& it,
    const v_decode_bf16::DecodeWalkState& st, const Fp8ScaleWalk& sc,
    uint32_t anc_base, uint32_t anc_count, const uint32_t* tree_anc_positions,
    const fp8* k_pool, const fp8* v_pool,
    const float (&q_reg)[v_decode_bf16::kElemsPerLane], uint32_t head_dim, float scale,
    const uint16_t* decode_lut,
    float& run_max, float& run_sum, float (&acc)[v_decode_bf16::kElemsPerLane],
    uint32_t& bad) {
    const uint32_t page_tokens = st.page_tokens;
    const uint32_t kv_heads_n = st.kv_heads;
    for (uint32_t ai = st.wave; ai < anc_count; ai += v_decode_bf16::kWaves) {
        const uint32_t pos = tree_anc_positions[anc_base + ai];
        const uint32_t anc_page_index = pos / page_tokens;
        const uint32_t anc_off = pos % page_tokens;
        const uint32_t page = st.block_table[anc_page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t elem =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page +
            static_cast<uint64_t>(anc_off) * st.elems_per_token;
        const size_t si =
            sc.scale_layer_base +
            (static_cast<size_t>(page) * page_tokens + anc_off) * kv_heads_n +
            st.kv_head;
        const float k_scale = sc.k_scale_pool[si];
        const float v_scale = sc.v_scale_pool[si];
        attention_attend_token_fp8<kFullHead>(
            it, q_reg, k_pool + elem + st.head_off, v_pool + elem + st.head_off,
            decode_lut, k_scale, v_scale, head_dim, scale,
            run_max, run_sum, acc);
    }
}

template <bool kFullHead, class IT>
inline void attention_paged_fp8_body(
    IT& it, v_decode_bf16::Locals& L, uint16_t* decode_lut,
    const PagedAttentionCommonArgs& a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    void* out_base) {
    const uint32_t head_dim = a.head_dim;
    const float scale = a.scale;

    const v_decode_bf16::DecodeTaskMap task = v_decode_bf16::attention_map_decode_task(it, a);
    if (!task.active) return;

    const Wave32 w = Wave32::current(it);
    const uint32_t tid = w.tid;
    const uint32_t lane = w.lane;
    const uint32_t wave = w.wave;

    attention_build_decode_lut(tid, kThreads,
        *reinterpret_cast<uint16_t (*)[256]>(decode_lut));

    if (tid == 0u) {
        v_decode_bf16::attention_initialize_visible_range(
            a, task.row, *L.row_valid, *L.visible, *L.invalid,
            *L.block_table, *L.anc_base, *L.anc_count);
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.row_valid == 0u) return;

    float q_reg[v_decode_bf16::kElemsPerLane];
    v_decode_bf16::attention_load_query_tile<kFullHead>(it, a, task, q_reg);

    v_decode_bf16::DecodeWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = *L.block_table;
    st.page_tokens = a.page_tokens;
    st.visible = *L.visible;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.kv_head = task.kv_head;
    st.kv_heads = a.kv_heads;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;

    Fp8ScaleWalk sc{};
    sc.k_scale_pool = k_scale_pool;
    sc.v_scale_pool = v_scale_pool;
    sc.scale_layer_base =
        static_cast<size_t>(a.layer) * a.num_pages * a.page_tokens * a.kv_heads;

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float acc[v_decode_bf16::kElemsPerLane] = {};
    uint32_t bad = 0u;

    attention_visit_pages_fp8<kFullHead>(it, st, sc, k_pool, v_pool, q_reg, head_dim,
                                         scale, decode_lut, run_max, run_sum, acc, bad);

    if (a.tree_anc_offsets != nullptr) {
        attention_visit_ancestors_fp8<kFullHead>(
            it, st, sc, *L.anc_base, *L.anc_count, a.tree_anc_positions,
            k_pool, v_pool, q_reg, head_dim, scale, decode_lut,
            run_max, run_sum, acc, bad);
    }

    v_decode_bf16::attention_stage_wave_partials(
        tid, lane, wave, head_dim, run_max, run_sum, bad, acc, L);
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.invalid != 0u) return;

    float weight_shared[v_decode_bf16::kWaves];
    float sum_total = 0.0f;
    v_decode_bf16::attention_merge_wave_partials(it.get_sub_group(), lane, L,
                                                 weight_shared, sum_total);
    v_decode_bf16::attention_store_output_bf16(a, task, out_base, tid, head_dim, L,
                                               weight_shared, sum_total);
}

} // namespace v_decode_fp8

namespace v_decode_fp8_split {

using namespace v_decode_fp8;
using fp8 = ::shelf::syclk::fp8e4m3_storage_t;
// (no `using namespace v_decode_bf16_split`: kHeadLimit/kElemsPerLane names
// would become ambiguous; split helpers are referenced fully qualified.)

constexpr uint32_t kSplitWaves = 4u;
constexpr uint32_t kSplitThreads = kSplitWaves * 32u;
static_assert(kSplitThreads == kPagedAttentionSplitBlock);

// ---------------------------------------------------------------------------
// FP8 split reuses the BF16 split skeleton (task / page window / Q tile /
// stage / merge-store helpers). The token body keeps FP8 semantics — note
// the V accumulation uses weight * v_scale * vf (left-assoc as written) and
// score folds k_scale after the wave sum: these differ from the decode body
// and MUST stay separate (contract §4).
// ---------------------------------------------------------------------------

template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_visit_pages_split_fp8(
    IT& it,
    const v_decode_bf16_split::SplitWalkState& st,
    const fp8* k_pool, const fp8* v_pool,
    const float* k_scale_pool, const float* v_scale_pool, size_t scale_layer_base,
    const float (&q_lds)[kQPerKV * kHeadLimit], uint32_t head_dim, float scale,
    const uint16_t* decode_lut,
    float (&run_max)[kQPerKV], float (&run_sum)[kQPerKV],
    float (&acc)[kQPerKV][kElemsPerLane], uint32_t& bad) {
    const uint32_t lane = st.lane;
    const uint32_t page_tokens = st.page_tokens;
    const uint32_t kv_heads = st.kv_heads;
    const auto sg = it.get_sub_group();
    for (uint32_t page_index = st.page_begin; page_index < st.page_end; ++page_index) {
        const uint32_t page = st.block_table[page_index];
        if (page >= st.num_pages) {
            bad = 1u;
            break;
        }
        const uint64_t page_base =
            st.layer_base + static_cast<uint64_t>(page) * st.elems_per_page;
        const size_t page_scale_base =
            scale_layer_base + static_cast<size_t>(page) * page_tokens * kv_heads;
        const uint32_t page_start = page_index * page_tokens;
        const uint32_t tokens_here =
            (st.visible - page_start < page_tokens) ? (st.visible - page_start) : page_tokens;
        for (uint32_t off = st.wave; off < tokens_here; off += kSplitWaves) {
            const uint64_t elem =
                page_base + static_cast<uint64_t>(off) * st.elems_per_token;
            const size_t si =
                page_scale_base + static_cast<size_t>(off) * kv_heads + st.kv_head;
            const float k_scale = k_scale_pool[si];
            const float v_scale = v_scale_pool[si];
            const fp8* k_head = k_pool + elem + st.head_off;
            const fp8* v_head = v_pool + elem + st.head_off;
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
                const float score = wave_sum_all(sg, partial) * k_scale * scale;
                const float next_max = ::sycl::fmax(run_max[qh], score);
                const float alpha = ::sycl::exp(run_max[qh] - next_max);
                const float weight = ::sycl::exp(score - next_max);
                run_sum[qh] = run_sum[qh] * alpha + weight;
                run_max[qh] = next_max;
#pragma unroll
                for (uint32_t j = 0u; j < kElemsPerLane; ++j)
                    acc[qh][j] = acc[qh][j] * alpha + weight * v_scale * vf[j];
            }
        }
    }
}

template <uint32_t kQPerKV, bool kFullHead, class IT>
inline void attention_paged_split_fp8_body(
    IT& it, v_decode_bf16_split::SplitLocals<kQPerKV>& L, uint16_t* decode_lut,
    const PagedAttentionCommonArgs& a,
    const fp8* k_pool,
    const fp8* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    uint32_t splits) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t kv_heads = a.kv_heads;
    const float scale = a.scale;

    const v_decode_bf16_split::SplitTaskMap task =
        v_decode_bf16_split::attention_map_split_task(it, a, splits);
    if (!task.active) return;

    const Wave32 w = Wave32::current(it);
    const uint32_t tid = w.tid;
    const uint32_t lane = w.lane;
    const uint32_t wave = w.wave;

    attention_build_decode_lut(tid, kSplitThreads,
        *reinterpret_cast<uint16_t (*)[256]>(decode_lut));

    if (tid == 0u) {
        v_decode_bf16_split::attention_split_init_page_range(
            a, task, splits, *L.row_valid, *L.visible, *L.page_begin,
            *L.page_end, *L.block_table);
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.row_valid == 0u) return;

    float* partial_base = partials
        + static_cast<size_t>((static_cast<size_t>(task.row) * kv_heads + task.kv_head) * splits + task.split)
          * (kQPerKV * (2u + kHeadLimit));

    v_decode_bf16_split::attention_load_q_tile_split<kQPerKV, kFullHead>(
        it, a, task, *reinterpret_cast<float (*)[kQPerKV * kHeadLimit]>(L.q_lds));
    it.barrier(::sycl::access::fence_space::local_space);

    v_decode_bf16_split::SplitWalkState st{};
    st.head_off = static_cast<uint64_t>(task.kv_head) * head_dim;
    st.layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    st.block_table = *L.block_table;
    st.page_tokens = a.page_tokens;
    st.visible = *L.visible;
    st.page_begin = *L.page_begin;
    st.page_end = *L.page_end;
    st.elems_per_page = a.elems_per_page;
    st.elems_per_token = a.elems_per_token;
    st.num_pages = a.num_pages;
    st.wave = wave;
    st.lane = lane;
    st.kv_head = task.kv_head;
    st.kv_heads = kv_heads;

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

    attention_visit_pages_split_fp8<kQPerKV, kFullHead>(
        it, st, k_pool, v_pool, k_scale_pool, v_scale_pool, scale_layer_base,
        *reinterpret_cast<float (*)[kQPerKV * kHeadLimit]>(L.q_lds),
        head_dim, scale, decode_lut, run_max, run_sum, acc, bad);

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

    v_decode_bf16_split::attention_stage_split_wave_partials<kQPerKV, kFullHead>(
        lane, wave, head_dim, run_max, run_sum, acc, L);
    it.barrier(::sycl::access::fence_space::local_space);

    v_decode_bf16_split::attention_merge_store_split_partials<kQPerKV, kFullHead>(
        tid, L, partial_base, head_dim);
}

} // namespace v_decode_fp8_split

} // namespace attn_detail
} // namespace shelf::syclk
