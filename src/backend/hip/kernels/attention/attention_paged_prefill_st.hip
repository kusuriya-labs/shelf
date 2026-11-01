// vendored from PhaseShift (own work); namespace/type adapted only.
//
// prefill 用 paged attention の転置コア版（bf16 KV・head_dim 256・page_tokens 16）。
// スコア行列を転置して `S^T = K·Q^T` で計算する（radiance の R4D prefill と同じ構成）。
//
// なぜ転置するか（radiance `libr4d/r4d_attn_paged_window.hip:455-470` の説明と同じ理由）:
//   * wave32 の WMMA は C フラグメントのレーン L が「列 N = L%16・行 8*(L/16)..+7」を
//     持つ。S^T = K·Q^T にするとレーン L が **1 クエリ行の 8 キー**（行 L%16、キー
//     8h..8h+7）を持ち、残り 8 キーは相方レーン L^16 にある。行方向の softmax は
//     shuffle 1 回で閉じ、5 段のツリー削減も 4 波分の LDS 部分和も要らない。
//   * S^T の C フラグメントは P·V の B フラグメント（P^T）と同一レイアウトなので、
//     **P が LDS を往復しない**（従来版は p_tile/p_tile_lo を書いて読み直していた）。
//   * O^T のレーンも 1 クエリ行の 8 dims だけを持つので、rescale は per-lane スカラー。
//   * P は f16 1 成分で持つ。gfx1201 に bf16 convert は無い（ソフトウェア RTNE で
//     6 VALU）が、f32→f16 は 2 値を 1 命令で丸める。f16 は 11 bit なので、従来版が
//     bf16 の hi/lo 二成分で確保していた精度を 1 成分で満たせる。
//   * V は [dim][key] に転置して LDS に置く。P·V はキー方向の縮約なので、A フラグメント
//     は「1 dim の 8 キー」= 16 B 連続になる。global からは global_load_tr_b128 で
//     直接転置して読む（8 レーン × 8 キーのブロック転置）。
//
// 形状: ワークグループ = 8 波 = 4 行グループ × 2 dim スライス。1 波は 16 クエリ行 ×
// 128 dims を持ち、O^T は 8 タイル × 8 f32 = 64 VGPR、Q^T は 8 断片 = 32 VGPR。
// タイル = 64 行（従来は 32 行）で、K/V のステージングは 2 倍に amortize される。
//
// 受理条件は従来の prefill kernel と同じ（rows > 1、head_dim 256、page_tokens 16、
// q_heads % kv_heads == 0、block table あり）。
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cfloat>

namespace shelf::hipk {
namespace {
namespace prefill_st {

using bf16 = __hip_bfloat16;

using wbf8_t = short __attribute__((ext_vector_type(8)));
using wf16_t = __fp16 __attribute__((ext_vector_type(8)));
using v2h_t = __fp16 __attribute__((ext_vector_type(2)));
using wf8_t = float __attribute__((ext_vector_type(8)));
using v8s_t = short __attribute__((ext_vector_type(8)));

constexpr uint32_t kWaves = 8u;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kSlices = 2u;                            // dim 方向の分割数
constexpr uint32_t kRowGroups = kWaves / kSlices;           // 4
constexpr uint32_t kTileRows = kRowGroups * 16u;            // 64
constexpr uint32_t kSliceDims = kHeadDim / kSlices;         // 128
constexpr uint32_t kSkSteps = kSliceDims / 16u;             // 8
constexpr uint32_t kKStride = kHeadDim + 8u;                // 264（16 B 境界 + bank ずらし）
constexpr uint32_t kVStride = kPageTokens + 8u;             // 24（48 B 行）
constexpr uint32_t kDimGroups = kHeadDim / 8u;              // 32: 1 行の 8 要素組
constexpr uint32_t kKeyGroups = kPageTokens / 8u;           // 2: 転置ロードのキー組
constexpr uint32_t kKItems = kPageTokens * kDimGroups;      // 512: 16 B の K 読込単位
constexpr uint32_t kKPerThread = kKItems / kThreads;        // 2
constexpr uint32_t kVUnits = kKeyGroups * kDimGroups;       // 64: 転置 V の単位
constexpr uint32_t kVThreads = kThreads / 8u;               // 32: 同時に走る単位
constexpr uint32_t kVPerThread = kVUnits / kVThreads;       // 2
// f16 の P が表現できる余裕（2^14 = 16834 < 65504）。reference max からの伸びが
// これを超えたときだけアキュムレータを rescale する（radiance の LAZY softmax）。
constexpr float kGrow = 14.0f;

static_assert(kThreads == kPagedAttentionPrefillBlock);
static_assert(kWaves == kRowGroups * kSlices, "geometry");
static_assert(kKItems % kThreads == 0u && kVUnits % kVThreads == 0u, "staging");

__device__ __forceinline__ wf8_t wmma_bf16(wbf8_t a, wbf8_t b, wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ wf8_t wmma_f16(wf16_t a, wf16_t b, wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ uint4 ld128(const void* p) {
    return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ wbf8_t as_bf8(uint4 u) {
    union { uint32_t w[4]; wbf8_t f; } q;
    q.w[0] = u.x; q.w[1] = u.y; q.w[2] = u.z; q.w[3] = u.w;
    return q.f;
}

__device__ __forceinline__ wf16_t as_f16(uint4 u) {
    union { uint32_t w[4]; wf16_t f; } q;
    q.w[0] = u.x; q.w[1] = u.y; q.w[2] = u.z; q.w[3] = u.w;
    return q.f;
}

// 16 B の転置ロード。8 レーンが「8 キー行 × 同じ 8 dims」を読み、レーン j が
// dim d0+j を 8 キー分受け取る（gfx1201 の global_load_tr_b128）。
__device__ __forceinline__ uint4 ld_tr128(const void* p) {
    const v8s_t v = __builtin_amdgcn_global_load_tr_b128_v8i16(
        (__attribute__((address_space(1))) v8s_t*)const_cast<void*>(p));
    union { v8s_t v; uint4 u; } q;
    q.v = v;
    return q.u;
}

// bf16 2 個（1 dword）-> f16 2 個（1 dword）。bf16->f32 は桁移動なので、f32 -> f16 の
// 丸め 1 命令（v_cvt_pkrtz_f16_f32）で 2 要素ぶんまとめて変換できる。
__device__ __forceinline__ uint32_t bf16w_to_f16w(uint32_t w) {
    union { v2h_t h; uint32_t u; } q;
    q.h = __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, w << 16u),
                                     __builtin_bit_cast(float, w & 0xFFFF0000u));
    return q.u;
}

__device__ __forceinline__ uint32_t pk_f16(float x, float y) {
    union { v2h_t h; uint32_t u; } q;
    q.h = __builtin_amdgcn_cvt_pkrtz(x, y);
    return q.u;
}

// パック済み f16 の行和（2 要素を 1 命令で足す）。分母は P·V が使った丸め後の P になる。
__device__ __forceinline__ float row_sum_f16(uint32_t p, float acc) {
    union { uint32_t u; v2h_t h; } q;
    q.u = p;
    const v2h_t one = {__fp16(1.0f), __fp16(1.0f)};
    return __builtin_amdgcn_fdot2(q.h, one, acc, false);
}

struct alignas(16) StShared {
    unsigned short k_tile[kPageTokens * kKStride];   // [key][dim] bf16, 8448 B
    _Float16 v_t[kHeadDim * kVStride];               // [dim][key] f16, 12288 B
    float s_part[kWaves * 32u * 8u];                 // スライス間の部分和, 8192 B
    uint32_t rows_here;
    uint32_t visible_max;
    uint32_t page_count;
    uint32_t invalid;
    uint32_t slot;
    const uint32_t* block_table;
};

__device__ __forceinline__
void attention_paged_prefill_st_bf16_body(
    const PagedAttentionCommonArgs a,
    const bf16* k_pool,
    const bf16* v_pool,
    void* out_base,
    const bf16* online_k = nullptr,
    const bf16* online_v = nullptr,
    uint32_t online_pages = 0u) {
    const uint32_t q_per_kv = a.q_heads / a.kv_heads;
    const float scale_log2e = a.scale * 1.4426950408889634f;

    const uint32_t row0 = blockIdx.x * kTileRows;
    const uint32_t q_head = blockIdx.y;
    const uint32_t kv_head = q_head / q_per_kv;
    if (row0 >= a.rows) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t c = lane & 15u;   // クエリ行 / dim（WMMA の列・行インデックス）
    const uint32_t h = lane >> 4u;   // キー / dim の 8 要素半分
    // 相方の dim スライスは warp^1（radiance と同じ割り当てにする）。
    const uint32_t dp = warp & (kSlices - 1u);   // dim スライス 0/1
    const uint32_t rg = warp >> 1u;              // 行グループ 0..3

    __shared__ StShared s;

    if (tid == 0u) {
        uint32_t rows = a.rows - row0;
        if (rows > kTileRows) rows = kTileRows;
        s.rows_here = rows;
        s.invalid = 0u;
        const uint32_t slot = a.row_sequence_slots[row0];
        uint32_t valid = (slot < a.max_sequences) ? 1u : 0u;
        uint32_t vis_max = 0u;
        if (valid != 0u) {
            vis_max = a.row_positions[row0 + rows - 1u] + 1u;
            const uint64_t cap = static_cast<uint64_t>(a.block_table_stride) * a.page_tokens;
            if (static_cast<uint64_t>(vis_max) > cap) valid = 0u;
        }
        if (valid == 0u) s.invalid = 1u;
        s.visible_max = vis_max;
        s.slot = slot;
        s.page_count = (vis_max + kPageTokens - 1u) / kPageTokens;
        s.block_table = a.block_tables
            + static_cast<size_t>(valid != 0u ? slot : 0u) * a.block_table_stride;
    }
    __syncthreads();
    if (s.invalid != 0u) return;

    const uint32_t rows_here = s.rows_here;
    const uint32_t visible_max = s.visible_max;
    const uint32_t page_count = s.page_count;
    const uint32_t slot = s.slot;
    const uint32_t* block_table = s.block_table;

    const uint32_t rloc = rg * 16u + c;   // このレーンが持つクエリ行（タイル内）
    const bool live = rloc < rows_here;
    const uint32_t grow = row0 + (live ? rloc : 0u);
    const uint32_t vis = live ? (a.row_positions[grow] + 1u) : 0u;

    // Q^T 断片: クエリ行は `c`、dim は担当スライスの 128 のうち kstep ごとに 8h..8h+7。
    // クエリ行の中で連続 8 要素 = 16 B なので 1 命令で読める。
    wbf8_t qf[kSkSteps];
    {
        const bf16* qrow = reinterpret_cast<const bf16*>(a.q)
            + static_cast<size_t>(grow) * a.q_row_stride
            + static_cast<size_t>(q_head) * kHeadDim;
        #pragma unroll
        for (uint32_t t = 0u; t < kSkSteps; ++t)
            qf[t] = as_bf8(ld128(qrow + dp * kSliceDims + t * 16u + 8u * h));
    }

    wf8_t oacc[kSkSteps];
    #pragma unroll
    for (uint32_t t = 0u; t < kSkSteps; ++t) oacc[t] = wf8_t{};

    float m_ref = -3.0e38f;
    float soff = 3.0e38f;
    float l_i = 0.0f;

    const uint64_t layer_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer;
    const uint64_t head_off = static_cast<uint64_t>(kv_head) * kHeadDim;
    const uint32_t vj = tid & 7u;
    const uint32_t lgi = tid >> 3u;

    const uint32_t window =
        (a.slot_window_end != nullptr && a.slot_window_end[slot] != 0u)
            ? a.slot_window_end[slot]
            : page_count;
    const uint32_t first_online =
        (online_pages != 0u && window > online_pages) ? window - online_pages : 0u;
    for (uint32_t page_index = 0u; page_index < page_count; ++page_index) {
        const bool mirrored = online_pages != 0u && page_index >= first_online;
        const uint32_t page = mirrored ? 0u : block_table[page_index];
        if (!mirrored && page >= a.num_pages) {
            if (tid == 0u) s.invalid = 1u;
            break;
        }
        const uint32_t page_start = page_index * kPageTokens;
        uint32_t tokens_here = (visible_max > page_start) ? (visible_max - page_start) : 0u;
        if (tokens_here > kPageTokens) tokens_here = kPageTokens;
        const uint64_t page_base = mirrored
            ? (static_cast<uint64_t>(slot) * online_pages +
               (page_index % online_pages)) * a.elems_per_page
            : layer_base + static_cast<uint64_t>(page) * a.elems_per_page;
        const bf16* k_src = (mirrored ? online_k : k_pool) + page_base + head_off;
        const bf16* v_src = (mirrored ? online_v : v_pool) + page_base + head_off;

        // K: [key][dim] のまま LDS へ（A フラグメントが 16 B 連続になる）。
        #pragma unroll
        for (uint32_t p = 0u; p < kKPerThread; ++p) {
            const uint32_t i = tid + p * kThreads;
            const uint32_t key = i / kDimGroups;
            const uint32_t gg = i % kDimGroups;
            const uint4 w = (key < tokens_here)
                ? ld128(k_src + static_cast<size_t>(key) * a.elems_per_token + gg * 8u)
                : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&s.k_tile[key * kKStride + gg * 8u]) = w;
        }
        // V: [dim][key] へ転置して置く。範囲外のキーは 0（P が 0 でも NaN を拾わない）。
        #pragma unroll
        for (uint32_t r = 0u; r < kVPerThread; ++r) {
            const uint32_t u = lgi + r * kVThreads;
            const uint32_t kg = u % kKeyGroups;
            const uint32_t dg = u / kKeyGroups;
            const uint32_t kk = kg * 8u + vj;
            // 転置ロードは 8 レーンのグループ全員が同じ形で参加する必要がある。分岐
            // させると不正アドレスで fault し、結果を 0 で潰すと転置で混ざったぶん
            // 有効なキーの成分まで消える。範囲外は「見えている最後のキー」に
            // クランプする（そのキーは p = 0 で落ちるので値は何でもよいが、有限で
            // ある必要がある）。
            const uint32_t kk_safe = (kk < tokens_here) ? kk : (tokens_here - 1u);
            const uint4 w = ld_tr128(v_src + static_cast<size_t>(kk_safe) * a.elems_per_token
                                     + dg * 8u);
            *reinterpret_cast<uint4*>(&s.v_t[(dg * 8u + vj) * kVStride + kg * 8u]) =
                make_uint4(bf16w_to_f16w(w.x), bf16w_to_f16w(w.y),
                           bf16w_to_f16w(w.z), bf16w_to_f16w(w.w));
        }
        __syncthreads();

        // S^T = K·Q^T。A は LDS の [key][dim] から、B は Q^T のレジスタ断片。
        wf8_t sr;
        {
            const unsigned short* krow =
                &s.k_tile[c * kKStride + dp * kSliceDims] + 8u * h;
            wbf8_t kf[kSkSteps];
            #pragma unroll
            for (uint32_t t = 0u; t < kSkSteps; ++t)
                kf[t] = as_bf8(ld128(krow + 16u * t));
            sr = wf8_t{};
            #pragma unroll
            for (uint32_t t = 0u; t < kSkSteps; ++t) sr = wmma_bf16(kf[t], qf[t], sr);
        }

        // dim スライス間の部分和を LDS で交換する。両スライスとも同じ順序で足すので
        // スコアは bit 一致する（IEEE の加算は可換）。
        {
            float* mine = &s.s_part[(warp * 32u + lane) * 8u];
            const float* theirs = &s.s_part[((warp ^ 1u) * 32u + lane) * 8u];
            *reinterpret_cast<float4*>(mine) = make_float4(sr[0], sr[1], sr[2], sr[3]);
            *reinterpret_cast<float4*>(mine + 4) = make_float4(sr[4], sr[5], sr[6], sr[7]);
            __syncthreads();
            const float4 pa = *reinterpret_cast<const float4*>(theirs);
            const float4 pb = *reinterpret_cast<const float4*>(theirs + 4);
            sr = wf8_t{sr[0] + pa.x, sr[1] + pa.y, sr[2] + pa.z, sr[3] + pa.w,
                       sr[4] + pb.x, sr[5] + pb.y, sr[6] + pb.z, sr[7] + pb.w};
        }

        // 可視範囲のマスク + log2 スケール。scalar は key = page_start + 8h + e。
        const uint32_t key0 = page_start + 8u * h;
        float sc[8];
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e)
            sc[e] = ((key0 + e) < vis) ? (sr[e] * scale_log2e) : -INFINITY;

        // reference max 方式の online softmax（rescale は f16 の余裕を超えたときだけ）。
        float smax = sc[0];
        #pragma unroll
        for (uint32_t e = 1u; e < 8u; ++e) smax = fmaxf(smax, sc[e]);
        if (__builtin_amdgcn_ballot_w32(smax > m_ref + kGrow) != 0u) {
            const float rmax = fmaxf(smax, __shfl_xor(smax, 16));
            const float mnew = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
            const float alpha = __builtin_amdgcn_exp2f(m_ref - mnew);
            #pragma unroll
            for (uint32_t t = 0u; t < kSkSteps; ++t)
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e) oacc[t][e] *= alpha;
            l_i *= alpha;
            m_ref = mnew;
            soff = -mnew;
        }

        uint32_t pw[4];
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; e += 2u)
            pw[e >> 1] = pk_f16(__builtin_amdgcn_exp2f(sc[e] + soff),
                                __builtin_amdgcn_exp2f(sc[e + 1u] + soff));
        #pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) l_i = row_sum_f16(pw[e], l_i);
        const wf16_t p16 = as_f16(make_uint4(pw[0], pw[1], pw[2], pw[3]));

        // O^T = V^T·P^T。A は LDS の [dim][key] から 16 B、B は上の P^T 断片。
        {
            const _Float16* vrow = &s.v_t[(dp * kSliceDims + c) * kVStride] + 8u * h;
            wf16_t vf[kSkSteps];
            #pragma unroll
            for (uint32_t t = 0u; t < kSkSteps; ++t)
                vf[t] = as_f16(ld128(vrow + t * 16u * kVStride));
            #pragma unroll
            for (uint32_t t = 0u; t < kSkSteps; ++t)
                oacc[t] = wmma_f16(vf[t], p16, oacc[t]);
        }
        __syncthreads();
    }

    if (s.invalid != 0u) return;
    const float ls = l_i + __shfl_xor(l_i, 16);
    if (!live || !(ls > 0.0f)) return;
    const float inv = 1.0f / ls;

    #pragma unroll
    for (uint32_t t = 0u; t < kSkSteps; ++t) {
        const uint32_t dim = dp * kSliceDims + t * 16u + 8u * h;
        const size_t o = static_cast<size_t>(grow) * a.output_row_stride
            + static_cast<size_t>(q_head) * kHeadDim + dim;
        if (a.output_dtype == PagedAttentionOutputDType::F32) {
            float* dst = reinterpret_cast<float*>(out_base) + o;
            *reinterpret_cast<float4*>(dst) =
                make_float4(oacc[t][0] * inv, oacc[t][1] * inv,
                            oacc[t][2] * inv, oacc[t][3] * inv);
            *reinterpret_cast<float4*>(dst + 4) =
                make_float4(oacc[t][4] * inv, oacc[t][5] * inv,
                            oacc[t][6] * inv, oacc[t][7] * inv);
        } else {
            union { uint32_t w[4]; unsigned short h[8]; } u;
            #pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e)
                u.h[e] = __bfloat16_as_ushort(__float2bfloat16(oacc[t][e] * inv));
            bf16* dst = reinterpret_cast<bf16*>(out_base) + o;
            *reinterpret_cast<uint4*>(dst) = make_uint4(u.w[0], u.w[1], u.w[2], u.w[3]);
        }
    }
}

__global__ __launch_bounds__(kThreads, 1)
void attention_paged_prefill_st_bf16_impl(const PagedAttentionCommonArgs a,
                                          const bf16* k_pool,
                                          const bf16* v_pool,
                                          void* out_base,
                                          const bf16* online_k = nullptr,
                                          const bf16* online_v = nullptr,
                                          uint32_t online_pages = 0u) {
    attention_paged_prefill_st_bf16_body(a, k_pool, v_pool, out_base, online_k, online_v,
                                         online_pages);
}

}  // namespace prefill_st
}  // namespace

extern "C" __global__ __launch_bounds__(kPagedAttentionPrefillBlock)
void phaseshift_attention_paged_prefill_st_bf16(
    ::shelf::hipk::PagedAttentionBf16Args args) {
    using namespace prefill_st;
    attention_paged_prefill_st_bf16_body(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.common.output,
        reinterpret_cast<const bf16*>(args.online_k),
        reinterpret_cast<const bf16*>(args.online_v), args.online_pages);
}

bool attention_paged_prefill_st_bf16_supported(const PagedAttentionCommonArgs& common) {
    using namespace prefill_st;
    return common.rows > 1u &&
           common.head_dim == kHeadDim &&
           common.page_tokens == kPageTokens &&
           common.q_heads > 0u && common.kv_heads > 0u &&
           common.q_heads % common.kv_heads == 0u &&
           common.block_table_stride > 0u && common.num_pages > 0u &&
           common.q != nullptr && common.output != nullptr &&
           common.row_sequence_slots != nullptr && common.row_positions != nullptr &&
           common.block_tables != nullptr;
}

hipError_t launch_attention_paged_prefill_st_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    hipStream_t stream,
    const bf16_t* online_k,
    const bf16_t* online_v,
    uint32_t online_pages) {
    using namespace prefill_st;
    if (!attention_paged_prefill_st_bf16_supported(common)) return hipErrorInvalidValue;
    const uint32_t tiles = (common.rows + kTileRows - 1u) / kTileRows;
    attention_paged_prefill_st_bf16_impl<<<dim3(tiles, common.q_heads), kThreads, 0, stream>>>(
        common, reinterpret_cast<const bf16*>(k_pool),
        reinterpret_cast<const bf16*>(v_pool), common.output,
        reinterpret_cast<const bf16*>(online_k),
        reinterpret_cast<const bf16*>(online_v), online_pages);
    return hipGetLastError();
}

}  // namespace shelf::hipk
