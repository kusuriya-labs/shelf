// vendored from PhaseShift (own work); namespace/type adapted only.
//
// decode（1 行）専用の GDN 再帰。参照 `phaseshift_gdn_recurrence_wmma_decode1` の移植。
//
// 1 行の数式（chunk 版の rows=1 と同じ）:
//   g    = -exp(a_log) * softplus(a + dt_bias),  dval = exp(g)
//   beta = sigmoid(b)
//   KS[j] = Σ_i k[i]·S[i][j]      QS[j] = Σ_i q[i]·S[i][j]      qk = Σ_i q[i]·k[i]
//   delta[j] = beta·(v[j] − dval·KS[j])
//   out[j]   = dval·QS[j] + qk·delta[j]        (1 行なので自己減衰 exp(g−g)=1)
//   S'[i][j] = dval·S[i][j] + k[i]·delta[j]
//
// スカラー f32 版（`gdn_recurrence.hip`）との違いは実行形式だけ:
//   * 状態 S の列を 64 列に分割して workgroup を分ける（LDS に張れるようにする）
//   * KS/QS・S'=S·rank1 を WMMA（16x16x16 f16）で実行する
//   * 積のオペランドは f16 断片。kLossy=false では hi/lo 二成分で 4 回 WMMA（f32 精度に寄せる）、
//     kLossy=true では hi のみ 1 回（chunk 版と同じ精度帯域）。
// 状態のメモリ上の型は shelf では f32 のまま（参照は f16 保存。そこは移植していない）。
#include <shelf/backend/hip/kernels/gdn/gdn_recurrence.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {
namespace {
namespace gdnw {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr std::uint32_t kWt = 32u;    // 16 行タイル数 × 16（= 128 = head_k）
constexpr std::uint32_t kWv = 64u;    // workgroup ごとに担う head_v 列数
constexpr std::uint32_t kWk = 128u;   // head_k（前提）
constexpr std::uint32_t kWthreads = 256u;
constexpr std::uint32_t kSp = 68u;    // LDS の S ピッチ

__device__ __forceinline__ float wbf2f(const shelf::bf16_t* p) {
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(p));
}

struct Frag2 {
    f16x8_t h;
    f16x8_t l;
};

__device__ __forceinline__ f32x8_t gdn_wmma(f16x8_t a, f16x8_t b, f32x8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

template <bool kLossy = false>
__device__ __forceinline__ Frag2 gdn_decomp8(const float* x) {
    Frag2 f;
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        const _Float16 h0 = static_cast<_Float16>(x[q]);
        f.h[q] = h0;
        if constexpr (kLossy) {
            f.l[q] = static_cast<_Float16>(0.0f);
        } else {
            f.l[q] = static_cast<_Float16>(x[q] - static_cast<float>(h0));
        }
    }
    return f;
}

template <bool kLossy = false>
__device__ __forceinline__ f32x8_t gdn_wmma2(f32x8_t acc, const Frag2& a, const Frag2& b) {
    acc = gdn_wmma(a.h, b.h, acc);
    if constexpr (!kLossy) {
        acc = gdn_wmma(a.h, b.l, acc);
        acc = gdn_wmma(a.l, b.h, acc);
        acc = gdn_wmma(a.l, b.l, acc);
    }
    return acc;
}

struct Decode1Smem {
    float S[kWk][kSp];
    float delta[kWv];
    float a00;
    _Float16 kqfrag[2][kWk / 16u][32][16];
};

template <bool kLossy>
__global__ __launch_bounds__(kWthreads, 1u)
void gdn_recurrence_wmma_decode1_impl(const GdnRecurrenceArgs a) {
    const std::uint32_t hd = a.head_k;
    const std::uint32_t hv = a.head_v;
    const std::uint32_t vgroups = hv / kWv;
    const std::uint32_t vg = blockIdx.x % vgroups;
    const std::uint32_t v_head = (blockIdx.x / vgroups) % a.num_v_heads;
    const std::uint32_t request_index = blockIdx.x / (vgroups * a.num_v_heads);
    if (request_index >= a.num_requests) return;
    const GdnRequestDescriptor req = a.requests[request_index];
    const std::uint32_t slot = req.slot;
    if (slot >= a.max_sequences) return;
    const std::uint32_t row_begin = req.row_begin;
    std::uint32_t row_count = req.row_count;
    if (row_begin >= a.actual_rows) return;
    if (row_begin + row_count > a.actual_rows) row_count = a.actual_rows - row_begin;
    if (row_count == 0u) return;
    const std::uint32_t row = row_begin;      // decode1 は 1 行目だけ使う
    if (row >= a.actual_rows) return;

    const std::uint32_t repeat = a.num_v_heads / a.key_heads;
    const std::uint32_t qk_head = v_head / repeat;
    const std::uint32_t v0 = vg * kWv;

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t warp = tid >> 5u;
    const std::uint32_t lo = lane & 15u;
    const std::uint32_t hi = lane >> 4u;
    const std::uint32_t kh = lane >> 4u;

    float* state_head = a.recurrent_state
        + static_cast<std::uint64_t>(slot) * a.recurrent_slot_stride
        + static_cast<std::uint64_t>(a.state_index) * a.recurrent_layer_stride
        + static_cast<std::uint64_t>(v_head) * hd * hv;

    const float* __restrict__ qp_g = a.q + qk_head * hd;
    const float* __restrict__ kp_g = a.k + qk_head * hd;

    const float dt_bias = wbf2f(a.dt_bias + v_head);
    const float neg_exp_a_log = -__expf(wbf2f(a.a_log + v_head));

    const float bv = wbf2f(a.b + static_cast<size_t>(row) * a.b_row_stride + v_head);
    const float av = wbf2f(a.a + static_cast<size_t>(row) * a.a_row_stride + v_head);
    const float beta = 1.0f / (1.0f + __expf(-bv));
    const float gval = neg_exp_a_log * __logf(1.0f + __expf(av + dt_bias));
    const float dval = __expf(gval);
    const float de0 = __expf(gval - gval);

    __shared__ Decode1Smem s;
    float* Sst = &s.S[0][0];
    const std::uint32_t spitch = kSp;

    const std::uint32_t mt = warp & 1u;
    const std::uint32_t nt = warp >> 1u;
    const std::uint32_t kt0 = (warp >> 1u) * 2u;
    const std::uint32_t vt0 = (warp & 1u) * 2u;

    // ---- 1. 状態を読む（グローバル f32 → レジスタ断片 + LDS） ------------------
    f32x8_t Sacc[4];
    #pragma unroll
    for (std::uint32_t ki = 0u; ki < 2u; ++ki)
        #pragma unroll
        for (std::uint32_t vj = 0u; vj < 2u; ++vj) {
            const std::uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
            const std::uint32_t n0 = (vt0 + vj) * 16u + lo;
            const float* sp = state_head + static_cast<std::size_t>(m0) * hv + v0 + n0;
            #pragma unroll
            for (std::uint32_t e = 0u; e < 8u; ++e)
                Sacc[ki * 2u + vj][e] = sp[static_cast<std::size_t>(e) * hv];
        }
    #pragma unroll
    for (std::uint32_t ki = 0u; ki < 2u; ++ki)
        #pragma unroll
        for (std::uint32_t vj = 0u; vj < 2u; ++vj) {
            const std::uint32_t kt = kt0 + ki, vt = vt0 + vj;
            #pragma unroll
            for (std::uint32_t e = 0u; e < 8u; ++e)
                Sst[static_cast<std::size_t>(kt * 16u + 8u * hi + e) * spitch + vt * 16u + lo] =
                    Sacc[ki * 2u + vj][e];
        }
    // ---- 2. k・q を f16 断片にして LDS に（他の波が B として使う） --------------
    {
        const std::uint32_t kk = warp;
        const std::uint32_t ti = lo;
        const bool iok = ti < 1u;
        const float* __restrict__ kp = kp_g + static_cast<size_t>(row + ti) * a.k_row_stride;
        const float* __restrict__ qp = qp_g + static_cast<size_t>(row + ti) * a.q_row_stride;
        float ktmp[8], qtmp[8];
        #pragma unroll
        for (std::uint32_t q = 0u; q < 4u; ++q) {
            ktmp[q] = iok ? kp[kk * 16u + kh * 4u + q] : 0.0f;
            ktmp[4u + q] = iok ? kp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
            qtmp[q] = iok ? qp[kk * 16u + kh * 4u + q] : 0.0f;
            qtmp[4u + q] = iok ? qp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
        }
        const Frag2 fk = gdn_decomp8<kLossy>(ktmp);
        const Frag2 fq = gdn_decomp8<kLossy>(qtmp);
        #pragma unroll
        for (std::uint32_t d = 0u; d < 8u; ++d) {
            s.kqfrag[0][kk][lane][d] = fk.h[d];
            s.kqfrag[1][kk][lane][d] = fq.h[d];
        }
        if constexpr (!kLossy) {
            #pragma unroll
            for (std::uint32_t d = 0u; d < 8u; ++d) {
                s.kqfrag[0][kk][lane][8u + d] = fk.l[d];
                s.kqfrag[1][kk][lane][8u + d] = fq.l[d];
            }
        }
    }
    __syncthreads();

    // ---- 3. KS = Kᵀ·S、QS = Qᵀ·S、qk = q·k ---------------------------------
    f32x8_t uacc = {};
    f32x8_t oacc = {};
    f32x8_t p_acc = {};
    if (mt == 0u) {
        const bool ppk = warp == 0u;
        #pragma unroll
        for (std::uint32_t kk = 0u; kk < kWk / 16u; ++kk) {
            float stmp[8];
            #pragma unroll
            for (std::uint32_t q = 0u; q < 4u; ++q) {
                stmp[q] = Sst[static_cast<std::size_t>(kk * 16u + kh * 4u + q) * spitch
                              + nt * 16u + lo];
                stmp[4u + q] = Sst[static_cast<std::size_t>(kk * 16u + 8u + kh * 4u + q) * spitch
                                   + nt * 16u + lo];
            }
            const Frag2 bs = gdn_decomp8<kLossy>(stmp);
            Frag2 ak, aq;
            #pragma unroll
            for (std::uint32_t d = 0u; d < 8u; ++d) {
                ak.h[d] = s.kqfrag[0][kk][lane][d];
                aq.h[d] = s.kqfrag[1][kk][lane][d];
            }
            if constexpr (!kLossy) {
                #pragma unroll
                for (std::uint32_t d = 0u; d < 8u; ++d) {
                    ak.l[d] = s.kqfrag[0][kk][lane][8u + d];
                    aq.l[d] = s.kqfrag[1][kk][lane][8u + d];
                }
            }
            uacc = gdn_wmma2<kLossy>(uacc, ak, bs);
            oacc = gdn_wmma2<kLossy>(oacc, aq, bs);
            if (ppk) p_acc = gdn_wmma2<kLossy>(p_acc, aq, ak);
        }
    }
    if (warp == 0u && lane == 0u) s.a00 = __expf(gval - gval) * p_acc[0];

    // ---- 4. delta = beta·(v − dval·KS) ---------------------------------------
    if (mt == 0u && hi == 0u) {
        const std::uint32_t v = nt * 16u + lo;
        const float vv = wbf2f(a.v + static_cast<size_t>(row) * a.v_row_stride
                               + static_cast<std::size_t>(v_head) * hv + v0 + v);
        s.delta[v] = beta * (vv - dval * uacc[0]);
    }
    __syncthreads();

    // ---- 5. out = dval·QS + qk·delta ----------------------------------------
    if (mt == 0u) {
        #pragma unroll
        for (std::uint32_t e = 0u; e < 8u; ++e) oacc[e] *= dval;

        float atmp0[8] = {};
        if (lane == 0u) atmp0[0] = s.a00;
        const Frag2 aab0 = gdn_decomp8<kLossy>(atmp0);
        float atmp1[8] = {};
        const Frag2 aab1 = gdn_decomp8<kLossy>(atmp1);
        #pragma unroll
        for (std::uint32_t jt = 0u; jt < 2u; ++jt) {
            float dtmp[8] = {};
            if (jt == 0u && kh == 0u) dtmp[0] = s.delta[nt * 16u + lo];
            const Frag2 bvb = gdn_decomp8<kLossy>(dtmp);
            oacc = gdn_wmma2<kLossy>(oacc, jt == 0u ? aab0 : aab1, bvb);
        }
        #pragma unroll
        for (std::uint32_t e = 0u; e < 8u; ++e) {
            const std::uint32_t i = 8u * hi + e;
            if (i < 1u)
                a.output[static_cast<size_t>(row) * a.output_row_stride
                         + static_cast<std::size_t>(v_head) * hv + v0 + nt * 16u + lo] = oacc[e];
        }
    }

    // ---- 6. S' = dval·S + k ⊗ delta（rank-1 更新を WMMA で） ------------------
    const float egv = dval;
    #pragma unroll
    for (std::uint32_t t = 0u; t < 4u; ++t)
        #pragma unroll
        for (std::uint32_t e = 0u; e < 8u; ++e) Sacc[t][e] *= egv;
    #pragma unroll
    for (std::uint32_t tstep = 0u; tstep < kWt / 16u; ++tstep) {
        Frag2 bv[2];
        #pragma unroll
        for (std::uint32_t vj = 0u; vj < 2u; ++vj) {
            float dtmp[8] = {};
            if (tstep == 0u && kh == 0u) dtmp[0] = s.delta[(vt0 + vj) * 16u + lo];
            bv[vj] = gdn_decomp8<kLossy>(dtmp);
        }
        #pragma unroll
        for (std::uint32_t ki = 0u; ki < 2u; ++ki) {
            float ktmp[8] = {};
            if (tstep == 0u && kh == 0u)
                ktmp[0] = a.k[static_cast<size_t>(row) * a.k_row_stride + qk_head * hd
                              + (kt0 + ki) * 16u + lo] * de0;
            const Frag2 akT = gdn_decomp8<kLossy>(ktmp);
            #pragma unroll
            for (std::uint32_t vj = 0u; vj < 2u; ++vj)
                Sacc[ki * 2u + vj] = gdn_wmma2<kLossy>(Sacc[ki * 2u + vj], akT, bv[vj]);
        }
    }
    #pragma unroll
    for (std::uint32_t ki = 0u; ki < 2u; ++ki)
        #pragma unroll
        for (std::uint32_t vj = 0u; vj < 2u; ++vj) {
            const std::uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
            const std::uint32_t n0 = (vt0 + vj) * 16u + lo;
            float* sp = state_head + static_cast<std::size_t>(m0) * hv + v0 + n0;
            #pragma unroll
            for (std::uint32_t e = 0u; e < 8u; ++e)
                sp[static_cast<std::size_t>(e) * hv] = Sacc[ki * 2u + vj][e];
        }
}

}  // namespace gdnw
}  // namespace

bool gdn_recurrence_wmma_decode1_supported(const GdnRecurrenceArgs& args) {
    using namespace gdnw;
    if (args.num_requests != 1u || args.requests == nullptr) return false;
    if (args.head_k != kWk) return false;                 // 断片構成が 128 固定
    if (args.head_v == 0u || args.head_v % kWv != 0u) return false;
    if (args.num_v_heads == 0u || args.key_heads == 0u) return false;
    if (args.num_v_heads % args.key_heads != 0u) return false;
    if (args.requests[0].row_count != 1u) return false;   // decode 専用
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr) return false;
    if (args.a == nullptr || args.b == nullptr) return false;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return false;
    if (args.recurrent_state == nullptr || args.output == nullptr) return false;
    if (args.output_row_stride != args.head_v * args.num_v_heads) return false;
    return true;
}

hipError_t launch_gdn_recurrence_wmma_decode1(const GdnRecurrenceArgs& args, bool lossy,
                                              hipStream_t stream) {
    using namespace gdnw;
    if (!gdn_recurrence_wmma_decode1_supported(args)) return hipErrorNotSupported;
    const GdnRecurrenceArgs a = args;
    const std::uint32_t vgroups = args.head_v / kWv;
    const dim3 grid(vgroups * args.num_v_heads * args.num_requests);
    if (lossy)
        gdn_recurrence_wmma_decode1_impl<true><<<grid, kWthreads, 0, stream>>>(a);
    else
        gdn_recurrence_wmma_decode1_impl<false><<<grid, kWthreads, 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace shelf::hipk
