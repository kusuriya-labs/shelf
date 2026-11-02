#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstdlib>

namespace shelf::kernel {

namespace {

__device__ __forceinline__ float bf16f(const bf16_t* p) {
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(p));
}

__device__ __forceinline__ float ld_f16(const fp16_t* p) {
    return static_cast<float>(p->value);
}

__device__ __forceinline__ void st_f16(fp16_t* p, float v) {
    p->value = static_cast<_Float16>(v);
}

__device__ __forceinline__ float f16_round(float v) {
    return static_cast<float>(static_cast<_Float16>(v));
}

__device__ __forceinline__ float gdn_softplus_exact(float x) {
    return __logf(1.0f + __expf(x));
}

__global__ void gdn_recurrence_f32_impl(const GdnRecurrenceArgs a) {
    const uint32_t head_k = a.head_k;
    const uint32_t head_v = a.head_v;
    const uint32_t v_tiles = (head_v + 31u) >> 5;

    const uint32_t task = blockIdx.x;
    const uint32_t tile = task % v_tiles;
    const uint32_t tmp = task / v_tiles;
    const uint32_t v_head = tmp % a.num_v_heads;
    const uint32_t request_index = tmp / a.num_v_heads;

    const uint32_t lane = threadIdx.x;
    const uint32_t vcol = tile * 32u + lane;
    const bool vcol_valid = vcol < head_v;

    if (request_index >= a.num_requests) return;
    const ::shelf::runtime::DeviceRequestDescriptor req = a.requests[request_index];
    const uint32_t slot = req.request_handle.slot;
    if (slot >= a.max_sequences) return;
    const uint32_t row_begin = req.row_begin;
    const uint32_t row_count = req.row_count;

    const uint32_t repeat = a.num_v_heads / a.key_heads;
    const uint32_t qk_head = v_head / repeat;

    fp16_t* state_head = a.recurrent_state
        + static_cast<uint64_t>(slot) * a.recurrent_slot_stride
        + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
        + static_cast<uint64_t>(v_head) * head_k * head_v;
    fp16_t* state_col = state_head + vcol;

    const float dt_bias = bf16f(a.dt_bias + v_head);
    const float neg_exp_a_log = -__expf(bf16f(a.a_log + v_head));

    const float* k_head = a.k + static_cast<size_t>(qk_head) * head_k;
    const float* q_head = a.q + static_cast<size_t>(qk_head) * head_k;

    for (uint32_t t = 0u; t < row_count; ++t) {
        const uint32_t row = row_begin + t;
        if (row >= a.actual_rows) continue;

        const float b_val = bf16f(a.b + static_cast<size_t>(row) * a.b_row_stride + v_head);
        const float a_val = bf16f(a.a + static_cast<size_t>(row) * a.a_row_stride + v_head);
        const float beta = 1.0f / (1.0f + __expf(-b_val));
        const float dt = a_val + dt_bias;
        const float g = neg_exp_a_log * gdn_softplus_exact(dt);
        const float d_t = __expf(g);

        const float* k_row = k_head + static_cast<size_t>(row) * a.k_row_stride;
        const float* q_row = q_head + static_cast<size_t>(row) * a.q_row_stride;

        const float v_val = vcol_valid
            ? bf16f(a.v + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(v_head) * head_v + vcol)
            : 0.0f;

        if (!vcol_valid) continue;

        float kv_mem = 0.0f;
        for (uint32_t kk = 0u; kk < head_k; ++kk) {
            const float s = ld_f16(state_col + static_cast<uint64_t>(kk) * head_v);
            kv_mem += s * d_t * k_row[kk];
        }

        const float delta = (v_val - kv_mem) * beta;

        for (uint32_t kk = 0u; kk < head_k; ++kk) {
            fp16_t* sc = state_col + static_cast<uint64_t>(kk) * head_v;
            st_f16(sc, ld_f16(sc) * d_t + k_row[kk] * delta);
        }

        float acc = 0.0f;
        for (uint32_t kk = 0u; kk < head_k; ++kk) {
            acc += ld_f16(state_col + static_cast<uint64_t>(kk) * head_v) * q_row[kk];
        }

        a.output[static_cast<size_t>(row) * a.output_row_stride + static_cast<size_t>(v_head) * head_v + vcol] = acc;
    }
}

}  // namespace

hipError_t launch_gdn_recurrence_f32(
    const GdnRecurrenceArgs& args, hipStream_t stream) {
    if (args.head_k == 0u || args.head_k > 128u) return hipErrorInvalidValue;
    if (args.head_v == 0u) return hipErrorInvalidValue;
    if (args.key_heads == 0u || args.num_v_heads == 0u) return hipErrorInvalidValue;
    if (args.num_v_heads % args.key_heads != 0u) return hipErrorInvalidValue;
    if (args.num_requests == 0u || args.actual_rows == 0u) return hipErrorInvalidValue;
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.a == nullptr || args.b == nullptr)
        return hipErrorInvalidValue;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return hipErrorInvalidValue;
    if (args.recurrent_state == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.requests == nullptr || args.max_sequences == 0u) return hipErrorInvalidValue;
    const uint32_t v_tiles = (args.head_v + 31u) >> 5;
    const uint64_t grid_x = static_cast<uint64_t>(args.num_requests) * args.num_v_heads * v_tiles;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    gdn_recurrence_f32_impl<<<dim3(static_cast<uint32_t>(grid_x)), 32, 0, stream>>>(a);
    return hipGetLastError();
}

namespace gdngfx12 {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWt = 32u;
constexpr uint32_t kWv = 64u;
constexpr uint32_t kWk = 128u;
constexpr uint32_t kWthreads = 256u;
static_assert(kWthreads == kGdnRecurrenceWmmaThreads);
static_assert(kWv == kGdnRecurrenceWmmaVGroup);
constexpr uint32_t kSp = 68u;
constexpr uint32_t kWp = 68u;
constexpr uint32_t kAp = 34u;
constexpr uint32_t kDp = 34u;

__device__ __forceinline__ float wbf2f(const bf16_t* p) {
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

__device__ __forceinline__ Frag2 gdn_decomp8_zero() {
    Frag2 f;
    f.h = {};
    f.l = {};
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

struct ChunkSmem {
    float A[kWt][kAp];
    float L[kWt][kAp];
    float W[kWt][kWp];
    float dT[kWv][kDp];
};

struct LdsSmem {
    union {
        ChunkSmem mid;
        float S[kWk][kSp];
    };
    float G[kWt];
    float de[kWt];
    float bt[kWt];
    float oe[kWt];
    float eg;
};

}  // namespace gdngfx12

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
template <bool kLossy, bool kSerialRows = false>
__device__ __forceinline__ void gdn_recurrence_wmma_body(
    const GdnRecurrenceArgs a) {
    namespace g = gdngfx12;
    using g::f32x8_t;
    using g::Frag2;

    const uint32_t hd = a.head_k;
    const uint32_t hv = a.head_v;
    const uint32_t vgroups = hv / g::kWv;
    const uint32_t vg = blockIdx.x % vgroups;
    const uint32_t v_head = (blockIdx.x / vgroups) % a.num_v_heads;
    const uint32_t request_index = blockIdx.x / (vgroups * a.num_v_heads);
    if (request_index >= a.num_requests) return;
    const ::shelf::runtime::DeviceRequestDescriptor req = a.requests[request_index];
    const uint32_t slot = req.request_handle.slot;
    if (slot >= a.max_sequences) return;
    const uint32_t row_begin = req.row_begin;
    uint32_t row_count = req.row_count;
    if (row_begin >= a.actual_rows) return;
    if (row_begin + row_count > a.actual_rows) row_count = a.actual_rows - row_begin;
    if (row_count == 0u) return;
    const uint32_t repeat = a.num_v_heads / a.key_heads;
    const uint32_t qk_head = v_head / repeat;
    const uint32_t v0 = vg * g::kWv;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lo = lane & 15u;
    const uint32_t hi = lane >> 4u;
    const uint32_t kh = lane >> 4u;

    fp16_t* state_head = a.recurrent_state
        + static_cast<uint64_t>(slot) * a.recurrent_slot_stride
        + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
        + static_cast<uint64_t>(v_head) * hd * hv;

    const float* __restrict__ qp_g = a.q + qk_head * hd;
    const float* __restrict__ kp_g = a.k + qk_head * hd;

    const float dt_bias = g::wbf2f(a.dt_bias + v_head);
    const float neg_exp_a_log = -__expf(g::wbf2f(a.a_log + v_head));

    __shared__ g::LdsSmem s;

    float* Sst = &s.S[0][0];
    const uint32_t spitch = g::kSp;

    const uint32_t mt = warp & 1u;
    const uint32_t nt = warp >> 1u;
    const uint32_t kt0 = (warp >> 1u) * 2u;
    const uint32_t vt0 = (warp & 1u) * 2u;

    f32x8_t Sacc[4];
    #pragma unroll
    for (uint32_t ki = 0; ki < 2u; ++ki)
        #pragma unroll
        for (uint32_t vj = 0; vj < 2u; ++vj) {
            const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
            const uint32_t n0 = (vt0 + vj) * 16u + lo;
            const fp16_t* sp = state_head + static_cast<size_t>(m0) * hv + v0 + n0;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) Sacc[ki * 2u + vj][e] = ld_f16(sp + static_cast<size_t>(e) * hv);
        }
    #pragma unroll
    for (uint32_t ki = 0; ki < 2u; ++ki)
        #pragma unroll
        for (uint32_t vj = 0; vj < 2u; ++vj) {
            const uint32_t kt = kt0 + ki, vt = vt0 + vj;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e)
                Sst[static_cast<size_t>(kt * 16u + 8u * hi + e) * spitch + vt * 16u + lo] =
                    Sacc[ki * 2u + vj][e];
        }
    detail::lds_barrier();

    const uint32_t nchunks = kSerialRows
        ? row_count
        : (row_count + g::kWt - 1u) / g::kWt;
    for (uint32_t ci = 0u; ci < nchunks; ++ci) {
        const uint32_t c0 = kSerialRows ? (row_begin + ci) : (row_begin + ci * g::kWt);
        uint32_t nval = kSerialRows ? 1u : (row_count - ci * g::kWt);
        if (nval > g::kWt) nval = g::kWt;

        if (tid < g::kWt) {
            const bool ok = tid < nval;
            const float av = ok ? g::wbf2f(a.a + static_cast<size_t>(c0 + tid) * a.a_row_stride + v_head) : 0.0f;
            const float bv = ok ? g::wbf2f(a.b + static_cast<size_t>(c0 + tid) * a.b_row_stride + v_head) : 0.0f;
            s.bt[tid] = ok ? 1.0f / (1.0f + __expf(-bv)) : 0.0f;
            s.oe[tid] = ok ? neg_exp_a_log * __logf(1.0f + __expf(av + dt_bias)) : 0.0f;
        }
        detail::lds_barrier();
        if (tid < 32u) {
            float val = s.oe[tid];
            #pragma unroll
            for (uint32_t d = 1u; d < 32u; d <<= 1u) {
                const float up = __shfl_up(val, d);
                if (tid >= d) val += up;
            }
            s.G[tid] = val;
        }
        detail::lds_barrier();
        if (tid < g::kWt) {
            const float gl = s.G[g::kWt - 1u];
            s.oe[tid] = __expf(s.G[tid]);
            s.de[tid] = __expf(gl - s.G[tid]);
        }
        if (tid == 0u) s.eg = __expf(s.G[g::kWt - 1u]);
        detail::lds_barrier();

        f32x8_t p_acc = {};
        f32x8_t pk_acc = {};
        f32x8_t uacc = {};
        f32x8_t oacc = {};
        {
            const uint32_t ti = mt * 16u + lo;
            const uint32_t tj = nt * 16u + lo;
            const bool iok = ti < nval;
            const bool jok = tj < nval;
            const bool ppk = warp < 4u;
            const float* __restrict__ kp = kp_g + static_cast<size_t>(c0 + ti) * a.k_row_stride;
            const float* __restrict__ qp = qp_g + static_cast<size_t>(c0 + ti) * a.q_row_stride;
            const float* __restrict__ kpj = kp_g + static_cast<size_t>(c0 + tj) * a.k_row_stride;
            #pragma unroll
            for (uint32_t kk = 0; kk < g::kWk / 16u; ++kk) {
                float stmp[8];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    stmp[q] = Sst[static_cast<size_t>(kk * 16u + kh * 4u + q) * spitch + nt * 16u + lo];
                    stmp[4u + q] = Sst[static_cast<size_t>(kk * 16u + 8u + kh * 4u + q) * spitch + nt * 16u + lo];
                }
                const Frag2 bs = g::gdn_decomp8<kLossy>(stmp);
                float ktmp[8], qtmp[8];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    ktmp[q] = iok ? kp[kk * 16u + kh * 4u + q] : 0.0f;
                    ktmp[4u + q] = iok ? kp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
                    qtmp[q] = iok ? qp[kk * 16u + kh * 4u + q] : 0.0f;
                    qtmp[4u + q] = iok ? qp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
                }
                const Frag2 ak = g::gdn_decomp8<kLossy>(ktmp);
                const Frag2 aq = g::gdn_decomp8<kLossy>(qtmp);
                uacc = g::gdn_wmma2<kLossy>(uacc, ak, bs);
                oacc = g::gdn_wmma2<kLossy>(oacc, aq, bs);
                if (ppk) {
                    float kjtmp[8];
                    #pragma unroll
                    for (uint32_t q = 0; q < 4u; ++q) {
                        kjtmp[q] = jok ? kpj[kk * 16u + kh * 4u + q] : 0.0f;
                        kjtmp[4u + q] = jok ? kpj[kk * 16u + 8u + kh * 4u + q] : 0.0f;
                    }
                    const Frag2 akj = g::gdn_decomp8<kLossy>(kjtmp);
                    p_acc = g::gdn_wmma2<kLossy>(p_acc, aq, akj);
                    pk_acc = g::gdn_wmma2<kLossy>(pk_acc, ak, akj);
                }
            }
        }
        detail::lds_barrier();
        if (warp < 4u) {
            const uint32_t i0 = mt * 16u + 8u * hi;
            const uint32_t j0 = nt * 16u + lo;
            const float gj = s.G[j0];
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) {
                const uint32_t i = i0 + e;
                const float d = __expf(s.G[i] - gj);
                s.mid.A[i][j0] = (j0 <= i) ? (d * p_acc[e]) : 0.0f;
                s.mid.L[i][j0] = (j0 < i) ? (s.bt[i] * d * pk_acc[e]) : 0.0f;
            }
        }
        #pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t i = mt * 16u + 8u * hi + e;
            const uint32_t v = nt * 16u + lo;
            const float vv = (i < nval)
                ? g::wbf2f(a.v + static_cast<size_t>(c0 + i) * a.v_row_stride + static_cast<size_t>(v_head) * hv + v0 + v)
                : 0.0f;
            s.mid.W[i][v] = s.bt[i] * (vv - s.oe[i] * uacc[e]);
        }
        #pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) oacc[e] *= s.oe[mt * 16u + 8u * hi + e];
        detail::lds_barrier();

        if (tid < g::kWv) {
            float dlt[g::kWt];
            #pragma unroll
            for (uint32_t i = 0; i < g::kWt; ++i) {
                float acc = s.mid.W[i][tid];
                #pragma unroll
                for (uint32_t l = 0; l < g::kWt; ++l)
                    if (l < i) acc -= s.mid.L[i][l] * dlt[l];
                dlt[i] = acc;
                s.mid.dT[tid][i] = acc;
            }
        }
        detail::lds_barrier();

        #pragma unroll
        for (uint32_t jt = 0; jt < g::kWt / 16u; ++jt) {
            float atmp[8];
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                atmp[q] = s.mid.A[mt * 16u + lo][jt * 16u + kh * 4u + q];
                atmp[4u + q] = s.mid.A[mt * 16u + lo][jt * 16u + 8u + kh * 4u + q];
            }
            float dtmp[8];
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                dtmp[q] = s.mid.dT[nt * 16u + lo][jt * 16u + kh * 4u + q];
                dtmp[4u + q] = s.mid.dT[nt * 16u + lo][jt * 16u + 8u + kh * 4u + q];
            }
            const Frag2 aab = g::gdn_decomp8<kLossy>(atmp);
            const Frag2 bvb = g::gdn_decomp8<kLossy>(dtmp);
            oacc = g::gdn_wmma2<kLossy>(oacc, aab, bvb);
        }
        #pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t i = mt * 16u + 8u * hi + e;
            if (i < nval)
                a.output[static_cast<size_t>(c0 + i) * a.output_row_stride
                         + static_cast<size_t>(v_head) * hv + v0 + nt * 16u + lo] = oacc[e];
        }

        const float egv = s.eg;
        #pragma unroll
        for (uint32_t t = 0; t < 4u; ++t)
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] *= egv;
        #pragma unroll
        for (uint32_t tstep = 0; tstep < g::kWt / 16u; ++tstep) {
            Frag2 bv[2];
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                float dtmp[8];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    dtmp[q] = s.mid.dT[(vt0 + vj) * 16u + lo][tstep * 16u + kh * 4u + q];
                    dtmp[4u + q] = s.mid.dT[(vt0 + vj) * 16u + lo][tstep * 16u + 8u + kh * 4u + q];
                }
                bv[vj] = g::gdn_decomp8<kLossy>(dtmp);
            }
            #pragma unroll
            for (uint32_t ki = 0; ki < 2u; ++ki) {
                float ktmp[8];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    const uint32_t ta = tstep * 16u + kh * 4u + q;
                    const uint32_t tb = tstep * 16u + 8u + kh * 4u + q;
                    ktmp[q] = (ta < nval)
                        ? a.k[static_cast<size_t>(c0 + ta) * a.k_row_stride + qk_head * hd
                              + (kt0 + ki) * 16u + lo] * s.de[ta]
                        : 0.0f;
                    ktmp[4u + q] = (tb < nval)
                        ? a.k[static_cast<size_t>(c0 + tb) * a.k_row_stride + qk_head * hd
                              + (kt0 + ki) * 16u + lo] * s.de[tb]
                        : 0.0f;
                }
                const Frag2 akT = g::gdn_decomp8<kLossy>(ktmp);
                #pragma unroll
                for (uint32_t vj = 0; vj < 2u; ++vj)
                    Sacc[ki * 2u + vj] = g::gdn_wmma2<kLossy>(Sacc[ki * 2u + vj], akT, bv[vj]);
            }
        }
        if constexpr (kSerialRows) {
            if (a.recurrent_history != nullptr && ci < a.capture_rows &&
                (a.capture_verify_only == 0u ||
                 req.execution_class == ::shelf::runtime::ExecutionClass::SPEC_VERIFY)) {
                fp16_t* hist_head = a.recurrent_history
                    + static_cast<uint64_t>(ci) * a.recurrent_history_row_stride
                    + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
                    + static_cast<uint64_t>(v_head) * hd * hv;
                #pragma unroll
                for (uint32_t ki = 0; ki < 2u; ++ki)
                    #pragma unroll
                    for (uint32_t vj = 0; vj < 2u; ++vj) {
                        const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                        const uint32_t n0 = (vt0 + vj) * 16u + lo;
                        fp16_t* hp = hist_head + static_cast<size_t>(m0) * hv + v0 + n0;
                        #pragma unroll
                        for (uint32_t e = 0; e < 8u; ++e)
                            st_f16(hp + static_cast<size_t>(e) * hv, Sacc[ki * 2u + vj][e]);
                    }
            }
            if (a.compact_delta != nullptr) {
                const uint64_t dbase =
                    static_cast<uint64_t>(a.state_index) * a.compact_delta_layer_stride;
                const uint64_t vdim = static_cast<uint64_t>(a.num_v_heads) * hv;
                if (tid < g::kWv) {
                    a.compact_delta[dbase + static_cast<size_t>(c0) * vdim +
                                    static_cast<size_t>(v_head) * hv + v0 + tid] =
                        s.mid.dT[tid][0];
                }
                if (tid == 0u && a.compact_a != nullptr) {
                    a.compact_a[static_cast<uint64_t>(a.state_index) * a.compact_a_layer_stride +
                                static_cast<size_t>(c0) * a.num_v_heads + v_head] = s.eg;
                }
            }
            if (a.compact_k != nullptr) {
                const uint64_t ckbase =
                    static_cast<uint64_t>(a.state_index) * a.compact_k_layer_stride;
                const uint64_t cqkdim = static_cast<uint64_t>(a.num_v_heads) * hd;
                const float de0 = s.de[0u];
                const float* ksrc = a.k + static_cast<size_t>(c0) * a.k_row_stride +
                                    static_cast<size_t>(qk_head) * hd;
                float* kdst = a.compact_k + ckbase + static_cast<size_t>(c0) * cqkdim +
                              static_cast<size_t>(v_head) * hd;
                for (uint32_t col = tid; col < hd; col += g::kWthreads) {
                    kdst[col] = ksrc[col] * de0;
                }
            }
            #pragma unroll
            for (uint32_t t = 0; t < 4u; ++t)
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] = f16_round(Sacc[t][e]);
        }
        detail::lds_barrier();
        #pragma unroll
        for (uint32_t ki = 0; ki < 2u; ++ki)
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                const uint32_t kt = kt0 + ki, vt = vt0 + vj;
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    Sst[static_cast<size_t>(kt * 16u + 8u * hi + e) * spitch + vt * 16u + lo] =
                        Sacc[ki * 2u + vj][e];
            }
        detail::lds_barrier();
    }

    const uint32_t slot_store = a.requests[request_index].request_handle.slot;
    fp16_t* const store_head = a.recurrent_state
        + static_cast<uint64_t>(slot_store) * a.recurrent_slot_stride
        + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
        + static_cast<uint64_t>(v_head) * hd * hv;
    #pragma unroll
    for (uint32_t ki = 0; ki < 2u; ++ki)
        #pragma unroll
        for (uint32_t vj = 0; vj < 2u; ++vj) {
            const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
            const uint32_t n0 = (vt0 + vj) * 16u + lo;
            fp16_t* sp = store_head + static_cast<size_t>(m0) * hv + v0 + n0;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) st_f16(sp + static_cast<size_t>(e) * hv, Sacc[ki * 2u + vj][e]);
        }
}
#pragma clang diagnostic pop

template <bool kLossy, bool kSerialRows>
__global__ __launch_bounds__(gdngfx12::kWthreads, 1u)
__attribute__((amdgpu_num_vgpr(256))) void gdn_recurrence_wmma_impl(
    const GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_body<kLossy, kSerialRows>(a);
}

extern "C" __global__ __launch_bounds__(256)
__attribute__((amdgpu_num_vgpr(256)))
void shelf_gdn_recurrence_wmma_lossy(GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_body<true, false>(a);
}

extern "C" __global__ __launch_bounds__(256)
__attribute__((amdgpu_num_vgpr(256)))
void shelf_gdn_recurrence_wmma_exact(GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_body<false, false>(a);
}

extern "C" __global__ __launch_bounds__(256)
__attribute__((amdgpu_num_vgpr(256)))
void shelf_gdn_recurrence_wmma_serial_lossy(GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_body<true, true>(a);
}

extern "C" __global__ __launch_bounds__(256)
__attribute__((amdgpu_num_vgpr(256)))
void shelf_gdn_recurrence_wmma_serial_exact(GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_body<false, true>(a);
}

struct RowSmem {
    float S[gdngfx12::kWk][gdngfx12::kSp];
    float delta[gdngfx12::kWv];
    float a00;
    _Float16 kqfrag[2][gdngfx12::kWk / 16u][32][16];
};

extern "C" __global__ __launch_bounds__(gdngfx12::kWthreads, 1u)
void shelf_gdn_recurrence_wmma_commit(GdnRecurrenceArgs a) {
    namespace g = gdngfx12;
    using g::f32x8_t;
    using g::Frag2;
    constexpr bool kLossy = true;

    const uint32_t hd = a.head_k;
    const uint32_t hv = a.head_v;
    const uint32_t vgroups = hv / g::kWv;
    const uint32_t vg = blockIdx.x % vgroups;
    const uint32_t v_head = (blockIdx.x / vgroups) % a.num_v_heads;
    const uint32_t request_index = blockIdx.x / (vgroups * a.num_v_heads);
    if (request_index >= a.num_requests) return;
    const ::shelf::runtime::DeviceRequestDescriptor req = a.requests[request_index];
    const uint32_t slot = req.request_handle.slot;
    if (slot >= a.max_sequences) return;
    if (a.compact_delta == nullptr || a.compact_k == nullptr || a.compact_a == nullptr)
        return;
    if (a.compact_rows == 0u) return;
    __shared__ uint32_t bad_index;
    if (a.compact_row_indices != nullptr) {
        if (threadIdx.x == 0u) {
            bad_index = 0u;
            for (uint32_t i = 0u; i < a.compact_rows; ++i) {
                const int32_t idx = a.compact_row_indices[i];
                if (idx < 0 || static_cast<uint32_t>(idx) >= a.compact_row_capacity) {
                    bad_index = 1u;
                    break;
                }
            }
        }
        detail::lds_barrier();
        if (bad_index != 0u) return;
    }

    const uint32_t v0 = vg * g::kWv;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lo = lane & 15u;
    const uint32_t hi = lane >> 4u;
    const uint32_t kh = lane >> 4u;

    const uint64_t vdim = static_cast<uint64_t>(a.num_v_heads) * hv;
    const uint64_t qkdim = static_cast<uint64_t>(a.num_v_heads) * hd;

    const uint32_t kt0 = (warp >> 1u) * 2u;
    const uint32_t vt0 = (warp & 1u) * 2u;
    const uint32_t row_begin = req.row_begin;

    for (uint32_t li = 0u; li < a.num_gdn_states; ++li) {
        fp16_t* state_head = a.recurrent_state
            + static_cast<uint64_t>(slot) * a.recurrent_slot_stride
            + static_cast<uint64_t>(li) * a.recurrent_layer_stride
            + static_cast<uint64_t>(v_head) * hd * hv;
        const uint64_t dbase = static_cast<uint64_t>(li) * a.compact_delta_layer_stride;
        const uint64_t kbase = static_cast<uint64_t>(li) * a.compact_k_layer_stride;
        const uint64_t abase = static_cast<uint64_t>(li) * a.compact_a_layer_stride;

        f32x8_t Sacc[4];
        #pragma unroll
        for (uint32_t ki = 0; ki < 2u; ++ki)
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                const uint32_t n0 = (vt0 + vj) * 16u + lo;
                const fp16_t* sp = state_head + static_cast<size_t>(m0) * hv + v0 + n0;
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    Sacc[ki * 2u + vj][e] = ld_f16(sp + static_cast<size_t>(e) * hv);
            }

        for (uint32_t i = 0u; i < a.compact_rows; ++i) {
            const uint32_t row = a.compact_row_indices != nullptr
                                     ? static_cast<uint32_t>(a.compact_row_indices[i])
                                     : (row_begin + i);
            const float egv =
                a.compact_a[abase + static_cast<size_t>(row) * a.num_v_heads +
                            static_cast<size_t>(v_head)];
            #pragma unroll
            for (uint32_t t = 0; t < 4u; ++t)
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] *= egv;
            #pragma unroll
            for (uint32_t tstep = 0; tstep < g::kWt / 16u; ++tstep) {
                Frag2 bv[2];
                #pragma unroll
                for (uint32_t vj = 0; vj < 2u; ++vj) {
                    float dtmp[8] = {};
                    if (tstep == 0u && kh == 0u)
                        dtmp[0] = a.compact_delta[
                            dbase + static_cast<size_t>(row) * vdim +
                            static_cast<size_t>(v_head) * hv + v0 + (vt0 + vj) * 16u + lo];
                    bv[vj] = g::gdn_decomp8<kLossy>(dtmp);
                }
                #pragma unroll
                for (uint32_t ki = 0; ki < 2u; ++ki) {
                    float ktmp[8] = {};
                    if (tstep == 0u && kh == 0u)
                        ktmp[0] = a.compact_k[
                            kbase + static_cast<size_t>(row) * qkdim +
                            static_cast<size_t>(v_head) * hd + (kt0 + ki) * 16u + lo];
                    const Frag2 akT = g::gdn_decomp8<kLossy>(ktmp);
                    #pragma unroll
                    for (uint32_t vj = 0; vj < 2u; ++vj)
                        Sacc[ki * 2u + vj] =
                            g::gdn_wmma2<kLossy>(Sacc[ki * 2u + vj], akT, bv[vj]);
                }
            }
            #pragma unroll
            for (uint32_t t = 0; t < 4u; ++t)
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] = f16_round(Sacc[t][e]);
        }

        #pragma unroll
        for (uint32_t ki = 0; ki < 2u; ++ki)
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                const uint32_t n0 = (vt0 + vj) * 16u + lo;
                fp16_t* sp = state_head + static_cast<size_t>(m0) * hv + v0 + n0;
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    st_f16(sp + static_cast<size_t>(e) * hv, Sacc[ki * 2u + vj][e]);
            }
    }
}

template <uint32_t kRows>
__device__ __forceinline__ void gdn_recurrence_wmma_decode_rows_exact_body(
    const GdnRecurrenceArgs a) {
    namespace g = gdngfx12;
    using g::f32x8_t;
    using g::Frag2;
    static_assert(kRows == 2u || kRows == 4u || kRows == 8u,
                  "decode rows kernel covers verify rows 2/4/8");

    const uint32_t hd = a.head_k;
    const uint32_t hv = a.head_v;
    const uint32_t vgroups = hv / g::kWv;
    const uint32_t vg = blockIdx.x % vgroups;
    const uint32_t v_head = (blockIdx.x / vgroups) % a.num_v_heads;
    const uint32_t request_index = blockIdx.x / (vgroups * a.num_v_heads);
    if (request_index >= a.num_requests) return;
    const ::shelf::runtime::DeviceRequestDescriptor req = a.requests[request_index];
    const uint32_t slot = req.request_handle.slot;
    if (slot >= a.max_sequences) return;
    const uint32_t row_begin = req.row_begin;
    if (row_begin >= a.actual_rows) return;
    if (row_begin + kRows > a.actual_rows) return;

    const uint32_t repeat = a.num_v_heads / a.key_heads;
    const uint32_t qk_head = v_head / repeat;
    const uint32_t v0 = vg * g::kWv;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lo = lane & 15u;
    const uint32_t hi = lane >> 4u;
    const uint32_t kh = lane >> 4u;

    fp16_t* state_head = a.recurrent_state
        + static_cast<uint64_t>(slot) * a.recurrent_slot_stride
        + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
        + static_cast<uint64_t>(v_head) * hd * hv;

    const float* __restrict__ qp_g = a.q + qk_head * hd;
    const float* __restrict__ kp_g = a.k + qk_head * hd;

    const float dt_bias = g::wbf2f(a.dt_bias + v_head);
    const float neg_exp_a_log = -__expf(g::wbf2f(a.a_log + v_head));

    __shared__ RowSmem s;
    float* Sst = &s.S[0][0];
    const uint32_t spitch = g::kSp;

    const uint32_t mt = warp & 1u;
    const uint32_t nt = warp >> 1u;
    const uint32_t kt0 = (warp >> 1u) * 2u;
    const uint32_t vt0 = (warp & 1u) * 2u;

    f32x8_t Sacc[4];
    #pragma unroll
    for (uint32_t ki = 0; ki < 2u; ++ki)
        #pragma unroll
        for (uint32_t vj = 0; vj < 2u; ++vj) {
            const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
            const uint32_t n0 = (vt0 + vj) * 16u + lo;
            const fp16_t* sp = state_head + static_cast<size_t>(m0) * hv + v0 + n0;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e)
                Sacc[ki * 2u + vj][e] = ld_f16(sp + static_cast<size_t>(e) * hv);
        }

    #pragma unroll 1
    for (uint32_t r = 0u; r < kRows; ++r) {
        const uint32_t row = row_begin + r;
        if (a.row_parent != nullptr && r > 0u) {
            const int32_t parent = a.row_parent[row];
            if (parent != static_cast<int32_t>(row) - 1) {
                const fp16_t* src = state_head;
                if (parent >= 0 && a.recurrent_history != nullptr &&
                    static_cast<uint32_t>(parent) < a.capture_rows) {
                    src = a.recurrent_history +
                          static_cast<uint64_t>(parent) * a.recurrent_history_row_stride +
                          static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride +
                          static_cast<uint64_t>(v_head) * hd * hv;
                }
                #pragma unroll
                for (uint32_t ki = 0; ki < 2u; ++ki)
                    #pragma unroll
                    for (uint32_t vj = 0; vj < 2u; ++vj) {
                        const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                        const uint32_t n0 = (vt0 + vj) * 16u + lo;
                        #pragma unroll
                        for (uint32_t e = 0; e < 8u; ++e)
                            Sacc[ki * 2u + vj][e] =
                                ld_f16(src + static_cast<size_t>(m0) * hv + v0 + n0 +
                                       static_cast<size_t>(e) * hv);
                    }
            }
        }

        const float bv = g::wbf2f(a.b + static_cast<size_t>(row) * a.b_row_stride + v_head);
        const float av = g::wbf2f(a.a + static_cast<size_t>(row) * a.a_row_stride + v_head);
        const float beta = 1.0f / (1.0f + __expf(-bv));
        const float gval = neg_exp_a_log * __logf(1.0f + __expf(av + dt_bias));
        const float dval = __expf(gval);
        const float de0 = __expf(gval - gval);

        #pragma unroll
        for (uint32_t ki = 0; ki < 2u; ++ki)
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                const uint32_t kt = kt0 + ki, vt = vt0 + vj;
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    Sst[static_cast<size_t>(kt * 16u + 8u * hi + e) * spitch + vt * 16u + lo] =
                        Sacc[ki * 2u + vj][e];
            }

        fp16_t* hist_head = nullptr;
        if (a.recurrent_history != nullptr && r < a.capture_rows &&
            (a.capture_verify_only == 0u ||
             req.execution_class ==
                 ::shelf::runtime::ExecutionClass::SPEC_VERIFY)) {
            hist_head = a.recurrent_history
                + static_cast<uint64_t>(r) * a.recurrent_history_row_stride
                + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
                + static_cast<uint64_t>(v_head) * hd * hv;
        }

        {
            const uint32_t kk = warp;
            const uint32_t ti = lo;
            const bool iok = ti < 1u;
            const float* __restrict__ kp = kp_g + static_cast<size_t>(row + ti) * a.k_row_stride;
            const float* __restrict__ qp = qp_g + static_cast<size_t>(row + ti) * a.q_row_stride;
            float ktmp[8], qtmp[8];
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                ktmp[q] = iok ? kp[kk * 16u + kh * 4u + q] : 0.0f;
                ktmp[4u + q] = iok ? kp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
                qtmp[q] = iok ? qp[kk * 16u + kh * 4u + q] : 0.0f;
                qtmp[4u + q] = iok ? qp[kk * 16u + 8u + kh * 4u + q] : 0.0f;
            }
            const Frag2 fk = g::gdn_decomp8<true>(ktmp);
            const Frag2 fq = g::gdn_decomp8<true>(qtmp);
            #pragma unroll
            for (uint32_t d = 0; d < 8u; ++d) {
                s.kqfrag[0][kk][lane][d] = fk.h[d];
                s.kqfrag[1][kk][lane][d] = fq.h[d];
            }
        }
        detail::lds_barrier();

        f32x8_t uacc = {};
        f32x8_t oacc = {};
        f32x8_t p_acc = {};
        if (mt == 0u) {
            const bool ppk = warp == 0u;
            #pragma unroll
            for (uint32_t kk = 0; kk < g::kWk / 16u; ++kk) {
                float stmp[8];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    stmp[q] = Sst[static_cast<size_t>(kk * 16u + kh * 4u + q) * spitch + nt * 16u + lo];
                    stmp[4u + q] = Sst[static_cast<size_t>(kk * 16u + 8u + kh * 4u + q) * spitch + nt * 16u + lo];
                }
                const Frag2 bs = g::gdn_decomp8<true>(stmp);
                Frag2 ak, aq;
                #pragma unroll
                for (uint32_t d = 0; d < 8u; ++d) {
                    ak.h[d] = s.kqfrag[0][kk][lane][d];
                    aq.h[d] = s.kqfrag[1][kk][lane][d];
                }
                uacc = g::gdn_wmma2<true>(uacc, ak, bs);
                oacc = g::gdn_wmma2<true>(oacc, aq, bs);
                if (ppk) {
                    p_acc = g::gdn_wmma2<true>(p_acc, aq, ak);
                }
            }
        }

        if (warp == 0u && lane == 0u) s.a00 = __expf(gval - gval) * p_acc[0];

        if (mt == 0u && hi == 0u) {
            const uint32_t v = nt * 16u + lo;
            const float vv = g::wbf2f(a.v + static_cast<size_t>(row) * a.v_row_stride
                                      + static_cast<size_t>(v_head) * hv + v0 + v);
            s.delta[v] = beta * (vv - dval * uacc[0]);
        }
        detail::lds_barrier();

        if (mt == 0u) {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) oacc[e] *= dval;

            float atmp0[8] = {};
            if (lane == 0u) atmp0[0] = s.a00;
            const Frag2 aab0 = g::gdn_decomp8<true>(atmp0);
            float atmp1[8] = {};
            const Frag2 aab1 = g::gdn_decomp8<true>(atmp1);
            #pragma unroll
            for (uint32_t jt = 0; jt < 2u; ++jt) {
                float dtmp[8] = {};
                if (jt == 0u && kh == 0u) dtmp[0] = s.delta[nt * 16u + lo];
                const Frag2 bvb = g::gdn_decomp8<true>(dtmp);
                oacc = g::gdn_wmma2<true>(oacc, jt == 0u ? aab0 : aab1, bvb);
            }

            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) {
                const uint32_t i = 8u * hi + e;
                if (i < 1u)
                    a.output[static_cast<size_t>(row) * a.output_row_stride
                             + static_cast<size_t>(v_head) * hv + v0 + nt * 16u + lo] = oacc[e];
            }
        }

        const float egv = dval;
        const uint64_t ckbase =
            a.compact_k != nullptr
                ? static_cast<uint64_t>(a.state_index) * a.compact_k_layer_stride
                : 0u;
        const uint64_t cqkdim = static_cast<uint64_t>(a.num_v_heads) * hd;
        #pragma unroll
        for (uint32_t t = 0; t < 4u; ++t)
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] *= egv;
        #pragma unroll
        for (uint32_t tstep = 0; tstep < g::kWt / 16u; ++tstep) {
            Frag2 bvv[2];
            #pragma unroll
            for (uint32_t vj = 0; vj < 2u; ++vj) {
                float dtmp[8] = {};
                if (tstep == 0u && kh == 0u) dtmp[0] = s.delta[(vt0 + vj) * 16u + lo];
                bvv[vj] = g::gdn_decomp8<true>(dtmp);
            }
            #pragma unroll
            for (uint32_t ki = 0; ki < 2u; ++ki) {
                float ktmp[8] = {};
                if (tstep == 0u && kh == 0u) {
                    ktmp[0] = a.k[static_cast<size_t>(row) * a.k_row_stride + qk_head * hd
                                  + (kt0 + ki) * 16u + lo] * de0;
                    if (a.compact_k != nullptr)
                        a.compact_k[ckbase + static_cast<size_t>(row) * cqkdim +
                                    static_cast<size_t>(v_head) * hd +
                                    (kt0 + ki) * 16u + lo] = ktmp[0];
                }
                const Frag2 akT = g::gdn_decomp8<true>(ktmp);
                #pragma unroll
                for (uint32_t vj = 0; vj < 2u; ++vj)
                    Sacc[ki * 2u + vj] = g::gdn_wmma2<true>(Sacc[ki * 2u + vj], akT, bvv[vj]);
            }
        }

        if (a.compact_delta != nullptr) {
            const uint64_t vdim = static_cast<uint64_t>(a.num_v_heads) * hv;
            const uint64_t dbase =
                static_cast<uint64_t>(a.state_index) * a.compact_delta_layer_stride;
            const uint64_t abase =
                static_cast<uint64_t>(a.state_index) * a.compact_a_layer_stride;
            for (uint32_t v = tid; v < g::kWv; ++v)
                a.compact_delta[dbase + static_cast<size_t>(row) * vdim +
                                static_cast<size_t>(v_head) * hv + v0 + v] = s.delta[v];
            if (tid == 0u && a.compact_a != nullptr)
                a.compact_a[abase + static_cast<size_t>(row) * a.num_v_heads + v_head] = dval;
        }

        if (hist_head != nullptr) {
            #pragma unroll
            for (uint32_t ki = 0; ki < 2u; ++ki)
                #pragma unroll
                for (uint32_t vj = 0; vj < 2u; ++vj) {
                    const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                    const uint32_t n0 = (vt0 + vj) * 16u + lo;
                    fp16_t* hp = hist_head + static_cast<size_t>(m0) * hv + v0 + n0;
                    #pragma unroll
                    for (uint32_t e = 0; e < 8u; ++e)
                        st_f16(hp + static_cast<size_t>(e) * hv, Sacc[ki * 2u + vj][e]);
                }
        }

        #pragma unroll
        for (uint32_t t = 0; t < 4u; ++t)
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) Sacc[t][e] = f16_round(Sacc[t][e]);

        if (r + 1u == kRows) {
            #pragma unroll
            for (uint32_t ki = 0; ki < 2u; ++ki)
                #pragma unroll
                for (uint32_t vj = 0; vj < 2u; ++vj) {
                    const uint32_t m0 = (kt0 + ki) * 16u + 8u * hi;
                    const uint32_t n0 = (vt0 + vj) * 16u + lo;
                    fp16_t* sp = state_head + static_cast<size_t>(m0) * hv + v0 + n0;
                    #pragma unroll
                    for (uint32_t e = 0; e < 8u; ++e)
                        st_f16(sp + static_cast<size_t>(e) * hv, Sacc[ki * 2u + vj][e]);
                }
        }
    }
}

template <uint32_t kRows>
__global__ __launch_bounds__(gdngfx12::kWthreads, 1u) void
gdn_recurrence_wmma_decode_rows_exact_impl(const GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_decode_rows_exact_body<kRows>(a);
}

extern "C" __global__ __launch_bounds__(gdngfx12::kWthreads, 1u)
void shelf_gdn_recurrence_decode_rows_exact_r2(
    GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_decode_rows_exact_body<2u>(a);
}

extern "C" __global__ __launch_bounds__(gdngfx12::kWthreads, 1u)
void shelf_gdn_recurrence_decode_rows_exact_r4(
    GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_decode_rows_exact_body<4u>(a);
}

extern "C" __global__ __launch_bounds__(gdngfx12::kWthreads, 1u)
void shelf_gdn_recurrence_decode_rows_exact_r8(
    GdnRecurrenceArgs a) {
    gdn_recurrence_wmma_decode_rows_exact_body<8u>(a);
}

bool gdn_recurrence_lossy_enabled() {
    static const bool enabled = []() {
        const char* x = std::getenv("SHELF_GDN_RECURRENCE_EXACT");
        if (x != nullptr && x[0] != '\0' && x[0] != '0') return false;
        const char* l = std::getenv("SHELF_GDN_RECURRENCE_LOSSY");
        if (l != nullptr && l[0] == '0') return false;
        return true;
    }();
    return enabled;
}

hipError_t launch_gdn_recurrence_f32_wmma(
    const GdnRecurrenceArgs& args, hipStream_t stream) {
    if (args.head_k != gdngfx12::kWk) return hipErrorInvalidValue;
    if (args.head_v == 0u || args.head_v % gdngfx12::kWv != 0u) return hipErrorInvalidValue;
    if (args.key_heads == 0u || args.num_v_heads == 0u) return hipErrorInvalidValue;
    if (args.num_v_heads % args.key_heads != 0u) return hipErrorInvalidValue;
    if (args.num_requests == 0u || args.actual_rows == 0u) return hipErrorInvalidValue;
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.a == nullptr || args.b == nullptr)
        return hipErrorInvalidValue;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return hipErrorInvalidValue;
    if (args.recurrent_state == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.requests == nullptr || args.max_sequences == 0u) return hipErrorInvalidValue;
    const uint32_t vgroups = args.head_v / gdngfx12::kWv;
    const uint64_t grid_x = static_cast<uint64_t>(args.num_requests) * args.num_v_heads * vgroups;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (gdn_recurrence_lossy_enabled())
        shelf_gdn_recurrence_wmma_lossy<<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
    else
        shelf_gdn_recurrence_wmma_exact<<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
    return hipGetLastError();
}

hipError_t launch_gdn_recurrence_f32_wmma_serial(
    const GdnRecurrenceArgs& args, hipStream_t stream) {
    if (args.head_k != gdngfx12::kWk) return hipErrorInvalidValue;
    if (args.head_v == 0u || args.head_v % gdngfx12::kWv != 0u) return hipErrorInvalidValue;
    if (args.key_heads == 0u || args.num_v_heads == 0u) return hipErrorInvalidValue;
    if (args.num_v_heads % args.key_heads != 0u) return hipErrorInvalidValue;
    if (args.num_requests == 0u || args.actual_rows == 0u) return hipErrorInvalidValue;
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.a == nullptr || args.b == nullptr)
        return hipErrorInvalidValue;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return hipErrorInvalidValue;
    if (args.recurrent_state == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.requests == nullptr || args.max_sequences == 0u) return hipErrorInvalidValue;
    const uint32_t vgroups = args.head_v / gdngfx12::kWv;
    const uint64_t grid_x = static_cast<uint64_t>(args.num_requests) * args.num_v_heads * vgroups;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (gdn_recurrence_lossy_enabled())
        gdn_recurrence_wmma_impl<true, true><<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
    else
        gdn_recurrence_wmma_impl<false, true><<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
    return hipGetLastError();
}

bool gdn_recurrence_decode_rows_supported(const GdnRecurrenceArgs& args) {
    if (!gdn_recurrence_lossy_enabled()) return false;
    const bool tp1 = args.key_heads == 16u && args.num_v_heads == 48u;
    const bool tp2 = args.key_heads == 8u && args.num_v_heads == 24u;
    if (!tp1 && !tp2) return false;
    if (args.head_k != gdngfx12::kWk || args.head_v != 128u) return false;
    if (args.num_requests != 1u) return false;
    return args.actual_rows == 2u || args.actual_rows == 4u || args.actual_rows == 8u;
}

hipError_t launch_gdn_recurrence_f32_wmma_decode_rows_exact(
    const GdnRecurrenceArgs& args, hipStream_t stream) {
    if (!gdn_recurrence_decode_rows_supported(args)) return hipErrorNotSupported;
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.a == nullptr ||
        args.b == nullptr)
        return hipErrorInvalidValue;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return hipErrorInvalidValue;
    if (args.recurrent_state == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.requests == nullptr || args.max_sequences == 0u) return hipErrorInvalidValue;
    const uint32_t vgroups = args.head_v / gdngfx12::kWv;
    const uint64_t grid_x = static_cast<uint64_t>(args.num_requests) * args.num_v_heads * vgroups;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    switch (args.actual_rows) {
        case 2u:
            gdn_recurrence_wmma_decode_rows_exact_impl<2>
                <<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
            break;
        case 4u:
            gdn_recurrence_wmma_decode_rows_exact_impl<4>
                <<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
            break;
        case 8u:
            gdn_recurrence_wmma_decode_rows_exact_impl<8>
                <<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
            break;
        default:
            return hipErrorNotSupported;
    }
    return hipGetLastError();
}

bool gdn_recurrence_commit_supported(const GdnRecurrenceArgs& args) {
    return args.head_k == gdngfx12::kWk && args.head_v == 128u &&
           args.num_v_heads != 0u;
}

hipError_t launch_gdn_recurrence_f32_wmma_commit(
    const GdnRecurrenceArgs& args, hipStream_t stream) {
    if (!gdn_recurrence_commit_supported(args)) return hipErrorNotSupported;
    if (args.num_requests == 0u || args.compact_rows == 0u) return hipErrorInvalidValue;
    if (args.num_gdn_states == 0u) return hipErrorInvalidValue;
    if (args.recurrent_state == nullptr || args.requests == nullptr)
        return hipErrorInvalidValue;
    if (args.compact_delta == nullptr || args.compact_k == nullptr || args.compact_a == nullptr)
        return hipErrorInvalidValue;
    if (args.compact_delta_layer_stride == 0u || args.compact_k_layer_stride == 0u ||
        args.compact_a_layer_stride == 0u)
        return hipErrorInvalidValue;
    const uint32_t vgroups = args.head_v / gdngfx12::kWv;
    const uint64_t grid_x = static_cast<uint64_t>(args.num_requests) * args.num_v_heads * vgroups;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    shelf_gdn_recurrence_wmma_commit<<<grid, gdngfx12::kWthreads, 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace shelf::kernel
