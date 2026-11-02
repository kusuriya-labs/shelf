#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint32_t kDim = 128u;
constexpr uint32_t kState = kDim * kDim;

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float rndf(uint32_t& s) {
    return ((float)(int32_t)(xr(s) % 2001u) - 1000.0f) / 1000.0f;
}

__global__ void poc_seq(const float* s0, const float* k, const float* v,
                        const float* arow, const float* brow, const float* dtb,
                        const float* alog, uint32_t rows, float* traj, float* sp_a,
                        float* sp_delta) {
    const uint32_t vv = threadIdx.x;
    __shared__ float kk[kDim];
    __shared__ float vrow[kDim];
    __shared__ float sc[2];
    float scol[kDim];
    #pragma unroll 1
    for (uint32_t d = 0u; d < kDim; ++d) scol[d] = s0[d * kDim + vv];
    #pragma unroll 1
    for (uint32_t d = 0u; d < kDim; ++d) traj[d * kDim + vv] = scol[d];
    __syncthreads();
    for (uint32_t row = 0u; row < rows; ++row) {
        if (vv == 0u) {
            for (uint32_t d = 0u; d < kDim; ++d) kk[d] = k[row * kDim + d];
            const float av = arow[row];
            const float bv = brow[row];
            const float beta = 1.0f / (1.0f + __expf(-bv));
            const float gval = -__expf(*alog) * __logf(1.0f + __expf(av + *dtb));
            sc[0] = __expf(gval);
            sc[1] = beta;
        }
        __syncthreads();
        const float a = sc[0];
        const float beta = sc[1];
        if (vv < kDim) vrow[vv] = v[row * kDim + vv];
        __syncthreads();
        float dot = 0.0f;
        #pragma unroll 1
        for (uint32_t d = 0u; d < kDim; ++d) dot += kk[d] * scol[d];
        const float delta = beta * (vrow[vv] - a * dot);
        #pragma unroll 1
        for (uint32_t d = 0u; d < kDim; ++d) scol[d] = a * scol[d] + kk[d] * delta;
        #pragma unroll 1
        for (uint32_t d = 0u; d < kDim; ++d)
            traj[(row + 1u) * kState + d * kDim + vv] = scol[d];
        sp_delta[row * kDim + vv] = delta;
        if (vv == 0u) sp_a[row] = a;
        __syncthreads();
    }
}

__global__ void poc_recon_seq(const float* s0, const float* k, const float* sp_a,
                              const float* sp_delta, uint32_t m, float* out) {
    const uint32_t vv = threadIdx.x;
    __shared__ float kk[kDim];
    float scol[kDim];
    #pragma unroll 1
    for (uint32_t d = 0u; d < kDim; ++d) scol[d] = s0[d * kDim + vv];
    for (uint32_t row = 0u; row < m; ++row) {
        if (vv == 0u)
            for (uint32_t d = 0u; d < kDim; ++d) kk[d] = k[row * kDim + d];
        __syncthreads();
        const float a = sp_a[row];
        const float delta = sp_delta[row * kDim + vv];
        #pragma unroll 1
        for (uint32_t d = 0u; d < kDim; ++d) scol[d] = a * scol[d] + kk[d] * delta;
        __syncthreads();
    }
    #pragma unroll 1
    for (uint32_t d = 0u; d < kDim; ++d) out[d * kDim + vv] = scol[d];
}

__global__ void poc_recon_gemm(const float* s0, const float* k, const float* sp_delta,
                               const float* coef, const float* am, uint32_t m,
                               float* out) {
    const uint32_t vv = threadIdx.x;
    const float total = *am;
    #pragma unroll 1
    for (uint32_t d = 0u; d < kDim; ++d) {
        float acc = total * s0[d * kDim + vv];
        for (uint32_t i = 0u; i < m; ++i)
            acc += coef[i] * k[i * kDim + d] * sp_delta[i * kDim + vv];
        out[d * kDim + vv] = acc;
    }
}

struct Diff {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool exact = true;
};

Diff compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Diff d;
    float ref_max = 0.0f;
    for (float x : ref) ref_max = std::max(ref_max, std::fabs(x));
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(ref[i] - got[i]);
        d.max_abs = std::max(d.max_abs, e);
        if (ref[i] != got[i]) d.exact = false;
    }
    d.max_rel = ref_max > 0.0f ? d.max_abs / ref_max : d.max_abs;
    return d;
}

}  // namespace

int main() {
    std::printf("=== GDN compact-log commit PoC (fp32, 1 layer / 1 head) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    const uint32_t rows = 16u;
    const uint32_t nf = rows * kDim;
    const uint32_t ntraj = (rows + 1u) * kState;

    float* d_s0 = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_dtb = nullptr;
    float* d_alog = nullptr;
    float* d_traj = nullptr;
    float* d_spa = nullptr;
    float* d_spd = nullptr;
    float* d_out = nullptr;
    float* d_coef = nullptr;
    float* d_am = nullptr;
    auto alloc = [&](float** p, size_t n) {
        return hipMalloc(reinterpret_cast<void**>(p), n * sizeof(float)) == hipSuccess;
    };
    bool ok = alloc(&d_s0, kState) && alloc(&d_k, nf) && alloc(&d_v, nf) && alloc(&d_a, rows) &&
              alloc(&d_b, rows) && alloc(&d_dtb, 1u) && alloc(&d_alog, 1u) &&
              alloc(&d_traj, ntraj) && alloc(&d_spa, rows) && alloc(&d_spd, nf) &&
              alloc(&d_out, kState) && alloc(&d_coef, rows) && alloc(&d_am, 1u);
    if (!ok) {
        std::printf("FAIL: hipMalloc\n");
        return 1;
    }

    uint32_t seed = 20260929u;
    std::vector<float> h_s0(kState), h_k(nf), h_v(nf), h_a(rows), h_b(rows);
    std::vector<float> h_dtb(1), h_alog(1);
    for (float& x : h_s0) x = rndf(seed) * 0.25f;
    for (float& x : h_k) x = rndf(seed);
    for (float& x : h_v) x = rndf(seed);
    for (float& x : h_a) x = rndf(seed) * 0.5f;
    for (float& x : h_b) x = rndf(seed) * 0.5f;
    h_dtb[0] = rndf(seed) * 0.5f;
    h_alog[0] = rndf(seed) * 0.5f;

    ps::gpu::discard_cleanup_result(hipMemcpy(d_s0, h_s0.data(), kState * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_k, h_k.data(), nf * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_v, h_v.data(), nf * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_a, h_a.data(), rows * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_b, h_b.data(), rows * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_dtb, h_dtb.data(), 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_alog, h_alog.data(), 4u, hipMemcpyHostToDevice));

    poc_seq<<<1u, kDim>>>(d_s0, d_k, d_v, d_a, d_b, d_dtb, d_alog, rows, d_traj, d_spa, d_spd);
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL: poc_seq launch\n");
        return 1;
    }

    std::vector<float> h_traj(ntraj), h_spa(rows), h_spd(nf);
    ps::gpu::discard_cleanup_result(hipMemcpy(h_traj.data(), d_traj, ntraj * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(h_spa.data(), d_spa, rows * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(h_spd.data(), d_spd, nf * 4u, hipMemcpyDeviceToHost));

    std::vector<float> h_ref(kState), h_out(kState), h_coef(rows);
    double worst_seq = 0.0;
    double worst_gemm = 0.0;
    for (uint32_t m = 0u; m <= rows; ++m) {
        std::copy(h_traj.begin() + static_cast<size_t>(m) * kState,
                  h_traj.begin() + static_cast<size_t>(m + 1u) * kState, h_ref.begin());

        poc_recon_seq<<<1u, kDim>>>(d_s0, d_k, d_spa, d_spd, m, d_out);
        ps::gpu::discard_cleanup_result(hipMemcpy(h_out.data(), d_out, kState * 4u, hipMemcpyDeviceToHost));
        const Diff ds = compare(h_ref, h_out);

        double am = 1.0;
        for (uint32_t i = 0u; i < m; ++i) am *= static_cast<double>(h_spa[i]);
        const float amf = static_cast<float>(am);
        for (uint32_t i = 0u; i < m; ++i) {
            double c = 1.0;
            for (uint32_t j = i + 1u; j < m; ++j) c *= static_cast<double>(h_spa[j]);
            h_coef[i] = static_cast<float>(c);
        }
        ps::gpu::discard_cleanup_result(hipMemcpy(d_coef, h_coef.data(), m * 4u, hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(hipMemcpy(d_am, &amf, 4u, hipMemcpyHostToDevice));
        poc_recon_gemm<<<1u, kDim>>>(d_s0, d_k, d_spd, d_coef, d_am, m, d_out);
        ps::gpu::discard_cleanup_result(hipMemcpy(h_out.data(), d_out, kState * 4u, hipMemcpyDeviceToHost));
        const Diff dg = compare(h_ref, h_out);

        std::printf("m=%2u  seq: max_abs=%.3e max_rel=%.3e exact=%s  |  gemm: max_abs=%.3e "
                    "max_rel=%.3e exact=%s\n",
                    m, ds.max_abs, ds.max_rel, ds.exact ? "yes" : "NO", dg.max_abs, dg.max_rel,
                    dg.exact ? "yes" : "NO");
        worst_seq = std::max(worst_seq, static_cast<double>(ds.max_rel));
        worst_gemm = std::max(worst_gemm, static_cast<double>(dg.max_rel));
    }

    check(worst_seq < 1e-5, "sequential replay matches reference (max_rel < 1e-5)");
    check(worst_gemm < 1e-3, "gemm regroup error bounded (max_rel < 1e-3)");

    void* frees[] = {d_s0, d_k, d_v, d_a, d_b, d_dtb, d_alog, d_traj,
                     d_spa, d_spd, d_out, d_coef, d_am};
    for (void* p : frees)
        if (p) ps::gpu::discard_cleanup_result(hipFree(p));

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
