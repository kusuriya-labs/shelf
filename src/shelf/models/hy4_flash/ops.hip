#include <shelf/models/hy4_flash/ops.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
__device__ float read(Param param, size_t i) {
    if (param.type == io::SType::F32)
        return static_cast<const float *>(param.p)[i];
    return bf16_bits_to_f32(static_cast<const uint16_t *>(param.p)[i]);
}
__device__ float value(Mat a, size_t i) { return bf16_bits_to_f32(a.p[i]); }
__device__ float blocksum(float x, float *shared) {
    for (int m = 16; m; m /= 2)
        x += __shfl_down(x, m, 32);
    if (threadIdx.x % 32 == 0)
        shared[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.f;
    for (int m = 16; m; m /= 2)
        x += __shfl_down(x, m, 32);
    if (threadIdx.x == 0)
        shared[0] = x;
    __syncthreads();
    return shared[0];
}
__global__ void dense(Mat x, Param w, Mat out) {
    unsigned row = blockIdx.y, col = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (col >= unsigned(out.cols))
        return;
    float acc = 0;
    for (unsigned j = lane; j < unsigned(x.cols); j += 32)
        acc += value(x, size_t(row) * x.cols + j) * read(w, size_t(col) * x.cols + j);
    for (int d = 16; d; d /= 2)
        acc += __shfl_down(acc, d, 32);
    if (!lane)
        out.p[size_t(row) * out.cols + col] = f32_to_bf16_rne_device(acc);
}
template <int RT, int CT, int WR, int WC>
__global__ void __launch_bounds__((WR * WC) * 32) dense_tiled(Mat x, const uint16_t *w, Mat out) {
    constexpr int CTA_R = RT * WR;
    constexpr int CTA_C = CT * WC;
    constexpr int THREADS = WR * WC * 32;
    __shared__ uint16_t s_x[CTA_R][72];
    __shared__ uint16_t s_w[CTA_C][72];
    int t = threadIdx.x, wave = t >> 5, lane = t & 31;
    int wr = wave % WR, wc = wave / WR;
    int r_base = blockIdx.y * CTA_R, c_base = blockIdx.x * CTA_C;
    float acc[RT][CT]{};
    int k_off = (t & 7) * 8;
    int r_step = THREADS / 8;
    for (int k = 0; k < x.cols; k += 64) {
        for (int xr = t >> 3; xr < CTA_R; xr += r_step) {
            int r = r_base + xr;
            *reinterpret_cast<uint4 *>(&s_x[xr][k_off]) =
                r < x.rows ? *reinterpret_cast<const uint4 *>(x.p + size_t(r) * x.cols + k + k_off) : uint4{};
        }
        for (int wc_r = t >> 3; wc_r < CTA_C; wc_r += r_step) {
            int c = c_base + wc_r;
            *reinterpret_cast<uint4 *>(&s_w[wc_r][k_off]) =
                c < out.cols ? *reinterpret_cast<const uint4 *>(w + size_t(c) * x.cols + k + k_off) : uint4{};
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 64; ks += 32) {
            float vx[RT], vw[CT];
#pragma unroll
            for (int r = 0; r < RT; ++r)
                vx[r] = bf16_bits_to_f32(s_x[wr * RT + r][ks + lane]);
#pragma unroll
            for (int c = 0; c < CT; ++c)
                vw[c] = bf16_bits_to_f32(s_w[wc * CT + c][ks + lane]);
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int c = 0; c < CT; ++c)
                    acc[r][c] += vx[r] * vw[c];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        int row = r_base + wr * RT + r;
        if (row >= x.rows)
            continue;
#pragma unroll
        for (int c = 0; c < CT; ++c) {
            int col = c_base + wc * CT + c;
            if (col >= out.cols)
                continue;
            float sum = acc[r][c];
            for (int d = 16; d; d /= 2)
                sum += __shfl_down(sum, d, 32);
            if (!lane)
                out.p[size_t(row) * out.cols + col] = f32_to_bf16_rne_device(sum);
        }
    }
}
__global__ void normalize(Mat x, Param weight, Mat y, float eps) {
    __shared__ float scratch[32];
    size_t base = size_t(blockIdx.x) * x.cols;
    if (x.cols <= 3072) {
        uint16_t vals[12];
        int n = 0;
        float sum = 0;
        for (int c = threadIdx.x; c < x.cols; c += blockDim.x, ++n) {
            vals[n] = x.p[base + c];
            float v = bf16_bits_to_f32(vals[n]);
            sum += v * v;
        }
        float inv = rsqrtf(blocksum(sum, scratch) / x.cols + eps);
        for (int i = 0, c = threadIdx.x; i < n; ++i, c += blockDim.x)
            y.p[base + c] =
                f32_to_bf16_rne_device(round_bf16(bf16_bits_to_f32(vals[i]) * inv) * (weight.p ? read(weight, c) : 1.f));
        return;
    }
    float sum = 0;
    for (int c = threadIdx.x; c < x.cols; c += blockDim.x) {
        float v = value(x, base + c);
        sum += v * v;
    }
    float inv = rsqrtf(blocksum(sum, scratch) / x.cols + eps);
    for (int c = threadIdx.x; c < x.cols; c += blockDim.x)
        y.p[base + c] =
            f32_to_bf16_rne_device(round_bf16(value(x, base + c) * inv) * (weight.p ? read(weight, c) : 1.f));
}
__global__ void normalize_from(Mat x, Param weight, Mat y, float eps, int first, int width) {
    __shared__ float scratch[32];
    size_t bx = size_t(blockIdx.x) * x.cols + first, by = size_t(blockIdx.x) * width;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float v = value(x, bx + c);
        sum += v * v;
    }
    float inv = rsqrtf(blocksum(sum, scratch) / width + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        y.p[by + c] =
            f32_to_bf16_rne_device(round_bf16(value(x, bx + c) * inv) * (weight.p ? read(weight, c) : 1.f));
}
__global__ void norm_bias_kernel(Mat x, const uint16_t *bias, Param weight, Mat y, float eps) {
    __shared__ float scratch[32];
    size_t base = size_t(blockIdx.x) * x.cols;
    uint16_t vals[12];
    int n = 0;
    float sum = 0;
    for (int c = threadIdx.x; c < x.cols; c += 256) {
        float v = bf16_bits_to_f32(x.p[base + c]) + bf16_bits_to_f32(bias[c]);
        vals[n] = f32_to_bf16_rne_device(v);
        float r = bf16_bits_to_f32(vals[n]);
        sum += r * r;
        ++n;
    }
    float inv = rsqrtf(blocksum(sum, scratch) / x.cols + eps);
    for (int i = 0, c = threadIdx.x; c < x.cols; c += 256, ++i)
        y.p[base + c] = f32_to_bf16_rne_device(round_bf16(bf16_bits_to_f32(vals[i]) * inv) *
                                              (weight.p ? read(weight, c) : 1.f));
}
__global__ void norm_slice_kernel(Mat x, Param weight, Mat y, float eps, int first) {
    __shared__ float scratch[32];
    size_t bx = size_t(blockIdx.x) * x.cols + first, by = size_t(blockIdx.x) * y.cols;
    uint16_t vals[12];
    int n = 0;
    float sum = 0;
    for (int c = threadIdx.x; c < y.cols; c += 256) {
        vals[n] = x.p[bx + c];
        float r = bf16_bits_to_f32(vals[n]);
        sum += r * r;
        ++n;
    }
    float inv = rsqrtf(blocksum(sum, scratch) / y.cols + eps);
    for (int i = 0, c = threadIdx.x; c < y.cols; c += 256, ++i)
        y.p[by + c] = f32_to_bf16_rne_device(round_bf16(bf16_bits_to_f32(vals[i]) * inv) *
                                            (weight.p ? read(weight, c) : 1.f));
}
__global__ void rotary_from(Mat x, Mat y, int heads, int start, float theta, int first,
                            const Hy4StepParams *params, const int *positions = nullptr) {
    if (params)
        start = params->position;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(y.rows) * y.cols)
        return;
    int row = i / y.cols, width = y.cols / heads, c = i % width;
    size_t src = size_t(row) * x.cols + first + c;
    if (c < width - 64) {
        y.p[i] = x.p[src];
        return;
    }
    int j = c - (width - 64);
    float phase = (positions ? positions[row] : start + row) * powf(theta, -float(j / 2) / 32);
    float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
    float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
    y.p[i] = f32_to_bf16_rne_device(value(x, src) * co + ((j & 1) ? 1.f : -1.f) * value(x, src ^ 1) * si);
}
__global__ void select_cols(Mat x, Mat y, int first) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(y.rows) * y.cols)
        y.p[i] = x.p[(i / y.cols) * x.cols + first + i % y.cols];
}
__global__ void join_cols(Mat a, Mat b, Mat y) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(y.rows) * y.cols)
        return;
    int c = i % y.cols;
    y.p[i] = c < a.cols ? a.p[(i / y.cols) * a.cols + c] : b.p[(i / y.cols) * b.cols + c - a.cols];
}
__global__ void element(Mat a, Mat b, Mat y, int mode) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(a.cols) * a.rows)
        return;
    float g = value(a, i), u = value(b, i);
    if (mode == 2) {
        g = fminf(g, 10.f);
        u = fminf(10.f, fmaxf(-10.f, u));
    }
    y.p[i] = f32_to_bf16_rne_device(mode ? round_bf16(g / (1.f + expf(-g))) * u : g + u);
}
__global__ void gather_rows(Mat x, const int *ids, Mat y, int repeat) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(y.rows) * y.cols) {
        int id = ids[i / (repeat * x.cols)];
        y.p[i] = id >= 0 && id < x.rows ? x.p[size_t(id) * x.cols + i % x.cols] : 0;
    }
}
template <int Rows>
__global__ void hc_mix_multi(Mat x, Param fn, Param scale, Param base, float *gates, int count) {
    __shared__ float s_dot[Rows][8][8], s_norm[Rows][8];
    int r0 = blockIdx.y * Rows, t = threadIdx.x, lane = t & 31, wave = t >> 5;
    float dot[Rows][8]{}, norm[Rows]{};
    for (int i = t; i < x.cols; i += 256) {
        float vx[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            vx[r] = r0 + r < x.rows ? value(x, size_t(r0 + r) * x.cols + i) : 0.f;
        for (int g = 0; g < count; ++g) {
            float f = read(fn, size_t(g) * x.cols + i);
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                dot[r][g] += vx[r] * f;
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            norm[r] += vx[r] * vx[r];
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float nv = norm[r];
        for (int m = 16; m; m /= 2)
            nv += __shfl_down(nv, m, 32);
        if (!lane)
            s_norm[r][wave] = nv;
        for (int g = 0; g < count; ++g) {
            float dv = dot[r][g];
            for (int m = 16; m; m /= 2)
                dv += __shfl_down(dv, m, 32);
            if (!lane)
                s_dot[r][g][wave] = dv;
        }
    }
    __syncthreads();
    if (t < Rows * count) {
        int r = t / count, g = t % count;
        if (r0 + r < x.rows) {
            float d0 = s_dot[r][g][0] + s_dot[r][g][4], d1 = s_dot[r][g][1] + s_dot[r][g][5];
            float d2 = s_dot[r][g][2] + s_dot[r][g][6], d3 = s_dot[r][g][3] + s_dot[r][g][7];
            float d = (d0 + d2) + (d1 + d3);
            float n0 = s_norm[r][0] + s_norm[r][4], n1 = s_norm[r][1] + s_norm[r][5];
            float n2 = s_norm[r][2] + s_norm[r][6], n3 = s_norm[r][3] + s_norm[r][7];
            float n = (n0 + n2) + (n1 + n3);
            float f = d * rsqrtf(n / x.cols + 1e-5f);
            float v = __fadd_rn(__fmul_rn(f, read(scale, g / 4)), read(base, g));
            gates[(r0 + r) * count + g] = (g < 4 ? 1.f : 2.f) / (1.f + expf(-v)) + 1e-6f;
        }
    }
}
__global__ void hc_mix(Mat x, Param fn, Param scale, Param base, float *gates, int count) {
    __shared__ float s1[32], s2[32];
    int gate = blockIdx.x, row = blockIdx.y;
    float dot = 0, norm = 0;
    for (int i = threadIdx.x; i < x.cols; i += blockDim.x) {
        float v = value(x, size_t(row) * x.cols + i);
        dot += v * read(fn, size_t(gate) * x.cols + i);
        norm += v * v;
    }
    float f = blocksum(dot, s1) * rsqrtf(blocksum(norm, s2) / x.cols + 1e-5f);
    if (threadIdx.x == 0) {
        float v = __fadd_rn(__fmul_rn(f, read(scale, gate / 4)), read(base, gate));
        gates[row * count + gate] = (gate < 4 ? 1.f : 2.f) / (1.f + expf(-v)) + 1e-6f;
    }
}
__global__ void hc_reduce(Mat x, Mat y, const float *gates, int count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(y.rows) * y.cols)
        return;
    int row = i / y.cols, c = i % y.cols;
    float acc = 0;
    for (int h = 0; h < 4; ++h)
        acc =
            __fadd_rn(acc, __fmul_rn(gates[row * count + h], value(x, size_t(row) * x.cols + h * y.cols + c)));
    y.p[i] = f32_to_bf16_rne_device(acc);
}
__global__ void hc_reduce_norm(Mat x, const float *gates, Param weight, Mat y, float eps, int count) {
    __shared__ float scratch[32];
    int row = blockIdx.x, tid = threadIdx.x;
    uint16_t vals[12];
    int n = 0;
    float sum = 0;
    for (int c = tid; c < y.cols; c += 256) {
        float acc = 0;
        for (int h = 0; h < 4; ++h)
            acc = __fadd_rn(acc, __fmul_rn(gates[row * count + h], value(x, size_t(row) * x.cols + h * y.cols + c)));
        vals[n] = f32_to_bf16_rne_device(acc);
        float v = bf16_bits_to_f32(vals[n]);
        sum += v * v;
        ++n;
    }
    float inv = rsqrtf(blocksum(sum, scratch) / y.cols + eps);
    for (int i = 0, c = tid; c < y.cols; c += 256, ++i)
        y.p[size_t(row) * y.cols + c] = f32_to_bf16_rne_device(
            round_bf16(bf16_bits_to_f32(vals[i]) * inv) * (weight.p ? read(weight, c) : 1.f));
}
__global__ void hc_expand_vec(Mat x, Mat y, const float *post) {
    size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    size_t total = size_t(x.rows) * x.cols;
    if (idx >= total)
        return;
    int r = idx / x.cols, h = (idx / y.cols) % 4;
    float g = post[r * 8 + 4 + h];
    uint4 xv = *reinterpret_cast<const uint4 *>(x.p + idx);
    uint4 yv = *reinterpret_cast<const uint4 *>(y.p + size_t(r) * y.cols + idx % y.cols);
    uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    uint32_t yw[4] = {yv.x, yv.y, yv.z, yv.w};
    uint32_t out[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint16_t a0 = f32_to_bf16_rne_device(
            __fadd_rn(bf16_bits_to_f32(uint16_t(xw[p] & 0xffffu)),
                      __fmul_rn(g, bf16_bits_to_f32(uint16_t(yw[p] & 0xffffu)))));
        uint16_t a1 = f32_to_bf16_rne_device(
            __fadd_rn(bf16_bits_to_f32(uint16_t(xw[p] >> 16)),
                      __fmul_rn(g, bf16_bits_to_f32(uint16_t(yw[p] >> 16)))));
        out[p] = uint32_t(a0) | (uint32_t(a1) << 16);
    }
    *reinterpret_cast<uint4 *>(x.p + idx) = uint4{out[0], out[1], out[2], out[3]};
}
__global__ void hc_expand(Mat x, Mat y, const float *post) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int r = i / x.cols, h = (i / y.cols) % 4;
    x.p[i] = f32_to_bf16_rne_device(
        __fadd_rn(value(x, i), __fmul_rn(post[r * 8 + 4 + h], value(y, size_t(r) * y.cols + i % y.cols))));
}
__global__ void rotary(Mat x, Mat y, int heads, int start, float theta, const Hy4StepParams *params, const int *positions = nullptr) {
    if (params)
        start = params->position;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int row = i / x.cols, width = x.cols / heads, c = i % width;
    if (c < width - 64) {
        y.p[i] = x.p[i];
        return;
    }
    int j = c - (width - 64);
    float phase = (positions ? positions[row] : start + row) * powf(theta, -float(j / 2) / 32);
    float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
    float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
    y.p[i] = f32_to_bf16_rne_device(value(x, i) * co + ((j & 1) ? 1.f : -1.f) * value(x, i ^ 1) * si);
}
template <int Width>
__global__ void rotary_tiled(Mat x, Mat y, int heads, int start, float theta, const Hy4StepParams *params,
                             const int *positions = nullptr) {
    __shared__ float co_s[32], si_s[32];
    if (params)
        start = params->position;
    constexpr int HPB = Width == 256 ? 4 : 8;
    constexpr int LPW = Width / 8;
    int row = blockIdx.y, h_base = blockIdx.x * HPB, t = threadIdx.x;
    if (t < 32) {
        float phase = (positions ? positions[row] : start + row) * powf(theta, -float(t) / 32);
        co_s[t] = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
        si_s[t] = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
    }
    __syncthreads();
    int h = h_base + t / LPW, lane = t % LPW, c = lane * 8;
    if (h >= heads)
        return;
    size_t off = (size_t(row) * heads + h) * Width + c;
    uint4 raw = *reinterpret_cast<const uint4 *>(x.p + off);
    if (c < Width - 64) {
        *reinterpret_cast<uint4 *>(y.p + off) = raw;
        return;
    }
    int p_base = (c - (Width - 64)) >> 1;
    uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t out[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int pair = p_base + p;
        float v0 = bf16_bits_to_f32(uint16_t(w[p] & 0xffffu));
        float v1 = bf16_bits_to_f32(uint16_t(w[p] >> 16));
        float co = co_s[pair], si = si_s[pair];
        uint16_t o0 = f32_to_bf16_rne_device(v0 * co - v1 * si);
        uint16_t o1 = f32_to_bf16_rne_device(v1 * co + v0 * si);
        out[p] = uint32_t(o0) | (uint32_t(o1) << 16);
    }
    *reinterpret_cast<uint4 *>(y.p + off) = uint4{out[0], out[1], out[2], out[3]};
}
unsigned grid(Mat x) { return (size_t(x.rows) * x.cols + 255) / 256; }
}
void checked(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
}
namespace {
__global__ void peer_copy_kernel(uint8_t *dst, const uint8_t *src, size_t bytes) {
    size_t offset = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (offset + 16 <= bytes) {
        *reinterpret_cast<uint4 *>(dst + offset) = *reinterpret_cast<const uint4 *>(src + offset);
        return;
    }
    for (size_t i = offset; i < bytes; ++i)
        dst[i] = src[i];
}
}
void peer_copy(void *dst, const void *src, size_t bytes, hipStream_t stream) {
    if (!bytes)
        return;
    size_t threads = (bytes + 15) / 16;
    peer_copy_kernel<<<(threads + 255) / 256, 256, 0, stream>>>(static_cast<uint8_t *>(dst),
                                                     static_cast<const uint8_t *>(src), bytes);
    checked(hipGetLastError());
}
namespace {
__global__ void add_bf16_kernel(uint16_t *dst, const uint16_t *src, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = f32_to_bf16_rne_device(bf16_bits_to_f32(dst[i]) + bf16_bits_to_f32(src[i]));
}
}
void tp_add_bf16(void *dst, const void *src, size_t bytes, hipStream_t stream) {
    if (!bytes)
        return;
    size_t count = bytes / 2;
    add_bf16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<uint16_t *>(dst),
                                                  static_cast<const uint16_t *>(src), count);
    checked(hipGetLastError());
}
struct alignas(8) TpArgmaxPair {
    float val;
    int id;
};
__global__ void argmax_pack_kernel(const float *vals, const int *ids, TpArgmaxPair *out, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows)
        out[r] = TpArgmaxPair{vals[r], ids[r]};
}
__global__ void argmax_reduce_kernel(const TpArgmaxPair *src, int stride_pairs, int tp_size, int rows, int *out_ids) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows)
        return;
    TpArgmaxPair best = src[r];
    for (int t = 1; t < tp_size; ++t) {
        TpArgmaxPair cur = src[size_t(t) * stride_pairs + r];
        if (cur.val > best.val || (cur.val == best.val && cur.id < best.id))
            best = cur;
    }
    out_ids[r] = best.id;
}
__global__ void scatter_logits_kernel(const float *src, float *dst, int rows, int local_vocab, int full_vocab, int offset) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(rows) * local_vocab;
    if (idx >= total)
        return;
    int r = int(idx / local_vocab);
    int c = int(idx - size_t(r) * local_vocab);
    dst[size_t(r) * full_vocab + offset + c] = src[idx];
}
__global__ void scatter_logits_vec4_kernel(const float4 *src, float4 *dst, int rows, int local_v4, int full_v4, int off_v4) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(rows) * local_v4;
    if (idx >= total)
        return;
    int r = int(idx / local_v4);
    int c = int(idx - size_t(r) * local_v4);
    dst[size_t(r) * full_v4 + off_v4 + c] = src[idx];
}
void tp_argmax_pack(const float *vals, const int *ids, void *out_pairs, int rows, hipStream_t stream) {
    if (rows <= 0)
        return;
    argmax_pack_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        vals, ids, static_cast<TpArgmaxPair *>(out_pairs), rows);
    checked(hipGetLastError());
}
void tp_argmax_reduce(const void *src_pairs, int stride_pairs, int tp_size, int rows, int *out_ids, hipStream_t stream) {
    if (rows <= 0)
        return;
    argmax_reduce_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        static_cast<const TpArgmaxPair *>(src_pairs), stride_pairs, tp_size, rows, out_ids);
    checked(hipGetLastError());
}
void tp_scatter_logits(const float *src, float *dst, int rows, int local_vocab, int full_vocab, int offset, hipStream_t stream) {
    if (rows <= 0 || local_vocab <= 0)
        return;
    if ((local_vocab & 3) == 0 && (full_vocab & 3) == 0 && (offset & 3) == 0) {
        size_t total = size_t(rows) * (local_vocab / 4);
        scatter_logits_vec4_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const float4 *>(src), reinterpret_cast<float4 *>(dst),
            rows, local_vocab / 4, full_vocab / 4, offset / 4);
    } else {
        size_t total = size_t(rows) * local_vocab;
        scatter_logits_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            src, dst, rows, local_vocab, full_vocab, offset);
    }
    checked(hipGetLastError());
}
Mat Ops::mat(int rows, int cols) {
    if (rows < 1 || cols < 1)
        throw std::runtime_error("空の作業行列です");
    return {static_cast<uint16_t *>(arena.next(size_t(rows) * cols * 2)), rows, cols};
}
Param Ops::param(const std::string &key) const {
    const auto &w = gpu.model.at(key, rank);
    if (w.rank != rank)
        throw std::runtime_error("重みの rank が違います: " + key);
    return {gpu.device(w), w.spec.dtype};
}
Mat Ops::linear(const std::string &name, Mat x) {
    auto it = gpu.model.find(name + ".trellis", rank);
    if (!it) {
        const auto &w = gpu.model.at(name + ".weight", rank);
        if (w.spec.shape.size() != 2 || w.spec.shape[1] != size_t(x.cols) ||
            (w.spec.dtype != io::SType::F32 && w.spec.dtype != io::SType::BF16))
            throw std::runtime_error("線形の形状・型が不正です: " + name);
        Mat y = mat(x.rows, w.spec.shape[0]);
        Param wp = param(name + ".weight");
        if (wp.type == io::SType::BF16 && x.rows >= 8 && (x.cols % 64) == 0 && (y.cols % 32) == 0) {
            dense_tiled<8, 8, 2, 4><<<dim3(y.cols / 32, (x.rows + 15) / 16), 256, 0, stream>>>(
                x, static_cast<const uint16_t *>(wp.p), y);
        } else if (wp.type == io::SType::BF16 && x.rows >= 8 && (x.cols % 64) == 0 && y.cols == 16) {
            dense_tiled<2, 8, 4, 2><<<dim3(1, (x.rows + 7) / 8), 256, 0, stream>>>(
                x, static_cast<const uint16_t *>(wp.p), y);
        } else {
            dense<<<dim3((y.cols + 3) / 4, y.rows), 128, 0, stream>>>(x, wp, y);
        }
        checked(hipGetLastError());
        return y;
    }
    auto logical = gpu.model.shape_of(name, rank) ? *gpu.model.shape_of(name, rank)
                                                  : std::array<int, 2>{1536, 1536};
    if (x.cols != logical[0])
        throw std::runtime_error("線形入力の幅が異なります: " + name);
    Mat y = mat(x.rows, logical[1]);
    size_t save = arena.cursor;
    int pk = it->spec.shape[0] * 16, pn = it->spec.shape[1] * 16;
    auto codes = static_cast<int8_t *>(arena.next(size_t(x.rows) * pk));
    auto scales = static_cast<float *>(arena.next(size_t(x.rows) * pk / 128 * 4));
    int splits = 1;
    while (splits < 16 && (pn / 64) * splits < 128 && (pk / 128) % (2 * splits) == 0)
        splits *= 2;
    auto partial = static_cast<float *>(arena.next(size_t(x.rows) * pn * 4 * splits));
    auto expanded = x.rows >= 256 ? static_cast<int8_t *>(arena.next(size_t(pk) * pn)) : nullptr;
    checked(shelf_hy4_linear_workspace(x.p, static_cast<const uint16_t *>(param(name + ".suh").p),
                             static_cast<const uint16_t *>(param(name + ".svh").p),
                             static_cast<const uint32_t *>(param(name + ".trellis").p), codes, scales, partial,
                             y.p, x.rows, x.cols, y.cols, pk, pn, it->spec.shape[2] / 16, stream,
                             splits, expanded));
    if (gpu.model.row_parallel(name, rank))
        gpu.tp_sum(y.p, rank, y.bytes());
    arena.cursor = save;
    return y;
}
Mat Ops::norm(Mat x, const std::string &key, float eps) {
    Mat y = mat(x.rows, x.cols);
    Param w = key.empty() ? Param{} : param(key);
    normalize<<<x.rows, 256, 0, stream>>>(x, w, y, eps);
    checked(hipGetLastError());
    return y;
}
Mat Ops::norm_from(Mat x, int first, int width, const std::string &key, float eps) {
    Mat y = mat(x.rows, width);
    Param w = key.empty() ? Param{} : param(key);
    normalize_from<<<x.rows, 256, 0, stream>>>(x, w, y, eps, first, width);
    checked(hipGetLastError());
    return y;
}
Mat Ops::norm_bias(Mat x, const uint16_t *bias, const std::string &key, float eps) {
    Mat y = mat(x.rows, x.cols);
    Param w = key.empty() ? Param{} : param(key);
    norm_bias_kernel<<<x.rows, 256, 0, stream>>>(x, bias, w, y, eps);
    checked(hipGetLastError());
    return y;
}
Mat Ops::norm_slice(Mat x, int first, int width, const std::string &key, float eps) {
    Mat y = mat(x.rows, width);
    Param w = key.empty() ? Param{} : param(key);
    norm_slice_kernel<<<x.rows, 256, 0, stream>>>(x, w, y, eps, first);
    checked(hipGetLastError());
    return y;
}
Mat Ops::slice(Mat x, int first, int width) {
    Mat y = mat(x.rows, width);
    select_cols<<<grid(y), 256, 0, stream>>>(x, y, first);
    checked(hipGetLastError());
    return y;
}
Mat Ops::join(Mat a, Mat b) {
    Mat y = mat(a.rows, a.cols + b.cols);
    join_cols<<<grid(y), 256, 0, stream>>>(a, b, y);
    checked(hipGetLastError());
    return y;
}
Mat Ops::add(Mat a, Mat b) {
    Mat y = mat(a.rows, a.cols);
    element<<<grid(y), 256, 0, stream>>>(a, b, y, 0);
    checked(hipGetLastError());
    return y;
}
Mat Ops::swiglu(Mat a, Mat b, bool clip) {
    Mat y = mat(a.rows, a.cols);
    element<<<grid(y), 256, 0, stream>>>(a, b, y, clip ? 2 : 1);
    checked(hipGetLastError());
    return y;
}
int *Ops::indices(const std::vector<int> &ids) {
    auto p = static_cast<int *>(arena.next(ids.size() * sizeof(int)));
    copy(p, ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice);
    return p;
}
void *Ops::reserve_persistent(size_t bytes) {
    size_t aligned = (bytes + 255) & ~size_t(255);
    if (aligned > arena.size || arena.cursor > arena.size - aligned)
        throw std::runtime_error("作業域の容量が不足しています");
    arena.size -= aligned;
    return arena.base + arena.size;
}
void Ops::copy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind) {
    checked(hipMemcpyAsync(dst, src, bytes, kind, stream));
    checked(hipStreamSynchronize(stream));
}
Mat Ops::embed(const std::vector<int> &ids, bool expand) {
    for (int id : ids)
        if (id < 0 || id >= 120832)
            throw std::runtime_error("token ID が範囲外です");
    return embed_device(indices(ids), ids.size(), expand);
}
Mat Ops::embed_device(const int *list, int rows, bool expand) {
    const auto &w = gpu.model.at("model.embed_tokens.weight", rank);
    if (w.spec.shape != std::vector<size_t>{120832, 3072} || w.spec.dtype != io::SType::BF16)
        throw std::runtime_error("embedding の形状・型が異なります");
    Mat x{static_cast<uint16_t *>(const_cast<void *>(gpu.device(w))), 120832, 3072};
    Mat y = mat(rows, expand ? 12288 : 3072);
    gather_rows<<<grid(y), 256, 0, stream>>>(x, list, y, expand ? 4 : 1);
    checked(hipGetLastError());
    return y;
}
static void launch_hc_mix(Mat x, Param fn, Param scale, Param base, float *post, int count, hipStream_t stream) {
    if (x.rows >= 256)
        hc_mix_multi<4><<<dim3(1, (x.rows + 3) / 4), 256, 0, stream>>>(x, fn, scale, base, post, count);
    else
        hc_mix<<<dim3(count, x.rows), 256, 0, stream>>>(x, fn, scale, base, post, count);
}
Mat Ops::collapse(Mat x, const std::string &prefix, float *&post) {
    int count = prefix == "model.hc_head.hc_head_" ? 4 : 8;
    post = static_cast<float *>(arena.next(size_t(x.rows) * count * 4));
    Mat y = mat(x.rows, 3072);
    launch_hc_mix(x, param(prefix + "fn"), param(prefix + "scale"), param(prefix + "base"), post, count, stream);
    checked(hipGetLastError());
    hc_reduce<<<grid(y), 256, 0, stream>>>(x, y, post, count);
    checked(hipGetLastError());
    return y;
}
Mat Ops::collapse_norm(Mat x, const std::string &prefix, float *&post, const std::string &key, float eps) {
    int count = prefix == "model.hc_head.hc_head_" ? 4 : 8;
    post = static_cast<float *>(arena.next(size_t(x.rows) * count * 4));
    Mat y = mat(x.rows, 3072);
    Param w = key.empty() ? Param{} : param(key);
    launch_hc_mix(x, param(prefix + "fn"), param(prefix + "scale"), param(prefix + "base"), post, count, stream);
    checked(hipGetLastError());
    hc_reduce_norm<<<x.rows, 256, 0, stream>>>(x, post, w, y, eps, count);
    checked(hipGetLastError());
    return y;
}
Mat Ops::expand(Mat x, Mat y, const float *post) {
    if ((y.cols & 7) == 0 && (x.cols & 7) == 0)
        hc_expand_vec<<<((size_t(x.rows) * x.cols) / 8 + 255) / 256, 256, 0, stream>>>(x, y, post);
    else
        hc_expand<<<grid(x), 256, 0, stream>>>(x, y, post);
    checked(hipGetLastError());
    return x;
}
void Ops::add_into(Mat a, Mat b, Mat out) {
    element<<<grid(a), 256, 0, stream>>>(a, b, out, 0);
    checked(hipGetLastError());
}
Mat Ops::rope_batch(Mat x, int heads, float theta, const int *row_positions) {
    Mat y = mat(x.rows, x.cols);
    if (x.cols / heads == 256 && (heads & 3) == 0)
        rotary_tiled<256><<<dim3(heads / 4, x.rows), 128, 0, stream>>>(x, y, heads, 0, theta, nullptr, row_positions);
    else if (x.cols / heads == 128 && (heads & 7) == 0)
        rotary_tiled<128><<<dim3(heads / 8, x.rows), 128, 0, stream>>>(x, y, heads, 0, theta, nullptr, row_positions);
    else
        rotary<<<grid(y), 256, 0, stream>>>(x, y, heads, 0, theta, nullptr, row_positions);
    checked(hipGetLastError());
    return y;
}
Mat Ops::rope_from_batch(Mat x, int first, int width, int heads, float theta, const int *row_positions) {
    Mat y = mat(x.rows, width);
    rotary_from<<<grid(y), 256, 0, stream>>>(x, y, heads, 0, theta, first, nullptr, row_positions);
    checked(hipGetLastError());
    return y;
}
Mat Ops::rope_from(Mat x, int first, int width, int heads, int start, float theta) {
    Mat y = mat(x.rows, width);
    rotary_from<<<grid(y), 256, 0, stream>>>(x, y, heads, start, theta, first, params);
    checked(hipGetLastError());
    return y;
}
Mat Ops::rope(Mat x, int heads, int start, float theta) {
    Mat y = mat(x.rows, x.cols);
    if (x.cols / heads == 256 && (heads & 3) == 0)
        rotary_tiled<256><<<dim3(heads / 4, x.rows), 128, 0, stream>>>(x, y, heads, start, theta, params);
    else if (x.cols / heads == 128 && (heads & 7) == 0)
        rotary_tiled<128><<<dim3(heads / 8, x.rows), 128, 0, stream>>>(x, y, heads, start, theta, params);
    else
        rotary<<<grid(y), 256, 0, stream>>>(x, y, heads, start, theta, params);
    checked(hipGetLastError());
    return y;
}
}
