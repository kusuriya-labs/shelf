#include <shelf/models/hy4_flash/ops.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <cmath>
#include <stdexcept>

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
__global__ void normalize(Mat x, Param weight, Mat y, float eps) {
    __shared__ float scratch[32];
    size_t base = size_t(blockIdx.x) * x.cols;
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
__global__ void rotary_from(Mat x, Mat y, int heads, int start, float theta, int first) {
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
    float phase = (start + row) * powf(theta, -float(j / 2) / 32);
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
    if (i < size_t(y.rows) * y.cols)
        y.p[i] = x.p[size_t(ids[i / (repeat * x.cols)]) * x.cols + i % x.cols];
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
__global__ void hc_expand(Mat x, Mat y, const float *post) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int r = i / x.cols, h = (i / y.cols) % 4;
    x.p[i] = f32_to_bf16_rne_device(
        __fadd_rn(value(x, i), __fmul_rn(post[r * 8 + 4 + h], value(y, size_t(r) * y.cols + i % y.cols))));
}
__global__ void rotary(Mat x, Mat y, int heads, int start, float theta) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int row = i / x.cols, width = x.cols / heads, c = i % width;
    if (c < width - 64) {
        y.p[i] = x.p[i];
        return;
    }
    int j = c - (width - 64);
    float phase = (start + row) * powf(theta, -float(j / 2) / 32);
    float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
    float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
    y.p[i] = f32_to_bf16_rne_device(value(x, i) * co + ((j & 1) ? 1.f : -1.f) * value(x, i ^ 1) * si);
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
void peer_copy(void *dst, const void *src, size_t bytes) {
    if (!bytes)
        return;
    size_t threads = (bytes + 15) / 16;
    peer_copy_kernel<<<(threads + 255) / 256, 256>>>(static_cast<uint8_t *>(dst),
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
void tp_add_bf16(void *dst, const void *src, size_t bytes) {
    if (!bytes)
        return;
    size_t count = bytes / 2;
    add_bf16_kernel<<<(count + 255) / 256, 256>>>(static_cast<uint16_t *>(dst),
                                                  static_cast<const uint16_t *>(src), count);
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
        dense<<<dim3((y.cols + 3) / 4, y.rows), 128>>>(x, param(name + ".weight"), y);
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
    if (x.rows <= 16)
        while (splits < 16 && (pn / 64) * splits < 128 && (pk / 128) % (2 * splits) == 0)
            splits *= 2;
    auto partial = static_cast<float *>(arena.next(size_t(x.rows) * pn * 4 * splits));
    checked(shelf_hy4_linear(x.p, static_cast<const uint16_t *>(param(name + ".suh").p),
                             static_cast<const uint16_t *>(param(name + ".svh").p),
                             static_cast<const uint32_t *>(param(name + ".trellis").p), codes, scales, partial,
                             y.p, x.rows, x.cols, y.cols, pk, pn, it->spec.shape[2] / 16, nullptr,
                             splits));
    if (gpu.model.row_parallel(name, rank))
        gpu.tp_sum(y.p, rank, y.bytes());
    arena.cursor = save;
    return y;
}
Mat Ops::norm(Mat x, const std::string &key, float eps) {
    Mat y = mat(x.rows, x.cols);
    Param w = key.empty() ? Param{} : param(key);
    normalize<<<x.rows, 256>>>(x, w, y, eps);
    checked(hipGetLastError());
    return y;
}
Mat Ops::norm_from(Mat x, int first, int width, const std::string &key, float eps) {
    Mat y = mat(x.rows, width);
    Param w = key.empty() ? Param{} : param(key);
    normalize_from<<<x.rows, 256>>>(x, w, y, eps, first, width);
    checked(hipGetLastError());
    return y;
}
Mat Ops::slice(Mat x, int first, int width) {
    Mat y = mat(x.rows, width);
    select_cols<<<grid(y), 256>>>(x, y, first);
    checked(hipGetLastError());
    return y;
}
Mat Ops::join(Mat a, Mat b) {
    Mat y = mat(a.rows, a.cols + b.cols);
    join_cols<<<grid(y), 256>>>(a, b, y);
    checked(hipGetLastError());
    return y;
}
Mat Ops::add(Mat a, Mat b) {
    Mat y = mat(a.rows, a.cols);
    element<<<grid(y), 256>>>(a, b, y, 0);
    checked(hipGetLastError());
    return y;
}
Mat Ops::swiglu(Mat a, Mat b, bool clip) {
    Mat y = mat(a.rows, a.cols);
    element<<<grid(y), 256>>>(a, b, y, clip ? 2 : 1);
    checked(hipGetLastError());
    return y;
}
int *Ops::indices(const std::vector<int> &ids) {
    auto p = static_cast<int *>(arena.next(ids.size() * sizeof(int)));
    copy(p, ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice);
    return p;
}
void Ops::copy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind) {
    if (kind == hipMemcpyDeviceToHost)
        checked(hipStreamSynchronize(nullptr));
    checked(hipMemcpy(dst, src, bytes, kind));
}
Mat Ops::embed(const std::vector<int> &ids, bool expand) {
    const auto &w = gpu.model.at("model.embed_tokens.weight", rank);
    if (w.spec.shape != std::vector<size_t>{120832, 3072} || w.spec.dtype != io::SType::BF16)
        throw std::runtime_error("embedding の形状・型が異なります");
    for (int id : ids)
        if (id < 0 || id >= 120832)
            throw std::runtime_error("token ID が範囲外です");
    Mat x{static_cast<uint16_t *>(const_cast<void *>(gpu.device(w))), 120832, 3072};
    Mat y = mat(ids.size(), expand ? 12288 : 3072);
    int *list = indices(ids);
    gather_rows<<<grid(y), 256>>>(x, list, y, expand ? 4 : 1);
    checked(hipGetLastError());
    return y;
}
Mat Ops::collapse(Mat x, const std::string &prefix, float *&post) {
    int count = prefix == "model.hc_head.hc_head_" ? 4 : 8;
    post = static_cast<float *>(arena.next(size_t(x.rows) * count * 4));
    Mat y = mat(x.rows, 3072);
    hc_mix<<<dim3(count, x.rows), 256>>>(x, param(prefix + "fn"), param(prefix + "scale"),
                                         param(prefix + "base"), post, count);
    checked(hipGetLastError());
    hc_reduce<<<grid(y), 256>>>(x, y, post, count);
    checked(hipGetLastError());
    return y;
}
Mat Ops::expand(Mat x, Mat y, const float *post) {
    hc_expand<<<grid(x), 256>>>(x, y, post);
    checked(hipGetLastError());
    return x;
}
void Ops::add_into(Mat a, Mat b, Mat out) {
    element<<<grid(a), 256>>>(a, b, out, 0);
    checked(hipGetLastError());
}
Mat Ops::rope_from(Mat x, int first, int width, int heads, int start, float theta) {
    Mat y = mat(x.rows, width);
    rotary_from<<<grid(y), 256>>>(x, y, heads, start, theta, first);
    checked(hipGetLastError());
    return y;
}
Mat Ops::rope(Mat x, int heads, int start, float theta) {
    Mat y = mat(x.rows, x.cols);
    rotary<<<grid(y), 256>>>(x, y, heads, start, theta);
    checked(hipGetLastError());
    return y;
}
}
