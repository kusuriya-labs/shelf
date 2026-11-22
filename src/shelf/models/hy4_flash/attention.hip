#include <hip/hip_fp16.h>
#include <shelf/models/hy4_flash/mla_prefill.h>
#include <shelf/models/hy4_flash/attention.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/kernels/fused/detail/octave_common.h>
#include <shelf/kernels/fused/detail/radix_select.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
using shelf::kernel::detail::octave::sign_of;
__device__ int step_at(int value, const Hy4StepParams *params) { return params ? params->position : value; }
__device__ int kv_base(int first, int row_offset, bool sliding, const Hy4StepParams *params) {
    if (!params)
        return first;
    return sliding ? max(0, params->position + row_offset - 511) : 0;
}
__device__ constexpr float levels[] = {-2.73258948f, -2.06901717f, -1.61804640f, -1.25623119f,
                                       -0.94234043f, -0.65675914f, -0.38804829f, -0.12839504f,
                                       0.12839504f,  0.38804829f,  0.65675914f,  0.94234043f,
                                       1.25623119f,  1.61804640f,  2.06901717f,  2.73258948f};
__device__ float read(Param p, size_t i) {
    return p.type == io::SType::F32 ? static_cast<const float *>(p.p)[i]
                                    : bf16_bits_to_f32(static_cast<const uint16_t *>(p.p)[i]);
}
__device__ float bv(Mat x, int row, int col) { return bf16_bits_to_f32(x.p[size_t(row) * x.cols + col]); }
__device__ float wht(float v, float *scratch, int width) {
    int c = threadIdx.x;
    for (int step = 1; step < width; step *= 2) {
        scratch[c] = v;
        __syncthreads();
        float other = scratch[c ^ step];
        __syncthreads();
        v = (c & step) ? other - v : v + other;
    }
    return v * rsqrtf(float(width));
}
__device__ float wht_shared(float v, float *scratch, int width, int c, bool active) {
    for (int step = 1; step < width; step *= 2) {
        scratch[c] = active ? v : 0.f;
        __syncthreads();
        float other = scratch[c ^ step];
        __syncthreads();
        if (active)
            v = (c & step) ? other - v : v + other;
    }
    return v * rsqrtf(float(width));
}
template <bool Max> __device__ float reduce(float v, float *scratch, int width) {
    int c = threadIdx.x;
    scratch[c] = v;
    __syncthreads();
    for (int step = width / 2; step; step /= 2) {
        if (c < step)
            scratch[c] = Max ? fmaxf(scratch[c], scratch[c + step]) : scratch[c] + scratch[c + step];
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}
template <bool Max> __device__ float reduce_shared(float v, float *scratch, int width, int c, bool active) {
    scratch[c] = active ? v : (Max ? -INFINITY : 0.f);
    __syncthreads();
    for (int step = width / 2; step; step /= 2) {
        if (c < step && c < width)
            scratch[c] = Max ? fmaxf(scratch[c], scratch[c + step]) : scratch[c] + scratch[c + step];
        __syncthreads();
    }
    float result = active ? scratch[0] : 0.f;
    __syncthreads();
    return result;
}
template <int Width, int Salt>
__device__ void pack_part(Mat x, uint8_t *records, int offset, int sub, int row, int c, bool active, float *scratch,
                          uint8_t *codes) {
    uint8_t *dst = records + size_t(row) * 294 + offset + sub * (Width / 2 + 2);
    float v = wht_shared(active ? bv(x, row, sub * Width + c) * sign_of(c, Salt) : 0.f, scratch, Width, c, active);
    float ss = reduce_shared<false>(v * v, scratch, Width, c, active);
    float z = ss > 0 ? v * rsqrtf(ss / Width) : 0;
    int lo = 0, hi = 15;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (z > (levels[mid] + levels[mid + 1]) * 0.5f)
            lo = mid + 1;
        else
            hi = mid;
    }
    codes[c] = active && ss > 0 ? lo : 0;
    float dot = reduce_shared<false>(active ? v * levels[lo] : 0.f, scratch, Width, c, active);
    if (active && c < Width / 2)
        dst[c] = codes[c * 2] | (codes[c * 2 + 1] << 4);
    if (active && !c) {
        uint16_t h = __half_as_ushort(__float2half(dot > 0 ? ss / dot : 0));
        dst[Width / 2] = h & 255;
        dst[Width / 2 + 1] = h >> 8;
    }
}
__global__ void pack_kv_combined(Mat lat, Mat rot, uint8_t *records) {
    __shared__ float scratch[256];
    __shared__ uint8_t codes[256];
    int sub = blockIdx.x, row = blockIdx.y, c = threadIdx.x;
    if (sub < 2)
        pack_part<256, 29>(lat, records, 0, sub, row, c, true, scratch, codes);
    else
        pack_part<64, 11>(rot, records, 260, 0, row, c, c < 64, scratch, codes);
}
template <int Width, int Salt>
__global__ void unpack(const uint8_t *cache, const uint8_t *fresh, const int *ids, int first, int start, int slots,
                       Mat out, int offset, const Hy4StepParams *params, bool sliding, int row_offset, int chunk_rows) {
    __shared__ float scratch[Width];
    start = step_at(start, params);
    int base = kv_base(first, row_offset, sliding, params);
    int c = threadIdx.x, sub = blockIdx.x, row = blockIdx.y, id = ids ? ids[row] : base + row;
    if (!ids && id >= start + row_offset + chunk_rows) {
        out.p[size_t(row) * out.cols + sub * Width + c] = 0;
        return;
    }
    if (id < 0) {
        out.p[size_t(row) * out.cols + sub * Width + c] = 0;
        return;
    }
    const uint8_t *rec = id >= start ? fresh + size_t(id - start) * 294 : cache + size_t(id % slots) * 294;
    const uint8_t *src = rec + offset + sub * (Width / 2 + 2);
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[Width / 2] | (src[Width / 2 + 1] << 8)));
    float value = wht(levels[code] * scale, scratch, Width) * sign_of(c, Salt);
    out.p[size_t(row) * out.cols + sub * Width + c] = f32_to_bf16_rne_device(value);
}
__global__ void save_records(uint8_t *cache, const uint8_t *fresh, int rows, int start, int slots,
                             const Hy4StepParams *params) {
    start = step_at(start, params);
    int i = (rows - min(rows, slots)) * 294 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 294 * rows)
        cache[size_t((start + i / 294) % slots) * 294 + i % 294] = fresh[i];
}
__global__ void pool(Mat keys, uint16_t *history, int first, int start, Param values, Param gates, Param ape,
                     Param norm, const uint16_t *pending, const Hy4StepParams *params, int rows) {
    __shared__ float pooled[128], scratch[128];
    start = step_at(start, params);
    first = start / 4;
    if (blockIdx.x >= (start + rows) / 4 - first)
        return;
    int c = threadIdx.x, group = first + blockIdx.x;
    float v[4], g[4], peak = -INFINITY;
    for (int i = 0; i < 4; ++i) {
        int pos = 4 * group + i;
        auto x = pos >= start ? keys.p + size_t(pos - start) * 128 : pending + size_t(pos % 12) * 128;
        v[i] = 0;
        g[i] = read(ape, i * 128 + c);
        for (int j = 0; j < 128; ++j) {
            float a = bf16_bits_to_f32(x[j]);
            v[i] += a * read(values, c * 128 + j);
            g[i] += a * read(gates, c * 128 + j);
        }
        peak = fmaxf(peak, g[i]);
    }
    float sum = 0, total = 0;
    for (int i = 0; i < 4; ++i) {
        float p = expf(g[i] - peak);
        sum += p * v[i];
        total += p;
    }
    float out = sum / total;
    float inv = rsqrtf(reduce<false>(out * out, scratch, 128) / 128 + 1e-5f);
    pooled[c] = bf16_bits_to_f32(f32_to_bf16_rne_device(out * inv * read(norm, c)));
    __syncthreads();
    out = pooled[c];
    if (c >= 64) {
        float phase = group * 4 * powf(1e7f, -float((c - 64) / 2) / 32);
        float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
        float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
        out = out * co + pooled[c ^ 1] * ((c & 1) ? si : -si);
    }
    history[size_t(group) * 128 + c] = f32_to_bf16_rne_device(out);
}
__global__ void save_pending(Mat k, int start, uint16_t *pending, const Hy4StepParams *params) {
    start = step_at(start, params);
    int row = k.rows - 1 - blockIdx.x;
    pending[((start + row) % 12) * 128 + threadIdx.x] = k.p[row * 128 + threadIdx.x];
}
__global__ void index_scores(Mat q, Mat weights, const uint16_t *groups, int first, int start, int stride,
                             float *scores, const Hy4StepParams *params) {
    start = step_at(start, params);
    int row = first + blockIdx.y;
    int count = (start + row) / 4;
    int group = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (group >= count || group >= stride)
        return;
    float sum = 0;
    for (int h = 0; h < 16; ++h) {
        float dot = 0;
        for (int j = lane; j < 128; j += 32)
            dot += bv(q, row, h * 128 + j) * bf16_bits_to_f32(groups[size_t(group) * 128 + j]);
        for (int dist = 16; dist; dist /= 2)
            dot += __shfl_down(dot, dist, 32);
        sum += fmaxf(dot, 0) * bv(weights, row, h);
    }
    if (!lane)
        scores[size_t(blockIdx.y) * stride + group] = sum;
}
__global__ void index_topk(const float *scores, int first, int start, int stride, int *selected,
                           const Hy4StepParams *params) {
    __shared__ uint32_t histogram[256], state[4];
    start = step_at(start, params);
    int row = blockIdx.x;
    unsigned count = (start + first + row) / 4;
    count = min(count, (unsigned)stride);
    ::shelf::kernel::detail::radix_select<3>(scores + size_t(row) * stride, count, 511, histogram, state,
                                             selected + row * 511, nullptr, 0);
}
__global__ void contiguous_offsets(int *offsets, int n, int stride, int pos0, bool sliding, int first, int row_offset,
                                   const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= stride)
        return;
    int pos = step_at(pos0, params) + row_offset + r;
    int base = kv_base(first, row_offset, sliding, params);
    int len = sliding ? min(pos + 1, 512) : pos + 1;
    int first_r = pos + 1 - len;
    offsets[r * stride + t] = t < len ? first_r - base + t : -1;
}
__global__ void draft_contiguous_offsets(int *ids, int *offsets, int stride, bool sliding,
                                         const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= stride)
        return;
    int start = params->position;
    int history = sliding ? min(start, 512) : start;
    if (!r)
        ids[t] = t < history + int(gridDim.y) ? start - history + t : -1;
    offsets[r * stride + t] = t <= history + r ? t : -1;
}
__global__ void sparse_clear(int *mask, int mask_count, unsigned *bits, int word_count, int *meta, int *rev,
                             int rev_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mask_count)
        mask[i] = 0;
    if (i < word_count)
        bits[i] = 0;
    if (i < 516)
        meta[i] = 0;
    if (i < rev_count)
        rev[i] = -1;
}
__global__ void sort_top(const int *top, int *groups, int *mask, int row_begin, int rows, int width,
                         const Hy4StepParams *draft = nullptr) {
    __shared__ int g[512];
    int r = blockIdx.x, t = threadIdx.x;
    if (draft)
        row_begin = draft->draft_last;
    g[t] = t < 511 ? top[(size_t(row_begin + r)) * 511 + t] : INT_MAX;
    __syncthreads();
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            int other = g[t ^ j], v = g[t];
            bool lower = (t & j) == 0;
            bool gt = lower ? v > other : other > v;
            bool lt = lower ? v < other : other < v;
            bool swap = (t & k) == 0 ? gt : lt;
            __syncthreads();
            g[t] = swap ? other : v;
        }
    }
    __syncthreads();
    groups[size_t(r) * 512 + t] = g[t];
    if (t < 511 && g[t] < width)
        mask[size_t(r) * width + g[t]] = 1;
}
__global__ void sparse_group_ids(int *mask, int rows, int width, int *meta, int *ids, int *rev,
                                 unsigned *bits, int pos0, int row_offset, const Hy4StepParams *params) {
    pos0 = step_at(pos0, params) + row_offset;
    for (int g = threadIdx.x; g < width; g += blockDim.x) {
        int v = 0;
        for (int r = 0; r < rows; ++r)
            v |= mask[size_t(r) * width + g];
        mask[g] = v;
    }
    __syncthreads();
    if (!threadIdx.x) {
        int count = 0;
        for (int g = 0; g < width; ++g) {
            int present = mask[g];
            mask[g] = present ? count * 4 : -1;
            count += present;
        }
        meta[512] = count * 4;
    }
    __syncthreads();
    for (int g = threadIdx.x; g < width; g += blockDim.x)
        if (mask[g] >= 0)
            for (int k = 0; k < 4; ++k) {
                int token = 4 * g + k;
                ids[mask[g] + k] = token;
                rev[token] = mask[g] + k;
            }
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        int pos = pos0 + r, base = pos & ~3;
        for (int k = base; k <= pos; ++k)
            atomicOr(bits + (k >> 5), 1u << (k & 31));
    }
}
__global__ void sparse_tail_emit(const unsigned *bits, int *ids, int *rev, int *meta, int lo, int hi, bool dedup,
                                 int start, int row_offset, int rows, const Hy4StepParams *params) {
    int pos = step_at(start, params) + row_offset;
    if (dedup)
        lo = 2048, hi = pos + rows;
    else
        lo = pos & ~3, hi = lo + (pos & 3) + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x, token = lo + i;
    if (token >= hi)
        return;
    if ((!dedup || rev[token] < 0) && (bits[token >> 5] & (1u << (token & 31)))) {
        int p = meta[512] + (dedup ? atomicAdd(meta + 513, 1) : i);
        if (!dedup && i == 0)
            meta[513] = hi - lo;
        ids[p] = token;
        rev[token] = p;
    }
}
__global__ void draft_extra_ids(int start, int rows, int *ids, int *rev, int *meta,
                                const Hy4StepParams *params) {
    int i = threadIdx.x;
    start = step_at(start, params);
    if (i < rows) {
        int p = meta[512] + meta[513] + i, token = start + i;
        ids[p] = token;
        rev[token] = p;
    }
    __syncthreads();
    if (!i)
        atomicAdd(meta + 513, rows);
}
__global__ void sparse_offsets(const int *groups, const int *rev, int *offsets, int rows, int stride, int pos0,
                               int *ids, int *meta, int max_ids, int row_offset, const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    pos0 = step_at(pos0, params) + row_offset;
    if (t < max_ids && t >= meta[512] + meta[513])
        ids[t] = -1;
    if (t >= stride)
        return;
    int token;
    if (t < 2044) {
        int g = groups[size_t(r) * 512 + t / 4];
        token = 4 * g + t % 4;
    } else {
        int pos = pos0 + r;
        token = (pos & ~3) + (t - 2044);
        if (token > pos)
            token = -1;
    }
    offsets[r * stride + t] = token >= 0 && rev[token] >= 0 ? rev[token] : -1;
}
__global__ void draft_offsets(int *offsets, int rows, int stride, int base_len, int *ids, int *meta, int max_ids,
                              int start, const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = step_at(start, params) - 1;
    base_len = 2044 + (pos & 3) + 1;
    if (t < max_ids && t >= meta[512] + meta[513])
        ids[t] = -1;
    if (t >= stride)
        return;
    offsets[r * stride + t] = t < base_len + r + 1 ? t : -1;
}
Selection select(Ops &op, const std::string &p, int layer, int start, Mat x, Mat qres) {
    int end = start + x.rows;
    int bucket = op.max_kv_len > 0 ? op.max_kv_len : std::max(2048, (end + 63) / 64 * 64);
    Selection selected(x.rows);
    bool sliding = layer != 0 && layer % 4 != 1;
    int begin = sliding ? x.rows : std::min(x.rows, std::max(0, 2048 - start));
    if (!sliding) {
        Mat k = op.norm(op.linear(p + "token_key_projection", x), p + "token_key_norm.weight");
        auto groups = op.gpu.index_keys[op.rank][layer];
        auto pending = groups + (op.gpu.tokens / 4) * 128;
        int first = start / 4, ng = end / 4 - first;
        int pool_blocks = op.params ? 2 : ng;
        if (pool_blocks) {
            pool<<<pool_blocks, 128, 0, op.stream>>>(k, groups, first, start, op.param(p + "key_value_projection.weight"),
                                            op.param(p + "key_gate_projection.weight"),
                                            op.param(p + "intra_group_ape"), op.param(p + "key_norm.weight"),
                                            pending, op.params, x.rows);
            checked(hipGetLastError());
        }
        save_pending<<<std::min(k.rows, 12), 128, 0, op.stream>>>(k, start, pending, op.params);
        checked(hipGetLastError());
        if (end > 2048) {
            Mat q = op.rope(op.linear(p + "query_projection", qres), 16, start, 1e7f);
            Mat w = op.linear(p + "head_weight_projection", x);
            int stride = std::max(512, (bucket - 1) / 4);
            int batch = std::min(16, x.rows - begin);
            auto buffer = static_cast<float *>(op.arena.next(size_t(stride) * batch * 4));
            if (!op.gpu.sparse_top[op.rank][layer])
                op.gpu.sparse_top[op.rank][layer] =
                    static_cast<int *>(op.reserve_persistent(size_t(max_prefill_chunk) * 511 * 4));
            auto top = op.gpu.sparse_top[op.rank][layer];
            op.sparse_last_layer[layer] = x.rows - 1;
            for (int row = begin; row < x.rows; row += batch) {
                int rows = std::min(batch, x.rows - row);
                index_scores<<<dim3((stride + 3) / 4, rows), 128, 0, op.stream>>>(q, w, groups, row, start, stride,
                                                                                  buffer, op.params);
                checked(hipGetLastError());
                index_topk<<<rows, 256, 0, op.stream>>>(buffer, row, start, stride, top + size_t(row) * 511,
                                                        op.params);
                checked(hipGetLastError());
            }
        }
    }
    for (int i = 0; i < begin; ++i) {
        int pos = start + i, from = sliding ? std::max(0, pos - 511) : 0;
        selected[i].resize(pos - from + 1);
        std::iota(selected[i].begin(), selected[i].end(), from);
    }
    return selected;
}
}
Mat attention(Ops &op, const std::string &prefix, int layer, int start, Mat hidden, Selection &selected,
              bool draft) {
    std::string p = prefix + "self_attn.";
    int bucket = op.max_kv_len > 0 ? op.max_kv_len
                                   : std::max(2048, (start + hidden.rows + 63) / 64 * 64);
    bool sliding = layer != 0 && layer % 4 != 1;
    const auto *projection = op.gpu.model.shape_of(p + "q_b_proj", op.rank);
    int heads = projection ? projection->at(1) / 256 : (sliding ? 64 : 32);
    float theta = sliding ? 1e4f : 1e7f;
    Mat qres = op.norm(op.linear(p + "q_a_proj", hidden), p + "q_a_layernorm.weight");
    Mat q = op.rope(op.linear(p + "q_b_proj", qres), heads, start, theta);
    Mat ka = op.linear(p + "kv_a_proj_with_mqa", hidden);
    Mat lat = op.norm_from(ka, 0, 512, p + "kv_a_layernorm.weight");
    Mat rot = op.rope_from(ka, 512, 64, 1, start, theta);
    Mat gate = op.linear(p + "linear_gate", hidden);
    Param sinks = op.param(p + "learnable_sink_param");
    if (op.gpu.tp_size > 1 && !draft) {
        const int first = (op.rank % op.gpu.tp_size) * heads;
        sinks.p = sinks.type == io::SType::F32
                      ? static_cast<const void *>(static_cast<const float *>(sinks.p) + first)
                      : static_cast<const void *>(static_cast<const uint16_t *>(sinks.p) + first);
    }
    auto fresh = static_cast<uint8_t *>(op.arena.next(size_t(hidden.rows) * 294));
    pack_kv_combined<<<dim3(3, hidden.rows), 256, 0, op.stream>>>(lat, rot, fresh);
    checked(hipGetLastError());
    if (!draft)
        selected = select(op, p + "compressed_indexer.", layer, start, hidden, qres);
    Mat out = op.mat(hidden.rows, heads * 256);
    int slots = sliding ? std::min(op.gpu.tokens, size_t(518)) : op.gpu.tokens;
    auto cache = op.gpu.cache[op.rank][layer];
    bool contiguous = sliding || (draft ? start <= 2048 : start + hidden.rows <= 2048);
    int begin = std::min(hidden.rows, std::max(0, 2048 - start));
    for (int i = 0; i < hidden.rows;) {
        size_t save = op.arena.cursor;
        bool chunk_contiguous = contiguous;
        int n = hidden.rows - i;
        if (!contiguous && !draft) {
            if (i < begin) {
                chunk_contiguous = true;
                n = std::min(n, begin - i);
            } else if (start + hidden.rows > 4096)
                n = std::min(n, 16);
        }
        int stride = 0;
        for (int r = i; r < i + n && !selected.empty(); ++r)
            stride = std::max(stride, int(selected[r].size()));
        bool tiles = n >= 16 && chunk_contiguous && !draft;
        bool dynamic_sizes = !op.params && (tiles || (chunk_contiguous && n == 1));
        if (!dynamic_sizes)
            stride = chunk_contiguous ? (sliding ? 512 : 2048) : 2048;
        int first = chunk_contiguous && !selected.empty() ? selected[i][0] : 0;
        int count = chunk_contiguous && !selected.empty() ? selected[i + n - 1].back() + 1 - first : 0;
        int ids_count = dynamic_sizes ? count
                                      : (chunk_contiguous ? (sliding ? 518 : 2048) : (2044 * n + 4 * n + 8));
        int *ids_gpu = nullptr, *offsets_gpu = nullptr, *groups = nullptr;
        int *mask = nullptr, *meta = nullptr, *rev = nullptr;
        unsigned *bits = nullptr;
        if (draft && !chunk_contiguous && op.gpu.sparse_top[op.rank][layer]) {
            int last = op.sparse_last_layer[layer], pos = start - 1;
            int tail_len = (pos & 3) + 1, tail_base = pos & ~3;
            int base_len = 2044 + tail_len;
            stride = (op.params ? 2048 : base_len) + n;
            int width = op.params ? bucket / 4 + 2 : (pos + 3) / 4 + 2;
            ids_count = std::min(4 * width, 2044) + 4 + n + 8;
            int capacity = op.params ? bucket : start + n;
            int words = (capacity + 31) / 32, rev_count = capacity + 8;
            groups = static_cast<int *>(op.arena.next(size_t(512) * 4));
            mask = static_cast<int *>(op.arena.next(size_t(width) * 4));
            bits = static_cast<unsigned *>(op.arena.next(size_t(words) * 4));
            meta = static_cast<int *>(op.arena.next(size_t(516) * 4));
            rev = static_cast<int *>(op.arena.next(size_t(rev_count) * 4));
            ids_gpu = static_cast<int *>(op.arena.next(size_t(ids_count) * 4));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * 4));
            sparse_clear<<<(std::max({width, words, rev_count, 516}) + 255) / 256, 256, 0, op.stream>>>(
                mask, width, bits, words, meta, rev, rev_count);
            sort_top<<<1, 512, 0, op.stream>>>(op.gpu.sparse_top[op.rank][layer], groups, mask, last, 1, width, op.params);
            sparse_group_ids<<<1, 256, 0, op.stream>>>(mask, 1, width, meta, ids_gpu, rev, bits,
                                                                       start, -1, op.params);
            sparse_tail_emit<<<(tail_len + 255) / 256, 256, 0, op.stream>>>(bits, ids_gpu, rev, meta, tail_base,
                                                                            tail_base + tail_len, false, start, -1, n,
                                                                            op.params);
            draft_extra_ids<<<1, 64, 0, op.stream>>>(start, n, ids_gpu, rev, meta, op.params);
            draft_offsets<<<dim3((std::max(stride, ids_count) + 255) / 256, n), 256, 0, op.stream>>>(
                offsets_gpu, n, stride, base_len, ids_gpu, meta, ids_count, start, op.params);
        } else if (draft && op.params) {
            ids_count = stride = sliding ? 518 : 2054;
            ids_gpu = static_cast<int *>(op.arena.next(size_t(stride) * sizeof(int)));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * sizeof(int)));
            draft_contiguous_offsets<<<dim3((stride + 255) / 256, n), 256, 0, op.stream>>>(
                ids_gpu, offsets_gpu, stride, sliding, op.params);
        } else if (draft) {
            stride = 0;
            for (int r = i; r < i + n; ++r)
                stride = std::max(stride, int(selected[r].size()));
            std::vector<int> ids, offsets(n * stride, -1);
            ids = selected[i + n - 1];
            for (int r = 0; r < n; ++r)
                std::iota(offsets.begin() + r * stride, offsets.begin() + r * stride + selected[i + r].size(), 0);
            ids_gpu = op.indices(ids);
            offsets_gpu = op.indices(offsets);
            ids_count = int(ids.size());
        } else if (chunk_contiguous) {
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * 4));
            contiguous_offsets<<<dim3((stride + 255) / 256, n), 256, 0, op.stream>>>(
                offsets_gpu, n, stride, start, sliding, first, i, op.params);
        } else {
            int rows = n, pos0 = start + i;
            stride = 2048;
            int width = bucket / 4 + 8;
            ids_count = std::min(4 * width, 2044 * rows) + 4 * rows + 8;
            int words = (bucket + 31) / 32;
            int rev_count = bucket + 8;
            groups = static_cast<int *>(op.arena.next(size_t(rows) * 512 * 4));
            mask = static_cast<int *>(op.arena.next(size_t(rows) * width * 4));
            bits = static_cast<unsigned *>(op.arena.next(size_t(words) * 4));
            meta = static_cast<int *>(op.arena.next(size_t(516) * 4));
            rev = static_cast<int *>(op.arena.next(size_t(rev_count) * 4));
            ids_gpu = static_cast<int *>(op.arena.next(size_t(ids_count) * 4));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(rows) * stride * 4));
            int mask_count = rows * width;
            sparse_clear<<<(std::max({mask_count, words, rev_count, 516}) + 255) / 256, 256, 0, op.stream>>>(
                mask, mask_count, bits, words, meta, rev, rev_count);
            sort_top<<<rows, 512, 0, op.stream>>>(op.gpu.sparse_top[op.rank][layer], groups, mask, i, rows, width);
            sparse_group_ids<<<1, 256, 0, op.stream>>>(mask, rows, width, meta, ids_gpu, rev, bits,
                                                                       start, i, op.params);
            sparse_tail_emit<<<(bucket + rows - 2048 + 255) / 256, 256, 0, op.stream>>>(
                bits, ids_gpu, rev, meta, 2048, pos0 + rows, true, start, i, rows, op.params);
            sparse_offsets<<<dim3((std::max(stride, ids_count) + 255) / 256, rows), 256, 0, op.stream>>>(
                groups, rev, offsets_gpu, rows, stride, start, ids_gpu, meta, ids_count, i, op.params);
        }
        checked(hipGetLastError());
        Mat a = op.mat(ids_count, 512), b = op.mat(ids_count, 64);
        unpack<256, 29><<<dim3(2, ids_count), 256, 0, op.stream>>>(cache, fresh, ids_gpu, first, start, slots, a, 0,
                                                                    op.params, sliding, i, n);
        checked(hipGetLastError());
        unpack<64, 11><<<dim3(1, ids_count), 64, 0, op.stream>>>(cache, fresh, ids_gpu, first, start, slots, b, 260,
                                                                  op.params, sliding, i, n);
        checked(hipGetLastError());
        if (tiles) {
            Mat expanded = op.linear(p + "kv_b_proj", a);
            size_t size = size_t(n) * heads * ((ids_count + 15) / 16 * 16);
            auto scores = static_cast<float *>(op.arena.next(size * 4));
            auto probs = static_cast<uint16_t *>(op.arena.next(size * 2));
            checked(shelf_hy4_mla_prefill(q.row(i, n).p, expanded.p, b.p, gate.row(i, n).p, sinks.p,
                                          sinks.type == io::SType::F32, out.row(i, n).p, scores, probs, n, heads,
                                          ids_count, first, start, sliding, op.stream,
                                           op.params ? &op.params->position : nullptr, i, op.params,
                                           chunk_contiguous, n));
        } else
            attend_absorbed(op, p + "kv_b_proj", a, b, q.row(i, n), gate.row(i, n), offsets_gpu, stride,
                            sinks, out.row(i, n), chunk_contiguous, sliding, i, n);
        op.arena.cursor = save;
        i += n;
    }
    if (!draft) {
        save_records<<<(std::min(hidden.rows, slots) * 294 + 255) / 256, 256, 0, op.stream>>>(
            cache, fresh, hidden.rows, start, slots, op.params);
        checked(hipGetLastError());
    }
    return op.norm(op.linear(p + "o_proj", out), p + "post_proj_norm.weight");
}
}
