#include <phaseshift/kernels/optimized/attention/qsa_indexer.h>
#include <phaseshift/kernels/optimized/attention/rope.h>
#include <phaseshift/kernels/optimized/rmsnorm.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kIndexerThreads = 256u;

struct QsaScratchLayout {
    size_t q_normed = 0;
    size_t q_rope = 0;
    size_t pooled = 0;
    size_t k_normed = 0;
    size_t block_keys = 0;
    size_t block_positions = 0;
    size_t total = 0;
};

struct QsaSelectArgs {
    const bf16* q_rope = nullptr;
    const bf16* block_keys = nullptr;
    uint32_t query_begin = 0;
    uint32_t query_rows = 0;
    uint32_t key_tokens = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t block_topk = 0;
    float score_scale = 1.0f;
    float* scores = nullptr;
    uint32_t scores_stride = 0;
    uint32_t* selected_blocks = nullptr;
    uint32_t* selected_block_counts = nullptr;
    uint8_t* mask = nullptr;
};

size_t align_up(size_t v, size_t a) {
    return (v + a - 1u) / a * a;
}

QsaScratchLayout make_layout(uint32_t query_rows, uint32_t key_tokens, uint32_t num_heads,
                             uint32_t head_dim, uint32_t compress_ratio) {
    QsaScratchLayout l;
    const size_t n_blocks = static_cast<size_t>(key_tokens / compress_ratio);
    const size_t q_elems = static_cast<size_t>(query_rows) * num_heads * head_dim;
    size_t off = 0;
    l.q_normed = off;
    off += align_up(q_elems * sizeof(float), 16u);
    l.q_rope = off;
    off += align_up(q_elems * sizeof(bf16_t), 16u);
    l.pooled = off;
    off += align_up(n_blocks * head_dim * sizeof(float), 16u);
    l.k_normed = off;
    off += align_up(n_blocks * head_dim * sizeof(float), 16u);
    l.block_keys = off;
    off += align_up(n_blocks * head_dim * sizeof(bf16_t), 16u);
    l.block_positions = off;
    off += align_up(n_blocks * sizeof(uint32_t), 16u);
    l.total = off;
    return l;
}

__global__ void qsa_block_positions_kernel(const uint32_t* __restrict__ key_positions,
                                           uint32_t key_tokens, uint32_t compress_ratio,
                                           uint32_t n_blocks, uint32_t* __restrict__ out) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint32_t start = b * compress_ratio;
    if (start >= key_tokens) return;
    out[b] = key_positions[start];
}

__global__ void qsa_block_pool_kernel(const bf16* __restrict__ key_cache, uint32_t key_tokens,
                                      uint32_t compress_ratio, uint32_t head_dim,
                                      uint32_t n_blocks, float* __restrict__ out) {
    const uint32_t b = blockIdx.x;
    if (b >= n_blocks) return;
    const uint32_t start = b * compress_ratio;
    if (start + compress_ratio > key_tokens) return;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < compress_ratio; ++t) {
            const size_t idx = static_cast<size_t>(start + t) * head_dim + d;
            acc += __bfloat162float(key_cache[idx]);
        }
        out[static_cast<size_t>(b) * head_dim + d] = acc / static_cast<float>(compress_ratio);
    }
}

__global__ void qsa_select_kernel(QsaSelectArgs a) {
    const uint32_t r = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t i = a.query_begin + r;
    const uint32_t visible = i + 1u;
    const uint32_t ncb = visible / a.compress_ratio;
    const uint32_t tail_begin = ncb * a.compress_ratio;
    const uint32_t k = a.block_topk < ncb ? a.block_topk : ncb;
    const uint32_t q_features = a.num_heads * a.head_dim;
    const bf16* q_row = a.q_rope + static_cast<size_t>(r) * q_features;
    float* row_scores = a.scores + static_cast<size_t>(r) * a.scores_stride;

    for (uint32_t b = tid; b < ncb; b += blockDim.x) {
        const bf16* bk = a.block_keys + static_cast<size_t>(b) * a.head_dim;
        float acc = 0.0f;
        for (uint32_t h = 0; h < a.num_heads; ++h) {
            const bf16* qh = q_row + static_cast<size_t>(h) * a.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < a.head_dim; ++d) {
                dot += __bfloat162float(qh[d]) * __bfloat162float(bk[d]);
            }
            acc += dot > 0.0f ? dot : 0.0f;
        }
        row_scores[b] = acc * a.score_scale;
    }
    for (uint32_t b = ncb + tid; b < a.scores_stride; b += blockDim.x) row_scores[b] = 0.0f;
    __syncthreads();

    for (uint32_t b = tid; b < ncb; b += blockDim.x) {
        const float sb = row_scores[b];
        uint32_t rank = 0u;
        for (uint32_t bp = 0; bp < ncb; ++bp) {
            const float sp = row_scores[bp];
            if (sp > sb || (sp == sb && bp < b)) ++rank;
        }
        if (rank < k) {
            a.selected_blocks[static_cast<size_t>(r) * a.block_topk + rank] = b;
        }
    }
    if (tid == 0u) a.selected_block_counts[r] = k;
    __syncthreads();

    for (uint32_t t = tid; t < k * a.compress_ratio; t += blockDim.x) {
        const uint32_t blk = t / a.compress_ratio;
        const uint32_t off = t % a.compress_ratio;
        const uint32_t tok =
            a.selected_blocks[static_cast<size_t>(r) * a.block_topk + blk] * a.compress_ratio + off;
        a.mask[static_cast<size_t>(r) * a.key_tokens + tok] = 1u;
    }
    for (uint32_t tok = tail_begin + tid; tok <= i && tok < a.key_tokens;
         tok += blockDim.x) {
        a.mask[static_cast<size_t>(r) * a.key_tokens + tok] = 1u;
    }
}

}

uint32_t qsa_indexer_block_count(uint32_t key_tokens, uint32_t compress_ratio) noexcept {
    if (compress_ratio == 0u) return 0u;
    return key_tokens / compress_ratio;
}

uint32_t qsa_indexer_scores_stride(uint32_t key_tokens, uint32_t compress_ratio) noexcept {
    return qsa_indexer_block_count(key_tokens, compress_ratio);
}

size_t qsa_indexer_scratch_bytes(uint32_t query_rows, uint32_t key_tokens, uint32_t num_heads,
                                 uint32_t head_dim, uint32_t compress_ratio) noexcept {
    if (compress_ratio == 0u || num_heads == 0u || head_dim == 0u) return 0u;
    return make_layout(query_rows, key_tokens, num_heads, head_dim, compress_ratio).total;
}

hipError_t launch_qsa_indexer(const QsaIndexerArgs& a) {
    if (a.query_rows == 0u) return hipSuccess;
    if (a.q_raw == nullptr || a.key_cache == nullptr || a.key_positions == nullptr ||
        a.q_weight == nullptr || a.k_weight == nullptr)
        return hipErrorInvalidValue;
    if (a.num_heads == 0u || a.head_dim < kQsaIndexerRotaryDim || a.compress_ratio == 0u ||
        a.block_topk == 0u)
        return hipErrorNotSupported;
    if (a.key_tokens == 0u || a.query_begin + a.query_rows > a.key_tokens)
        return hipErrorInvalidValue;
    const uint32_t n_blocks = qsa_indexer_block_count(a.key_tokens, a.compress_ratio);
    if (a.scores_stride < n_blocks) return hipErrorInvalidValue;
    if (a.scores == nullptr || a.selected_blocks == nullptr ||
        a.selected_block_counts == nullptr || a.mask == nullptr)
        return hipErrorInvalidValue;
    const QsaScratchLayout l = make_layout(a.query_rows, a.key_tokens, a.num_heads, a.head_dim,
                                           a.compress_ratio);
    if (a.scratch == nullptr || a.scratch_bytes < l.total) return hipErrorInvalidValue;

    char* base = static_cast<char*>(a.scratch);
    float* q_normed = reinterpret_cast<float*>(base + l.q_normed);
    bf16_t* q_rope = reinterpret_cast<bf16_t*>(base + l.q_rope);
    float* pooled = reinterpret_cast<float*>(base + l.pooled);
    float* k_normed = reinterpret_cast<float*>(base + l.k_normed);
    bf16_t* block_keys = reinterpret_cast<bf16_t*>(base + l.block_keys);
    uint32_t* block_positions = reinterpret_cast<uint32_t*>(base + l.block_positions);

    const uint32_t q_features = a.num_heads * a.head_dim;
    float scale = a.score_scale;
    if (!(scale > 0.0f)) scale = 1.0f / sqrtf(static_cast<float>(a.head_dim));

    hipError_t err = launch_rmsnorm(
        a.q_raw, RmsNormDataType::BF16, a.q_weight, RmsNormDataType::BF16,
        RmsNormWeightLayout::PER_GROUP, RmsNormWeightMode::ONE_PLUS, q_normed,
        RmsNormDataType::F32, a.query_rows, q_features, a.head_dim, q_features, q_features,
        a.eps, a.stream);
    if (err != hipSuccess) return err;

    err = launch_rope_f32_bf16_generic(
        q_normed, q_features, q_rope, q_features, a.key_positions + a.query_begin, a.query_rows,
        q_features, a.head_dim, kQsaIndexerRotaryDim, a.rope_theta, nullptr, 1.0f, a.stream);
    if (err != hipSuccess) return err;

    if (n_blocks > 0u) {
        const uint32_t grid = (n_blocks + kIndexerThreads - 1u) / kIndexerThreads;
        qsa_block_positions_kernel<<<grid, kIndexerThreads, 0, a.stream>>>(
            a.key_positions, a.key_tokens, a.compress_ratio, n_blocks, block_positions);
        err = hipGetLastError();
        if (err != hipSuccess) return err;

        qsa_block_pool_kernel<<<n_blocks, kIndexerThreads, 0, a.stream>>>(
            reinterpret_cast<const bf16*>(a.key_cache), a.key_tokens, a.compress_ratio,
            a.head_dim, n_blocks, pooled);
        err = hipGetLastError();
        if (err != hipSuccess) return err;

        err = launch_rmsnorm(
            pooled, RmsNormDataType::F32, a.k_weight, RmsNormDataType::BF16,
            RmsNormWeightLayout::PER_GROUP, RmsNormWeightMode::ONE_PLUS, k_normed,
            RmsNormDataType::F32, n_blocks, a.head_dim, a.head_dim, a.head_dim, a.head_dim,
            a.eps, a.stream);
        if (err != hipSuccess) return err;

        err = launch_rope_f32_bf16_generic(
            k_normed, a.head_dim, block_keys, a.head_dim, block_positions, n_blocks, a.head_dim,
            a.head_dim, kQsaIndexerRotaryDim, a.rope_theta, nullptr, 1.0f, a.stream);
        if (err != hipSuccess) return err;
    }

    const size_t mask_bytes = static_cast<size_t>(a.query_rows) * a.key_tokens;
    err = hipMemsetAsync(a.mask, 0, mask_bytes, a.stream);
    if (err != hipSuccess) return err;

    QsaSelectArgs s;
    s.q_rope = reinterpret_cast<const bf16*>(q_rope);
    s.block_keys = reinterpret_cast<const bf16*>(block_keys);
    s.query_begin = a.query_begin;
    s.query_rows = a.query_rows;
    s.key_tokens = a.key_tokens;
    s.num_heads = a.num_heads;
    s.head_dim = a.head_dim;
    s.compress_ratio = a.compress_ratio;
    s.block_topk = a.block_topk;
    s.score_scale = scale;
    s.scores = a.scores;
    s.scores_stride = a.scores_stride;
    s.selected_blocks = a.selected_blocks;
    s.selected_block_counts = a.selected_block_counts;
    s.mask = a.mask;
    qsa_select_kernel<<<a.query_rows, kIndexerThreads, 0, a.stream>>>(s);
    return hipGetLastError();
}

}
