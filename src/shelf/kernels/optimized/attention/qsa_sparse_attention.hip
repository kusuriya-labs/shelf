#include <shelf/kernels/optimized/attention/qsa_sparse_attention.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = __hip_bfloat16;

__global__ __launch_bounds__(256)
void qsa_sparse_attention_kernel(QsaSparseAttentionArgs a) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = a.rows * a.q_heads;
    if (idx >= total) return;
    const uint32_t row = idx / a.q_heads;
    const uint32_t h = idx % a.q_heads;
    const uint32_t group = a.q_heads / a.kv_heads;
    const uint32_t kv_head = h / group;
    const uint32_t head_dim = a.head_dim;

    if (a.row_sequence_slots[row] >= a.max_sequences) return;
    const uint32_t slot = a.row_sequence_slots[row];
    const uint32_t pos = a.row_positions[row];
    const uint32_t* block_table = a.block_tables + static_cast<size_t>(slot) * a.block_table_stride;
    const bf16* q = reinterpret_cast<const bf16*>(a.q) +
                    static_cast<size_t>(row) * a.q_row_stride +
                    static_cast<size_t>(h) * head_dim;
    float qv[256];
    for (uint32_t d = 0; d < head_dim; ++d) qv[d] = __bfloat162float(q[d]);

    const uint8_t* mask_row = a.mask + static_cast<size_t>(row) * a.mask_row_stride;
    const bf16* k_pool = reinterpret_cast<const bf16*>(a.k_pool);
    const bf16* v_pool = reinterpret_cast<const bf16*>(a.v_pool);

    float max_score = -3.402823466e+38f;
    float sum_exp = 0.0f;
    float acc[256];
    for (uint32_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    for (uint32_t pass = 0; pass < 2u; ++pass) {
        sum_exp = 0.0f;
        if (pass == 1u) {
            for (uint32_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
        }
        for (uint32_t p = 0; p <= pos; ++p) {
            if (mask_row[p] == 0u) continue;
            const uint32_t lb = p / a.page_tokens;
            const uint32_t off = p % a.page_tokens;
            if (lb >= a.block_table_stride) return;
            const uint32_t page_id = block_table[lb];
            if (page_id >= a.num_pages) return;
            const size_t kb = static_cast<size_t>(a.layer) * a.elems_per_layer +
                              static_cast<size_t>(page_id) * a.elems_per_page +
                              static_cast<size_t>(off) * a.elems_per_token +
                              static_cast<size_t>(kv_head) * head_dim;
            float sc = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d)
                sc += qv[d] * __bfloat162float(k_pool[kb + d]);
            sc *= a.scale;
            if (pass == 0u) {
                if (sc > max_score) max_score = sc;
            } else {
                const float pv = __expf(sc - max_score);
                sum_exp += pv;
                for (uint32_t d = 0; d < head_dim; ++d)
                    acc[d] += pv * __bfloat162float(v_pool[kb + d]);
            }
        }
    }

    const size_t out_base = static_cast<size_t>(row) * a.output_row_stride +
                            static_cast<size_t>(h) * head_dim;
    if (a.output_dtype == QsaSparseAttentionOutputDType::F32) {
        float* out = static_cast<float*>(a.output);
        const float inv = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) out[out_base + d] = acc[d] * inv;
    } else {
        bf16* out = reinterpret_cast<bf16*>(a.output);
        const float inv = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) out[out_base + d] = __float2bfloat16(acc[d] * inv);
    }
}

}

hipError_t launch_qsa_sparse_attention_bf16(const QsaSparseAttentionArgs& a) {
    if (a.rows == 0u) return hipSuccess;
    if (a.q == nullptr || a.output == nullptr || a.mask == nullptr ||
        a.row_sequence_slots == nullptr || a.row_positions == nullptr ||
        a.block_tables == nullptr || a.k_pool == nullptr || a.v_pool == nullptr)
        return hipErrorInvalidValue;
    if (a.q_heads == 0u || a.kv_heads == 0u || a.q_heads % a.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (a.head_dim == 0u || a.head_dim > 256u) return hipErrorNotSupported;
    if (a.q_row_stride < a.q_heads * a.head_dim) return hipErrorInvalidValue;
    if (a.page_tokens == 0u || a.num_pages == 0u || a.block_table_stride == 0u ||
        a.num_attention_layers == 0u || a.layer >= a.num_attention_layers)
        return hipErrorInvalidValue;
    if (a.elems_per_token == 0u || a.elems_per_page == 0u || a.elems_per_layer == 0u)
        return hipErrorInvalidValue;
    if (a.mask_row_stride == 0u || a.output_row_stride == 0u) return hipErrorInvalidValue;
    if (a.scale == 0.0f) return hipErrorInvalidValue;

    const uint64_t total = static_cast<uint64_t>(a.rows) * a.q_heads;
    const uint32_t threads = 256u;
    const uint32_t grid = static_cast<uint32_t>((total + threads - 1u) / threads);
    qsa_sparse_attention_kernel<<<grid, threads, 0, a.stream>>>(a);
    return hipGetLastError();
}

}
