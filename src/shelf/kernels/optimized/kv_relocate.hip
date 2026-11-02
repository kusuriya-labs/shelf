#include <shelf/kernels/optimized/kv_relocate.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kKvRelocateThreads = 128u;

template <typename Pool>
__global__ void kv_relocate_kernel(const KvRelocateArgs a, Pool* k_pool, Pool* v_pool,
                                   float* k_scale_pool, float* v_scale_pool) {
    const uint32_t layer = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t part = blockIdx.z;
    if (layer >= a.num_attention_layers || head >= a.kv_heads) return;
    if (a.slot >= a.max_sequences) return;
    if (a.src_positions == nullptr || a.dst_positions == nullptr ||
        a.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return;

    const uint32_t* bt =
        a.block_tables + static_cast<std::size_t>(a.slot) * a.block_table_stride;
    const uint64_t layer_base = static_cast<uint64_t>(layer) * a.elems_per_layer;
    const uint64_t head_off = static_cast<uint64_t>(head) * a.head_dim;
    Pool* pool = (part == 0u) ? k_pool : v_pool;
    float* scales = (part == 0u) ? k_scale_pool : v_scale_pool;

    for (uint32_t i = 0; i < a.rows; ++i) {
        const uint32_t src_pos = a.src_positions[i];
        const uint32_t dst_pos = a.dst_positions[i];
        const uint32_t src_page_idx = src_pos / a.page_tokens;
        const uint32_t dst_page_idx = dst_pos / a.page_tokens;
        if (src_page_idx >= a.block_table_stride ||
            dst_page_idx >= a.block_table_stride)
            break;
        const uint32_t src_page = bt[src_page_idx];
        const uint32_t dst_page = bt[dst_page_idx];
        if (src_page >= a.num_pages || dst_page >= a.num_pages) break;
        if (src_pos != dst_pos) {
            const uint32_t src_off = src_pos % a.page_tokens;
            const uint32_t dst_off = dst_pos % a.page_tokens;
            const uint64_t src_elem = layer_base +
                                      static_cast<uint64_t>(src_page) * a.elems_per_page +
                                      static_cast<uint64_t>(src_off) * a.elems_per_token +
                                      head_off;
            const uint64_t dst_elem = layer_base +
                                      static_cast<uint64_t>(dst_page) * a.elems_per_page +
                                      static_cast<uint64_t>(dst_off) * a.elems_per_token +
                                      head_off;
            for (uint32_t d = threadIdx.x; d < a.head_dim; d += blockDim.x) {
                pool[dst_elem + d] = pool[src_elem + d];
            }
            if (scales != nullptr && threadIdx.x == 0u) {
                const std::size_t src_si =
                    ((static_cast<std::size_t>(layer) * a.num_pages + src_page) *
                         a.page_tokens +
                     src_off) *
                        a.kv_heads +
                    head;
                const std::size_t dst_si =
                    ((static_cast<std::size_t>(layer) * a.num_pages + dst_page) *
                         a.page_tokens +
                     dst_off) *
                        a.kv_heads +
                    head;
                scales[dst_si] = scales[src_si];
            }
        }
        __syncthreads();
    }
}

}  // namespace

hipError_t launch_kv_relocate_bf16(const KvRelocateBf16Args& args, hipStream_t stream) {
    const KvRelocateArgs& a = args.common;
    if (a.rows == 0u || a.head_dim == 0u || a.kv_heads == 0u) return hipErrorInvalidValue;
    if (a.page_tokens == 0u || a.block_table_stride == 0u || a.num_pages == 0u ||
        a.num_attention_layers == 0u)
        return hipErrorInvalidValue;
    if (a.elems_per_token == 0u || a.elems_per_page == 0u || a.elems_per_layer == 0u)
        return hipErrorInvalidValue;
    if (a.src_positions == nullptr || a.dst_positions == nullptr ||
        a.block_tables == nullptr || args.k_pool == nullptr || args.v_pool == nullptr)
        return hipErrorInvalidValue;
    if (a.slot >= a.max_sequences) return hipErrorInvalidValue;
    const dim3 grid(a.num_attention_layers, a.kv_heads, 2u);
    kv_relocate_kernel<bf16_t><<<grid, kKvRelocateThreads, 0, stream>>>(
        a, reinterpret_cast<bf16_t*>(args.k_pool), reinterpret_cast<bf16_t*>(args.v_pool),
        nullptr, nullptr);
    return hipGetLastError();
}

hipError_t launch_kv_relocate_fp8_e4m3(const KvRelocateFp8Args& args, hipStream_t stream) {
    const KvRelocateArgs& a = args.common;
    if (a.rows == 0u || a.head_dim == 0u || a.kv_heads == 0u) return hipErrorInvalidValue;
    if (a.page_tokens == 0u || a.block_table_stride == 0u || a.num_pages == 0u ||
        a.num_attention_layers == 0u)
        return hipErrorInvalidValue;
    if (a.elems_per_token == 0u || a.elems_per_page == 0u || a.elems_per_layer == 0u)
        return hipErrorInvalidValue;
    if (a.src_positions == nullptr || a.dst_positions == nullptr ||
        a.block_tables == nullptr || args.k_pool == nullptr || args.v_pool == nullptr ||
        args.k_scale_pool == nullptr || args.v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    if (a.slot >= a.max_sequences) return hipErrorInvalidValue;
    const dim3 grid(a.num_attention_layers, a.kv_heads, 2u);
    kv_relocate_kernel<::ps::kernel::fp8e4m3_storage_t><<<grid, kKvRelocateThreads, 0, stream>>>(
        a, args.k_pool, args.v_pool, args.k_scale_pool, args.v_scale_pool);
    return hipGetLastError();
}

}  // namespace ps::kernel
