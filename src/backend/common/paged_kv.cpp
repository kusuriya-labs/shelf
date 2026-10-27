// shelf backend/common — PagedKVLayout validation.
#include <shelf/backend/common/paged_kv.hpp>

namespace shelf::backend::common {

Status PagedKVLayout::check() const {
    if (block_size == 0) return Status::invalid_argument("paged-kv: block_size 0", __FILE__, __LINE__);
    if (kv_heads == 0 || head_dim == 0)
        return Status::invalid_argument("paged-kv: kv_heads/head_dim 0", __FILE__, __LINE__);
    if (num_blocks == 0)
        return Status::invalid_argument("paged-kv: pool has no blocks", __FILE__, __LINE__);
    const DataType dt = dtype;
    if (dt != DataType::bf16 && dt != DataType::f32 && dt != DataType::f64 &&
        dt != DataType::u8) {  // u8 = fp8 e4m3 payload later
        return Status::unsupported("paged-kv: unsupported KV dtype", __FILE__, __LINE__);
    }
    if (slot_stride == 0 || head_stride == 0 || block_stride == 0)
        return Status::invalid_argument("paged-kv: strides unset (call prepare)",
                                        __FILE__, __LINE__);
    // packing は SeparatePools のみ（InterleavedKV は削除済み）。
    if (head_stride < head_dim)
        return Status::invalid_argument("paged-kv: head_stride < head_dim",
                                        __FILE__, __LINE__);
    if (slot_stride < kv_heads * head_stride)
        return Status::invalid_argument("paged-kv: slot_stride overlaps heads",
                                        __FILE__, __LINE__);
    if (block_stride < block_size * slot_stride)
        return Status::invalid_argument("paged-kv: block_stride overlaps slots",
                                        __FILE__, __LINE__);
    return Status::make_ok();
}

Status PagedKVLayout::prepare() {
    const bool all_zero = block_stride == 0 && head_stride == 0 && slot_stride == 0;
    if (all_zero) {
        head_stride = head_dim;
        slot_stride = kv_heads * head_stride;
        block_stride = block_size * slot_stride;
    }
    return check();
}

} // namespace shelf::backend::common
