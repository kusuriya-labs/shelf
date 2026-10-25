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
    const std::size_t canonical_slot = 2 * head_dim;
    if (slot_stride == 0 || head_stride == 0 || block_stride == 0)
        return Status::invalid_argument("paged-kv: strides unset (call prepare)",
                                        __FILE__, __LINE__);
    if (slot_stride < canonical_slot)
        return Status::invalid_argument("paged-kv: slot_stride must fit K+V",
                                        __FILE__, __LINE__);
    if (head_stride < block_size * slot_stride)
        return Status::invalid_argument("paged-kv: head_stride overlaps slots",
                                        __FILE__, __LINE__);
    if (block_stride < kv_heads * head_stride)
        return Status::invalid_argument("paged-kv: block_stride overlaps heads",
                                        __FILE__, __LINE__);
    return Status::make_ok();
}

Status PagedKVLayout::prepare() {
    const bool all_zero = block_stride == 0 && head_stride == 0 && slot_stride == 0;
    if (all_zero) {
        slot_stride = 2 * head_dim;
        head_stride = block_size * slot_stride;
        block_stride = kv_heads * head_stride;
    }
    return check();
}

} // namespace shelf::backend::common
