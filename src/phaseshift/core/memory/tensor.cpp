#include <phaseshift/core/memory/tensor.h>
#include <cstdint>
#include <limits>

namespace ps {
namespace gpu {

Status Tensor::compute_required_span_bytes(
    const std::size_t* shape,
    std::size_t ndim,
    const std::size_t* strides,
    std::size_t elem_size,
    std::size_t& out) {
    if (ndim == 0) {
        out = 0;
        return Status::make_ok();
    }

    std::size_t max_offset = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        std::size_t last_idx = shape[i] - 1;
        std::size_t elem_offset;
        auto s = checked_mul(last_idx, strides[i], elem_offset, __FILE__, __LINE__);
        if (!s.ok()) return s;

        s = checked_add(max_offset, elem_offset, __FILE__, __LINE__);
        if (!s.ok()) return s;
        max_offset += elem_offset;
    }

    auto s = checked_add(max_offset, std::size_t(1), __FILE__, __LINE__);
    if (!s.ok()) return s;
    std::size_t count = max_offset + 1;

    std::size_t byte_span = 0;
    s = checked_mul(count, elem_size, byte_span, __FILE__, __LINE__);
    if (!s.ok()) return s;

    out = byte_span;
    return Status::make_ok();
}

Result<Tensor> Tensor::slice(std::size_t dim, std::size_t start, std::size_t count) const {
    if (dim >= ndim_) {
        return Status::out_of_range("slice dim out of range", __FILE__, __LINE__);
    }

    if (count == 0) {
        return Status::invalid_argument("slice count must be non-zero", __FILE__, __LINE__);
    }

    if (start >= shape_[dim]) {
        return Status::out_of_range("slice start exceeds dimension", __FILE__, __LINE__);
    }

    if (count > shape_[dim] - start) {
        return Status::out_of_range("slice range exceeds dimension", __FILE__, __LINE__);
    }

    Tensor sliced;
    sliced.allocation_base_ = allocation_base_;
    sliced.allocation_bytes_ = allocation_bytes_;
    for (std::size_t i = 0; i < ndim_; ++i) {
        sliced.shape_[i] = shape_[i];
        sliced.strides_[i] = strides_[i];
    }
    sliced.ndim_ = ndim_;
    sliced.elem_size_ = elem_size_;
    sliced.physical_device_ = physical_device_;

    std::size_t stride_bytes = 0;
    auto s = checked_mul(strides_[dim], elem_size_, stride_bytes, __FILE__, __LINE__);
    if (!s.ok()) return s;

    std::size_t elem_offset = 0;
    s = checked_mul(start, stride_bytes, elem_offset, __FILE__, __LINE__);
    if (!s.ok()) return s;

    s = checked_add(view_offset_bytes_, elem_offset, __FILE__, __LINE__);
    if (!s.ok()) return s;
    std::size_t new_view_offset = view_offset_bytes_ + elem_offset;
    sliced.view_offset_bytes_ = new_view_offset;

    sliced.shape_[dim] = count;

    std::size_t span;
    s = sliced.required_span_bytes(span);
    if (!s.ok()) return s;

    s = checked_add(sliced.view_offset_bytes_, span, __FILE__, __LINE__);
    if (!s.ok()) return s;
    std::size_t end_offset = sliced.view_offset_bytes_ + span;

    if (end_offset > sliced.allocation_bytes_) {
        return Status::out_of_range("slice exceeds allocation", __FILE__, __LINE__);
    }

    return sliced;
}

} // namespace gpu
} // namespace ps
