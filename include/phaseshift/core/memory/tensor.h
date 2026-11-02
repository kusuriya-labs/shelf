#pragma once

#include <phaseshift/core/memory/device_allocation_view.h>
#include <phaseshift/core/status.h>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ps {
namespace gpu {

constexpr std::size_t kMaxTensorDims = 8;

class Tensor {
 public:
    template <typename T>
    static Result<Tensor> create(
        const DeviceAllocationView& backing,
        const std::initializer_list<std::size_t>& shape,
        const std::initializer_list<std::size_t>& strides);

    template <typename T>
    static Result<Tensor> create(
        const DeviceAllocationView& backing,
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides);

    template <typename T>
    static Result<Tensor> create(
        const DeviceAllocationView& backing,
        const std::size_t* shape,
        std::size_t ndim,
        const std::size_t* strides);

    template <typename T>
    T* data() const {
        return reinterpret_cast<T*>(
            reinterpret_cast<std::byte*>(allocation_base_) + view_offset_bytes_);
    }

    const void* data() const {
        return reinterpret_cast<const std::byte*>(allocation_base_) + view_offset_bytes_;
    }

    bool same_shape(const Tensor& other) const noexcept {
        if (ndim_ != other.ndim_) {
            return false;
        }
        for (std::size_t i = 0; i < ndim_; ++i) {
            if (shape_[i] != other.shape_[i]) {
                return false;
            }
        }
        return true;
    }

    const std::size_t* shape_data() const noexcept { return shape_.data(); }
    const std::size_t* strides_data() const noexcept { return strides_.data(); }
    std::size_t dim(std::size_t i) const noexcept { return shape_[i]; }
    std::size_t ndim() const noexcept { return ndim_; }
    std::size_t allocation_bytes() const noexcept { return allocation_bytes_; }
    std::size_t view_offset_bytes() const noexcept { return view_offset_bytes_; }
    std::size_t element_size() const noexcept { return elem_size_; }
    int physical_device() const { return physical_device_; }
    void* allocation_base() const { return allocation_base_; }

    Result<Tensor> slice(std::size_t dim, std::size_t start, std::size_t count) const;

    Tensor() = default;

    Status required_span_bytes(std::size_t& out) const {
        return compute_required_span_bytes(shape_.data(), ndim_, strides_.data(), elem_size_, out);
    }

   private:
    static Status compute_required_span_bytes(
        const std::size_t* shape,
        std::size_t ndim,
        const std::size_t* strides,
        std::size_t elem_size,
        std::size_t& out);

    friend Status TensorTestCheckInternalFields(const Tensor& t,
                                                void* expected_base,
                                                std::size_t expected_alloc_bytes,
                                                std::size_t expected_view_offset,
                                                int expected_device);

    void* allocation_base_ = nullptr;
    std::size_t allocation_bytes_ = 0;
    std::size_t view_offset_bytes_ = 0;
    std::array<std::size_t, kMaxTensorDims> shape_{};
    std::array<std::size_t, kMaxTensorDims> strides_{};
    std::size_t ndim_ = 0;
    std::size_t elem_size_ = 0;
    int physical_device_ = -1;
};

template <typename T>
Result<Tensor> Tensor::create(
    const DeviceAllocationView& backing,
    const std::size_t* shape,
    std::size_t ndim,
    const std::size_t* strides) {
    if (!backing) {
        return Status::invalid_argument("backing must be valid", __FILE__, __LINE__);
    }

    if (shape == nullptr || strides == nullptr) {
        return Status::invalid_argument("shape and strides must not be null", __FILE__, __LINE__);
    }

    if (ndim == 0 || ndim > kMaxTensorDims) {
        return Status::invalid_argument("ndim must be in (0, kMaxTensorDims]", __FILE__, __LINE__);
    }

    for (std::size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            return Status::invalid_argument("shape element must be non-zero", __FILE__, __LINE__);
        }
        if (strides[i] == 0) {
            return Status::invalid_argument("stride element must be non-zero", __FILE__, __LINE__);
        }
    }

    std::size_t required_bytes;
    auto span_status = compute_required_span_bytes(shape, ndim, strides, sizeof(T), required_bytes);
    if (!span_status.ok()) return span_status;

    if (required_bytes > backing.bytes()) {
        return Status::insufficient_memory(
            "required_bytes exceeds backing_bytes", __FILE__, __LINE__);
    }

    Tensor t;
    t.allocation_base_ = backing.data();
    t.allocation_bytes_ = backing.bytes();
    t.view_offset_bytes_ = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        t.shape_[i] = shape[i];
        t.strides_[i] = strides[i];
    }
    t.ndim_ = ndim;
    t.elem_size_ = sizeof(T);
    t.physical_device_ = backing.physical_device();
    return t;
}

template <typename T>
Result<Tensor> Tensor::create(
    const DeviceAllocationView& backing,
    const std::initializer_list<std::size_t>& shape,
    const std::initializer_list<std::size_t>& strides) {
    if (shape.size() != strides.size()) {
        return Status::invalid_argument(
            "shape and strides must have same size", __FILE__, __LINE__);
    }
    return create<T>(backing, shape.begin(), shape.size(), strides.begin());
}

template <typename T>
Result<Tensor> Tensor::create(
    const DeviceAllocationView& backing,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if (shape.size() != strides.size()) {
        return Status::invalid_argument(
            "shape and strides must have same size", __FILE__, __LINE__);
    }
    return create<T>(backing, shape.data(), shape.size(), strides.data());
}

} // namespace gpu
} // namespace ps
