// shelf foundation — TensorShape: rank-limited, validated, overflow-checked.
#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include <shelf/status.hpp>

namespace shelf {

// Logical tensor shape. Row-major semantics is decided by consumers;
// shape itself only validates rank and dimension values.
class TensorShape {
public:
    static constexpr std::size_t kMaxRank = 8;

    // Every dimension must be >= 1; rank must be within 1..kMaxRank.
    static Result<TensorShape> create(const std::size_t* dims, std::size_t rank) {
        if (rank == 0 || rank > kMaxRank) {
            return Status::out_of_range("shape rank must be within 1..kMaxRank",
                                        __FILE__, __LINE__);
        }
        std::array<std::size_t, kMaxRank> stored{};
        for (std::size_t i = 0; i < rank; ++i) {
            if (dims[i] == 0) {
                return Status::invalid_argument("shape dimension must be >= 1",
                                                __FILE__, __LINE__);
            }
            stored[i] = dims[i];
        }
        return TensorShape{stored, rank};
    }

    static Result<TensorShape> create(std::initializer_list<std::size_t> dims) {
        return create(dims.begin(), dims.size());
    }

    static Result<TensorShape> create(const std::vector<std::size_t>& dims) {
        return create(dims.data(), dims.size());
    }

    TensorShape() = default;

    std::size_t rank() const noexcept { return rank_; }
    bool empty() const noexcept { return rank_ == 0; }

    std::size_t dim(std::size_t i) const {
        return i < rank_ ? dims_[i] : 0;
    }

    const std::size_t* dims() const noexcept { return dims_.data(); }

    // Element count with overflow detection (Phase 1 completion criterion).
    Result<std::size_t> num_elements() const {
        if (rank_ == 0) {
            return Status::invalid_state("num_elements on default-constructed shape",
                                         __FILE__, __LINE__);
        }
        std::size_t acc = 1;
        for (std::size_t i = 0; i < rank_; ++i) {
            std::size_t next = 0;
            SHELF_RETURN_IF_ERROR(checked_mul(acc, dims_[i], next, __FILE__, __LINE__));
            acc = next;
        }
        return acc;
    }

    bool operator==(const TensorShape& other) const noexcept {
        if (rank_ != other.rank_) return false;
        for (std::size_t i = 0; i < rank_; ++i) {
            if (dims_[i] != other.dims_[i]) return false;
        }
        return true;
    }

private:
    TensorShape(std::array<std::size_t, kMaxRank> dims, std::size_t rank)
        : dims_(dims), rank_(rank) {}

    std::array<std::size_t, kMaxRank> dims_{};
    std::size_t rank_ = 0;
};

} // namespace shelf
