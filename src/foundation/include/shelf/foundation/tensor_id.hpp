// shelf foundation — TensorId: stable strong identity for logical tensors.
//
// Contract §4.1: logical tensors never carry raw pointers or residency
// flags. Identity is a plain id resolved later by weights/ binding.
#pragma once

#include <compare>
#include <cstdint>

#include <shelf/status.hpp>

namespace shelf {

class TensorId {
public:
    static constexpr std::uint32_t kInvalidValue = 0xFFFFFFFFu;

    constexpr TensorId() = default;  // invalid by default

    static constexpr TensorId from_uint32(std::uint32_t value) {
        TensorId id;
        id.value_ = value;
        return id;
    }

    static Result<TensorId> parse(std::uint32_t value) {
        if (value == kInvalidValue) {
            return Status::invalid_argument("TensorId value is reserved (invalid)",
                                            __FILE__, __LINE__);
        }
        return from_uint32(value);
    }

    std::uint32_t value() const noexcept { return value_; }
    bool valid() const noexcept { return value_ != kInvalidValue; }

    auto operator<=>(const TensorId&) const = default;

private:
    std::uint32_t value_ = kInvalidValue;
};

} // namespace shelf
