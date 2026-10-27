// shelf artifact/common — Raw Tensor Catalog contracts (plan §4.2 path).
//
//   Artifact Reader -> Raw Tensor Catalog -> (name/axis mapping) -> Canonical Catalog
//
// Entries describe what the CONTAINER holds: name, logical dtype, shape and a
// byte range. No forward, no model knowledge, no execution details.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/status.hpp>

namespace shelf::artifact {

enum class ContainerFormat : std::uint8_t {
    safetensors = 0,
};

struct RawTensorEntry {
    std::string name;
    DataType dtype = DataType::unknown;
    TensorShape shape;
    std::uint64_t begin = 0;  // payload range within the container's data region
    std::uint64_t end = 0;    // exclusive
};

// Shape/dtype/size consistency for one entry.
inline Status validate_entry(const RawTensorEntry& entry) {
    if (entry.name.empty()) {
        return Status::invalid_argument("tensor name is empty", __FILE__, __LINE__);
    }
    if (entry.shape.empty()) {
        return Status::invalid_argument("tensor shape is empty", __FILE__, __LINE__);
    }
    if (entry.dtype == DataType::unknown) {
        return Status::unsupported("unknown dtype for tensor " + entry.name,
                                   __FILE__, __LINE__);
    }
    const std::size_t elem = element_size(entry.dtype);
    if (elem == 0) {
        return Status::unsupported("opaque dtype not expected in raw catalog",
                                   __FILE__, __LINE__);
    }
    auto n = entry.shape.num_elements();
    SHELF_RETURN_IF_ERROR(n.status());
    std::size_t bytes = 0;
    SHELF_RETURN_IF_ERROR(checked_mul(n.value(), elem, bytes, __FILE__, __LINE__));
    if (entry.end < entry.begin) {
        return Status::invalid_argument("inverted byte range", __FILE__, __LINE__);
    }
    if (static_cast<std::size_t>(entry.end - entry.begin) != bytes) {
        return Status::invalid_argument("byte range does not match shape*dtype",
                                        __FILE__, __LINE__);
    }
    return Status::make_ok();
}

// Ordered entries with duplicate-name detection. Overlap checks are the
// reader's job (only a reader knows the data region layout).
class RawTensorCatalog {
public:
    Status add(RawTensorEntry entry) {
        SHELF_RETURN_IF_ERROR(validate_entry(entry));
        for (const auto& existing : entries_) {
            if (existing.name == entry.name) {
                return Status::invalid_state("duplicate tensor name: " + entry.name,
                                             __FILE__, __LINE__);
            }
        }
        entries_.push_back(std::move(entry));
        return Status::make_ok();
    }

    const std::vector<RawTensorEntry>& entries() const noexcept { return entries_; }
    std::size_t size() const noexcept { return entries_.size(); }

    const RawTensorEntry* find(const std::string& name) const {
        for (const auto& e : entries_) {
            if (e.name == name) return &e;
        }
        return nullptr;
    }

private:
    std::vector<RawTensorEntry> entries_;
};

} // namespace shelf::artifact
