// shelf foundation — TensorSpec (Tensor Contract, plan §4.1).
//
//   struct TensorSpec { TensorId id; TensorShape shape; DataType logical_dtype; TensorRole role; }
//
// A logical TensorSpec must NOT carry: raw pointers, GPU residency flags,
// container-format details, or quantization payloads (those belong to
// weights/QuantSpec in later phases).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/foundation/tensor_id.hpp>
#include <shelf/status.hpp>

namespace shelf {

enum class TensorRole : std::uint8_t {
    weight = 0,     // resolved from artifacts via weights/ binding
    activation,     // transient computation buffer
    state,          // session-persistent (KV cache, recurrent state)
    workspace,      // compiler-allocated scratch
};

struct TensorSpec {
    TensorId id;
    TensorShape shape;
    DataType logical_dtype = DataType::unknown;
    TensorRole role = TensorRole::activation;

    Status validate() const {
        if (!id.valid()) {
            return Status::invalid_argument("TensorSpec has invalid TensorId",
                                            __FILE__, __LINE__);
        }
        if (shape.empty()) {
            return Status::invalid_argument("TensorSpec has empty shape",
                                            __FILE__, __LINE__);
        }
        if (logical_dtype == DataType::unknown) {
            return Status::invalid_argument("TensorSpec has unknown dtype",
                                            __FILE__, __LINE__);
        }
        SHELF_RETURN_IF_ERROR(shape.num_elements().status());  // overflow check
        return Status::make_ok();
    }
};

// Detect duplicate identities inside one catalog (Tensor ID inconsistency,
// Phase 1 completion criterion). O(n log n); catalogs are small at load time.
inline Status validate_unique_ids(const std::vector<TensorSpec>& specs) {
    std::vector<std::uint32_t> ids;
    ids.reserve(specs.size());
    for (const auto& spec : specs) {
        SHELF_RETURN_IF_ERROR(spec.validate());
        ids.push_back(spec.id.value());
    }
    // sort + adjacent scan keeps foundation free of extra containers beyond <vector>
    std::sort(ids.begin(), ids.end());
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (ids[i] == ids[i - 1]) {
            return Status::invalid_state("duplicate TensorId in catalog",
                                         __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

} // namespace shelf
