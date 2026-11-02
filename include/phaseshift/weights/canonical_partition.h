#pragma once

#include <phaseshift/quantization/bundle/quantized_model_reader.h>
#include <phaseshift/weights/tensor_partition.h>

#include <cstdint>
#include <vector>

namespace ps::weights {

struct OwnedCanonicalTensor {
    std::vector<int64_t> logical_shape;
    uint64_t k_padded = 0;
    std::vector<uint8_t> data;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
    std::vector<uint8_t> metadata2;
    std::vector<uint8_t> metadata3;
    std::vector<uint8_t> metadata4;
};

Status materialize_rank_local_canonical(
    const TensorPartitionDesc& partition,
    const quantization::bundle::QuantizedTensorView& global,
    OwnedCanonicalTensor& owned,
    quantization::bundle::QuantizedTensorView& local);

}  // namespace ps::weights
