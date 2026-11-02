#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/tensor_partition.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps::weights {

class SafetensorsCollection {
public:
    static Result<SafetensorsCollection> open(const std::string& model_dir);

    SafetensorsCollection() = default;
    SafetensorsCollection(SafetensorsCollection&&) = default;
    SafetensorsCollection& operator=(SafetensorsCollection&&) = default;
    SafetensorsCollection(const SafetensorsCollection&) = delete;
    SafetensorsCollection& operator=(const SafetensorsCollection&) = delete;

    Result<io::StTensorSpec> tensor_spec(const std::string& name) const;
    Result<const void*> tensor_data(const std::string& name, std::size_t& out_bytes) const;

    // Enumerates all logical tensor names across the referenced shards.
    Result<std::vector<std::string>> tensor_names() const;

    Result<std::string> tensor_shard(const std::string& name) const;
    Result<io::StAxisSpan> tensor_span(const std::string& name) const;
    Result<io::StAxisSpan> axis_span(const std::string& name, std::uint64_t unit_begin,
                                     std::uint64_t unit_count) const;
    Result<std::uint64_t> shard_file_size(const std::string& shard) const;

    const std::string& model_dir() const noexcept { return model_dir_; }

private:
    Result<const io::SafetensorsReader*> reader_for(const std::string& name) const;
    Result<const io::SafetensorsReader*> open_shard(const std::string& shard) const;

    std::string shard_for(const std::string& name) const;

    std::string model_dir_;
    std::map<std::string, std::string> weight_map_;
    std::string single_shard_;
    mutable std::map<std::string, io::SafetensorsReader> shard_readers_;
};

bool is_quantized_model_dir(const std::string& model_dir);

Status validate_quantized_model(const quantization::bundle::QuantizedModelReader& reader);

// Options for loading matrix weights from disk. Preshuffle converts the
// canonical payload to the GPU-native WMMA layout at load time; it is a
// property of the weight loading layer, not of any model family.
// `partition_plan` selects an optional logical TP partition that is
// materialized before any preshuffle; nullptr keeps the global load path.
struct WeightLoadOptions {
    bool preshuffle = true;
    const TensorPartitionPlan* partition_plan = nullptr;
};

Result<gpu::Tensor> load_bf16_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream);

Result<gpu::Tensor> load_bf16_or_f32_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream);

Result<MatrixWeight> load_bf16_matrix(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options);

Result<MatrixWeight> load_quantized_matrix(
    const quantization::bundle::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options);

Result<gpu::Tensor> load_quantized_small(
    const quantization::bundle::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options = {});

Result<MatrixWeight> load_bf16_matrix_axis_slice(
    const SafetensorsCollection& collection,
    const std::string& name,
    std::uint64_t unit_begin,
    std::uint64_t unit_count,
    gpu::GpuArena& arena,
    hipStream_t stream);

Result<MatrixWeight> load_fp8_matrix_axis_slice(
    const SafetensorsCollection& collection,
    const std::string& codes_name,
    const std::string& scale_name,
    std::uint64_t unit_begin,
    std::uint64_t unit_count,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options = {});

}  // namespace ps::weights
