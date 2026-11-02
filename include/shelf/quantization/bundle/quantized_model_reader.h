#pragma once
#include <shelf/core/status.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::quantization::bundle {

using ps::quantization::manifest::QuantizedEncoding;
using ps::quantization::manifest::QuantizedManifest;

struct HfQuantizedSource;

struct ByteSpan {
    const void* data = nullptr;
    std::size_t size = 0;
};

struct QuantizedTensorView {
    std::string name;
    QuantizedEncoding encoding = QuantizedEncoding::Bf16;
    std::vector<int64_t> logical_shape;
    int64_t k_padded = 0;
    uint32_t scale_block_n = 0;
    uint32_t scale_block_k = 0;
    ByteSpan data;
    ByteSpan codes;
    ByteSpan metadata1;
    ByteSpan metadata2;
    ByteSpan metadata3;
    ByteSpan metadata4;
};

// Output-only reader for a self-contained Shelf quantized safetensors
// model directory. The input is the finished model directory alone; no source
// BF16 model path, bundle dir, or manifest is required.
class QuantizedModelReader {
 public:
    static Result<QuantizedModelReader> open(
        const std::string& output_dir, bool verify_payload_crc = true);

    static Result<QuantizedModelReader> open_hf_source(
        HfQuantizedSource&& source, bool verify_payload_crc = true);

    ~QuantizedModelReader() noexcept;
    QuantizedModelReader(QuantizedModelReader&& other) noexcept;
    QuantizedModelReader& operator=(QuantizedModelReader&& other) noexcept;
    QuantizedModelReader(const QuantizedModelReader&) = delete;
    QuantizedModelReader& operator=(const QuantizedModelReader&) = delete;

    const QuantizedManifest& manifest() const;

    // Resolves a logical tensor (after alias resolution) to its physical bytes.
    Result<QuantizedTensorView> resolve(const std::string& logical_name) const;

    // Enumerates all logical tensor names in stable (lexical) order.
    Result<std::vector<std::string>> list_logical_tensors() const;

    // Resolves an alias target (follows aliases to the canonical tensor name).
    Result<std::string> resolve_alias(const std::string& name) const;

    // Total payload bytes across all shards referenced by the manifest.
    uint64_t payload_bytes() const;

 private:
    QuantizedModelReader() noexcept : pimpl_(nullptr) {}

    struct Impl;
    Impl* pimpl_;
};

}
