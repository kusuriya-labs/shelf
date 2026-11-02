#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/quantization/fp8/fp8.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace expert_slice_fixture {

namespace fs = std::filesystem;

inline constexpr std::uint32_t kExperts = 4;
inline constexpr std::uint32_t kGateUpRows = 1280;
inline constexpr std::uint32_t kDownRows = 2560;
inline constexpr std::uint32_t kHidden = 2560;
inline constexpr std::uint32_t kInter = 640;
inline constexpr std::uint32_t kTokens = 6;
inline constexpr std::uint32_t kScaleBlockN = 128;
inline constexpr std::uint32_t kScaleBlockK = 128;

inline constexpr const char* kGateUpName = "model.layers.0.mlp.experts.gate_up_proj";
inline constexpr const char* kGateUpScaleName =
    "model.layers.0.mlp.experts.gate_up_proj_scale_inv";
inline constexpr const char* kDownName = "model.layers.0.mlp.experts.down_proj";
inline constexpr const char* kDownScaleName = "model.layers.0.mlp.experts.down_proj_scale_inv";

struct Fixture {
    std::vector<std::uint8_t> gate_up_bf16;
    std::vector<std::uint8_t> down_bf16;
    std::vector<std::uint8_t> gate_up_codes;
    std::vector<std::uint8_t> down_codes;
    std::vector<std::uint16_t> gate_up_scales;
    std::vector<std::uint16_t> down_scales;
    std::vector<std::uint16_t> input_bf16;
};

struct PlannedTensor {
    std::string name;
    shelf::io::SType dtype;
    std::vector<std::size_t> shape;
    const std::uint8_t* data;
    std::size_t bytes;
};

inline std::uint16_t f32_to_bf16(float value) {
    return shelf::f32_to_bf16_rne(value);
}

inline float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline std::vector<float> random_source(std::uint64_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> out(count);
    for (float& value : out) value = dist(rng);
    return out;
}

inline void append_bf16(std::vector<std::uint8_t>& dst, const std::vector<float>& src) {
    const std::size_t at = dst.size() / 2;
    dst.resize(dst.size() + src.size() * 2);
    auto* out = reinterpret_cast<std::uint16_t*>(dst.data()) + at;
    for (std::size_t i = 0; i < src.size(); ++i) out[i] = f32_to_bf16(src[i]);
}

inline void append_fp8(std::vector<std::uint8_t>& codes_dst,
                       std::vector<std::uint16_t>& scales_dst,
                       const std::vector<float>& src,
                       std::uint64_t rows, std::uint64_t k) {
    const std::size_t codes_at = codes_dst.size();
    codes_dst.resize(codes_at + static_cast<std::size_t>(rows * k));
    const std::size_t scale_n = (static_cast<std::size_t>(rows) + kScaleBlockN - 1) / kScaleBlockN;
    const std::size_t scale_k = static_cast<std::size_t>(k) / kScaleBlockK;
    std::vector<float> scales(scale_n * scale_k);
    shelf::quantization::fp8::quantize_fp8_matrix(
        src.data(), k, rows, k, k, codes_dst.data() + codes_at, scales.data());
    const std::size_t scales_at = scales_dst.size();
    scales_dst.resize(scales_at + scales.size());
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales_dst[scales_at + i] = f32_to_bf16(scales[i]);
    }
}

inline std::uint64_t gate_up_expert_elements() {
    return static_cast<std::uint64_t>(kGateUpRows) * kHidden;
}

inline std::uint64_t down_expert_elements() {
    return static_cast<std::uint64_t>(kDownRows) * kInter;
}

inline std::uint64_t gate_up_scale_elements() {
    return static_cast<std::uint64_t>((kGateUpRows + kScaleBlockN - 1) / kScaleBlockN) *
           (kHidden / kScaleBlockK);
}

inline std::uint64_t down_scale_elements() {
    return static_cast<std::uint64_t>((kDownRows + kScaleBlockN - 1) / kScaleBlockN) *
           (kInter / kScaleBlockK);
}

inline Fixture make_fixture() {
    Fixture f;
    f.gate_up_bf16.reserve(static_cast<std::size_t>(kExperts) * gate_up_expert_elements() * 2);
    f.down_bf16.reserve(static_cast<std::size_t>(kExperts) * down_expert_elements() * 2);
    f.gate_up_codes.reserve(static_cast<std::size_t>(kExperts) * gate_up_expert_elements());
    f.down_codes.reserve(static_cast<std::size_t>(kExperts) * down_expert_elements());
    f.gate_up_scales.reserve(static_cast<std::size_t>(kExperts) * gate_up_scale_elements());
    f.down_scales.reserve(static_cast<std::size_t>(kExperts) * down_scale_elements());
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        const std::vector<float> gate_up =
            random_source(gate_up_expert_elements(), 1000u + e);
        append_bf16(f.gate_up_bf16, gate_up);
        append_fp8(f.gate_up_codes, f.gate_up_scales, gate_up, kGateUpRows, kHidden);
        const std::vector<float> down = random_source(down_expert_elements(), 2000u + e);
        append_bf16(f.down_bf16, down);
        append_fp8(f.down_codes, f.down_scales, down, kDownRows, kInter);
    }
    const std::vector<float> input =
        random_source(static_cast<std::uint64_t>(kTokens) * kHidden, 3000u);
    f.input_bf16.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) f.input_bf16[i] = f32_to_bf16(input[i]);
    return f;
}

inline bool write_model_file(const std::string& path, const std::vector<PlannedTensor>& tensors) {
    auto writer_result = shelf::io::SafetensorsWriter::create(path);
    if (!writer_result.ok()) return false;
    shelf::io::SafetensorsWriter writer = writer_result.release();
    for (const PlannedTensor& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) return false;
    }
    if (!writer.write_header().ok()) return false;
    for (const PlannedTensor& t : tensors) {
        if (!writer.write_tensor(t.name, t.data, t.bytes).ok()) return false;
    }
    if (!writer.finish().ok()) return false;
    return true;
}

inline bool write_bf16_model_dir(const fs::path& dir, const Fixture& f) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::vector<PlannedTensor> tensors = {
        {kGateUpName, shelf::io::SType::BF16, {kExperts, kGateUpRows, kHidden},
         f.gate_up_bf16.data(), f.gate_up_bf16.size()},
        {kDownName, shelf::io::SType::BF16, {kExperts, kDownRows, kInter},
         f.down_bf16.data(), f.down_bf16.size()},
    };
    return write_model_file((dir / "model.safetensors").string(), tensors);
}

inline std::string fp8_config_json() {
    return "{\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
           "\"activation_scheme\":\"dynamic\",\"weight_block_size\":[128,128]}}";
}

inline bool write_fp8_model_dir(const fs::path& dir, const Fixture& f) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    {
        std::ofstream config((dir / "config.json").string(),
                             std::ios::binary | std::ios::trunc);
        if (!config) return false;
        config << fp8_config_json();
    }
    const std::vector<PlannedTensor> tensors = {
        {kGateUpName, shelf::io::SType::F8_E4M3, {kExperts, kGateUpRows, kHidden},
         f.gate_up_codes.data(), f.gate_up_codes.size()},
        {kGateUpScaleName, shelf::io::SType::BF16,
         {kExperts, (kGateUpRows + kScaleBlockN - 1) / kScaleBlockN, kHidden / kScaleBlockK},
         reinterpret_cast<const std::uint8_t*>(f.gate_up_scales.data()),
         f.gate_up_scales.size() * 2},
        {kDownName, shelf::io::SType::F8_E4M3, {kExperts, kDownRows, kInter},
         f.down_codes.data(), f.down_codes.size()},
        {kDownScaleName, shelf::io::SType::BF16,
         {kExperts, (kDownRows + kScaleBlockN - 1) / kScaleBlockN, kInter / kScaleBlockK},
         reinterpret_cast<const std::uint8_t*>(f.down_scales.data()),
         f.down_scales.size() * 2},
    };
    return write_model_file((dir / "model.safetensors").string(), tensors);
}

inline std::vector<float> bf16_region_f32(const std::vector<std::uint8_t>& bytes,
                                          std::uint64_t element_count,
                                          std::uint64_t element_begin) {
    const auto* src = reinterpret_cast<const std::uint16_t*>(bytes.data()) + element_begin;
    std::vector<float> out(static_cast<std::size_t>(element_count));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_f32(src[i]);
    return out;
}

inline std::vector<float> scale_region_f32(const std::vector<std::uint16_t>& scales,
                                           std::uint64_t element_count,
                                           std::uint64_t element_begin) {
    std::vector<float> out(static_cast<std::size_t>(element_count));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = bf16_to_f32(scales[static_cast<std::size_t>(element_begin) + i]);
    }
    return out;
}

inline std::vector<float> bf16_region_f32(const std::vector<std::uint16_t>& bits,
                                          std::uint64_t element_count,
                                          std::uint64_t element_begin) {
    return scale_region_f32(bits, element_count, element_begin);
}

}
