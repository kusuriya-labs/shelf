#pragma once

#include <shelf/io/safetensors_writer.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/quantization/manifest/quantized_manifest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace tp_fixture {

namespace fs = std::filesystem;
using namespace shelf;

inline int g_fail = 0;

inline void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}


static void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

static std::vector<uint8_t> bf16_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 2u);
    auto* u = reinterpret_cast<uint16_t*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        u[i] = static_cast<uint16_t>(0x3D40u + ((seed + static_cast<uint32_t>(i)) % 64u));
    }
    return out;
}

static std::string crc_hex(const std::vector<uint8_t>& bytes) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x",
                  quantization::manifest::crc32_iso_hdlc(bytes.data(), bytes.size()));
    return buf;
}

namespace {

struct PhysicalTensor {
    std::string name;
    io::SType dtype = io::SType::BF16;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

struct TensorPlan {
    quantization::manifest::QuantizedTensorMetadata meta;
    std::vector<PhysicalTensor> physical;
};

TensorPlan make_bf16_matrix(uint64_t rows, uint64_t k, const std::string& logical) {
    TensorPlan plan;
    std::vector<uint8_t> bytes = bf16_bytes(rows * k, static_cast<uint32_t>(rows * 7 + k));
    plan.meta.role = "weight";
    plan.meta.encoding = quantization::manifest::QuantizedEncoding::Bf16;
    plan.meta.logical_shape = {static_cast<int64_t>(rows), static_cast<int64_t>(k)};
    plan.meta.k_padded = k;
    PhysicalTensor data;
    data.name = logical + ".__shelf_data";
    data.dtype = io::SType::BF16;
    data.shape = {static_cast<std::size_t>(rows), static_cast<std::size_t>(k)};
    data.bytes = bytes;
    quantization::manifest::QuantizedTensorRef ref;
    ref.tensor = data.name;
    ref.dtype = "BF16";
    ref.crc32 = crc_hex(bytes);
    plan.meta.data = ref;
    plan.physical.push_back(std::move(data));
    return plan;
}

TensorPlan make_bf16_small(const std::vector<std::size_t>& shape,
                           const std::string& logical, uint32_t seed) {
    TensorPlan plan;
    uint64_t n = 1;
    for (std::size_t d : shape) n *= d;
    std::vector<uint8_t> bytes = bf16_bytes(n, seed);
    plan.meta.role = "norm";
    plan.meta.encoding = quantization::manifest::QuantizedEncoding::Bf16;
    for (std::size_t d : shape) plan.meta.logical_shape.push_back(static_cast<int64_t>(d));
    plan.meta.k_padded = static_cast<uint64_t>(shape.back());
    PhysicalTensor data;
    data.name = logical + ".__shelf_data";
    data.dtype = io::SType::BF16;
    data.shape = shape;
    data.bytes = bytes;
    quantization::manifest::QuantizedTensorRef ref;
    ref.tensor = data.name;
    ref.dtype = "BF16";
    ref.crc32 = crc_hex(bytes);
    plan.meta.data = ref;
    plan.physical.push_back(std::move(data));
    return plan;
}

std::string config_json() {
    return std::string(
               "{"
               "\"model_type\":\"qwen3_5\","
               "\"tie_word_embeddings\":false,"
               "\"text_config\":{"
               "\"model_type\":\"qwen3_5_text\","
               "\"vocab_size\":128,"
               "\"hidden_size\":64,"
               "\"intermediate_size\":128,"
               "\"num_hidden_layers\":4,"
               "\"full_attention_interval\":2,"
               "\"layer_types\":[\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\"],"
               "\"num_attention_heads\":4,"
               "\"num_key_value_heads\":2,"
               "\"head_dim\":16,"
               "\"rms_norm_eps\":1e-6,"
               "\"rope_parameters\":{\"rope_theta\":10000000,\"partial_rotary_factor\":0.25},"
               "\"linear_num_key_heads\":4,"
               "\"linear_num_value_heads\":8,"
               "\"linear_key_head_dim\":128,"
               "\"linear_value_head_dim\":128,"
               "\"linear_conv_kernel_dim\":4,"
               "\"tie_word_embeddings\":false"
               "}}");
}

Status build_fixture(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);

    const std::string p = "model.language_model.";
    const std::string l0 = p + "layers.0.";
    const std::string l1 = p + "layers.1.";
    const std::string l2 = p + "layers.2.";
    const std::string l3 = p + "layers.3.";

    quantization::manifest::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "bf16";
    manifest.source_dtype = "BF16";
    std::vector<PhysicalTensor> shard;

    const auto add = [&](const std::string& logical, TensorPlan plan) {
        for (auto& phys : plan.physical) shard.push_back(std::move(phys));
        manifest.tensors[logical] = plan.meta;
    };

    add(p + "embed_tokens.weight", make_bf16_matrix(128, 64, p + "embed_tokens.weight"));
    add(p + "lm_head.weight", make_bf16_matrix(128, 64, p + "lm_head.weight"));
    add(p + "norm.weight", make_bf16_small({64}, p + "norm.weight", 11));

    for (const std::string& layer : {l0, l1, l2, l3}) {
        add(layer + "input_layernorm.weight",
            make_bf16_small({64}, layer + "input_layernorm.weight", 21));
        add(layer + "post_attention_layernorm.weight",
            make_bf16_small({64}, layer + "post_attention_layernorm.weight", 22));
        add(layer + "mlp.gate_proj.weight",
            make_bf16_matrix(128, 64, layer + "mlp.gate_proj.weight"));
        add(layer + "mlp.up_proj.weight",
            make_bf16_matrix(128, 64, layer + "mlp.up_proj.weight"));
        add(layer + "mlp.down_proj.weight",
            make_bf16_matrix(64, 128, layer + "mlp.down_proj.weight"));
    }

    const auto add_linear_attn = [&](const std::string& lp) {
        add(lp + "linear_attn.in_proj_qkv.weight",
            make_bf16_matrix(2048, 64, lp + "linear_attn.in_proj_qkv.weight"));
        add(lp + "linear_attn.in_proj_z.weight",
            make_bf16_matrix(1024, 64, lp + "linear_attn.in_proj_z.weight"));
        add(lp + "linear_attn.in_proj_a.weight",
            make_bf16_matrix(8, 64, lp + "linear_attn.in_proj_a.weight"));
        add(lp + "linear_attn.in_proj_b.weight",
            make_bf16_matrix(8, 64, lp + "linear_attn.in_proj_b.weight"));
        add(lp + "linear_attn.out_proj.weight",
            make_bf16_matrix(64, 1024, lp + "linear_attn.out_proj.weight"));
        add(lp + "linear_attn.conv1d.weight",
            make_bf16_small({2048, 1, 4}, lp + "linear_attn.conv1d.weight", 31));
        add(lp + "linear_attn.norm.weight",
            make_bf16_small({128}, lp + "linear_attn.norm.weight", 32));
        add(lp + "linear_attn.dt_bias",
            make_bf16_small({8}, lp + "linear_attn.dt_bias", 33));
        add(lp + "linear_attn.A_log",
            make_bf16_small({8}, lp + "linear_attn.A_log", 34));
    };
    add_linear_attn(l0);
    add_linear_attn(l2);

    const auto add_full_attn = [&](const std::string& lp) {
        add(lp + "self_attn.q_proj.weight",
            make_bf16_matrix(128, 64, lp + "self_attn.q_proj.weight"));
        add(lp + "self_attn.k_proj.weight",
            make_bf16_matrix(32, 64, lp + "self_attn.k_proj.weight"));
        add(lp + "self_attn.v_proj.weight",
            make_bf16_matrix(32, 64, lp + "self_attn.v_proj.weight"));
        add(lp + "self_attn.o_proj.weight",
            make_bf16_matrix(64, 64, lp + "self_attn.o_proj.weight"));
        add(lp + "self_attn.q_norm.weight",
            make_bf16_small({16}, lp + "self_attn.q_norm.weight", 41));
        add(lp + "self_attn.k_norm.weight",
            make_bf16_small({16}, lp + "self_attn.k_norm.weight", 42));
    };
    add_full_attn(l1);
    add_full_attn(l3);

    auto serialized = quantization::manifest::serialize_quantized_manifest(manifest);
    if (!serialized.ok()) return serialized.status();

    auto writer_result = io::SafetensorsWriter::create((dir / "model.safetensors").string());
    if (!writer_result.ok()) return writer_result.status();
    io::SafetensorsWriter writer = writer_result.release();
    for (const PhysicalTensor& phys : shard) {
        Status st = writer.plan_tensor(phys.name, phys.dtype, phys.shape);
        if (!st.ok()) return st;
    }
    Status st = writer.set_metadata("format", "pt");
    if (!st.ok()) return st;
    st = writer.set_metadata("shelf.format", quantization::manifest::kQuantizedSafetensorsFormat);
    if (!st.ok()) return st;
    st = writer.set_metadata(
        "shelf.format_version",
        std::to_string(quantization::manifest::kQuantizedSafetensorsFormatVersion));
    if (!st.ok()) return st;
    st = writer.set_metadata("shelf.quantization_metadata",
                             quantization::manifest::kQuantizedMetadataFile);
    if (!st.ok()) return st;
    auto header_result = writer.write_header();
    if (!header_result.ok()) return header_result.status();
    for (const PhysicalTensor& phys : shard) {
        Status write_status = writer.write_tensor(phys.name, phys.bytes.data(),
                                                  phys.bytes.size());
        if (!write_status.ok()) return write_status;
    }
    auto finish_result = writer.finish();
    if (!finish_result.ok()) return finish_result.status();

    write_text((dir / quantization::manifest::kQuantizedMetadataFile).string(),
               serialized.value());
    write_text((dir / "config.json").string(), config_json());
    return Status::make_ok();
}

}
}
