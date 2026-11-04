#include <shelf/io/safetensors_reader.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/manifest/runtime_resolution.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace moe_expert_fixture;
using shelf::io::SType;
using shelf::io::StTensorSpec;
using shelf::qwen35moe::build_moe_expert_group_plan;
using shelf::qwen35moe::MoeExpertGroupPlan;
using shelf::qwen35moe::MoeExpertProjection;
using shelf::qwen35moe::MoeExpertTensorSource;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

bool in_fused_plane(MoeExpertProjection projection) noexcept {
    return projection == MoeExpertProjection::Gate || projection == MoeExpertProjection::Up;
}

std::vector<std::string> logical_names() {
    std::vector<std::string> names;
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (const auto& name : expert_names(layer))
            names.push_back(name);
        for (const auto& name : mlp_aux_names(layer))
            names.push_back(name);
    }
    return names;
}

// The two physical tensors of one source, in checkpoint file order: codes then scale.
std::vector<std::string> physical_names(const MoeExpertTensorSource& src) {
    return {src.name, src.name + "_scale"};
}

// A layer's expert tensors must be one contiguous run in the shard, in the order the
// plan enumerates them. That is what lets the loader read the group without scattering
// reads across the file.
void test_file_order(const shelf::io::SafetensorsReader& reader, const MoeExpertGroupPlan& plan) {
    std::uint64_t expect_begin = 0;
    std::uint64_t span_begin = 0;
    std::uint64_t span_end = 0;
    bool first = true;
    for (const auto& src : plan.sources) {
        const std::uint64_t cols = in_fused_plane(src.projection) ? plan.w13_k_padded : plan.w2_k_padded;
        const std::vector<std::string> names = physical_names(src);
        for (std::size_t which = 0; which < names.size(); ++which) {
            const bool scale = which == 1u;
            auto spec = reader.tensor_spec(names[which]);
            if (!spec.ok()) {
                check(false, "expert tensor present in shard: " + names[which]);
                continue;
            }
            const StTensorSpec& s = spec.value();
            check(s.dtype == SType::U8, names[which] + ": dtype is U8");
            if (scale) {
                check(s.shape == std::vector<std::size_t>{src.rows, cols / 32u},
                      names[which] + ": scale shape is [rows][K/32]");
            } else {
                check(s.shape == std::vector<std::size_t>{src.rows, cols / 2u},
                      names[which] + ": codes shape is [rows][K/2]");
            }
            if (first) {
                span_begin = s.data_begin;
                first = false;
            } else {
                check(s.data_begin == expect_begin,
                      names[which] + ": follows the previous expert tensor");
            }
            expect_begin = s.data_end;
            span_end = s.data_end;
        }
    }
    const std::uint64_t plan_bytes = plan.w13_codes_bytes + plan.w13_scales_bytes +
                                     plan.w2_codes_bytes + plan.w2_scales_bytes;
    check(span_end - span_begin == plan_bytes,
          "one layer of experts is contiguous and plan-sized");
}

void test_manifest(const fs::path& dir, const MoeExpertGroupPlan& plan, const Fixture& f) {
    check(shelf::quantization::bundle::is_hf_quantized_model_dir(dir.string()),
          "fixture dir is an HF quantized dir");
    auto method = shelf::quantization::bundle::detect_hf_quant_method(dir.string());
    check(method.ok() && method.value() == shelf::quantization::bundle::HfQuantMethod::QuarkMxfp4,
          "fixture config is quark mxfp4");

    auto source = shelf::quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), logical_names(), false);
    if (!source.ok()) {
        check(false, "synthesize hf source: " + source.status().message());
        return;
    }
    auto reader = shelf::quantization::bundle::QuantizedModelReader::open_hf_source(
        source.release(), false);
    if (!reader.ok()) {
        check(false, "open hf source: " + reader.status().message());
        return;
    }
    const shelf::quantization::bundle::QuantizedModelReader& qreader = reader.value();
    check(shelf::quantization::manifest::validate_quantized_manifest_contract(qreader.manifest()).ok(),
          "fixture manifest contract validates");
    check(qreader.manifest().tensors.size() == kLayers * (kExperts * 3u + 5u),
          "manifest carries every expert projection and mlp aux tensor");

    // One logical tensor per expert projection: it owns both the codes plane and the
    // E8M0 scale plane, so the loader resolves it once.
    for (const auto& src : plan.sources) {
        auto view = qreader.resolve(src.name);
        if (!view.ok()) {
            check(false, "manifest resolves " + src.name);
            continue;
        }
        const shelf::quantization::bundle::QuantizedTensorView& v = view.value();
        const std::uint64_t cols = in_fused_plane(src.projection) ? plan.w13_k_padded : plan.w2_k_padded;
        check(v.encoding == shelf::quantization::manifest::QuantizedEncoding::Mxfp4,
              src.name + ": encoding is Mxfp4");
        check(v.logical_shape == std::vector<std::int64_t>{static_cast<std::int64_t>(src.rows),
                                                           static_cast<std::int64_t>(cols)},
              src.name + ": logical shape is [rows][k_padded]");
        check(v.k_padded == static_cast<std::int64_t>(cols), src.name + ": k_padded");
        check(v.scale_block_k == plan.scale_block_k, src.name + ": scale block K is 32");
        check(v.scale_block_n == 1u, src.name + ": scale block N is 1");
        check(v.codes.size == src.rows * (cols / 2u), src.name + ": codes payload size");
        check(v.metadata1.size == src.rows * (cols / 32u), src.name + ": scale payload size");

        const auto it_codes = f.codes.find(src.name);
        const auto it_scales = f.scales.find(src.name + "_scale");
        check(it_codes != f.codes.end() && it_scales != f.scales.end(),
              src.name + ": fixture payload recorded");
        if (it_codes == f.codes.end() || it_scales == f.scales.end())
            continue;
        check(std::memcmp(v.codes.data, it_codes->second.data(), it_codes->second.size()) == 0,
              src.name + ": codes bytes match the fixture");
        check(std::memcmp(v.metadata1.data, it_scales->second.data(),
                          it_scales->second.size()) == 0,
              src.name + ": scale bytes match the fixture");
    }

    // The scale tensor is physical: resolving it as a logical tensor must fail, which is
    // why the plan enumerates projections and not on-disk tensors.
    check(!qreader.resolve(plan.sources.front().name + "_scale").ok(),
          "the scale tensor is not a logical tensor");
}

}  // namespace

int main() {
    std::printf("=== MoE expert slice fixture (CPU) ===\n");
    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_expert_fixture_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    Fixture f;
    check(write_fixture(f, root / "mxfp4"), "fixture written");

    {
        auto cfg = shelf::qwen35moe::read_qwen35_moe_text_config((root / "mxfp4").string());
        check(cfg.ok(),
              "the fixture config.json is accepted by the MoE reader: " +
                  (cfg.ok() ? "" : cfg.status().message()));
        if (cfg.ok()) {
            check(cfg.value().ffn.num_experts == f.config.ffn.num_experts &&
                      cfg.value().ffn.num_experts_per_tok == f.config.ffn.num_experts_per_tok &&
                      cfg.value().ffn.moe_intermediate_size == f.config.ffn.moe_intermediate_size &&
                      cfg.value().ffn.shared_expert_intermediate_size ==
                          f.config.ffn.shared_expert_intermediate_size,
                  "the fixture config.json carries the expert geometry the test uses");
            check(cfg.value().hidden_size == f.config.hidden_size &&
                      cfg.value().num_hidden_layers == f.config.num_hidden_layers,
                  "the fixture config.json carries the base geometry the test uses");
            check(cfg.value().output_gate_silu, "a fixture config without output_gate_type is silu");
        }
    }

    const fs::path shard = root / "mxfp4" / "model.safetensors";
    auto reader = shelf::io::SafetensorsReader::open(shard.string());
    check(reader.ok(), "shard open");

    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        auto plan = build_moe_expert_group_plan(f.config, layer);
        if (!plan.ok()) {
            check(false, "plan for layer " + std::to_string(layer) + ": " + plan.status().message());
            continue;
        }
        if (reader.ok()) {
            test_file_order(reader.value(), plan.value());
        }
        if (layer == 0)
            test_manifest(root / "mxfp4", plan.value(), f);
    }

    fs::remove_all(root, ec);
    std::printf("test_moe_expert_slice_fixture: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
