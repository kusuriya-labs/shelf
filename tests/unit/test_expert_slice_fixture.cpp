#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/quantization/bundle/hf_manifest.h>
#include <phaseshift/quantization/fp8/fp8.h>
#include <phaseshift/quantization/fp8/fp8_native.h>
#include <phaseshift/weights/weight_loader.h>

#include "expert_slice_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace expert_slice_fixture;
using ps::io::SType;
using ps::io::StAxisSpan;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static void check_span(const StAxisSpan& span, std::uint64_t unit_bytes,
                       std::uint64_t byte_length, const std::string& msg) {
    check(span.unit_bytes == unit_bytes, msg + " unit_bytes");
    check(span.byte_length == byte_length, msg + " byte_length");
}

static void test_bf16_spans(const ps::weights::SafetensorsCollection& col) {
    auto gate_spec = col.tensor_spec(kGateUpName);
    check(gate_spec.ok(), "bf16 gate_up spec");
    if (gate_spec.ok()) {
        check(gate_spec.value().dtype == SType::BF16, "bf16 gate_up dtype");
        check(gate_spec.value().shape == std::vector<std::size_t>{kExperts, kGateUpRows, kHidden},
              "bf16 gate_up shape");
    }
    auto down_spec = col.tensor_spec(kDownName);
    check(down_spec.ok() &&
              down_spec.value().shape == std::vector<std::size_t>{kExperts, kDownRows, kInter},
          "bf16 down shape");

    auto gate_base = col.tensor_span(kGateUpName);
    check(gate_base.ok(), "bf16 gate_up tensor_span");
    const std::uint64_t gate_unit = gate_up_expert_elements() * 2;
    for (std::uint64_t e = 0; e < kExperts; ++e) {
        auto span = col.axis_span(kGateUpName, e, 1);
        check(span.ok(), "bf16 gate_up expert span");
        if (!span.ok() || !gate_base.ok()) continue;
        check(span.value().absolute_offset == gate_base.value().absolute_offset + e * gate_unit,
              "bf16 gate_up expert offset e=" + std::to_string(e));
        check_span(span.value(), gate_unit, gate_unit,
                   "bf16 gate_up expert e=" + std::to_string(e));
    }
    auto all = col.axis_span(kGateUpName, 0, kExperts);
    check(all.ok() && all.value().byte_length == gate_base.value().byte_length,
          "bf16 gate_up full span covers tensor");
    check(!col.axis_span(kGateUpName, kExperts, 1).ok(), "bf16 gate_up invalid expert rejected");
    check(!col.axis_span(kGateUpName, 0, 0).ok(), "bf16 gate_up empty range rejected");

    auto down_base = col.tensor_span(kDownName);
    check(down_base.ok(), "bf16 down tensor_span");
    const std::uint64_t down_unit = down_expert_elements() * 2;
    auto down_span = col.axis_span(kDownName, 2, 1);
    check(down_span.ok(), "bf16 down expert span");
    if (down_span.ok() && down_base.ok()) {
        check(down_span.value().absolute_offset ==
                  down_base.value().absolute_offset + 2 * down_unit,
              "bf16 down expert offset");
        check_span(down_span.value(), down_unit, down_unit, "bf16 down expert");
    }
}

static void test_fp8_manifest_contract(const fs::path& dir) {
    check(ps::quantization::bundle::is_hf_quantized_model_dir(dir.string()),
          "fp8 dir recognized as hf quantized");
    auto method = ps::quantization::bundle::detect_hf_quant_method(dir.string());
    check(method.ok() &&
              method.value() == ps::quantization::bundle::HfQuantMethod::Fp8,
          "fp8 config accepted by detect_hf_quant_method");

    auto source = ps::quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), {kGateUpName}, false);
    check(!source.ok(), "rank-3 fp8 expert rejected by hf manifest");
    if (!source.ok()) {
        check(source.status().code() == ps::Status::Code::unsupported,
              "rank-3 fp8 expert rejection is unsupported");
    }

    fs::path bad_dir = dir.parent_path() / "fp8_bad_config";
    std::error_code ec;
    fs::create_directories(bad_dir, ec);
    {
        std::ofstream config((bad_dir / "config.json").string(),
                             std::ios::binary | std::ios::trunc);
        config << "{\"quantization_config\":{\"quant_method\":\"fp8\","
                  "\"weight_block_size\":[128,128]}}";
    }
    check(!ps::quantization::bundle::detect_hf_quant_method(bad_dir.string()).ok(),
          "fp8 config without fmt e4m3 rejected");
}

static void test_fp8_spans(const ps::weights::SafetensorsCollection& col) {
    auto codes_spec = col.tensor_spec(kGateUpName);
    check(codes_spec.ok() && codes_spec.value().dtype == SType::F8_E4M3,
          "fp8 gate_up codes dtype");
    check(codes_spec.ok() &&
              codes_spec.value().shape == std::vector<std::size_t>{kExperts, kGateUpRows, kHidden},
          "fp8 gate_up codes shape");
    auto scale_spec = col.tensor_spec(kGateUpScaleName);
    check(scale_spec.ok() && scale_spec.value().dtype == SType::BF16,
          "fp8 gate_up scale dtype is BF16");
    check(scale_spec.ok() &&
              scale_spec.value().shape ==
                  std::vector<std::size_t>{kExperts, kGateUpRows / kScaleBlockN,
                                           kHidden / kScaleBlockK},
          "fp8 gate_up scale grid shape");

    auto codes_base = col.tensor_span(kGateUpName);
    auto scale_base = col.tensor_span(kGateUpScaleName);
    check(codes_base.ok() && scale_base.ok(), "fp8 tensor spans");
    const std::uint64_t codes_unit = gate_up_expert_elements();
    const std::uint64_t scale_unit = gate_up_scale_elements() * 2;
    for (std::uint64_t e = 0; e < kExperts; ++e) {
        auto codes_span = col.axis_span(kGateUpName, e, 1);
        auto scale_span = col.axis_span(kGateUpScaleName, e, 1);
        check(codes_span.ok() && scale_span.ok(), "fp8 expert spans");
        if (!codes_span.ok() || !scale_span.ok() || !codes_base.ok() || !scale_base.ok()) continue;
        check(codes_span.value().absolute_offset ==
                  codes_base.value().absolute_offset + e * codes_unit,
              "fp8 codes expert offset e=" + std::to_string(e));
        check(scale_span.value().absolute_offset ==
                  scale_base.value().absolute_offset + e * scale_unit,
              "fp8 scale expert offset e=" + std::to_string(e));
        check_span(codes_span.value(), codes_unit, codes_unit,
                   "fp8 codes expert e=" + std::to_string(e));
        check_span(scale_span.value(), scale_unit, scale_unit,
                   "fp8 scale expert e=" + std::to_string(e));
    }
    check(!col.axis_span(kGateUpName, kExperts, 1).ok(), "fp8 invalid expert rejected");
    check(!col.axis_span(kGateUpScaleName, kExperts, 1).ok(),
          "fp8 invalid scale expert rejected");
}

static void test_preshuffle_contract(const Fixture& f) {
    using ps::quantization::fp8::Fp8BlockCanonicalView;
    using ps::quantization::fp8::Fp8BlockCanonicalStore;
    using ps::quantization::fp8::Fp8NativeHost;

    const std::uint64_t codes_per = gate_up_expert_elements();
    const std::uint64_t scales_per = gate_up_scale_elements();

    for (std::uint64_t e :
         {std::uint64_t{0}, std::uint64_t{2}, static_cast<std::uint64_t>(kExperts - 1)}) {
        const std::vector<float> scales_f =
            scale_region_f32(f.gate_up_scales, scales_per, e * scales_per);
        Fp8BlockCanonicalView cv;
        cv.batch = 1;
        cv.n = kGateUpRows;
        cv.padded_k = kHidden;
        cv.codes = f.gate_up_codes.data() + e * codes_per;
        cv.codes_bytes = codes_per;
        cv.scales = scales_f.data();
        cv.scales_bytes = scales_f.size() * sizeof(float);
        cv.scale_n = kGateUpRows / kScaleBlockN;
        cv.scale_k = kHidden / kScaleBlockK;
        cv.scale_block_k = kScaleBlockK;
        check(cv.validate(), "expert canonical view validates e=" + std::to_string(e));

        Fp8NativeHost native;
        const bool ok = ps::quantization::fp8::preshuffle_fp8_native(cv, native);
        check(ok, "expert preshuffle accepts one expert e=" + std::to_string(e));
        if (!ok) continue;
        check(native.n == kGateUpRows, "preshuffled rows");
        check(native.padded_k == kHidden, "preshuffled padded_k");
        check(native.codes_row_stride_bytes == (kHidden / 32u) * 512u,
              "preshuffled codes stride");
        check(native.scale_n == kGateUpRows / kScaleBlockN &&
                  native.scale_k == kHidden / kScaleBlockK,
              "preshuffled scale grid");
    }

    const std::vector<float> scales_all =
        scale_region_f32(f.gate_up_scales, scales_per * kExperts, 0);
    Fp8BlockCanonicalView whole;
    whole.batch = kExperts;
    whole.n = kGateUpRows;
    whole.padded_k = kHidden;
    whole.codes = f.gate_up_codes.data();
    whole.codes_bytes = f.gate_up_codes.size();
    whole.scales = scales_all.data();
    whole.scales_bytes = scales_all.size() * sizeof(float);
    whole.scale_n = kGateUpRows / kScaleBlockN;
    whole.scale_k = kHidden / kScaleBlockK;
    whole.scale_block_k = kScaleBlockK;
    Fp8NativeHost native;
    check(!ps::quantization::fp8::preshuffle_fp8_native(whole, native),
          "fused rank-3 payload rejected by preshuffle");
}

int main() {
    std::printf("=== Expert axis-slice fixture (CPU) ===\n");
    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("ps_expert_slice_cpu_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    const Fixture f = make_fixture();
    const fs::path bf16_dir = root / "bf16";
    const fs::path fp8_dir = root / "fp8";
    check(write_bf16_model_dir(bf16_dir, f), "bf16 fixture written");
    check(write_fp8_model_dir(fp8_dir, f), "fp8 fixture written");

    auto bf16_col = ps::weights::SafetensorsCollection::open(bf16_dir.string());
    check(bf16_col.ok(), "bf16 collection open");
    if (bf16_col.ok()) test_bf16_spans(bf16_col.value());

    auto fp8_col = ps::weights::SafetensorsCollection::open(fp8_dir.string());
    check(fp8_col.ok(), "fp8 collection open");
    if (fp8_col.ok()) test_fp8_spans(fp8_col.value());

    test_fp8_manifest_contract(fp8_dir);
    test_preshuffle_contract(f);

    fs::remove_all(root, ec);
    std::printf("%s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
