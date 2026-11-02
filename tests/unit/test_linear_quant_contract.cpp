#include <phaseshift/quantization/quant_spec.h>
#include <phaseshift/quantization/quantization_types.h>
#include <phaseshift/runtime/program/linear_quant_contract.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace rt = ps::runtime;
namespace quant = ps::quantization;

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", message.c_str());
    }
}

uint8_t enc(quant::manifest::QuantizedEncoding value) {
    return static_cast<uint8_t>(value);
}

uint8_t spec(quant::ComputeSpecId value) {
    return static_cast<uint8_t>(value);
}

void expect_quantized(
    uint8_t weight_encoding,
    uint8_t compute_spec,
    rt::KernelId expected_kernel,
    uint32_t expected_alignment,
    const std::string& tag) {
    auto resolved = rt::resolve_linear_quant_contract(weight_encoding, compute_spec);
    check(resolved.ok(), tag + ": contract resolves");
    if (!resolved.ok()) return;
    const rt::ResolvedLinearQuantContract& c = resolved.value();
    check(c.linear_kernel == expected_kernel, tag + ": linear kernel");
    check(c.requires_activation_quant, tag + ": activation quant required");
    check(c.activation.codes_dtype == quant::QuantScalarType::FP8E4M3,
          tag + ": activation codes are E4M3");
    check(c.activation.group_k == 128u,
          tag + ": activation scale group is per-token-group 128");
    check(c.activation.scale_dtype == quant::QuantScalarType::F32,
          tag + ": activation scale dtype");
    check(c.activation_k_alignment == expected_alignment, tag + ": activation k alignment");
    check(quant::validate_activation_quant_spec(c.activation).ok(),
          tag + ": activation spec validates");
}

void expect_error(
    uint8_t weight_encoding,
    uint8_t compute_spec,
    const std::string& text,
    const std::string& tag) {
    auto resolved = rt::resolve_linear_quant_contract(weight_encoding, compute_spec);
    check(!resolved.ok(), tag + ": rejected");
    if (resolved.ok()) return;
    check(resolved.status().code() == ps::Status::Code::invalid_argument,
          tag + ": invalid argument status");
    check(resolved.status().message().find(text) != std::string::npos,
          tag + ": message keeps '" + text + "'");
}

}

int main() {
    {
        auto resolved = rt::resolve_linear_quant_contract(
            enc(quant::manifest::QuantizedEncoding::Bf16),
            spec(quant::ComputeSpecId::BF16_F32_BF16));
        check(resolved.ok(), "bf16: contract resolves");
        if (resolved.ok())
            check(!resolved.value().requires_activation_quant,
                  "bf16: no activation quant");
    }

    expect_quantized(
        enc(quant::manifest::QuantizedEncoding::Fp8),
        spec(quant::ComputeSpecId::FP8_W8A8_F32_BF16),
        rt::KernelId::LINEAR_FP8, 128u, "fp8");

    expect_error(
        enc(quant::manifest::QuantizedEncoding::Fp8),
        spec(quant::ComputeSpecId::BF16_F32_BF16),
        "fp8 linear weight requires fp8 compute spec", "fp8 bf16 spec");
    expect_error(
        enc(quant::manifest::QuantizedEncoding::Bf16),
        spec(quant::ComputeSpecId::FP8_W8A8_F32_BF16),
        "bf16 linear weight requires bf16 compute spec", "bf16 fp8 spec");

    expect_error(
        1u, spec(quant::ComputeSpecId::FP8_W8A8_F32_BF16),
        "bf16 linear weight requires bf16 compute spec", "unknown encoding fp8 spec");

    {
        auto resolved = rt::resolve_linear_quant_contract(2u, 0u);
        check(resolved.ok(), "unknown encoding with bf16 spec resolves");
        if (resolved.ok()) {
            check(resolved.value().linear_kernel == rt::KernelId::LINEAR_BF16,
                  "unknown encoding maps to the bf16 linear kernel");
            check(!resolved.value().requires_activation_quant,
                  "unknown encoding needs no activation quant");
        }
    }

    check(rt::activation_k_padded_for(32u, 2500u) == 2528u, "pad 32: 2500 -> 2528");
    check(rt::activation_k_padded_for(128u, 2500u) == 2560u, "pad 128: 2500 -> 2560");
    check(rt::activation_k_padded_for(128u, 2560u) == 2560u, "pad 128: 2560 -> 2560");
    check(rt::activation_k_padded_for(32u, 0u) == 0u, "pad 32: 0 -> 0");

    check(rt::activation_quantize_kernel(quant::kE4m3ActivationQuantSpec) ==
              rt::KernelId::ACTIVATION_QUANTIZE_W4A8,
          "e4m3 activation spec selects ACTIVATION_QUANTIZE_W4A8");
    check(rt::activation_quantize_kernel(quant::kI8ActivationQuantSpec) ==
              rt::KernelId::ACTIVATION_QUANTIZE_FP8,
          "i8 activation spec selects ACTIVATION_QUANTIZE_FP8");

    check(rt::activation_quant_matches(
              quant::kE4m3ActivationQuantSpec, quant::kE4m3ActivationQuantSpec),
          "e4m3 spec matches itself");
    check(!rt::activation_quant_matches(
              quant::kE4m3ActivationQuantSpec, quant::kI8ActivationQuantSpec),
          "e4m3 spec does not match the i8 spec");
    {
        quant::ActivationQuantSpec grouped = quant::kE4m3ActivationQuantSpec;
        grouped.group_k = 64u;
        check(!rt::activation_quant_matches(quant::kE4m3ActivationQuantSpec, grouped),
              "e4m3 spec does not match a differently grouped spec");
    }

    std::printf("test_linear_quant_contract: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
