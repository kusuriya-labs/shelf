#include <phaseshift/quantization/manifest/quantized_manifest.h>
#include <phaseshift/quantization/quant_format.h>
#include <phaseshift/quantization/quantization_types.h>
#include <cstdio>

using ps::quantization::QuantFormatDesc;
using ps::quantization::QuantFormatId;
using ps::quantization::quant_codes_bytes;
using ps::quantization::quant_element_count;
using ps::quantization::quant_format_desc;
using ps::quantization::quant_metadata_bytes;
using ps::quantization::quant_metadata_count;
using ps::quantization::quant_padded_k;
using ps::quantization::quant_row_count;
using ps::quantization::manifest::QuantizedEncoding;
using ps::quantization::manifest::quant_format_id;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) { ++g_passed; } else { ++g_failed; std::printf("FAIL: %s\n", msg); }
}

int run() {
    check(quant_padded_k(1024, QuantFormatId::Bf16) == 1024, "bf16 k_padded==k");
    check(quant_padded_k(1000, QuantFormatId::Bf16) == 1000, "bf16 never pads");
    check(quant_padded_k(1000, QuantFormatId::Fp8) == 1024, "fp8 k_padded aligns 128");

    check(quant_element_count({1024}) == 1024, "1d element_count");
    check(quant_row_count({1024}) == 1, "1d row_count==1");
    check(quant_row_count({3, 1024}) == 3, "2d row_count");
    check(quant_row_count({6144, 1, 4}) == 6144, "3d conv1d row_count");
    check(quant_element_count({6144, 1, 4}) == 6144 * 4, "3d conv1d element_count");

    const QuantFormatDesc& fp8 = *quant_format_desc(QuantFormatId::Fp8);
    check(quant_codes_bytes(3, 1024, fp8) == 3 * 1024, "fp8 codes bytes");
    check(quant_metadata_count(1, 256, 256, fp8, 0) == 2 * 2, "fp8 scale grid 256x256");
    check(quant_metadata_bytes(1, 256, 256, fp8, 0) == 4 * 4, "fp8 scale bytes");
    check(quant_metadata_count(1, 17408, 5120, fp8, 0) == 136 * 40, "fp8 scale grid 136x40");
    check(quant_metadata_count(1, 255, 384, fp8, 0) == 2 * 3, "fp8 scale grid ceil n and k");

    const QuantFormatDesc& bf16 = *quant_format_desc(QuantFormatId::Bf16);
    check(quant_codes_bytes(3, 1024, bf16) == 3 * 1024 * 2, "bf16 data bytes");

    check(quant_format_id(QuantizedEncoding::Bf16) == QuantFormatId::Bf16, "map bf16");
    check(quant_format_id(QuantizedEncoding::Fp8) == QuantFormatId::Fp8, "map fp8");

    using ps::quantization::ComputeSpecId;
    using ps::quantization::QuantSpecId;
    using ps::quantization::is_gemm_compute_compatible;
    check(is_gemm_compute_compatible(QuantSpecId::BF16_V1, ComputeSpecId::BF16_F32_BF16),
          "bf16 storage maps to bf16 compute");
    check(is_gemm_compute_compatible(QuantSpecId::FP8_E4M3_F32_V1, ComputeSpecId::FP8_W8A8_F32_BF16),
          "fp8 storage maps to fp8 compute");
    check(!is_gemm_compute_compatible(QuantSpecId::BF16_V1, ComputeSpecId::FP8_W8A8_F32_BF16),
          "bf16 storage rejects fp8 compute");
    check(!is_gemm_compute_compatible(QuantSpecId::FP8_E4M3_F32_V1, ComputeSpecId::BF16_F32_BF16),
          "fp8 storage rejects bf16 compute");
    return 0;
}
}  // namespace

int main() {
    run();
    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
