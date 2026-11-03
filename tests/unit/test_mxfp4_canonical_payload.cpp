#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/manifest/manifest_v4.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/manifest/runtime_resolution.h>
#include <shelf/quantization/payload_descriptor.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/quant_spec.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/weights/matrix_weight.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace manifest = shelf::quantization::manifest;
namespace quant = shelf::quantization;
namespace weights = shelf::weights;

using quant::QuantPayloadRef;
using quant::QuantPayloadRole;
using quant::QuantScalarType;
using quant::QuantizedTensorDescriptor;
using quant::ScaleGranularity;
using quant::WeightQuantSpec;
using quant::WeightScaleSpec;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static QuantizedTensorDescriptor make_descriptor(
    uint64_t n, uint64_t k, uint64_t k_padded, uint64_t block_n, uint64_t block_k,
    QuantScalarType scale_dtype, const std::string& layout_id) {
    QuantizedTensorDescriptor desc;
    desc.domain = quant::QuantDomain::Weight;
    desc.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
    WeightQuantSpec spec;
    spec.encoding = manifest::QuantizedEncoding::Mxfp4;
    WeightScaleSpec scale;
    scale.granularity = ScaleGranularity::MatrixBlock;
    scale.block.n = static_cast<uint32_t>(block_n);
    scale.block.k = static_cast<uint32_t>(block_k);
    scale.dtype = scale_dtype;
    spec.scale = scale;
    desc.quant = spec;
    desc.layout.layout_id = layout_id;
    desc.layout.storage_order = quant::TensorStorageOrder::RowMajor;
    desc.layout.k_padded = k_padded;
    QuantPayloadRef codes;
    codes.role = QuantPayloadRole::Codes;
    codes.tensor = "x.weight";
    codes.dtype = QuantScalarType::U8;
    codes.shape = {n, k_padded / quant::kMxfp4WeightsPerCode};
    QuantPayloadRef scales;
    scales.role = QuantPayloadRole::Scale;
    scales.tensor = "x.weight_scale";
    scales.dtype = scale_dtype;
    scales.shape = {n, k_padded / block_k};
    desc.payloads = {codes, scales};
    return desc;
}

static void test_encoding_names() {
    check(manifest::quantized_encoding_name(manifest::QuantizedEncoding::Mxfp4) ==
              manifest::kMxfp4EncodingName,
          "on-disk mxfp4 encoding name");
    auto parsed = manifest::parse_quantized_encoding(manifest::kMxfp4EncodingName);
    check(parsed.ok() && parsed.value() == manifest::QuantizedEncoding::Mxfp4,
          "on-disk mxfp4 encoding parses");
    check(manifest::parse_quantized_encoding("nvfp4").status().code() ==
              shelf::Status::Code::invalid_argument,
          "unknown on-disk encoding rejected");
    check(std::string(manifest::v4_encoding_name(manifest::QuantizedEncoding::Mxfp4)) ==
              manifest::kMxfp4V4EncodingName,
          "v4 mxfp4 encoding name");
    auto v4 = manifest::parse_v4_encoding(manifest::kMxfp4V4EncodingName);
    check(v4.ok() && v4.value() == manifest::QuantizedEncoding::Mxfp4, "v4 mxfp4 encoding parses");
    check(quant::canonical_layout_name(manifest::QuantizedEncoding::Mxfp4) ==
              std::string(manifest::kMxfp4LayoutName),
          "mxfp4 canonical layout name");
    check(manifest::quant_format_id(manifest::QuantizedEncoding::Mxfp4) ==
              quant::QuantFormatId::Mxfp4Finegrained,
          "mxfp4 format id");
    const quant::QuantFormatDesc* format =
        quant::quant_format_desc(quant::QuantFormatId::Mxfp4Finegrained);
    check(format != nullptr, "mxfp4 format descriptor present");
    if (format == nullptr) return;
    check(format->block_elements == 32u, "mxfp4 format block elements");
    check(format->code_bytes_per_block == 16u, "mxfp4 format code bytes per block");
    check(format->meta[0].count_mode == quant::MetaCountMode::PerBlock,
          "mxfp4 scale is per (row, block)");
    check(format->meta[0].element_bytes == 1u, "mxfp4 scale element bytes");
    check(format->meta[0].storage == quant::MetaStorage::U8, "mxfp4 scale storage U8");
    check(format->meta[1].count_mode == quant::MetaCountMode::None, "mxfp4 has no metadata2");
}

static void test_weight_scale_spec() {
    const WeightScaleSpec expected = *quant::default_weight_scale_spec(manifest::QuantizedEncoding::Mxfp4);
    check(expected.granularity == ScaleGranularity::MatrixBlock, "mxfp4 scale granularity");
    check(expected.block.n == 1u && expected.block.k == 32u, "mxfp4 scale block 1x32");
    check(expected.dtype == QuantScalarType::E8M0, "mxfp4 scale dtype E8M0");
    check(expected.application == quant::ScaleApplication::Multiply, "mxfp4 scale multiplies");

    WeightQuantSpec spec = quant::default_weight_quant_spec(manifest::QuantizedEncoding::Mxfp4);
    check(quant::validate_weight_scale_spec(spec).ok(), "mxfp4 default weight spec accepted");

    spec.scale->dtype = QuantScalarType::F32;
    check(!quant::validate_weight_scale_spec(spec).ok(), "mxfp4 F32 scale rejected");
    spec = quant::default_weight_quant_spec(manifest::QuantizedEncoding::Mxfp4);

    spec.scale->block.k = 64u;
    check(quant::validate_weight_scale_spec(spec).ok(),
          "a non-default mxfp4 group is accepted at the format layer");
    check(!quant::weight_block_supported(manifest::QuantizedEncoding::Mxfp4, {1u, 64u}),
          "mxfp4 block k 64 is outside the kernel capability");
    spec = quant::default_weight_quant_spec(manifest::QuantizedEncoding::Mxfp4);

    spec.scale->block.n = 128u;
    check(quant::validate_weight_scale_spec(spec).ok(),
          "a shared-row mxfp4 scale tile is accepted at the format layer");
    check(!quant::weight_block_supported(manifest::QuantizedEncoding::Mxfp4, {128u, 32u}),
          "mxfp4 block n 128 is outside the kernel capability");
    spec.scale->block.n = 1u;
    check(quant::weight_block_supported(manifest::QuantizedEncoding::Mxfp4, {1u, 32u}),
          "mxfp4 block 1x32 supported");

    spec.scale.reset();
    check(!quant::validate_weight_scale_spec(spec).ok(), "mxfp4 without a scale rejected");

    const quant::ScaleGrid grid = quant::derive_scale_grid(130u, 2048u, {1u, 32u});
    check(grid.n == 130u && grid.k == 64u, "mxfp4 scale grid rows x groups");
    check(quant::scale_index(7u, 96u, {1u, 32u}, grid) == 7u * 64u + 3u,
          "mxfp4 scale index is row-major per row and group");
}

static void test_payload_contract() {
    QuantizedTensorDescriptor desc = make_descriptor(
        130u, 2048u, 2048u, 1u, 32u, QuantScalarType::E8M0, manifest::kMxfp4LayoutName);
    check(quant::validate_payload_contract(desc).ok(), "mxfp4 payload contract accepted");
    check(quant::validate_quantized_tensor(desc).ok(), "mxfp4 quantized tensor accepted");

    desc.payloads[0].shape = {130u, 2048u};
    check(!quant::validate_payload_contract(desc).ok(),
          "mxfp4 codes sized as one byte per weight rejected");

    desc = make_descriptor(130u, 2048u, 2048u, 1u, 32u, QuantScalarType::E8M0,
                           manifest::kMxfp4LayoutName);
    desc.payloads[1].shape = {2u, 64u};
    check(!quant::validate_payload_contract(desc).ok(),
          "mxfp4 FP8-style scale tile grid rejected");

    desc = make_descriptor(130u, 2048u, 2048u, 1u, 32u, QuantScalarType::E8M0,
                           manifest::kMxfp4LayoutName);
    QuantPayloadRef data;
    data.role = QuantPayloadRole::Data;
    data.tensor = "x.bf16";
    data.dtype = QuantScalarType::BF16;
    data.shape = {130u, 2048u};
    desc.payloads.push_back(data);
    check(!quant::validate_payload_contract(desc).ok(), "mxfp4 must not carry a data stream");

    desc = make_descriptor(130u, 2048u, 2048u, 1u, 32u, QuantScalarType::E8M0,
                           manifest::kFp8LayoutName);
    check(quant::validate_payload_contract(desc).ok(), "payload sizes ignore layout id");
    check(!quant::validate_quantized_tensor(desc).ok(), "mxfp4 layout id mismatch rejected");

    desc = make_descriptor(130u, 2050u, 2050u, 1u, 32u, QuantScalarType::E8M0,
                           manifest::kMxfp4LayoutName);
    check(!quant::validate_payload_contract(desc).ok(), "mxfp4 unaligned k_padded rejected");
}

static void test_kernel_capability() {
    using quant::QuantKernelKind;
    QuantizedTensorDescriptor desc = make_descriptor(
        130u, 2048u, 2048u, 1u, 32u, QuantScalarType::E8M0, manifest::kMxfp4LayoutName);
    check(quant::validate_kernel_capability(desc, QuantKernelKind::LinearMxfp4Finegrained).ok(),
          "mxfp4 kernel accepts mxfp4 weight");
    QuantizedTensorDescriptor wide_group = make_descriptor(
        130u, 2048u, 2048u, 1u, 64u, QuantScalarType::E8M0, manifest::kMxfp4LayoutName);
    check(quant::validate_payload_contract(wide_group).ok(),
          "group 64 payload sizes are self-consistent");
    check(quant::validate_kernel_capability(wide_group, QuantKernelKind::LinearMxfp4Finegrained)
              .code() == shelf::Status::Code::unsupported,
          "mxfp4 kernel rejects a group 64 weight");
    check(quant::validate_kernel_capability(desc, QuantKernelKind::LinearFp8Finegrained)
              .code() == shelf::Status::Code::unsupported,
          "fp8 kernel rejects mxfp4 weight");
    check(quant::weight_scale_group_supported(32u, QuantKernelKind::LinearMxfp4Finegrained),
          "mxfp4 group 32 supported");
    check(!quant::weight_scale_group_supported(128u, QuantKernelKind::LinearMxfp4Finegrained),
          "mxfp4 group 128 unsupported");
    check(quant::weight_scale_group_is_default(32u, QuantKernelKind::LinearMxfp4Finegrained),
          "mxfp4 group 32 is the default");
    check(quant::validate_weight_scale_group(32u, QuantKernelKind::LinearMxfp4Finegrained).ok(),
          "mxfp4 weight scale group accepted");
    check(!quant::validate_weight_scale_group(64u, QuantKernelKind::LinearMxfp4Finegrained).ok(),
          "mxfp4 weight scale group 64 rejected");

    quant::RuntimeWeightQuantDesc runtime =
        quant::make_runtime_weight_quant_desc(manifest::QuantizedEncoding::Mxfp4, 32u);
    check(runtime.block_n == 1u && runtime.block_k == 32u, "mxfp4 runtime block shape");
    check(runtime.scale_dtype == QuantScalarType::E8M0, "mxfp4 runtime scale dtype");
    check(quant::validate_runtime_kernel_capability(runtime, QuantKernelKind::LinearMxfp4Finegrained).ok(),
          "mxfp4 runtime capability accepted");
    runtime.block_k = 128u;
    check(!quant::validate_runtime_kernel_capability(runtime, QuantKernelKind::LinearMxfp4Finegrained).ok(),
          "mxfp4 runtime capability rejects group 128");
}

static void test_compute_spec_pairing() {
    check(quant::preferred_gemm_compute_spec(quant::QuantSpecId::MXFP4_E8M0_G32_V1).value() ==
              quant::ComputeSpecId::MXFP4_W4A8_F32_BF16,
          "mxfp4 storage maps to W4A8 compute");
    check(quant::is_gemm_compute_compatible(quant::QuantSpecId::MXFP4_E8M0_G32_V1,
                                            quant::ComputeSpecId::MXFP4_W4A8_F32_BF16),
          "mxfp4 W4A8 pairing compatible");
    check(!quant::is_gemm_compute_compatible(quant::QuantSpecId::MXFP4_E8M0_G32_V1,
                                             quant::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16),
          "mxfp4 with the FP8 finegrained compute spec is not compatible");
    check(!quant::is_gemm_compute_compatible(quant::QuantSpecId::FP8_FINEGRAINED_F32_V1,
                                             quant::ComputeSpecId::MXFP4_W4A8_F32_BF16),
          "FP8 storage cannot run the mxfp4 compute spec");
    check(weights::encoding_to_quant_spec(weights::MatrixEncoding::Mxfp4) ==
              quant::QuantSpecId::MXFP4_E8M0_G32_V1,
          "MatrixEncoding::Mxfp4 maps to the mxfp4 quant spec");
    check(weights::encoding_to_compute_spec(weights::MatrixEncoding::Mxfp4) ==
              quant::ComputeSpecId::MXFP4_W4A8_F32_BF16,
          "MatrixEncoding::Mxfp4 maps to the W4A8 compute spec");
    check(weights::quant_spec_to_encoding(quant::QuantSpecId::MXFP4_E8M0_G32_V1) ==
              weights::MatrixEncoding::Mxfp4,
          "mxfp4 quant spec maps back to the encoding");
    check(static_cast<uint8_t>(weights::MatrixEncoding::Mxfp4) == 3u,
          "MatrixEncoding::Mxfp4 keeps the weight table encoding value 3");
}

static manifest::QuantizedTensorMetadata make_manifest_tensor(
    const std::string& codes_dtype, const std::string& scale_dtype, uint64_t k_padded,
    const std::string& layout, bool with_scale) {
    manifest::QuantizedTensorMetadata t;
    t.role = "weight";
    t.encoding = manifest::QuantizedEncoding::Mxfp4;
    t.layout = layout;
    t.logical_shape = {130, 2048};
    t.k_padded = k_padded;
    t.scale_block_n = 1u;
    t.scale_block_k = 32u;
    t.codes = manifest::QuantizedTensorRef{"x.weight", codes_dtype, "0x00000000"};
    if (with_scale)
        t.metadata1 = manifest::QuantizedTensorRef{"x.weight_scale", scale_dtype, "0x00000000"};
    return t;
}

static void test_manifest_resolution() {
    manifest::QuantizedManifest m;
    m.architecture = "qwen35_dense";
    m.tensors["x"] = make_manifest_tensor("U8", "E8M0", 2048u, manifest::kMxfp4LayoutName, true);
    check(manifest::validate_quantized_manifest_contract(m).ok(), "mxfp4 manifest contract ok");
    auto resolved = manifest::resolve_quant_spec(m, m.tensors.at("x"));
    check(resolved.ok() &&
              resolved.value().spec_id == quant::QuantSpecId::MXFP4_E8M0_G32_V1,
          "mxfp4 manifest resolves to the mxfp4 spec");

    struct BadCase {
        const char* name;
        std::string codes_dtype;
        std::string scale_dtype;
        uint64_t k_padded;
        std::string layout;
        bool with_scale;
    };
    const BadCase cases[] = {
        {"mxfp4 codes dtype BF16 rejected", "BF16", "E8M0", 2048u, manifest::kMxfp4LayoutName, true},
        {"mxfp4 scale dtype F32 rejected", "U8", "F32", 2048u, manifest::kMxfp4LayoutName, true},
        {"mxfp4 without a scale ref rejected", "U8", "E8M0", 2048u, manifest::kMxfp4LayoutName, false},
        {"mxfp4 unaligned k_padded rejected", "U8", "E8M0", 2050u, manifest::kMxfp4LayoutName, true},
        {"mxfp4 fp8 layout rejected", "U8", "E8M0", 2048u, manifest::kFp8LayoutName, true},
    };
    for (const BadCase& c : cases) {
        manifest::QuantizedManifest bad;
        bad.tensors["x"] =
            make_manifest_tensor(c.codes_dtype, c.scale_dtype, c.k_padded, c.layout, c.with_scale);
        check(!manifest::validate_quantized_manifest_contract(bad).ok(), c.name);
    }
}

static void test_hf_method_enum() {
    check(quant::bundle::HfQuantMethod::QuarkMxfp4 != quant::bundle::HfQuantMethod::Fp8,
          "quark mxfp4 is a distinct HF method");
    check(quant::bundle::HfQuantMethod::None == static_cast<quant::bundle::HfQuantMethod>(0),
          "HF method none stays zero");
}


static uint8_t canonical_nibble(const std::vector<uint8_t>& codes, uint64_t row, uint64_t k,
                                uint64_t packed_k) {
    const uint8_t byte = codes[static_cast<std::size_t>(row) * packed_k + (k >> 1)];
    return (k & 1u) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0Fu);
}

static void test_native_preshuffle() {
    namespace mx = quant::mxfp4;
    const uint64_t n = 17;
    const uint64_t kp = 64;
    const uint64_t nb = kp / quant::kMxfp4ScaleBlockK;
    const uint64_t packed_k = kp / quant::kMxfp4WeightsPerCode;
    std::vector<uint8_t> codes(static_cast<std::size_t>(n) * packed_k, 0u);
    for (uint64_t r = 0; r < n; ++r)
        for (uint64_t b = 0; b < packed_k; ++b)
            codes[static_cast<std::size_t>(r) * packed_k + b] =
                static_cast<uint8_t>(((r * 7u + b * 3u) & 0xFu) |
                                     (((r + b * 5u) & 0xFu) << 4));
    std::vector<uint8_t> scales(static_cast<std::size_t>(n) * nb, 0u);
    for (uint64_t i = 0; i < scales.size(); ++i) scales[i] = static_cast<uint8_t>(119u + i);

    mx::Mxfp4CanonicalView cv;
    cv.batch = 1;
    cv.n = n;
    cv.padded_k = kp;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    cv.scales = scales.data();
    cv.scales_bytes = scales.size();
    check(cv.validate(), "canonical mxfp4 view of the preshuffle input");

    mx::Mxfp4NativeHost host;
    check(mx::preshuffle_mxfp4_native(cv, host), "mxfp4 native preshuffle succeeds");
    check(host.rows == 32 && host.blocks == nb, "native rows are padded to 16-row tiles");
    check(host.codes_row_stride_bytes == nb * 256u, "native code stride is one tile per block run");
    check(host.scale_row_stride_bytes == nb, "native scale stride stays per row");
    check(host.codes.size() == 2 * nb * 256u, "native codes bytes are 16 x 32 blocks of 256 B");
    check(host.scales == scales, "native E8M0 scales are copied unchanged");
    check(host.view().validate(), "native view satisfies its own contract");

    bool pairing = true;
    bool padded_zero = true;
    for (uint64_t t = 0; t < 2; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint32_t ol = 0; ol < 16; ++ol) {
                const uint64_t row = t * 16 + ol;
                for (uint32_t b = 0; b < 16; ++b) {
                    const uint8_t got = host.codes[(t * nb + ib) * 256 + ol * 16 + b];
                    if (row >= n) {
                        if (got != 0u) padded_zero = false;
                        continue;
                    }
                    const uint8_t lo = canonical_nibble(codes, row, ib * 32 + b, packed_k);
                    const uint8_t hi = canonical_nibble(codes, row, ib * 32 + 16 + b, packed_k);
                    if (got != static_cast<uint8_t>(lo | (hi << 4))) pairing = false;
                }
            }
        }
    }
    check(pairing, "native byte holds code[k] in the low nibble and code[k+16] in the high nibble");
    check(padded_zero, "native padding rows are zero filled");

    mx::Mxfp4CanonicalView two = cv;
    two.batch = 2;
    two.codes_bytes = 2 * codes.size();
    two.scales_bytes = 2 * scales.size();
    std::vector<uint8_t> big_codes(two.codes_bytes);
    for (uint64_t b = 0; b < 2; ++b)
        std::copy(codes.begin(), codes.end(),
                  big_codes.begin() + static_cast<std::ptrdiff_t>(b * codes.size()));
    std::vector<uint8_t> big_scales(two.scales_bytes);
    for (uint64_t b = 0; b < 2; ++b)
        std::copy(scales.begin(), scales.end(),
                  big_scales.begin() + static_cast<std::ptrdiff_t>(b * scales.size()));
    two.codes = big_codes.data();
    two.scales = big_scales.data();
    check(two.validate(), "batched canonical mxfp4 view is accepted");
    mx::Mxfp4NativeHost batched;
    check(mx::preshuffle_mxfp4_native(two, batched), "batched native preshuffle succeeds");
    check(batched.codes.size() == 2 * host.codes.size(), "batched native payload stacks matrices");
    bool batch_match = true;
    for (std::size_t i = 0; i < host.codes.size(); ++i) {
        if (batched.codes[i] != host.codes[i] ||
            batched.codes[i + host.codes.size()] != host.codes[i])
            batch_match = false;
    }
    check(batch_match, "each batch entry is preshuffled independently and identically");

    mx::Mxfp4NativeView bad = host.view();
    bad.codes_row_stride_bytes = packed_k;
    check(!bad.validate(), "a canonical code stride is rejected by the native layout contract");
    bad = host.view();
    bad.scale_row_stride_bytes = 0u;
    check(!bad.validate(), "a zero native scale stride is rejected");
}

int main() {
    test_encoding_names();
    test_weight_scale_spec();
    test_payload_contract();
    test_kernel_capability();
    test_compute_spec_pairing();
    test_manifest_resolution();
    test_hf_method_enum();
    test_native_preshuffle();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
