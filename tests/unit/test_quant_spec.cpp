#include <shelf/quantization/payload_descriptor.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/quantization/quant_spec.h>
#include <cstdio>

using namespace shelf::quantization;
using shelf::quantization::manifest::QuantizedEncoding;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) { ++g_passed; } else { ++g_failed; std::printf("FAIL: %s\n", msg); }
}

QuantizedTensorDescriptor make_desc(QuantizedEncoding enc, uint64_t n, uint64_t k) {
    QuantizedTensorDescriptor d;
    d.domain = QuantDomain::Weight;
    d.logical_shape = {static_cast<int64_t>(n), static_cast<int64_t>(k)};
    d.quant = default_weight_quant_spec(enc);
    d.layout.storage_order = TensorStorageOrder::RowMajor;
    d.layout.layout_id = canonical_layout_name(enc);
    if (enc == QuantizedEncoding::Bf16) {
        d.layout.k_padded = k;
        QuantPayloadRef data;
        data.role = QuantPayloadRole::Data;
        data.dtype = QuantScalarType::BF16;
        for (int64_t dim : d.logical_shape) data.shape.push_back(static_cast<uint64_t>(dim));
        d.layout.physical_shape = data.shape;
        d.payloads.push_back(std::move(data));
        return d;
    }
    const WeightBlockShape block = default_weight_block(enc);
    const uint64_t k_padded = ceil_div(k, block.k) * block.k;
    d.layout.k_padded = k_padded;
    const QuantFormatDesc* format = quant_format_desc(manifest::quant_format_id(enc));
    QuantPayloadRef codes;
    codes.role = QuantPayloadRole::Codes;
    codes.dtype = QuantScalarType::U8;
    codes.shape = {n, quant_codes_bytes(1, k_padded, *format)};
    d.layout.physical_shape = codes.shape;
    d.payloads.push_back(codes);
    QuantPayloadRef scale;
    scale.role = QuantPayloadRole::Scale;
    scale.dtype = d.quant.scale->dtype;
    const ScaleGrid grid = derive_scale_grid(n, k_padded, block);
    scale.shape = {grid.n, grid.k};
    d.payloads.push_back(std::move(scale));
    return d;
}

int run() {
    {
        const ScaleGrid g = derive_scale_grid(256, 256, WeightBlockShape{128, 128});
        check(g.n == 2 && g.k == 2, "grid 256x256 block 128x128 -> 2x2");
        const ScaleGrid g2 = derive_scale_grid(17408, 5120, WeightBlockShape{128, 128});
        check(g2.n == 136 && g2.k == 40, "grid 17408x5120 block 128x128 -> 136x40");
        const ScaleGrid g3 = derive_scale_grid(255, 384, WeightBlockShape{128, 128});
        check(g3.n == 2 && g3.k == 3, "grid ceil n and k");
        const ScaleGrid g4 = derive_scale_grid(256, 256, WeightBlockShape{1, 32});
        check(g4.n == 256 && g4.k == 8, "grid 256x256 block 1x32 -> 256x8");
        check(scale_index(200, 150, WeightBlockShape{128, 128}, ScaleGrid{2, 2}) == 3,
              "scale index row-major");
    }

    {
        check(!default_weight_scale_spec(QuantizedEncoding::Bf16).has_value(),
              "bf16 default spec has no scale");
        const WeightScaleSpec fp8 = *default_weight_scale_spec(QuantizedEncoding::Fp8);
        check(fp8.block.n == 128 && fp8.block.k == 128 && fp8.dtype == QuantScalarType::F32,
              "fp8 default scale 128x128 f32");
    }

    check(validate_quantized_tensor(make_desc(QuantizedEncoding::Bf16, 1024, 1024)).ok(),
          "bf16 descriptor valid");
    check(validate_quantized_tensor(make_desc(QuantizedEncoding::Fp8, 17408, 5120)).ok(),
          "fp8 descriptor valid");

    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.payloads.erase(d.payloads.begin());
        check(!validate_quantized_tensor(d).ok(), "fp8 rejects missing codes");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.payloads[1].shape = {256, 7};
        check(!validate_quantized_tensor(d).ok(), "fp8 rejects scale shape mismatch");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.payloads.push_back(d.payloads[0]);
        check(!validate_quantized_tensor(d).ok(), "fp8 rejects duplicate codes role");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.quant.scale->block = WeightBlockShape{1, 16};
        check(!validate_quantized_tensor(d).ok(), "rejects scale block k below 32");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.quant.scale->block = WeightBlockShape{3, 32};
        check(!validate_quantized_tensor(d).ok(), "rejects non power-of-two scale block");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.layout.k_padded = 320;
        check(!validate_quantized_tensor(d).ok(), "fp8 rejects k_padded not multiple of 128");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.layout.layout_id = "unknown_layout_v1";
        check(!validate_quantized_tensor(d).ok(), "rejects unknown layout_id");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Bf16, 256, 256);
        QuantPayloadRef codes;
        codes.role = QuantPayloadRole::Codes;
        codes.dtype = QuantScalarType::U8;
        codes.shape = {256, 256};
        d.payloads.push_back(std::move(codes));
        check(!validate_quantized_tensor(d).ok(), "bf16 rejects codes payload");
    }
    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.domain = QuantDomain::KvCache;
        check(!validate_quantized_tensor(d).ok(), "rejects non-weight domain");
    }

    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        check(validate_kernel_capability(d, QuantKernelKind::LinearFp8Finegrained).ok(),
              "fp8 kernel accepts fp8 spec");
        check(validate_kernel_capability(make_desc(QuantizedEncoding::Bf16, 256, 256),
                                         QuantKernelKind::LinearBf16).ok(),
              "bf16 kernel accepts bf16 spec");
        check(validate_weight_scale_group(128, QuantKernelKind::LinearFp8Finegrained).ok(),
              "fp8 dispatch accepts group 128");
        check(validate_weight_scale_group(64, QuantKernelKind::LinearFp8Finegrained).ok(),
              "fp8 dispatch accepts group 64");
        check(!validate_weight_scale_group(32, QuantKernelKind::LinearFp8Finegrained).ok(),
              "fp8 dispatch rejects group 32");
        check(!validate_weight_scale_group(256, QuantKernelKind::LinearFp8Finegrained).ok(),
              "fp8 dispatch rejects group 256");
    }

    {
        QuantizedTensorDescriptor d = make_desc(QuantizedEncoding::Fp8, 256, 256);
        d.quant.scale->block = WeightBlockShape{128, 64};
        d.layout.k_padded = 256;
        QuantPayloadRef* scale = const_cast<QuantPayloadRef*>(
            find_payload(d, QuantPayloadRole::Scale));
        scale->shape = {2, 4};
        check(validate_quantized_tensor(d).ok(),
              "format validation accepts the fp8 group-64 block");
        check(validate_kernel_capability(d, QuantKernelKind::LinearFp8Finegrained).ok(),
              "runtime capability accepts the fp8 group-64 block");

        QuantizedTensorDescriptor bad_n = make_desc(QuantizedEncoding::Fp8, 256, 256);
        bad_n.quant.scale->block = WeightBlockShape{64, 128};
        bad_n.layout.k_padded = 256;
        const_cast<QuantPayloadRef*>(find_payload(bad_n, QuantPayloadRole::Scale))->shape = {4, 2};
        check(validate_quantized_tensor(bad_n).ok(),
              "format validation accepts a power-of-two block");
        check(!validate_kernel_capability(bad_n, QuantKernelKind::LinearFp8Finegrained).ok(),
              "runtime capability rejects the unsupported block_n");

        QuantizedTensorDescriptor bad_group = make_desc(QuantizedEncoding::Fp8, 256, 256);
        bad_group.quant.scale->block = WeightBlockShape{128, 16};
        check(!validate_quantized_tensor(bad_group).ok(),
              "format validation rejects a scale block below 32");
    }

    {
        check(validate_activation_quant_spec(kFp8FinegrainedActivationQuantSpec).ok(),
              "e4m3 activation spec is valid");

        ActivationQuantSpec bad_codes = kFp8FinegrainedActivationQuantSpec;
        bad_codes.codes_dtype = QuantScalarType::BF16;
        check(!validate_activation_quant_spec(bad_codes).ok(),
              "activation spec rejects BF16 codes");

        ActivationQuantSpec bad_scale = kFp8FinegrainedActivationQuantSpec;
        bad_scale.scale_dtype = QuantScalarType::BF16;
        check(!validate_activation_quant_spec(bad_scale).ok(),
              "activation spec rejects non-F32 scale");

        ActivationQuantSpec bad_group = kFp8FinegrainedActivationQuantSpec;
        bad_group.group_k = 48;
        check(!validate_activation_quant_spec(bad_group).ok(),
              "activation spec rejects non-power-of-two group");

        ActivationQuantSpec group128 = kFp8FinegrainedActivationQuantSpec;
        group128.group_k = 128;
        check(validate_activation_quant_spec(group128).ok(),
              "activation spec accepts power-of-two group");
    }

    {
        KVQuantSpec bf16 = kv_quant_spec(QuantizedEncoding::Bf16, 128);
        check(validate_kv_quant_spec(bf16).ok(), "bf16 kv spec is valid");
        check(kv_scale_group(bf16) == 128, "bf16 kv scale group is full head");

        KVQuantSpec fp8 = kv_quant_spec(QuantizedEncoding::Fp8, 128);
        check(validate_kv_quant_spec(fp8).ok(), "fp8 kv spec is valid");
        check(fp8.scale_dtype == QuantScalarType::F32, "fp8 kv scale is F32");
        check(kv_scale_group(fp8) == 128, "fp8 kv scale group is full head");
        check(kv_blocks_per_head(fp8) == 1, "fp8 kv blocks per head");
    }

    return 0;
}
}  // namespace

int main() {
    run();
    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
