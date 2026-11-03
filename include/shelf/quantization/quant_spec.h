#pragma once

#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/quant_format.h>

#include <cstdint>
#include <optional>

namespace shelf::quantization {

enum class QuantDomain : uint8_t {
    Weight,
    Activation,
    KvCache,
    RecurrentState,
};

enum class QuantScalarType : uint8_t {
    U8,
    I8,
    BF16,
    F32,
    FP8E4M3,
    E8M0,
};

enum class ScaleGranularity : uint8_t {
    PerTensor,
    PerRow,
    MatrixBlock,
    PerTokenHead,
    PerHeadBlock,
};

enum class ScaleApplication : uint8_t {
    Multiply,
    Divide,
};

struct WeightBlockShape {
    uint32_t n = 1;
    uint32_t k = 1;
};

struct BlockScaleSpec {
    uint32_t group_n = 1;
    uint32_t group_k = 0;
    QuantScalarType dtype = QuantScalarType::F32;
    ScaleApplication application = ScaleApplication::Multiply;
};

struct WeightScaleSpec {
    ScaleGranularity granularity = ScaleGranularity::MatrixBlock;
    WeightBlockShape block;
    QuantScalarType dtype = QuantScalarType::F32;
    ScaleApplication application = ScaleApplication::Multiply;
};

struct WeightQuantSpec {
    manifest::QuantizedEncoding encoding = manifest::QuantizedEncoding::Bf16;
    std::optional<WeightScaleSpec> scale;
};

struct ScaleGrid {
    uint64_t n = 0;
    uint64_t k = 0;
};

constexpr uint64_t ceil_div(uint64_t value, uint64_t divisor) {
    return value / divisor + ((value % divisor != 0u) ? 1u : 0u);
}

constexpr WeightBlockShape default_weight_block(manifest::QuantizedEncoding enc) {
    switch (enc) {
        case manifest::QuantizedEncoding::Fp8:
            return WeightBlockShape{kFp8ScaleBlockN, kFp8ScaleBlockK};
        case manifest::QuantizedEncoding::Mxfp4:
            return WeightBlockShape{1u, kMxfp4ScaleBlockK};
        case manifest::QuantizedEncoding::Bf16:
            break;
    }
    return WeightBlockShape{1u, 1u};
}

inline ScaleGrid derive_scale_grid(uint64_t n, uint64_t k, WeightBlockShape block) {
    return ScaleGrid{ceil_div(n, block.n), ceil_div(k, block.k)};
}

inline uint64_t scale_index(uint64_t n, uint64_t k, WeightBlockShape block, ScaleGrid grid) {
    return (n / block.n) * grid.k + (k / block.k);
}

inline std::optional<WeightScaleSpec> default_weight_scale_spec(
    manifest::QuantizedEncoding enc) {
    switch (enc) {
        case manifest::QuantizedEncoding::Bf16:
            return std::nullopt;
        case manifest::QuantizedEncoding::Fp8:
            return WeightScaleSpec{
                ScaleGranularity::MatrixBlock,
                default_weight_block(enc),
                QuantScalarType::F32,
                ScaleApplication::Multiply};
        case manifest::QuantizedEncoding::Mxfp4:
            return WeightScaleSpec{
                ScaleGranularity::MatrixBlock,
                default_weight_block(enc),
                QuantScalarType::E8M0,
                ScaleApplication::Multiply};
    }
    return std::nullopt;
}

inline WeightQuantSpec default_weight_quant_spec(manifest::QuantizedEncoding enc) {
    WeightQuantSpec spec;
    spec.encoding = enc;
    spec.scale = default_weight_scale_spec(enc);
    return spec;
}

inline BlockScaleSpec block_scale_of(const WeightScaleSpec& scale) {
    BlockScaleSpec out;
    out.group_n = scale.block.n;
    out.group_k = scale.block.k;
    out.dtype = scale.dtype;
    out.application = scale.application;
    return out;
}

inline BlockScaleSpec block_scale_of(const WeightQuantSpec& spec) {
    return spec.scale.has_value() ? block_scale_of(*spec.scale) : BlockScaleSpec{};
}

struct ActivationQuantSpec {
    QuantScalarType codes_dtype = QuantScalarType::FP8E4M3;
    uint32_t group_k = 0;
    QuantScalarType scale_dtype = QuantScalarType::F32;
};

struct KVQuantSpec {
    manifest::QuantizedEncoding encoding = manifest::QuantizedEncoding::Bf16;
    uint32_t head_dim = 0;
    uint32_t group_k = 0;
    QuantScalarType scale_dtype = QuantScalarType::F32;
};

constexpr ActivationQuantSpec kFp8FinegrainedActivationQuantSpec{
    QuantScalarType::FP8E4M3, 128, QuantScalarType::F32};


inline Status validate_activation_quant_spec(const ActivationQuantSpec& spec) {
    if (spec.scale_dtype != QuantScalarType::F32) {
        return Status::invalid_argument("activation scale must be F32", __FILE__, __LINE__);
    }
    if (spec.codes_dtype != QuantScalarType::FP8E4M3) {
        return Status::invalid_argument(
            "activation codes must be FP8E4M3", __FILE__, __LINE__);
    }
    if (spec.group_k != 0 &&
        (spec.group_k < kQuantBlockElements || (spec.group_k & (spec.group_k - 1u)) != 0u)) {
        return Status::invalid_argument(
            "activation scale group must be full row or a power of two >= 32",
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

inline KVQuantSpec kv_quant_spec(manifest::QuantizedEncoding enc, uint32_t head_dim) {
    KVQuantSpec spec;
    spec.encoding = enc;
    spec.head_dim = head_dim;
    switch (enc) {
        case manifest::QuantizedEncoding::Fp8:
            spec.group_k = 0;
            spec.scale_dtype = QuantScalarType::F32;
            break;
        case manifest::QuantizedEncoding::Mxfp4:
            break;
        case manifest::QuantizedEncoding::Bf16:
            break;
    }
    return spec;
}

inline uint32_t kv_scale_group(const KVQuantSpec& spec) {
    return spec.group_k != 0u ? spec.group_k : spec.head_dim;
}

inline uint32_t kv_blocks_per_head(const KVQuantSpec& spec) {
    (void)spec;
    return 1u;
}

inline Status validate_kv_quant_spec(const KVQuantSpec& spec) {
    if (spec.head_dim == 0u) {
        return Status::invalid_argument("kv quant spec requires head_dim > 0", __FILE__, __LINE__);
    }
    if (spec.scale_dtype != QuantScalarType::F32 && spec.scale_dtype != QuantScalarType::BF16) {
        return Status::invalid_argument("kv scale must be F32 or BF16", __FILE__, __LINE__);
    }
    switch (spec.encoding) {
        case manifest::QuantizedEncoding::Bf16:
            if (spec.group_k != 0u || spec.scale_dtype != QuantScalarType::F32) {
                return Status::invalid_argument("bf16 kv must not carry a scale", __FILE__, __LINE__);
            }
            return Status::make_ok();
        case manifest::QuantizedEncoding::Fp8:
            if (spec.group_k != 0u || spec.scale_dtype != QuantScalarType::F32) {
                return Status::invalid_argument("fp8 kv scale must be full head F32", __FILE__, __LINE__);
            }
            return Status::make_ok();
        case manifest::QuantizedEncoding::Mxfp4:
            break;
    }
    return Status::unsupported("unsupported kv cache encoding", __FILE__, __LINE__);
}

}
