#pragma once

#include <shelf/models/common/qwen_hybrid/geometry_check.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>

namespace shelf {
namespace qwen_hybrid {

struct GdnGeometry {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
};

struct GdnWeightView {
    const ::shelf::weights::MatrixWeight* qkv = nullptr;
    const ::shelf::weights::MatrixWeight* z = nullptr;
    const ::shelf::weights::MatrixWeight* b = nullptr;
    const ::shelf::weights::MatrixWeight* a = nullptr;
    const ::shelf::weights::MatrixWeight* out = nullptr;
};

struct GdnFeatureSizes {
    std::uint32_t qk_features = 0;
    std::uint32_t value_features = 0;
    std::uint32_t conv_features = 0;
};

inline Status gdn_feature_sizes(const GdnGeometry& g, GdnFeatureSizes& out) {
    auto status = checked_product(
        g.key_heads, g.key_head_dim, "invalid GDN key geometry", out.qk_features);
    if (!status.ok()) return status;
    status = checked_product(
        g.value_heads, g.value_head_dim, "invalid GDN value geometry",
        out.value_features);
    if (!status.ok()) return status;
    if (out.qk_features >
        (std::numeric_limits<std::uint32_t>::max() - out.value_features) / 2) {
        return Status::out_of_range("invalid GDN convolution geometry", __FILE__,
                                    __LINE__);
    }
    out.conv_features = 2 * out.qk_features + out.value_features;
    return Status::make_ok();
}

inline Status validate_gdn_geometry(
    const GdnGeometry& geometry,
    std::uint32_t hidden_size,
    const GdnWeightView& weights) {

    std::uint32_t value_heads = 0;
    std::uint32_t key_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    auto status = checked_u32(geometry.value_heads, "invalid GDN value heads",
                              value_heads);
    if (!status.ok()) return status;
    status = checked_u32(geometry.key_heads, "invalid GDN key heads", key_heads);
    if (!status.ok()) return status;
    status = checked_u32(geometry.key_head_dim, "invalid GDN key head dimension",
                         key_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(geometry.value_head_dim,
                         "invalid GDN value head dimension", value_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(geometry.conv_kernel, "invalid GDN convolution kernel",
                         conv_kernel);
    if (!status.ok()) return status;

    GdnFeatureSizes features;
    status = gdn_feature_sizes(geometry, features);
    if (!status.ok()) return status;

    if (weights.qkv == nullptr || weights.z == nullptr || weights.b == nullptr ||
        weights.a == nullptr || weights.out == nullptr) {
        return Status::invalid_argument("invalid GDN weight view", __FILE__,
                                        __LINE__);
    }
    const MatrixwiseShapeKey qkv{features.conv_features, hidden_size};
    const MatrixwiseShapeKey z{features.value_features, hidden_size};
    const MatrixwiseShapeKey b{value_heads, hidden_size};
    const MatrixwiseShapeKey a{value_heads, hidden_size};
    const MatrixwiseShapeKey out{hidden_size, features.value_features};

    status = validate_matrix(*weights.qkv, qkv, "invalid GDN qkv projection");
    if (!status.ok()) return status;
    status = validate_matrix(*weights.z, z, "invalid GDN z projection");
    if (!status.ok()) return status;
    status = validate_matrix(*weights.b, b, "invalid GDN b projection");
    if (!status.ok()) return status;
    status = validate_matrix(*weights.a, a, "invalid GDN a projection");
    if (!status.ok()) return status;
    return validate_matrix(*weights.out, out, "invalid GDN output projection");
}

}
}
