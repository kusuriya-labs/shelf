#pragma once

#include <shelf/models/common/qwen_hybrid/geometry_check.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>

namespace shelf {
namespace qwen_hybrid {

struct GatedAttentionGeometry {
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    float partial_rotary_factor = 1.0f;
};

struct GatedAttentionWeightView {
    const ::shelf::weights::MatrixWeight* q_proj = nullptr;
    const ::shelf::weights::MatrixWeight* k_proj = nullptr;
    const ::shelf::weights::MatrixWeight* v_proj = nullptr;
    const ::shelf::weights::MatrixWeight* o_proj = nullptr;
};

struct GatedAttentionFeatureSizes {
    std::uint32_t query_features = 0;
    std::uint32_t kv_features = 0;
    std::uint32_t q_projection_features = 0;
    std::uint32_t rotary_dim = 0;
};

inline Status gated_attention_feature_sizes(
    const GatedAttentionGeometry& g,
    GatedAttentionFeatureSizes& out) {
    auto status = checked_product(
        g.query_heads, g.head_dim, "invalid full attention query geometry",
        out.query_features);
    if (!status.ok()) return status;
    status = checked_product(
        g.kv_heads, g.head_dim, "invalid full attention KV geometry",
        out.kv_features);
    if (!status.ok()) return status;
    status = checked_double(
        out.query_features, "invalid full attention gated query geometry",
        out.q_projection_features);
    if (!status.ok()) return status;
    if (!(g.partial_rotary_factor > 0.0f && g.partial_rotary_factor <= 1.0f)) {
        return Status::invalid_argument(
            "invalid full attention rotary factor", __FILE__, __LINE__);
    }
    const auto rotary_dim = static_cast<std::size_t>(
        static_cast<double>(g.head_dim) *
        static_cast<double>(g.partial_rotary_factor));
    status = checked_u32(rotary_dim, "invalid full attention rotary geometry",
                         out.rotary_dim);
    if (!status.ok()) return status;
    return Status::make_ok();
}

inline Status validate_gated_attention_geometry(
    const GatedAttentionGeometry& geometry,
    std::uint32_t hidden_size,
    const GatedAttentionWeightView& weights) {

    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    auto status = checked_u32(geometry.query_heads,
                              "invalid full attention query heads", query_heads);
    if (!status.ok()) return status;
    status = checked_u32(geometry.kv_heads, "invalid full attention KV heads",
                         kv_heads);
    if (!status.ok()) return status;
    status = checked_u32(geometry.head_dim,
                         "invalid full attention head dimension", head_dim);
    if (!status.ok()) return status;

    GatedAttentionFeatureSizes features;
    status = gated_attention_feature_sizes(geometry, features);
    if (!status.ok()) return status;

    if (weights.q_proj == nullptr || weights.k_proj == nullptr ||
        weights.v_proj == nullptr || weights.o_proj == nullptr) {
        return Status::invalid_argument("invalid full attention weight view",
                                        __FILE__, __LINE__);
    }
    const MatrixwiseShapeKey q_proj{features.q_projection_features, hidden_size};
    const MatrixwiseShapeKey k_proj{features.kv_features, hidden_size};
    const MatrixwiseShapeKey v_proj{features.kv_features, hidden_size};
    const MatrixwiseShapeKey o_proj{hidden_size, features.query_features};

    status = validate_matrix(*weights.q_proj, q_proj,
                             "invalid full attention q projection");
    if (!status.ok()) return status;
    status = validate_matrix(*weights.k_proj, k_proj,
                             "invalid full attention k projection");
    if (!status.ok()) return status;
    status = validate_matrix(*weights.v_proj, v_proj,
                             "invalid full attention v projection");
    if (!status.ok()) return status;
    return validate_matrix(*weights.o_proj, o_proj,
                           "invalid full attention o projection");
}

}
}
