#pragma once
#include <compare>
#include <cstdint>
#include <limits>

namespace ps::runtime {

template<std::uint32_t OutputFeatures, std::uint32_t InputFeatures>
struct MatrixwiseShapeSpec {
    static_assert(OutputFeatures != 0);
    static_assert(InputFeatures != 0);
    static constexpr std::uint32_t output_features = OutputFeatures;
    static constexpr std::uint32_t input_features = InputFeatures;
};

template<std::uint32_t Features>
struct RowwiseShapeSpec {
    static_assert(Features != 0);
    static constexpr std::uint32_t features = Features;
};

template<std::uint32_t QueryHeads, std::uint32_t KvHeads,
         std::uint32_t HeadDim, std::uint32_t RotaryDim>
struct AttentionShapeSpec {
    static_assert(QueryHeads != 0);
    static_assert(KvHeads != 0);
    static_assert(HeadDim != 0);
    static_assert(RotaryDim != 0);
    static_assert(static_cast<std::uint64_t>(QueryHeads) * HeadDim <=
                  std::numeric_limits<std::uint32_t>::max());
    static_assert(static_cast<std::uint64_t>(KvHeads) * HeadDim <=
                  std::numeric_limits<std::uint32_t>::max());
    static constexpr std::uint32_t query_heads = QueryHeads;
    static constexpr std::uint32_t kv_heads = KvHeads;
    static constexpr std::uint32_t head_dim = HeadDim;
    static constexpr std::uint32_t rotary_dim = RotaryDim;
    static constexpr std::uint32_t query_features = QueryHeads * HeadDim;
    static constexpr std::uint32_t kv_features = KvHeads * HeadDim;
};

template<std::uint32_t KeyHeads, std::uint32_t ValueHeads,
         std::uint32_t KeyHeadDim, std::uint32_t ValueHeadDim,
         std::uint32_t ConvKernel>
struct GdnShapeSpec {
    static_assert(KeyHeads != 0);
    static_assert(ValueHeads != 0);
    static_assert(KeyHeadDim != 0);
    static_assert(ValueHeadDim != 0);
    static_assert(ConvKernel != 0);
    static_assert(static_cast<std::uint64_t>(KeyHeads) * KeyHeadDim <=
                  std::numeric_limits<std::uint32_t>::max());
    static_assert(static_cast<std::uint64_t>(ValueHeads) * ValueHeadDim <=
                  std::numeric_limits<std::uint32_t>::max());
    static_assert(
        2ULL * KeyHeads * KeyHeadDim +
            static_cast<std::uint64_t>(ValueHeads) * ValueHeadDim <=
        std::numeric_limits<std::uint32_t>::max());
    static constexpr std::uint32_t key_heads = KeyHeads;
    static constexpr std::uint32_t value_heads = ValueHeads;
    static constexpr std::uint32_t key_head_dim = KeyHeadDim;
    static constexpr std::uint32_t value_head_dim = ValueHeadDim;
    static constexpr std::uint32_t conv_kernel = ConvKernel;
    static constexpr std::uint32_t qk_features = KeyHeads * KeyHeadDim;
    static constexpr std::uint32_t value_features = ValueHeads * ValueHeadDim;
    static constexpr std::uint32_t conv_features =
        2 * qk_features + value_features;
};

struct MatrixwiseShapeKey {
    std::uint32_t output_features = 0;
    std::uint32_t input_features = 0;
    auto operator<=>(const MatrixwiseShapeKey&) const = default;
};

struct RowwiseShapeKey {
    std::uint32_t features = 0;
    auto operator<=>(const RowwiseShapeKey&) const = default;
};

struct AttentionShapeKey {
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    auto operator<=>(const AttentionShapeKey&) const = default;
};

struct GdnShapeKey {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    auto operator<=>(const GdnShapeKey&) const = default;
};

constexpr bool valid_shape_key(MatrixwiseShapeKey key) noexcept {
    return key.output_features != 0 && key.input_features != 0;
}

constexpr bool valid_shape_key(RowwiseShapeKey key) noexcept {
    return key.features != 0;
}

constexpr bool valid_shape_key(AttentionShapeKey key) noexcept {
    if (key.query_heads == 0 || key.kv_heads == 0 ||
        key.head_dim == 0 || key.rotary_dim == 0 ||
        key.rotary_dim > key.head_dim) {
        return false;
    }
    return static_cast<std::uint64_t>(key.query_heads) * key.head_dim <=
               std::numeric_limits<std::uint32_t>::max() &&
           static_cast<std::uint64_t>(key.kv_heads) * key.head_dim <=
               std::numeric_limits<std::uint32_t>::max();
}

constexpr bool valid_shape_key(GdnShapeKey key) noexcept {
    if (key.key_heads == 0 || key.value_heads == 0 ||
        key.key_head_dim == 0 || key.value_head_dim == 0 ||
        key.conv_kernel == 0) {
        return false;
    }
    const auto qk_features =
        static_cast<std::uint64_t>(key.key_heads) * key.key_head_dim;
    const auto value_features =
        static_cast<std::uint64_t>(key.value_heads) * key.value_head_dim;
    return qk_features <= std::numeric_limits<std::uint32_t>::max() &&
           value_features <= std::numeric_limits<std::uint32_t>::max() &&
           2 * qk_features + value_features <=
               std::numeric_limits<std::uint32_t>::max();
}

}
