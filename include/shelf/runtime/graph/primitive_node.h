#pragma once
#include <shelf/runtime/graph/shape_spec.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/quantization/quantization_types.h>
#include <variant>
#include <cstdint>

namespace ps::runtime {

enum class PrimitiveKind : uint16_t {
    EMBEDDING_LOOKUP = 0,
    LINEAR = 1,
    RMS_NORM = 2,
    RESIDUAL_ADD = 3,
    SPLIT = 4,
    SILU = 5,
    SIGMOID = 6,
    MUL = 7,
    SWIGLU = 8,
    ROPE = 9,
    L2_NORMALIZE = 10,
    SCALE = 11,
    KV_APPEND = 12,
    PAGED_ATTENTION = 13,
    STATEFUL_CAUSAL_CONV1D = 14,
    GDN_RECURRENCE = 15,
    OUTPUT_GATHER = 16,
    SAMPLING = 17,
    CONCAT = 18,
    GROUP_MEAN = 19,
    GROUP_BROADCAST_MUL = 20,
    MOE_ROUTER = 21,
    MOE_EXPERTS = 22,
};

inline const char* to_string(PrimitiveKind k) noexcept {
    switch (k) {
        case PrimitiveKind::EMBEDDING_LOOKUP: return "EMBEDDING_LOOKUP";
        case PrimitiveKind::LINEAR: return "LINEAR";
        case PrimitiveKind::RMS_NORM: return "RMS_NORM";
        case PrimitiveKind::RESIDUAL_ADD: return "RESIDUAL_ADD";
        case PrimitiveKind::SPLIT: return "SPLIT";
        case PrimitiveKind::SILU: return "SILU";
        case PrimitiveKind::SIGMOID: return "SIGMOID";
        case PrimitiveKind::MUL: return "MUL";
        case PrimitiveKind::SWIGLU: return "SWIGLU";
        case PrimitiveKind::ROPE: return "ROPE";
        case PrimitiveKind::L2_NORMALIZE: return "L2_NORMALIZE";
        case PrimitiveKind::SCALE: return "SCALE";
        case PrimitiveKind::KV_APPEND: return "KV_APPEND";
        case PrimitiveKind::PAGED_ATTENTION: return "PAGED_ATTENTION";
        case PrimitiveKind::STATEFUL_CAUSAL_CONV1D: return "STATEFUL_CAUSAL_CONV1D";
        case PrimitiveKind::GDN_RECURRENCE: return "GDN_RECURRENCE";
        case PrimitiveKind::OUTPUT_GATHER: return "OUTPUT_GATHER";
        case PrimitiveKind::SAMPLING: return "SAMPLING";
        case PrimitiveKind::CONCAT: return "CONCAT";
        case PrimitiveKind::GROUP_MEAN: return "GROUP_MEAN";
        case PrimitiveKind::GROUP_BROADCAST_MUL: return "GROUP_BROADCAST_MUL";
        case PrimitiveKind::MOE_ROUTER: return "MOE_ROUTER";
        case PrimitiveKind::MOE_EXPERTS: return "MOE_EXPERTS";
    }
    return "UNKNOWN";
}

struct EmbeddingLookupNode {
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    MatrixwiseShapeKey weight_shape;
    uint32_t weight_index = 0;
};

struct LinearNode {
    MatrixwiseShapeKey shape;
    uint32_t weight_index = 0;
    ValueDType out_dtype = ValueDType::BF16;
};

enum class RmsNormWeightMode : uint8_t {
    ONE_PLUS = 0,
    DIRECT = 1,
};

struct RmsNormNode {
    RowwiseShapeKey shape;
    float eps = 1e-6f;
    uint32_t group_size = 0;
    uint32_t parameter_index = UINT32_MAX;
    RmsNormWeightMode weight_mode = RmsNormWeightMode::ONE_PLUS;
};

struct ResidualAddNode {
    RowwiseShapeKey shape;
};

enum class SplitLayout : uint32_t {
    HALVES = 0,
    INTERLEAVED_HEADS = 1,
};

struct SplitNode {
    RowwiseShapeKey input_shape;
    RowwiseShapeKey output_shape_a;
    RowwiseShapeKey output_shape_b;
    uint32_t split_dim = 0;
    SplitLayout layout = SplitLayout::HALVES;
    uint32_t head_dim = 0;
};

struct SiLUNode {
    RowwiseShapeKey shape;
};

struct SigmoidNode {
    RowwiseShapeKey shape;
};

struct MulNode {
    RowwiseShapeKey shape;
};

struct SwiGluNode {
    RowwiseShapeKey shape;
};

struct RoPENode {
    AttentionShapeKey attention;
    uint32_t rotary_dim = 0;
    uint32_t head_dim = 0;
    float theta = 10000000.0f;
    float attention_factor = 1.0f;
};

struct L2NormalizeNode {
    GdnShapeKey gdn;
    RowwiseShapeKey shape;
    float eps = 1e-6f;
    uint32_t group_size = 0;
};

struct ScaleNode {
    RowwiseShapeKey shape;
    float scale = 1.0f;
};

struct KvAppendNode {
    AttentionShapeKey attention;
    uint32_t state_index = 0;
};

struct PagedAttentionNode {
    AttentionShapeKey attention;
    float scale = 0.0f;
    uint32_t state_index = 0;
};

struct StatefulCausalConv1DNode {
    GdnShapeKey gdn;
    uint32_t parameter_index = UINT32_MAX;
    uint32_t state_index = 0;
};

struct GdnRecurrenceNode {
    GdnShapeKey gdn;
    uint32_t dt_bias_parameter_index = UINT32_MAX;
    uint32_t a_log_parameter_index = UINT32_MAX;
    uint32_t state_index = 0;
};

struct OutputGatherNode {
    RowwiseShapeKey shape;
};

struct SamplingNode {
    uint32_t vocab_size = 0;
};

struct ConcatNode {
    RowwiseShapeKey shape;
    RowwiseShapeKey input_shape_a;
    RowwiseShapeKey input_shape_b;
};

struct GroupMeanNode {
    RowwiseShapeKey shape;
    uint32_t group_size = 0;
};

struct GroupBroadcastMulNode {
    RowwiseShapeKey shape;
    uint32_t group_size = 0;
};

struct MoERouterNode {
    MatrixwiseShapeKey weight_shape;
    uint32_t weight_index = 0;
    uint32_t experts = 0;
    uint32_t top_k = 0;
    bool renorm = false;
};

struct MoEExpertsNode {
    MatrixwiseShapeKey gate_up_shape;
    MatrixwiseShapeKey down_shape;
    uint32_t weight_index = 0;
    uint32_t experts = 0;
    uint32_t top_k = 0;
};

using PrimitiveNode = std::variant<
    EmbeddingLookupNode,
    LinearNode,
    RmsNormNode,
    ResidualAddNode,
    SplitNode,
    SiLUNode,
    SigmoidNode,
    MulNode,
    SwiGluNode,
    RoPENode,
    L2NormalizeNode,
    ScaleNode,
    KvAppendNode,
    PagedAttentionNode,
    StatefulCausalConv1DNode,
    GdnRecurrenceNode,
    OutputGatherNode,
    SamplingNode,
    ConcatNode,
    GroupMeanNode,
    GroupBroadcastMulNode,
    MoERouterNode,
    MoEExpertsNode
>;

inline PrimitiveKind primitive_kind_of(const PrimitiveNode& n) noexcept {
    return std::visit([](auto&& v) -> PrimitiveKind {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, EmbeddingLookupNode>) return PrimitiveKind::EMBEDDING_LOOKUP;
        else if constexpr (std::is_same_v<T, LinearNode>) return PrimitiveKind::LINEAR;
        else if constexpr (std::is_same_v<T, RmsNormNode>) return PrimitiveKind::RMS_NORM;
        else if constexpr (std::is_same_v<T, ResidualAddNode>) return PrimitiveKind::RESIDUAL_ADD;
        else if constexpr (std::is_same_v<T, SplitNode>) return PrimitiveKind::SPLIT;
        else if constexpr (std::is_same_v<T, SiLUNode>) return PrimitiveKind::SILU;
        else if constexpr (std::is_same_v<T, SigmoidNode>) return PrimitiveKind::SIGMOID;
        else if constexpr (std::is_same_v<T, MulNode>) return PrimitiveKind::MUL;
        else if constexpr (std::is_same_v<T, SwiGluNode>) return PrimitiveKind::SWIGLU;
        else if constexpr (std::is_same_v<T, RoPENode>) return PrimitiveKind::ROPE;
        else if constexpr (std::is_same_v<T, L2NormalizeNode>) return PrimitiveKind::L2_NORMALIZE;
        else if constexpr (std::is_same_v<T, ScaleNode>) return PrimitiveKind::SCALE;
        else if constexpr (std::is_same_v<T, KvAppendNode>) return PrimitiveKind::KV_APPEND;
        else if constexpr (std::is_same_v<T, PagedAttentionNode>) return PrimitiveKind::PAGED_ATTENTION;
        else if constexpr (std::is_same_v<T, StatefulCausalConv1DNode>) return PrimitiveKind::STATEFUL_CAUSAL_CONV1D;
        else if constexpr (std::is_same_v<T, GdnRecurrenceNode>) return PrimitiveKind::GDN_RECURRENCE;
        else if constexpr (std::is_same_v<T, OutputGatherNode>) return PrimitiveKind::OUTPUT_GATHER;
        else if constexpr (std::is_same_v<T, SamplingNode>) return PrimitiveKind::SAMPLING;
        else if constexpr (std::is_same_v<T, ConcatNode>) return PrimitiveKind::CONCAT;
        else if constexpr (std::is_same_v<T, GroupMeanNode>) return PrimitiveKind::GROUP_MEAN;
        else if constexpr (std::is_same_v<T, GroupBroadcastMulNode>) return PrimitiveKind::GROUP_BROADCAST_MUL;
        else if constexpr (std::is_same_v<T, MoERouterNode>) return PrimitiveKind::MOE_ROUTER;
        else if constexpr (std::is_same_v<T, MoEExpertsNode>) return PrimitiveKind::MOE_EXPERTS;
        else return PrimitiveKind::LINEAR;
    }, n);
}

}
