#pragma once
#include <shelf/runtime/graph/shape_spec.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/quantization/quantization_types.h>
#include <variant>
#include <cstdint>

namespace shelf::runtime {

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
    MOE_ROUTER = 19,
    MOE_GROUPED_LINEAR = 20,
    MOE_COMBINE = 21,
    HYPER_MIX = 22,
    HYPER_APPLY = 23,
    PLE_GATE = 24,
    PLE_BROADCAST_MUL = 25,
    PLE_CONV = 26,
    QSA_KEY_CACHE = 27,
    QSA_SCORES = 28,
    QSA_TOPK = 29,
    QSA_EXPAND_MASK = 30,
    QSA_ROPE = 31,
    PLE_STREAM_DOT = 32,
    REPEAT_STREAMS = 33,
    HYPER_INJECT = 34,
    MEAN_STREAMS = 35,
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
        case PrimitiveKind::MOE_ROUTER: return "MOE_ROUTER";
        case PrimitiveKind::MOE_GROUPED_LINEAR: return "MOE_GROUPED_LINEAR";
        case PrimitiveKind::MOE_COMBINE: return "MOE_COMBINE";
        case PrimitiveKind::HYPER_MIX: return "HYPER_MIX";
        case PrimitiveKind::HYPER_APPLY: return "HYPER_APPLY";
        case PrimitiveKind::PLE_GATE: return "PLE_GATE";
        case PrimitiveKind::PLE_BROADCAST_MUL: return "PLE_BROADCAST_MUL";
        case PrimitiveKind::PLE_CONV: return "PLE_CONV";
        case PrimitiveKind::QSA_KEY_CACHE: return "QSA_KEY_CACHE";
        case PrimitiveKind::QSA_SCORES: return "QSA_SCORES";
        case PrimitiveKind::QSA_TOPK: return "QSA_TOPK";
        case PrimitiveKind::QSA_EXPAND_MASK: return "QSA_EXPAND_MASK";
        case PrimitiveKind::QSA_ROPE: return "QSA_ROPE";
        case PrimitiveKind::PLE_STREAM_DOT: return "PLE_STREAM_DOT";
        case PrimitiveKind::REPEAT_STREAMS: return "REPEAT_STREAMS";
        case PrimitiveKind::HYPER_INJECT: return "HYPER_INJECT";
        case PrimitiveKind::MEAN_STREAMS: return "MEAN_STREAMS";
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
    float scale = 1.0f;
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
    // When true, the second tensor input is a [rows, mask_features] U8 mask. A key is
    // visible only when its byte is non-zero, so the attention consumes the block
    // selection produced by the QSA indexer.
    bool has_token_mask = false;
    uint32_t mask_features = 0;
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

// Routing of one batch: softmax over `experts`, top_k selection, renormalization, and the
// group plan. The selection and the plan are not values: they live in the plan scratch the
// graph declares, so this node produces no tensor value.
//
// `experts` is the logits width and stays the whole expert count of the model even when the
// experts are split across devices: every rank must select the same experts. The rank's own
// range is `expert_begin` for `local_experts` experts, and the plan scratch is sized by that
// local count. `local_experts == 0` means the rank owns the whole width.
struct MoeRouterNode {
    uint32_t experts = 0;
    uint32_t top_k = 0;
    uint32_t local_experts = 0;
    uint32_t expert_begin = 0;

    // A node that states no range owns every expert.
    uint32_t owned_experts() const noexcept { return local_experts != 0u ? local_experts : experts; }
};

// One expert projection of a rank-3 expert plane. `shape` is one expert's {output, input}
// features; the expert count and the plane strides come from the weight slot. Its input is
// the bf16 activation, and the activation plane it quantizes into is addressed in group
// rows.
struct MoeGroupedLinearNode {
    MatrixwiseShapeKey shape;
    uint32_t weight_index = 0;
    ValueDType out_dtype = ValueDType::BF16;
};

// Weighted sum of a token's selected expert outputs in group row order, plus the shared
// expert scaled by its activated gate.
struct MoeCombineNode {
    RowwiseShapeKey shape;
    uint32_t top_k = 0;
};

// Qwen4Exp hyper-connection mix: gate * normed averaged over hc streams.
struct HyperMixNode {
    RowwiseShapeKey shape;  // hidden
    uint32_t hc = 0;
    uint32_t hidden = 0;
    bool sigmoid_gate = true;
};

// Qwen4Exp injection gate: 2 * sigmoid(x / hc) in one pass.
struct HyperInjectNode {
    RowwiseShapeKey shape;  // hc
    float scale = 1.0f;
};

// Qwen4Exp hyper-connection injection: raw[h] + y * injection[h].
struct HyperApplyNode {
    RowwiseShapeKey shape;  // hc * hidden
    uint32_t hc = 0;
    uint32_t hidden = 0;
};

// Repeat one hidden row across hc streams: out[r, h*hidden+i] = in[r, i].
struct RepeatStreamsNode {
    RowwiseShapeKey shape;  // hc * hidden
    uint32_t hc = 0;
    uint32_t hidden = 0;
};

// Mean one hidden row over hc streams: out[r, i] = mean_h in[r, h*hidden + i].
struct MeanStreamsNode {
    RowwiseShapeKey shape;  // hidden
    uint32_t hc = 0;
    uint32_t hidden = 0;
};

// PLE per-stream dot over one stream's hidden lanes.
struct PleStreamDotNode {
    RowwiseShapeKey shape;  // hc
    uint32_t hc = 0;
    uint32_t hidden = 0;
};

// PLE gate shaping: sign(x) * sqrt(max(|x|, 1e-6)).
struct PleGateNode {
    RowwiseShapeKey shape;  // hidden
};

// PLE sigmoid(gate) * value broadcast over hc streams.
struct PleBroadcastMulNode {
    RowwiseShapeKey shape;  // hc * hidden
    uint32_t hc = 0;
    uint32_t hidden = 0;
};

// PLE depthwise causal conv, kernel 4, dilation 3, SiLU; state is per sequence.
struct PleConvNode {
    RowwiseShapeKey shape;  // hc * hidden
    uint32_t parameter_index = UINT32_MAX;
    uint32_t state_index = 0;
};

// Partial rotary of the QSA indexer query heads. Unlike ROPE, this carries the position
// scale explicitly and runs the QSA rope kernel (up to 2x).
struct QsaRopeNode {
    RowwiseShapeKey shape;  // heads * head_dim
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
};

// Qwen4Exp QSA indexer key cache. Raw indexer keys are appended to the per-sequence
// partial block; every `compress_ratio` rows complete one pooled block key. The pooled
// key is mean -> RMSNorm(1 + parameter) -> partial RoPE at the block's first position.
struct QsaKeyCacheNode {
    RowwiseShapeKey shape;  // indexer head_dim
    uint32_t parameter_index = UINT32_MAX;
    uint32_t state_index = 0;
    uint32_t compress_ratio = 4;
    uint32_t rotary_dim = 0;
    uint32_t max_blocks = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
};

// Per-row indexer block scores from the query against the sequence's pooled keys.
struct QsaScoresNode {
    RowwiseShapeKey shape;  // index_heads * index_head_dim
    uint32_t state_index = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t blocks = 0;
    uint32_t compress_ratio = 4;
    float scale = 1.0f;
};

// Top-k block selection over the scores, with -1 padding past the visible blocks.
struct QsaTopkNode {
    RowwiseShapeKey shape;  // blocks
    uint32_t k = 0;
    uint32_t blocks = 0;
    uint32_t compress_ratio = 4;
};

// Expand the selected blocks to a token mask and AND it with causality.
struct QsaExpandMaskNode {
    RowwiseShapeKey shape;  // selected blocks
    uint32_t total_len = 0;
    uint32_t compress_ratio = 4;
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
    MoeRouterNode,
    MoeGroupedLinearNode,
    MoeCombineNode,
    HyperMixNode,
    HyperApplyNode,
    PleGateNode,
    PleBroadcastMulNode,
    PleConvNode,
    QsaKeyCacheNode,
    QsaScoresNode,
    QsaTopkNode,
    QsaExpandMaskNode,
    QsaRopeNode,
    PleStreamDotNode,
    RepeatStreamsNode,
    HyperInjectNode,
    MeanStreamsNode
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
        else if constexpr (std::is_same_v<T, MoeRouterNode>) return PrimitiveKind::MOE_ROUTER;
        else if constexpr (std::is_same_v<T, MoeGroupedLinearNode>) return PrimitiveKind::MOE_GROUPED_LINEAR;
        else if constexpr (std::is_same_v<T, MoeCombineNode>) return PrimitiveKind::MOE_COMBINE;
        else if constexpr (std::is_same_v<T, HyperMixNode>) return PrimitiveKind::HYPER_MIX;
        else if constexpr (std::is_same_v<T, HyperApplyNode>) return PrimitiveKind::HYPER_APPLY;
        else if constexpr (std::is_same_v<T, PleGateNode>) return PrimitiveKind::PLE_GATE;
        else if constexpr (std::is_same_v<T, PleBroadcastMulNode>) return PrimitiveKind::PLE_BROADCAST_MUL;
        else if constexpr (std::is_same_v<T, PleConvNode>) return PrimitiveKind::PLE_CONV;
        else if constexpr (std::is_same_v<T, QsaKeyCacheNode>) return PrimitiveKind::QSA_KEY_CACHE;
        else if constexpr (std::is_same_v<T, QsaScoresNode>) return PrimitiveKind::QSA_SCORES;
        else if constexpr (std::is_same_v<T, QsaTopkNode>) return PrimitiveKind::QSA_TOPK;
        else if constexpr (std::is_same_v<T, QsaExpandMaskNode>) return PrimitiveKind::QSA_EXPAND_MASK;
        else if constexpr (std::is_same_v<T, QsaRopeNode>) return PrimitiveKind::QSA_ROPE;
        else if constexpr (std::is_same_v<T, PleStreamDotNode>) return PrimitiveKind::PLE_STREAM_DOT;
        else if constexpr (std::is_same_v<T, RepeatStreamsNode>) return PrimitiveKind::REPEAT_STREAMS;
        else if constexpr (std::is_same_v<T, HyperInjectNode>) return PrimitiveKind::HYPER_INJECT;
        else if constexpr (std::is_same_v<T, MeanStreamsNode>) return PrimitiveKind::MEAN_STREAMS;
        else return PrimitiveKind::LINEAR;
    }, n);
}

}
