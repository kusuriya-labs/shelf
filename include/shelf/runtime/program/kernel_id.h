#pragma once
#include <cstdint>

namespace shelf::runtime {

enum class KernelId : uint32_t {
    EMBEDDING_LOOKUP,
    LINEAR_BF16,
    RMS_NORM,
    RESIDUAL_ADD,
    SPLIT,
    SILU,
    SIGMOID,
    MUL,
    SWIGLU,
    ROPE,
    L2_NORMALIZE,
    SCALE,
    KV_APPEND,
    PAGED_ATTENTION,
    STATEFUL_CAUSAL_CONV1D,
    GDN_RECURRENCE,
    OUTPUT_GATHER,
    SAMPLING,
    ACTIVATION_QUANTIZE_FP8_FINEGRAINED,
    CONCAT,
    LINEAR_FP8_FINEGRAINED,
    LINEAR_MXFP4_W4A8,
    VERIFY_ACCEPT,
    ACTIVATION_QUANTIZE_MXFP4_ROWREF,
    MOE_ROUTER,
    MOE_PLAN_SCAN,
    MOE_PLAN_SCATTER,
    LINEAR_MXFP4_W4A8_GROUPED,
    MOE_COMBINE,
    LINEAR_EXL3_W16A16,
    LINEAR_EXL3_W16A16_GROUPED,
    HYPER_MIX,
    HYPER_APPLY,
    PLE_GATE,
    PLE_BROADCAST_MUL,
    PLE_CONV,
    QSA_KEY_CACHE,
    QSA_SCORES,
    QSA_TOPK,
    QSA_EXPAND_MASK,
    QSA_ROPE,
    PLE_STREAM_DOT,
    REPEAT_STREAMS,
    HYPER_INJECT,
    MEAN_STREAMS,
    ACTIVATION_QUANTIZE_FP8_PERTOKEN,
    LINEAR_FP8_PTPC,
    COUNT,
};

constexpr uint32_t kernel_id_count() noexcept {
    return static_cast<uint32_t>(KernelId::COUNT);
}

inline const char* kernel_to_name(KernelId id) noexcept {
    switch (id) {
        case KernelId::EMBEDDING_LOOKUP: return "EMBEDDING_LOOKUP";
        case KernelId::LINEAR_BF16: return "LINEAR_BF16";
        case KernelId::RMS_NORM: return "RMS_NORM";
        case KernelId::RESIDUAL_ADD: return "RESIDUAL_ADD";
        case KernelId::SPLIT: return "SPLIT";
        case KernelId::SILU: return "SILU";
        case KernelId::SIGMOID: return "SIGMOID";
        case KernelId::MUL: return "MUL";
        case KernelId::SWIGLU: return "SWIGLU";
        case KernelId::ROPE: return "ROPE";
        case KernelId::L2_NORMALIZE: return "L2_NORMALIZE";
        case KernelId::SCALE: return "SCALE";
        case KernelId::KV_APPEND: return "KV_APPEND";
        case KernelId::PAGED_ATTENTION: return "PAGED_ATTENTION";
        case KernelId::STATEFUL_CAUSAL_CONV1D: return "STATEFUL_CAUSAL_CONV1D";
        case KernelId::GDN_RECURRENCE: return "GDN_RECURRENCE";
        case KernelId::OUTPUT_GATHER: return "OUTPUT_GATHER";
        case KernelId::SAMPLING: return "SAMPLING";
        case KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED: return "ACTIVATION_QUANTIZE_FP8_FINEGRAINED";
        case KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF: return "ACTIVATION_QUANTIZE_MXFP4_ROWREF";
        case KernelId::LINEAR_FP8_FINEGRAINED: return "LINEAR_FP8_FINEGRAINED";
        case KernelId::LINEAR_MXFP4_W4A8: return "LINEAR_MXFP4_W4A8";
        case KernelId::CONCAT: return "CONCAT";
        case KernelId::VERIFY_ACCEPT: return "VERIFY_ACCEPT";
        case KernelId::MOE_ROUTER: return "MOE_ROUTER";
        case KernelId::MOE_PLAN_SCAN: return "MOE_PLAN_SCAN";
        case KernelId::MOE_PLAN_SCATTER: return "MOE_PLAN_SCATTER";
        case KernelId::LINEAR_MXFP4_W4A8_GROUPED: return "LINEAR_MXFP4_W4A8_GROUPED";
        case KernelId::MOE_COMBINE: return "MOE_COMBINE";
        case KernelId::LINEAR_EXL3_W16A16: return "LINEAR_EXL3_W16A16";
        case KernelId::LINEAR_EXL3_W16A16_GROUPED: return "LINEAR_EXL3_W16A16_GROUPED";
        case KernelId::HYPER_MIX: return "HYPER_MIX";
        case KernelId::HYPER_APPLY: return "HYPER_APPLY";
        case KernelId::PLE_GATE: return "PLE_GATE";
        case KernelId::PLE_BROADCAST_MUL: return "PLE_BROADCAST_MUL";
        case KernelId::PLE_CONV: return "PLE_CONV";
        case KernelId::QSA_KEY_CACHE: return "QSA_KEY_CACHE";
        case KernelId::QSA_SCORES: return "QSA_SCORES";
        case KernelId::QSA_TOPK: return "QSA_TOPK";
        case KernelId::QSA_EXPAND_MASK: return "QSA_EXPAND_MASK";
        case KernelId::QSA_ROPE: return "QSA_ROPE";
        case KernelId::PLE_STREAM_DOT: return "PLE_STREAM_DOT";
        case KernelId::REPEAT_STREAMS: return "REPEAT_STREAMS";
        case KernelId::HYPER_INJECT: return "HYPER_INJECT";
        case KernelId::MEAN_STREAMS: return "MEAN_STREAMS";
        case KernelId::ACTIVATION_QUANTIZE_FP8_PERTOKEN: return "ACTIVATION_QUANTIZE_FP8_PERTOKEN";
        case KernelId::LINEAR_FP8_PTPC: return "LINEAR_FP8_PTPC";
        default: return "UNKNOWN";
    }
}

}

