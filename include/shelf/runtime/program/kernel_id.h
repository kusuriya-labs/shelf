#pragma once
#include <cstdint>

namespace ps::runtime {

enum class KernelId : uint32_t {
    EMBEDDING_LOOKUP = 0,
    LINEAR_BF16 = 1,
    RMS_NORM = 4,
    RESIDUAL_ADD = 5,
    SPLIT = 6,
    SILU = 7,
    SIGMOID = 8,
    MUL = 9,
    SWIGLU = 10,
    ROPE = 11,
    L2_NORMALIZE = 12,
    SCALE = 13,
    KV_APPEND = 14,
    PAGED_ATTENTION = 15,
    STATEFUL_CAUSAL_CONV1D = 16,
    GDN_RECURRENCE = 17,
    OUTPUT_GATHER = 18,
    SAMPLING = 19,
    ACTIVATION_QUANTIZE_FP8 = 20,
    ACTIVATION_QUANTIZE_W4A8 = 21,
    CONCAT = 24,
    LINEAR_FP8 = 25,
    LINEAR_PSQ5 = 26,
    VERIFY_ACCEPT = 27,
    GROUP_MEAN = 28,
    GROUP_BROADCAST_MUL = 29,
    MOE_ROUTER = 30,
    MOE_EXPERTS = 31,
    COUNT = 33,
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
        case KernelId::ACTIVATION_QUANTIZE_FP8: return "ACTIVATION_QUANTIZE_FP8";
        case KernelId::ACTIVATION_QUANTIZE_W4A8: return "ACTIVATION_QUANTIZE_W4A8";
        case KernelId::LINEAR_FP8: return "LINEAR_FP8";
        case KernelId::LINEAR_PSQ5: return "LINEAR_PSQ5";
        case KernelId::CONCAT: return "CONCAT";
        case KernelId::VERIFY_ACCEPT: return "VERIFY_ACCEPT";
        case KernelId::GROUP_MEAN: return "GROUP_MEAN";
        case KernelId::GROUP_BROADCAST_MUL: return "GROUP_BROADCAST_MUL";
        case KernelId::MOE_ROUTER: return "MOE_ROUTER";
        case KernelId::MOE_EXPERTS: return "MOE_EXPERTS";
        default: return "UNKNOWN";
    }
}

}

