#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/tp/tp_execution.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace ps::runtime {

inline uint64_t tp_target_bytes(const TpSumTarget& target) {
    return target.rows * static_cast<uint64_t>(target.row_stride) *
           value_dtype_bytes(target.dtype);
}

inline float tp_bf16_bits_to_f32(std::uint16_t v) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline void tp_load_bf16_f32(std::vector<float>& accum, const void* src,
                             std::size_t count) {
    const auto* s = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < count; ++i) {
        accum[i] = tp_bf16_bits_to_f32(s[i]);
    }
}

inline void tp_accumulate_bf16_f32(std::vector<float>& accum, const void* src,
                                   std::size_t count) {
    const auto* s = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < count; ++i) {
        accum[i] += tp_bf16_bits_to_f32(s[i]);
    }
}

inline void tp_load_f32(std::vector<float>& accum, const void* src, std::size_t count) {
    const auto* s = static_cast<const float*>(src);
    std::memcpy(accum.data(), s, count * sizeof(float));
}

inline void tp_accumulate_f32(std::vector<float>& accum, const void* src,
                              std::size_t count) {
    const auto* s = static_cast<const float*>(src);
    for (std::size_t i = 0; i < count; ++i) {
        accum[i] += s[i];
    }
}

inline void tp_store_bf16(void* dst, const std::vector<float>& accum,
                          std::size_t count) {
    auto* d = static_cast<std::uint16_t*>(dst);
    for (std::size_t i = 0; i < count; ++i) {
        d[i] = f32_to_bf16_rne(accum[i]);
    }
}

inline void tp_store_f32(void* dst, const std::vector<float>& accum,
                         std::size_t count) {
    std::memcpy(dst, accum.data(), count * sizeof(float));
}

inline void tp_reduce_f32_accumulate(const std::vector<const void*>& srcs,
                                     std::size_t count, ValueDType dtype, void* out) {
    if (srcs.empty() || count == 0 || out == nullptr) return;
    std::vector<float> accum(count, 0.0f);
    if (dtype == ValueDType::F32) {
        tp_load_f32(accum, srcs[0], count);
        for (std::size_t r = 1; r < srcs.size(); ++r) {
            tp_accumulate_f32(accum, srcs[r], count);
        }
        tp_store_f32(out, accum, count);
    } else {
        tp_load_bf16_f32(accum, srcs[0], count);
        for (std::size_t r = 1; r < srcs.size(); ++r) {
            tp_accumulate_bf16_f32(accum, srcs[r], count);
        }
        tp_store_bf16(out, accum, count);
    }
}

inline Status validate_tp_sum_invocation(const TpSumInvocation& invocation,
                                         std::size_t rank_count,
                                         std::size_t& bytes_out) {
    if (rank_count < 2) {
        return Status::invalid_argument("tp transport requires at least 2 ranks",
                                        __FILE__, __LINE__);
    }
    if (invocation.targets.size() != rank_count ||
        invocation.streams.size() != rank_count ||
        invocation.ready.size() != rank_count) {
        return Status::invalid_argument("tp sum invocation rank count mismatch",
                                        __FILE__, __LINE__);
    }
    const TpSumTarget& first = invocation.targets[0];
    if (first.ptr == nullptr || first.rows == 0 || first.row_stride == 0 ||
        first.feature_count == 0) {
        return Status::invalid_argument("tp sum target is empty", __FILE__, __LINE__);
    }
    if (first.dtype != ValueDType::BF16 && first.dtype != ValueDType::F32) {
        return Status::unsupported("tp sum supports BF16 and F32 targets only",
                                   __FILE__, __LINE__);
    }
    const uint64_t bytes = tp_target_bytes(first);
    for (std::size_t r = 1; r < rank_count; ++r) {
        const TpSumTarget& t = invocation.targets[r];
        if (t.ptr == nullptr || t.rows != first.rows ||
            t.row_stride != first.row_stride || t.feature_count != first.feature_count ||
            t.dtype != first.dtype) {
            return Status::invalid_argument("tp sum targets do not match", __FILE__, __LINE__);
        }
    }
    bytes_out = static_cast<std::size_t>(bytes);
    return Status::make_ok();
}

}  // namespace ps::runtime
