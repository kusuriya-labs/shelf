#pragma once

#include <shelf/core/status.h>
#include <shelf/runtime/tp/tp_execution.h>

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace shelf::runtime {

struct alignas(8) TpArgmaxPair {
    float value = -3.402823466e+38f;
    int32_t id = -1;
};

static_assert(sizeof(TpArgmaxPair) == 8);

constexpr std::size_t kTpArgmaxPairBytes = 8u;
constexpr std::size_t kTpReduceShotThresholdBytes = 2u * 1024u * 1024u;
constexpr std::size_t kTpPeerProbeBytes = 4096u;

struct TpGreedyProbe {
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0u;
    bool has_token_bitmask = false;
};

struct TpArgmaxInvocation {
    const float* local_logits = nullptr;
    int32_t* out_ids = nullptr;
    uint32_t rows = 0u;
    uint32_t row_stride = 0u;
    uint32_t vocab = 0u;
    uint32_t local_cols = 0u;
    uint32_t local_offset = 0u;
    uint32_t tp_size = 1u;
    hipStream_t stream = nullptr;
};

bool tp_greedy_argmax_eligible(const TpGreedyProbe& probe) noexcept;
uint32_t tp_reduce_shot_count(std::size_t bytes) noexcept;
uint32_t tp_reduce_four_input_shots(std::size_t bytes) noexcept;

void tp_argmax_pack_host(const float* logits, uint32_t rows, uint32_t row_stride,
                         uint32_t local_cols, uint32_t local_offset,
                         TpArgmaxPair* pairs);
void tp_argmax_reduce_host(const TpArgmaxPair* pairs, uint32_t pair_stride, uint32_t ranks,
                           uint32_t rows, int32_t* out_ids);
void tp_sum_bf16_4_host(uint16_t* dst, const uint16_t* s0, const uint16_t* s1,
                        const uint16_t* s2, const uint16_t* s3, std::size_t count);
void tp_sum_f32_4_host(float* dst, const float* s0, const float* s1, const float* s2,
                       const float* s3, std::size_t count);

hipError_t launch_tp_argmax_pack(const float* logits, TpArgmaxPair* pairs, uint32_t rows,
                                 uint32_t row_stride, uint32_t local_cols,
                                 uint32_t local_offset, hipStream_t stream);
hipError_t launch_tp_argmax_reduce(const TpArgmaxPair* pairs, uint32_t pair_stride,
                                   uint32_t ranks, uint32_t rows, int32_t* out_ids,
                                   hipStream_t stream);
hipError_t launch_tp_sum_bf16_4(uint16_t* dst, const uint16_t* s0, const uint16_t* s1,
                                const uint16_t* s2, const uint16_t* s3, std::size_t count,
                                hipStream_t stream);
hipError_t launch_tp_sum_f32_4(float* dst, const float* s0, const float* s1, const float* s2,
                               const float* s3, std::size_t count, hipStream_t stream);
hipError_t launch_tp_sum_f32_to_bf16_4(uint16_t* dst, const float* s0, const float* s1,
                                       const float* s2, const float* s3, std::size_t count,
                                       hipStream_t stream);

}  // namespace shelf::runtime
