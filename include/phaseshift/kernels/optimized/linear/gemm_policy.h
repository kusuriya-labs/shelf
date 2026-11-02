#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

using act_fragment_t = int32_t __attribute__((ext_vector_type(2)));

struct CanonicalA8ActivationPolicy {
    const uint8_t* codes = nullptr;
    uint32_t code_stride = 0;
    const float* scales = nullptr;
    uint32_t scale_stride = 0;

    __device__ __forceinline__ const uint8_t* row_base(uint32_t row,
                                                       uint32_t k_group) const {
        return codes +
               static_cast<size_t>(row >> 4) * (static_cast<size_t>(code_stride) * 16u) +
               static_cast<size_t>(row & 15u) * 16u + static_cast<size_t>(k_group) * 8u;
    }

    __device__ __forceinline__ void load_fragment(const uint8_t* base, uint32_t ib,
                                                  act_fragment_t& a0,
                                                  act_fragment_t& a1) const {
        const uint8_t* ap = base + static_cast<size_t>(ib) * 512u;
        a0 = *reinterpret_cast<const act_fragment_t*>(ap);
        a1 = *reinterpret_cast<const act_fragment_t*>(ap + 256u);
    }

    __device__ __forceinline__ float row_scale(uint32_t row) const {
        return scales[static_cast<size_t>(row) * (scale_stride >> 2u)];
    }
};

struct CanonicalStoreEpilogue {
    void* output = nullptr;
    bool bf16 = false;

    __device__ __forceinline__ static CanonicalStoreEpilogue make(
        void* output, bool bf16, const void* /*residual*/, uint32_t /*residual_row_stride*/) {
        return CanonicalStoreEpilogue{output, bf16};
    }

    __device__ __forceinline__ void store(size_t out_idx, uint32_t /*row*/, uint32_t /*col*/,
                                          float v) const {
        if (bf16) {
            static_cast<::ps::bf16_t*>(output)[out_idx] =
                ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            static_cast<float*>(output)[out_idx] = v;
        }
    }
};

}  // namespace ps::kernel
