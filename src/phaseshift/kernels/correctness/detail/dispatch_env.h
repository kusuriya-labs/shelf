#pragma once

#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/quantization/codec/e4m3_detail.h>
#include <phaseshift/quantization/codec/ue4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace ps::kernel::detail {

using bf16 = __hip_bfloat16;

using ::ps::runtime::DispatchBinding;
using ::ps::runtime::DeviceProgramView;
using ::ps::runtime::DeviceValueBinding;
using ::ps::runtime::DeviceWorkspaceRange;
using ::ps::runtime::DeviceStateBinding;
using ::ps::runtime::ProgramStatus;
using ::ps::runtime::ValueDType;
using ::ps::runtime::ValueRowDomain;
using ::ps::runtime::ValueStorage;
using ::ps::runtime::KernelId;
using ::ps::runtime::WeightSlot;
using ::ps::runtime::StaticParameterSlot;
using ::ps::runtime::kNoParameter;

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kScratchRowCapacity = 2048;
constexpr uint32_t kDispatchBlocks = 64;

struct ModelDispatchEnv {
    const ::ps::runtime::DeviceProgramView* program;
    ::ps::runtime::DeviceBatchContext* context;
    float* scratch;
    const ModelDispatchStateView* qwen_state;
    uint64_t workspace_base;
    uint32_t num_blocks;
    uint32_t* error_word;
};

__device__ __forceinline__ uint32_t global_tid(ModelDispatchEnv&) {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t global_stride(ModelDispatchEnv& env) {
    return env.num_blocks * blockDim.x;
}

struct ResolvedValue {
    void* ptr = nullptr;
    ValueDType dtype = ValueDType::BF16;
    ValueRowDomain row_domain = ValueRowDomain::TOKEN_ROWS;
    uint32_t features = 0;
    uint32_t row_stride = 0;
};

__device__ __forceinline__ bool resolve_value(
    ModelDispatchEnv& env, uint32_t logical_value, ResolvedValue& out) {
    const auto* p = env.program;
    for (uint32_t i = 0; i < p->value_count; ++i) {
        const auto& v = p->values[i];
        if (v.logical_value != logical_value) continue;
        out.dtype = v.dtype;
        out.row_domain = v.row_domain;
        out.features = v.feature_count;
        out.row_stride = v.row_stride;
        if (v.storage == ::ps::runtime::ValueStorage::WORKSPACE ||
            v.storage == ::ps::runtime::ValueStorage::WORKSPACE_VIEW) {
            out.ptr = reinterpret_cast<void*>(env.workspace_base + v.offset);
            return out.ptr != nullptr || v.bytes == 0;
        }
        out.ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(v.offset));
        return true;
    }
    return false;
}

__device__ __forceinline__ uint32_t value_rows(
    const ModelDispatchEnv& env, const ResolvedValue& value) {
    if (env.context == nullptr) return 1;
    if (value.row_domain == ValueRowDomain::OUTPUT_ROWS) {
        return env.context->num_outputs;
    }
    return env.context->actual_rows ? env.context->actual_rows : 1u;
}

__device__ __forceinline__ float load_elem(const ResolvedValue& v, size_t idx) {
    switch (v.dtype) {
        case ValueDType::BF16:
            return __bfloat162float(static_cast<const bf16*>(v.ptr)[idx]);
        case ValueDType::F32:
            return static_cast<const float*>(v.ptr)[idx];
        case ValueDType::I32:
            return static_cast<float>(static_cast<const int32_t*>(v.ptr)[idx]);
    }
    return 0.f;
}

__device__ __forceinline__ void store_elem(const ResolvedValue& v, size_t idx, float x) {
    switch (v.dtype) {
        case ValueDType::BF16:
            static_cast<bf16*>(v.ptr)[idx] = __float2bfloat16(x);
            break;
        case ValueDType::F32:
            static_cast<float*>(v.ptr)[idx] = x;
            break;
        case ValueDType::I32:
            static_cast<int32_t*>(v.ptr)[idx] = static_cast<int32_t>(x);
            break;
    }
}

__device__ __forceinline__ const void* resolve_parameter(
    const ModelDispatchEnv& env, uint32_t index, uint32_t* elements, ValueDType* dtype) {
    const auto* p = env.program;
    if (index == ::ps::runtime::kNoParameter) return nullptr;
    if (p->parameter_table == nullptr || index >= p->parameter_slot_count) return nullptr;
    const auto* table = static_cast<const StaticParameterSlot*>(p->parameter_table);
    if (elements != nullptr) *elements = table[index].elements;
    if (dtype != nullptr) *dtype = table[index].dtype;
    return table[index].device_ptr;
}

__device__ __forceinline__ float load_parameter_elem(
    const void* ptr, ValueDType dtype, uint32_t index) {
    switch (dtype) {
        case ValueDType::BF16:
            return __bfloat162float(static_cast<const bf16*>(ptr)[index]);
        case ValueDType::F32:
            return static_cast<const float*>(ptr)[index];
        case ValueDType::I32:
            return static_cast<float>(static_cast<const int32_t*>(ptr)[index]);
    }
    return 0.f;
}

__device__ __forceinline__ uint64_t a8_ws_scales_offset(
    uint32_t code_stride, uint32_t max_rows) {
    const uint64_t rows_padded = (static_cast<uint64_t>(max_rows) + 15ull) & ~15ull;
    return (static_cast<uint64_t>(code_stride) * rows_padded + 15ull) & ~15ull;
}

__device__ __forceinline__ float gdn_softplus(float x) {
    return __logf(1.0f + __expf(x));
}

__device__ __forceinline__ const ::ps::runtime::DeviceStateBinding* find_state(
    const ModelDispatchEnv& env, uint32_t descriptor_slot) {
    if (descriptor_slot == 0u || env.program->states == nullptr) return nullptr;
    for (uint32_t i = 0; i < env.program->state_count; ++i) {
        if (env.program->states[i].slot + 1u == descriptor_slot)
            return &env.program->states[i];
    }
    return nullptr;
}

__device__ __forceinline__ ModelDispatchEnv make_env(
    const DeviceProgramView& program,
    ::ps::runtime::DeviceBatchContext* context,
    const ModelDispatchStateView& state,
    float* scratch,
    uint32_t num_blocks,
    uint32_t* error_word = nullptr) {
    ModelDispatchEnv env{};
    env.program = &program;
    env.context = context;
    env.scratch = scratch + static_cast<size_t>(blockIdx.x) * kScratchRowCapacity;
    env.qwen_state = &state;
    env.workspace_base = program.workspace_base;
    env.num_blocks = num_blocks;
    env.error_word = error_word;
    return env;
}

__device__ __forceinline__ void record_status(uint32_t* error_word, ProgramStatus st) {
    if (st != ProgramStatus::COMPLETE) {
        atomicCAS(error_word,
                  static_cast<unsigned>(ProgramStatus::COMPLETE),
                  static_cast<unsigned>(st));
    }
}

}  // namespace ps::kernel::detail
