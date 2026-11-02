#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/kernels/optimized/kv_append.h>
#include <phaseshift/kernels/kv_types.h>
#include <phaseshift/inference/execution/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::execution {

struct HostExecutionContext;

struct KvAppendResolvedArgs {
    ::ps::kernel::KVCacheDType kv_dtype = ::ps::kernel::KVCacheDType::BF16;
    ::ps::kernel::KvAppendCommonArgs common;
    bf16_t* k_bf16_pool = nullptr;
    bf16_t* v_bf16_pool = nullptr;
    ::ps::kernel::fp8e4m3_storage_t* k_fp8_pool = nullptr;
    ::ps::kernel::fp8e4m3_storage_t* v_fp8_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;
};

Result<KvAppendResolvedArgs>
resolve_kv_append_args(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx);

Result<OptimizedLaunchResult>
try_launch_kv_append(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::execution
