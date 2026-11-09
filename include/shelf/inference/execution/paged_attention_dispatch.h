#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/program/program.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::execution {

struct HostExecutionContext;

struct PagedAttentionResolvedArgs {
    ::shelf::kernel::KVCacheDType kv_dtype = ::shelf::kernel::KVCacheDType::BF16;
    ::shelf::runtime::ValueDType output_dtype = ::shelf::runtime::ValueDType::BF16;
    ::shelf::kernel::PagedAttentionCommonArgs common;
    const bf16_t* k_bf16_pool = nullptr;
    const bf16_t* v_bf16_pool = nullptr;
    const ::shelf::kernel::fp8e4m3_storage_t* k_fp8_pool = nullptr;
    const ::shelf::kernel::fp8e4m3_storage_t* v_fp8_pool = nullptr;
    const float* k_scale_pool = nullptr;
    const float* v_scale_pool = nullptr;
    const uint8_t* k_octave_pool = nullptr;
    const uint8_t* v_octave_pool = nullptr;
    const float* octave_levels = nullptr;
    ::shelf::kernel::OctaveKVCacheFormat octave_format{};
    uint32_t octave_source_layer = 0u;
};

Result<PagedAttentionResolvedArgs>
resolve_paged_attention_args(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx);

struct PagedAttentionPlan {
    bool optimized = false;
    bool use_prefill = false;
    uint32_t splits = 1u;
    uint32_t prefill_kv_blocks = 0u;
};

constexpr uint32_t kPagedAttentionPrefillMinRows = 128u;
constexpr uint32_t kPagedAttentionSplitMinVisible = 2048u;

uint32_t make_paged_attention_plan_key(const PagedAttentionPlan& plan);

PagedAttentionPlan plan_paged_attention(
    const PagedAttentionResolvedArgs& args,
    const HostExecutionContext& ctx,
    bool use_prefill);

Result<uint32_t> paged_attention_plan_signature(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    bool use_prefill);

Result<OptimizedLaunchResult>
try_launch_paged_attention(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace shelf::execution
