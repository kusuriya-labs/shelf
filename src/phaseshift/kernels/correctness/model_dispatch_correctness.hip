#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include "detail/dispatch_env.h"
#include "detail/elementwise.inc"
#include "detail/normalization.inc"
#include "detail/linear.inc"
#include "detail/attention.inc"
#include "detail/gdn.inc"
#include "detail/embedding.inc"
#include "detail/output.inc"
#include "detail/group.inc"
#include "detail/moe.inc"

namespace ps::kernel {

namespace {

using ::ps::runtime::KernelId;
using ::ps::runtime::ProgramStatus;

__device__ ProgramStatus host_execute_binding(
    detail::ModelDispatchEnv& env, const ::ps::runtime::DispatchBinding& b) {
    switch (b.kernel_id) {
        case KernelId::EMBEDDING_LOOKUP: return detail::exec_embedding(env, b);
        case KernelId::LINEAR_BF16: return detail::exec_linear_weight(env, b);
        case KernelId::LINEAR_FP8: return detail::exec_linear_block_scaled(env, b);
        case KernelId::ACTIVATION_QUANTIZE_FP8: return detail::exec_activation_quantize_fp8(env, b);
        case KernelId::ACTIVATION_QUANTIZE_W4A8: return detail::exec_activation_quantize_w4a8(env, b);
        case KernelId::RMS_NORM: return detail::exec_row_norm_like(env, b, false);
        case KernelId::L2_NORMALIZE: return detail::exec_row_norm_like(env, b, true);
        case KernelId::SPLIT: return detail::exec_split(env, b);
        case KernelId::SILU:
        case KernelId::SIGMOID:
        case KernelId::SCALE: return detail::exec_unary(env, b.kernel_id, b);
        case KernelId::MUL:
        case KernelId::SWIGLU:
        case KernelId::RESIDUAL_ADD: return detail::exec_binary(env, b.kernel_id, b);
        case KernelId::CONCAT: return detail::exec_concat(env, b);
        case KernelId::ROPE: return detail::exec_rope(env, b);
        case KernelId::OUTPUT_GATHER:
            return detail::exec_gather_or_sample(env, b, KernelId::OUTPUT_GATHER);
        case KernelId::SAMPLING:
            return detail::exec_gather_or_sample(env, b, KernelId::SAMPLING);
        case KernelId::KV_APPEND: return detail::exec_kv_append(env, b);
        case KernelId::PAGED_ATTENTION: return detail::exec_paged_attention(env, b);
        case KernelId::STATEFUL_CAUSAL_CONV1D: return detail::exec_gdn_conv1d(env, b);
        case KernelId::GDN_RECURRENCE: return detail::exec_gdn_recurrence(env, b);
        case KernelId::GROUP_MEAN: return detail::exec_group_mean(env, b);
        case KernelId::GROUP_BROADCAST_MUL: return detail::exec_group_broadcast_mul(env, b);
        case KernelId::MOE_ROUTER: return detail::exec_moe_router(env, b);
        case KernelId::MOE_EXPERTS: return detail::exec_moe_experts(env, b);
        default:
            return ProgramStatus::DISPATCH_FAILED;
    }
}

}  // namespace

constexpr uint32_t kDispatchBlocks = 64;

__global__ void model_dispatch_kernel(
    ::ps::runtime::DeviceProgramView prog,
    ::ps::runtime::DispatchBinding binding,
    ::ps::runtime::DeviceBatchContext* ctx,
    ModelDispatchStateView state,
    float* scratch,
    uint32_t num_blocks,
    uint32_t* error_word) {
    detail::ModelDispatchEnv env =
        detail::make_env(prog, ctx, state, scratch, num_blocks, error_word);
    detail::record_status(error_word, host_execute_binding(env, binding));
}

Status launch_model_dispatch_correctness(
    const ::ps::runtime::DeviceProgramView& program,
    const ::ps::runtime::DispatchBinding& binding,
    ::ps::runtime::DeviceBatchContext* batch_context,
    const ModelDispatchStateView& state,
    float* scratch,
    uint32_t* error_word,
    hipStream_t stream) {
    model_dispatch_kernel<<<kDispatchBlocks, detail::kThreadsPerBlock, 0, stream>>>(
        program, binding, batch_context, state, scratch, kDispatchBlocks, error_word);
    hipError_t herr = hipGetLastError();
    if (herr != hipSuccess)
        return Status::hip_error("model_dispatch_kernel", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    return Status::make_ok();
}

}  // namespace ps::kernel
