#include <shelf/inference/execution/fp8_ptpc_dispatch.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/fp8_ptpc.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

namespace shelf::execution {
namespace {

using ::shelf::runtime::DispatchBinding;
using ::shelf::runtime::Int8ActivationWorkspaceLayout;
using ::shelf::runtime::KernelId;
using ::shelf::runtime::Program;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueId;
using ::shelf::runtime::ValueRowDomain;

Status bad(const char* msg) {
    return Status::invalid_state(msg, __FILE__, __LINE__);
}

constexpr uint32_t kRowTile = 16u;

uint32_t padded_row_count(uint32_t rows) noexcept {
    return (rows + kRowTile - 1u) & ~(kRowTile - 1u);
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_activation_quantize_fp8_pertoken(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 0u) return bad("fp8 pertoken: io");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    if (in.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count < b.activation_kp || in.value().ptr == nullptr) {
        return bad("fp8 pertoken: input contract");
    }
    if (b.activation_max_rows == 0u || b.workspace_slot >= program.workspace.ranges.size())
        return bad("fp8 pertoken: workspace");
    const uint64_t base = program.workspace.ranges[b.workspace_slot].offset;
    const auto layout =
        Int8ActivationWorkspaceLayout::make(b.activation_kp, b.activation_max_rows);
    uint8_t* workspace = ctx.workspace + base;
    const uint32_t rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    const uint32_t padded_rows = padded_row_count(rows);
    if (padded_rows > b.activation_max_rows)
        return bad("fp8 pertoken: row tile exceeds the workspace");
    
    const hipError_t error = ::shelf::kernel::launch_activation_quantize_fp8_pertoken(
        static_cast<const ::shelf::bf16_t*>(in.value().ptr),
        workspace + layout.codes_offset,
        reinterpret_cast<float*>(workspace + layout.scales_offset), rows, padded_rows,
        b.activation_max_rows, b.activation_kp, in.value().row_stride,
        layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
    if (error != hipSuccess)
        return Status::hip_error("fp8 pertoken quant launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_linear_fp8_ptpc(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 0u || b.output_count != 1u) return bad("fp8 ptpc: io");
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return bad("fp8 ptpc: weight");
    if (b.activation_max_rows == 0u || b.workspace_slot >= program.workspace.ranges.size())
        return bad("fp8 ptpc: workspace");
    const auto w = ctx.weights[b.weight_index];
    if (w.codes == nullptr || w.scales == nullptr || w.rows == 0u || w.cols == 0u) {
        return bad("fp8 ptpc: weight view");
    }
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if ((out.value().dtype != ValueDType::BF16 &&
         out.value().dtype != ValueDType::F32) ||
        out.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.value().feature_count != w.rows || out.value().ptr == nullptr) {
        return bad("fp8 ptpc: output contract");
    }
    const uint64_t base = program.workspace.ranges[b.workspace_slot].offset;
    const auto layout =
        Int8ActivationWorkspaceLayout::make(b.activation_kp, b.activation_max_rows);
    uint8_t* workspace = ctx.workspace + base;
    const uint32_t rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    const ::shelf::kernel::Fp8PertokenGemmOutputDType output_dtype =
        out.value().dtype == ValueDType::F32
            ? ::shelf::kernel::Fp8PertokenGemmOutputDType::F32
            : ::shelf::kernel::Fp8PertokenGemmOutputDType::BF16;
    const hipError_t error = ::shelf::kernel::launch_gemm_fp8_ptpc(
        static_cast<const uint8_t*>(w.codes), static_cast<const float*>(w.scales),
        workspace + layout.codes_offset,
        reinterpret_cast<const float*>(workspace + layout.scales_offset), out.value().ptr,
        output_dtype, rows, w.rows, w.cols, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, out.value().row_stride, stream);
    if (error != hipSuccess)
        return Status::hip_error("fp8 ptpc gemm launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
