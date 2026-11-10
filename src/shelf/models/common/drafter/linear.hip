#include <shelf/models/common/drafter/linear.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <algorithm>

namespace shelf::drafter {

Status quantized_linear(const weights::MatrixWeight& w, const bf16_t* input,
                        uint32_t input_stride, bf16_t* output, uint32_t output_stride,
                        uint32_t rows, LinearWorkspace& workspace, hipStream_t stream) {
    if (rows == 0u) return Status::make_ok();
    if (!input || !output || input_stride < w.cols || output_stride < w.rows ||
        w.k_padded < w.cols || w.k_padded % 128u || !w.codes.ndim())
        return Status::invalid_argument("drafter linear: invalid arguments", __FILE__, __LINE__);
    const auto layout = runtime::Int8ActivationWorkspaceLayout::make(w.k_padded, rows);
    if (!workspace.codes.ndim() || !workspace.scales.ndim() ||
        workspace.codes.dim(0) < layout.codes_bytes() ||
        workspace.scales.dim(0) * sizeof(float) < layout.scales_bytes())
        return Status::insufficient_memory("drafter linear: activation workspace", __FILE__, __LINE__);
    auto* codes = workspace.codes.data<int8_t>();
    auto* scales = workspace.scales.data<float>();
    hipError_t err = hipSuccess;
    if (w.encoding == weights::MatrixEncoding::Exl3 &&
        w.quant_spec == quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
        if (w.exl3_half || w.k_padded != w.cols || w.rows % 128u || !w.exl3_suh.ndim() ||
            !w.exl3_svh.ndim() || !workspace.accum.ndim() || w.exl3_ka < 1u || w.exl3_ka > 8u)
            return Status::invalid_argument("drafter linear: invalid EXL3-INT8 layout", __FILE__, __LINE__);
        const uint32_t scratch_rows = static_cast<uint32_t>(workspace.accum.dim(0) / w.rows);
        if (scratch_rows < rows)
            return Status::insufficient_memory("drafter linear: EXL3 workspace", __FILE__, __LINE__);
        const uint32_t splits = kernel::exl3_mul24_gemm_splits(rows, w.rows, w.cols, scratch_rows);
        err = kernel::launch_exl3_mul24_rotate_quantize(input, input_stride, w.exl3_suh.data<uint16_t>(),
            codes, scales, rows, w.cols, layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
        if (err == hipSuccess) err = kernel::launch_exl3_mul24_gemm(codes, scales,
            layout.code_row_stride_bytes, layout.scale_row_stride_bytes, w.codes.data<uint16_t>(),
            w.exl3_ka, workspace.accum.data<float>(), rows, w.rows, w.cols, splits, stream);
        if (err == hipSuccess) err = kernel::launch_exl3_mul24_rotate_out(workspace.accum.data<float>(),
            w.exl3_svh.data<uint16_t>(), output, kernel::Exl3OutputDType::BF16,
            rows, w.rows, output_stride, splits, stream);
    } else if (w.encoding == weights::MatrixEncoding::Fp8Finegrained) {
        if (!w.preshuffled || !w.scales.ndim() || w.weight_scale_group != 128u)
            return Status::invalid_argument("drafter linear: invalid FP8 layout", __FILE__, __LINE__);
        const auto config = execution::select_fp8_finegrained_gemm_config(
            {rows, w.rows, w.cols, w.k_padded});
        if (!config) return Status::unsupported("drafter linear: unsupported FP8 geometry", __FILE__, __LINE__);
        err = kernel::launch_activation_quantize_fp8_finegrained(input, reinterpret_cast<uint8_t*>(codes),
            scales, rows, w.cols, w.k_padded, input_stride, layout.code_row_stride_bytes,
            layout.scale_row_stride_bytes, stream);
        if (err == hipSuccess) err = kernel::launch_gemm_fp8_finegrained_w8a8_wmma_auto(
            *config, w.codes.data<uint8_t>(), w.scales.data<float>(), codes, scales, output,
            kernel::Fp8FinegrainedGemmOutputDType::BF16, rows, w.rows, w.k_padded,
            w.storage_scale_stride_bytes, layout.code_row_stride_bytes, layout.scale_row_stride_bytes,
            output_stride, stream);
    } else {
        return Status::unsupported("drafter linear: unsupported encoding", __FILE__, __LINE__);
    }
    if (err != hipSuccess) return Status::hip_error("drafter linear launch", hipGetErrorString(err), __FILE__, __LINE__);
    return Status::make_ok();
}

}
