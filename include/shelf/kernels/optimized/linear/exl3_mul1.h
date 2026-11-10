#pragma once

// mul1 (W16A16) codebook entry points. Shared EXL3 types and contracts live in
// exl3_family/types.h; mul24 entry points live in exl3_mul24.h.

#include <shelf/kernels/optimized/linear/exl3_family/types.h>

namespace shelf::kernel {

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream);
hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream);
hipError_t launch_rotate_exl3_rowsum(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                     hipStream_t stream);
hipError_t launch_recon_exl3_dense(const Exl3GemmArgs& args, void* dense, hipStream_t stream);
hipError_t launch_epilogue_exl3_dense(const Exl3GemmArgs& args, const void* rowsum,
                                      const void* accum, hipStream_t stream);
hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream);

// Verify entry: applies the policy's fixed tuning rows and runs the mul1 gemm. The
// kernel is the same as decode; only the split-K tuning input differs.
hipError_t launch_exl3_mul1_verify(const Exl3GemmArgs& args, const Exl3ExecutionPolicy& policy,
                                   hipStream_t stream);

// ---- grouped (MoE) ----

// Correctness-first grouped EXL3 linear. It dispatches the existing dense EXL3
// kernel once per group, so it preserves the packed expert layout but has a launch
// per group. A fused grouped kernel replaces this for throughput later.
hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream);

}  // namespace shelf::kernel
