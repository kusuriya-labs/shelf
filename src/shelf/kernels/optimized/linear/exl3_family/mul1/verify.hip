#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <shelf/kernels/optimized/linear/exl3_family/mul1/launch_plan.h>

namespace shelf::kernel {

hipError_t launch_exl3_mul1_verify(const Exl3GemmArgs& args, const Exl3ExecutionPolicy& policy,
                                   hipStream_t stream) {
    Exl3GemmArgs tuned = args;
    exl3_detail::apply_execution_policy(tuned, policy);
    return launch_gemm_exl3_w16a16(tuned, stream);
}

}  // namespace shelf::kernel
