#pragma once

#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_runtime.h>

namespace shelf::kernel {

hipError_t launch_qsa_key_cache(const QsaKeyCacheArgs& args, hipStream_t stream);
hipError_t launch_qsa_key_cache_batch(const QsaKeyCacheBatchArgs& args,
                                      hipStream_t stream);
hipError_t launch_qsa_key_cache_blocks(const QsaKeyCacheBatchArgs& args,
                                       hipStream_t stream);

}  // namespace shelf::kernel
