#pragma once

#include <shelf/runtime/tp/tensor_parallel.h>

namespace shelf::qwen35::runtime {

using ::shelf::runtime::DrafterFootprintFn;
using ::shelf::runtime::MtpBatchReserveConfig;
using ::shelf::runtime::MtpBatchReserveFn;
using ::shelf::runtime::PrivateWeightLoader;
using ::shelf::runtime::TpCoordinator;
using ::shelf::runtime::TpCoordinatorConfig;
using ::shelf::runtime::TpRankMemoryPlan;
using ::shelf::runtime::TpRankRuntime;
using ::shelf::runtime::TpRankRuntimeConfig;

Result<std::unique_ptr<shelf::models::TpRankModel>> build_dense_tp_rank_model(
    const TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream);

}  // namespace shelf::qwen35::runtime
