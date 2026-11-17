#pragma once

#include <shelf/runtime/tp/pipeline_parallel.h>

namespace shelf::qwen35::runtime {

using ::shelf::runtime::PipelineParallelConfig;
using ::shelf::runtime::PipelineStageModelBuilder;
using ::shelf::runtime::PipelineStageRuntime;
using ::shelf::runtime::PipelineStageRuntimeConfig;

::shelf::Result<std::unique_ptr<shelf::models::PipelineStageModel>>
build_dense_pipeline_stage_model(
    const std::string& model_dir,
    const ::shelf::qwen_hybrid::PipelineStage& stage,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ::shelf::qwen_hybrid::HybridLoadOptions& load_options);

}  // namespace shelf::qwen35::runtime
