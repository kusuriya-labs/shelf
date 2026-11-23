#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <string>

namespace shelf::models {

// Loads this rank's private target head / embedding. The tensor names belong to the
// checkpoint contract, not to the application, so they stay in the model layer.
Status load_private_target_weights(const std::string& model_dir,
                                   gpu::GpuArena& arena,
                                   hipStream_t stream,
                                   shelf::weights::MatrixWeight& lm_head,
                                   shelf::weights::MatrixWeight& embed_tokens,
                                   bool load_lm_head,
                                   bool load_embed_tokens);

}  // namespace shelf::models
