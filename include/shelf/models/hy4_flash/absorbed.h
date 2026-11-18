#pragma once
#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
void fold_kv_weights(Residency &gpu);
void attend_absorbed(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                     const int *positions, int stride, Param sinks, Mat output);
}
