#pragma once
#include <shelf/weights/matrix_weight.h>

namespace shelf {
namespace qwen35 {

// The dense FFN of one layer: gate_proj / up_proj / down_proj over the config's
// intermediate_size. Nothing else in a layer refers to these three.
struct DenseFfnWeights {
    shelf::weights::MatrixWeight gate;
    shelf::weights::MatrixWeight up;
    shelf::weights::MatrixWeight down;
};

}  // namespace qwen35
}  // namespace shelf
