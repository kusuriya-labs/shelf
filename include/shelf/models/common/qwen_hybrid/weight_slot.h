#pragma once

#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>

namespace shelf {
namespace qwen_hybrid {

// One MatrixWeight, described the same way for a dense matrix and for a rank-3 expert
// plane. Every model family lowers its weights through this, so the runtime sees one
// layout contract whatever the FFN is.
::shelf::runtime::WeightSlot make_weight_slot(const ::shelf::weights::MatrixWeight& w);

}
}
