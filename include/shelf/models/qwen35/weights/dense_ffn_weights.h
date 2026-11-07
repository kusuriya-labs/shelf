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

inline bool ffn_uses_exl3_linear(const DenseFfnWeights& ffn) noexcept {
    return ffn.gate.encoding == shelf::weights::MatrixEncoding::Exl3 ||
           ffn.up.encoding == shelf::weights::MatrixEncoding::Exl3 ||
           ffn.down.encoding == shelf::weights::MatrixEncoding::Exl3;
}

inline void note_ffn_exl3_linear(const DenseFfnWeights& ffn,
                                 shelf::weights::Exl3LinearShape& shape) noexcept {
    shelf::weights::note_exl3_linear(ffn.gate, shape);
    shelf::weights::note_exl3_linear(ffn.up, shape);
    shelf::weights::note_exl3_linear(ffn.down, shape);
}

}  // namespace qwen35
}  // namespace shelf
