#pragma once

#include <shelf/core/memory/tensor.h>

namespace shelf {
namespace state {

using GdnStateScalar = float;

struct GDNState {
    gpu::Tensor conv_state;
    gpu::Tensor recurrent_state;
};

}
}
