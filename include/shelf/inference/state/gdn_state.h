#pragma once

#include <shelf/core/memory/tensor.h>

namespace ps {
namespace state {

struct GDNState {
    gpu::Tensor conv_state;
    gpu::Tensor recurrent_state;
};

}
}
