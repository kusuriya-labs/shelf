#pragma once

#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
using Selection=std::vector<std::vector<int>>;
Mat attention(Ops& op,const std::string& prefix,int layer,int start,Mat hidden,Selection& selected,bool draft);
}
