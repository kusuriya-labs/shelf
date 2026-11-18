#pragma once
#include <shelf/models/hy4_flash/residency.h>

namespace shelf::hy4 {
void generate(Residency& gpu,const std::vector<int>& ids,int count,int chunk,float temperature,uint64_t seed);
}
