#pragma once
#include <shelf/models/hy4_flash/residency.h>
#include <cstdint>
#ifdef SHELF_HY4_RDMA
#include <shelf/models/hy4_flash/snapshot/transport.h>
#endif
#include <span>
#include <vector>

namespace shelf::hy4::snapshot {
#ifdef SHELF_HY4_RDMA
Digest fingerprint(Residency &gpu, const std::vector<int> &history);
#endif
std::vector<uint8_t> capture(Residency &gpu, const std::vector<int> &history);
std::vector<int> inspect(Residency &gpu, std::span<const uint8_t> state);
std::vector<int> restore(Residency &gpu, std::span<const uint8_t> state);
void clear_for_test(Residency &gpu);
}
