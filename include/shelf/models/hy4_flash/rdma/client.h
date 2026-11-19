#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace shelf::hy4::rdma {
using Indices = std::array<std::vector<uint64_t>, 4>;
using Tables = std::array<std::span<const uint8_t>, 4>;
class Client {
  public:
    Client();
    ~Client();
    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;
    void shutdown();
    Tables read(const Indices &indices);
    void validate_rows(const std::array<uint64_t, 4> &rows) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
