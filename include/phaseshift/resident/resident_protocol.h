#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/resident/resident_model_key.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps {
namespace resident {

constexpr std::uint32_t kWireMagic = 0x31575350u;
constexpr std::uint32_t kWireVersion = 1u;
constexpr std::uint64_t kMaxFrameBytes = 512ull * 1024ull * 1024ull;

enum class Command : std::uint32_t {
    Hello = 1,
    Acquire = 2,
    Release = 3,
    Health = 4,
    Stats = 5,
    Shutdown = 6,
};

enum class StatusCode : std::uint32_t {
    Ok = 0,
    Error = 1,
    NotFound = 2,
    Unhealthy = 3,
};

#pragma pack(push, 1)
struct WireHeader {
    std::uint32_t magic = kWireMagic;
    std::uint32_t version = kWireVersion;
    std::uint32_t cmd = 0;
    std::uint32_t status = 0;
    std::uint64_t payload_size = 0;
};

struct HelloInfo {
    std::uint32_t proto_version = kWireVersion;
    std::int32_t device = 0;
    std::uint32_t reserved = 0;
    std::uint32_t pad = 0;
    std::uint64_t resident_bytes_total = 0;
};

struct HealthInfo {
    std::uint32_t healthy = 0;
    std::int32_t device = 0;
    std::uint64_t elapsed_us = 0;
};
#pragma pack(pop)

static_assert(sizeof(WireHeader) == 24, "WireHeader layout");
static_assert(sizeof(HelloInfo) == 24, "HelloInfo layout");
static_assert(sizeof(HealthInfo) == 16, "HealthInfo layout");

Status write_bytes(int fd, const void* data, std::size_t size);
Status read_bytes(int fd, void* data, std::size_t size);
Status write_frame(int fd, Command cmd, StatusCode status, const void* payload,
                   std::size_t size);
Result<WireHeader> read_frame(int fd, std::vector<std::byte>& payload);

std::vector<std::byte> serialize_key(const ResidentModelKey& key);
Result<ResidentModelKey> parse_key(const std::byte* data, std::size_t size);

std::string default_socket_path();

}
}
