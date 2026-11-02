#pragma once

#include <phaseshift/core/status.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps {
namespace runtime {

enum class StreamSignalKind : uint8_t {
    None = 0,
    ExtSignal = 1,
    HostCoherent = 2,
    HostMapped = 3,
    Device = 4,
};

const char* stream_signal_kind_name(StreamSignalKind kind);

struct StreamSignal {
    uint32_t* ptr = nullptr;
    uint32_t* device_ptr = nullptr;
    std::size_t bytes = 0;
    StreamSignalKind kind = StreamSignalKind::None;
};

Result<int> stream_wait_value_supported(int device);

Result<StreamSignal> stream_signal_alloc(std::size_t bytes);

Status stream_signal_free(const StreamSignal& memory) noexcept;

Status stream_write_value32(hipStream_t stream,
                                    const StreamSignal& memory,
                                    uint32_t value);

Status stream_wait_value32(hipStream_t stream,
                                   const StreamSignal& memory,
                                   uint32_t value);

uint32_t stream_signal_load(const StreamSignal& memory) noexcept;

}  // namespace runtime
}  // namespace ps
