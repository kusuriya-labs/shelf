// shelf backend/hip — device / buffer / queue / event (plan Phase 2).
//
// HIP types are confined to this module: only src/backend/hip/ may include
// <hip/...>. Higher layers consume these RAII handles, which produce
// backend/common value types (DeviceInfo, BufferView) as the neutral
// vocabulary. No polymorphic device abstraction on purpose (plan §1.2).
#pragma once

#include <cstddef>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <shelf/backend/common/device.hpp>
#include <shelf/status.hpp>

namespace shelf::hip {

using backend::DeviceId;
using backend::DeviceInfo;

// ---- device enumeration ---------------------------------------------------
Result<std::vector<DeviceInfo>> enumerate_devices();
Status set_device(DeviceId id);
Status get_device(DeviceId& out);
Status device_synchronize();

// ---- event ----------------------------------------------------------------
// Recorded lazily; host/stream waits before recording are rejected
// (invalid_state) so event/buffer lifetime bugs surface early.
class Event {
public:
    static Result<Event> create(bool with_timing = false);

    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event();

    bool recorded() const noexcept { return recorded_; }

    Status record(const class Stream& stream) const;
    Status wait(const class Stream& stream) const;    // stream-side dependency
    Status synchronize() const;                       // host-side wait
    // Requires both events created with timing and both recorded.
    static Result<double> elapsed_us(const Event& start, const Event& end);

private:
    Event() = default;
    hipEvent_t handle_ = nullptr;
    bool with_timing_ = false;
    mutable bool recorded_ = false;
};

// ---- stream (queue) --------------------------------------------------------
class DeviceBuffer;  // fwd for friendship

class Stream {
public:
    static Result<Stream> create(bool non_blocking = true);

    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream();  // destroys the stream; in-flight work is drained by the runtime

    Status synchronize() const;

private:
    Stream() = default;
    hipStream_t handle_ = nullptr;
    friend class Event;
    friend class DeviceBuffer;
    friend Status copy_h2d(DeviceBuffer&, const void*, std::size_t, const Stream&);
    friend Status copy_d2h(void*, const DeviceBuffer&, std::size_t, const Stream&);
    friend Status copy_d2d(DeviceBuffer&, const DeviceBuffer&, std::size_t, const Stream&);
    friend Status fill_zero(DeviceBuffer&, std::size_t, const Stream&);
};

// ---- device buffer (VRAM) ---------------------------------------------------
// Move-only. Destruction is safe with work still in flight (the runtime
// drains before storage is reusable); double-free is structurally impossible.
class DeviceBuffer {
public:
    static Result<DeviceBuffer> allocate(std::size_t bytes, DeviceId device);

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer();

    backend::BufferView view() const;
    std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    DeviceId device() const noexcept { return device_; }

private:
    DeviceBuffer() = default;
    void* ptr_ = nullptr;
    std::size_t capacity_bytes_ = 0;
    DeviceId device_ = backend::kInvalidDevice;
    friend Status copy_h2d(DeviceBuffer&, const void*, std::size_t, const Stream&);
    friend Status copy_d2h(void*, const DeviceBuffer&, std::size_t, const Stream&);
    friend Status copy_d2d(DeviceBuffer&, const DeviceBuffer&, std::size_t, const Stream&);
    friend Status fill_zero(DeviceBuffer&, std::size_t, const Stream&);
};

// ---- transfers (async on stream, sizes validated against capacity) ---------
// Zero-byte transfers are rejected: in shelf they always indicate a plan bug.
Status copy_h2d(DeviceBuffer& dst, const void* src, std::size_t bytes, const Stream& stream);
Status copy_d2h(void* dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream);
Status copy_d2d(DeviceBuffer& dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream);
Status fill_zero(DeviceBuffer& dst, std::size_t bytes, const Stream& stream);

} // namespace shelf::hip
