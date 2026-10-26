// shelf backend/sycl — device / buffer / queue / event (SYCL port of
// backend/hip's hip_api.hpp; plan Phase 2 / §4.5).
//
// SYCL types are confined to this module: only src/backend/sycl/ may include
// <sycl/...>. Higher layers consume these RAII handles, which produce
// backend/common value types (DeviceInfo, BufferView) as the neutral
// vocabulary. No polymorphic device abstraction on purpose (plan §1.2).
//
// Porting notes vs backend/hip:
//   * Stream  -> an in-order ::sycl::queue (commands on one stream execute in
//     submission order; independent streams carry no implicit order, same as
//     independent HIP streams).
//   * Event   -> a captured ::sycl::event. There is no native "record an
//     event into a queue" in SYCL 2020; record() submits a no-op marker, so
//     the event completes after every command submitted to that stream before
//     the record. wait()/synchronize() map to depends_on()/event::wait().
//   * Event timing: when the device exposes the queue_profiling aspect, the
//     stream queue is created with sycl::property::queue::enable_profiling and
//     elapsed_us() uses the standard SYCL 2020 event profiling
//     (info::event_profiling::command_start/command_end around the marker
//     commands: start = beginning of the start-marker, end = completion of
//     the done-marker, so the in-order stream's work between records is
//     measured). Host-timestamp approximation is only a fallback when the
//     aspect is unavailable.
//   * DeviceBuffer -> USM device allocation (sycl::malloc_device). The
//     kernels take raw device pointers, mirroring the HIP kernel ABI.
#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

#include <sycl/sycl.hpp>

#include <shelf/backend/common/device.hpp>
#include <shelf/status.hpp>

namespace shelf::sycl {

using backend::DeviceId;
using backend::DeviceInfo;

// ---- device enumeration ---------------------------------------------------
// Device ids index the backend's stable device registry (SYCL discovery
// order, GPUs included; kind reflects sycl device type).
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
    mutable std::shared_ptr<::sycl::event> handle_;
    bool with_timing_ = false;
    mutable bool recorded_ = false;
    mutable std::chrono::steady_clock::time_point recorded_at_{};
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

    // Queue is shared-mutable: submit() is non-const in SYCL; independent
    // Streams do not imply ordering (same contract as separate HIP streams).
    ::sycl::queue& queue() const { return *q_; }
    DeviceId device() const { return device_; }

private:
    Stream() = default;
    std::shared_ptr<::sycl::queue> q_;
    DeviceId device_ = backend::kInvalidDevice;
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

} // namespace shelf::sycl
