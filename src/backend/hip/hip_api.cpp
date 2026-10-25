// shelf backend/hip — implementation (host-side runtime API only; kernels
// arrive in Phase 5 and will use the HIP language TU path).
#include <utility>

#include <shelf/backend/hip/hip_api.hpp>

namespace shelf::hip {
namespace {

Status map_err(hipError_t err, const char* expr, const char* file, int line) {
    if (err == hipSuccess) return Status::make_ok();
    return Status::device_error(expr, hipGetErrorString(err), file, line);
}

#define SHELF_HIP_CHECK(expr) \
    map_err((expr), #expr, __FILE__, __LINE__)

Status same_device_as(const DeviceBuffer& buf) {
    DeviceId current = backend::kInvalidDevice;
    SHELF_RETURN_IF_ERROR(get_device(current));
    if (current != buf.device()) {
        return Status::invalid_state("buffer device does not match current device",
                                     __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status valid_bytes(std::size_t bytes) {
    if (bytes == 0) {
        return Status::invalid_argument("zero-byte transfer rejected", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

} // namespace

// ---- device ---------------------------------------------------------------

Result<std::vector<DeviceInfo>> enumerate_devices() {
    int count = 0;
    SHELF_RETURN_IF_ERROR(SHELF_HIP_CHECK(hipGetDeviceCount(&count)));
    std::vector<DeviceInfo> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        SHELF_RETURN_IF_ERROR(SHELF_HIP_CHECK(hipGetDeviceProperties(&prop, i)));
        DeviceInfo info;
        info.id = static_cast<DeviceId>(i);
        info.kind = backend::DeviceKind::gpu;
        info.name = prop.name;
        info.total_memory_bytes = prop.totalGlobalMem;
        info.max_block_dim = prop.maxThreadsPerBlock;
        info.supports_unified_memory = prop.unifiedAddressing != 0;
        out.push_back(std::move(info));
    }
    return out;
}

Status set_device(DeviceId id) {
    return SHELF_HIP_CHECK(hipSetDevice(static_cast<int>(id)));
}

Status get_device(DeviceId& out) {
    int current = 0;
    SHELF_RETURN_IF_ERROR(SHELF_HIP_CHECK(hipGetDevice(&current)));
    out = static_cast<DeviceId>(current);
    return Status::make_ok();
}

Status device_synchronize() {
    return SHELF_HIP_CHECK(hipDeviceSynchronize());
}

// ---- Event ----------------------------------------------------------------

Result<Event> Event::create(bool with_timing) {
    Event ev;
    const unsigned flags = with_timing ? hipEventDefault : hipEventDisableTiming;
    Status st = SHELF_HIP_CHECK(hipEventCreateWithFlags(&ev.handle_, flags));
    if (!st.ok()) return st;
    ev.with_timing_ = with_timing;
    return ev;
}

Event::Event(Event&& other) noexcept
    : handle_(other.handle_), with_timing_(other.with_timing_), recorded_(other.recorded_) {
    other.handle_ = nullptr;
    other.recorded_ = false;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (handle_) (void)hipEventDestroy(handle_);
        handle_ = other.handle_;
        with_timing_ = other.with_timing_;
        recorded_ = other.recorded_;
        other.handle_ = nullptr;
        other.recorded_ = false;
    }
    return *this;
}

Event::~Event() {
    if (handle_) (void)hipEventDestroy(handle_);
}

Status Event::record(const Stream& stream) const {
    if (!handle_) return Status::invalid_state("Event has no handle", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(SHELF_HIP_CHECK(hipEventRecord(handle_, stream.handle_)));
    recorded_ = true;
    return Status::make_ok();
}

Status Event::wait(const Stream& stream) const {
    if (!handle_) return Status::invalid_state("Event has no handle", __FILE__, __LINE__);
    if (!recorded_) {
        return Status::invalid_state("Event::wait before record", __FILE__, __LINE__);
    }
    return SHELF_HIP_CHECK(hipStreamWaitEvent(stream.handle_, handle_, 0));
}

Status Event::synchronize() const {
    if (!handle_) return Status::invalid_state("Event has no handle", __FILE__, __LINE__);
    if (!recorded_) {
        return Status::invalid_state("Event::synchronize before record", __FILE__, __LINE__);
    }
    return SHELF_HIP_CHECK(hipEventSynchronize(handle_));
}

Result<double> Event::elapsed_us(const Event& start, const Event& end) {
    if (!start.with_timing_ || !end.with_timing_) {
        return Status::invalid_state("elapsed_us requires timing events", __FILE__, __LINE__);
    }
    if (!start.recorded_ || !end.recorded_) {
        return Status::invalid_state("elapsed_us requires recorded events", __FILE__, __LINE__);
    }
    float ms = 0.0f;
    SHELF_RETURN_IF_ERROR(SHELF_HIP_CHECK(hipEventElapsedTime(&ms, start.handle_, end.handle_)));
    return static_cast<double>(ms) * 1000.0;
}

// ---- Stream ---------------------------------------------------------------

Result<Stream> Stream::create(bool non_blocking) {
    Stream s;
    const unsigned flags = non_blocking ? hipStreamNonBlocking : hipStreamDefault;
    Status st = SHELF_HIP_CHECK(hipStreamCreateWithFlags(&s.handle_, flags));
    if (!st.ok()) return st;
    return s;
}

Stream::Stream(Stream&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (handle_) (void)hipStreamDestroy(handle_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

Stream::~Stream() {
        if (handle_) (void)hipStreamDestroy(handle_);
}

Status Stream::synchronize() const {
    if (!handle_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    return SHELF_HIP_CHECK(hipStreamSynchronize(handle_));
}

// ---- DeviceBuffer -----------------------------------------------------------

Result<DeviceBuffer> DeviceBuffer::allocate(std::size_t bytes, DeviceId device) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    DeviceId previous = backend::kInvalidDevice;
    SHELF_RETURN_IF_ERROR(get_device(previous));
    if (previous != device) {
        SHELF_RETURN_IF_ERROR(set_device(device));
    }
    void* ptr = nullptr;
    Status st = SHELF_HIP_CHECK(hipMalloc(&ptr, bytes));
    if (!st.ok()) {
        if (previous != device) (void)set_device(previous);  // best-effort restore
        return st;
    }
    if (previous != device) set_device(previous);
    DeviceBuffer buf;
    buf.ptr_ = ptr;
    buf.capacity_bytes_ = bytes;
    buf.device_ = device;
    return buf;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), capacity_bytes_(other.capacity_bytes_), device_(other.device_) {
    other.ptr_ = nullptr;
    other.capacity_bytes_ = 0;
    other.device_ = backend::kInvalidDevice;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_) (void)hipFree(ptr_);
        ptr_ = other.ptr_;
        capacity_bytes_ = other.capacity_bytes_;
        device_ = other.device_;
        other.ptr_ = nullptr;
        other.capacity_bytes_ = 0;
        other.device_ = backend::kInvalidDevice;
    }
    return *this;
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_) (void)hipFree(ptr_);  // synchronous; safe with in-flight work
}

backend::BufferView DeviceBuffer::view() const {
    return backend::BufferView{ptr_, capacity_bytes_, backend::MemoryKind::device, device_};
}

// ---- transfers -------------------------------------------------------------

Status copy_h2d(DeviceBuffer& dst, const void* src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    if (src == nullptr) return Status::invalid_argument("h2d null source", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    return SHELF_HIP_CHECK(hipMemcpyAsync(dst.ptr_, src, bytes, hipMemcpyHostToDevice, stream.handle_));
}

Status copy_d2h(void* dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    if (dst == nullptr) return Status::invalid_argument("d2h null destination", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(src.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(src));
    return SHELF_HIP_CHECK(hipMemcpyAsync(dst, src.ptr_, bytes, hipMemcpyDeviceToHost, stream.handle_));
}

Status copy_d2d(DeviceBuffer& dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(src.view().contains(0, bytes));
    if (dst.device() != src.device()) {
        return Status::unsupported("cross-device d2d not wired yet", __FILE__, __LINE__);
    }
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    return SHELF_HIP_CHECK(hipMemcpyAsync(dst.ptr_, src.ptr_, bytes, hipMemcpyDeviceToDevice, stream.handle_));
}

Status fill_zero(DeviceBuffer& dst, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    return SHELF_HIP_CHECK(hipMemsetAsync(dst.ptr_, 0, bytes, stream.handle_));
}

#undef SHELF_HIP_CHECK

} // namespace shelf::hip
