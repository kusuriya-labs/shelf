// shelf backend/sycl — implementation (host-side runtime API; the kernel
// layer lives in kernels/ and consumes the raw device pointers handed out
// here, mirroring backend/hip).
#include <algorithm>
#include <mutex>
#include <utility>

#include <shelf/backend/sycl/sycl_api.hpp>

namespace shelf::sycl {
namespace {

std::mutex g_mu;

// Stable device registry: ids are indices into this vector, built once from
// sycl::device::get_devices(). Each entry owns an in-order allocation queue
// so malloc/free and synchronous transfers do not depend on user streams.
struct DeviceEntry {
    ::sycl::device dev;
    std::shared_ptr<::sycl::queue> alloc_q;
    DeviceInfo info;
};

std::vector<DeviceEntry>& registry() {
    static std::vector<DeviceEntry> reg = [] {
        std::vector<DeviceEntry> out;
        // Keep only devices this binary can actually execute kernels on:
        // the SSCP/generic flow is portable to every backend; classic flows
        // carry launchers only for the compiled targets.
        const bool portable =
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
            true;
#else
            false;
#endif
        for (const auto& d : ::sycl::device::get_devices()) {
            if (!portable) {
                const bool gpu_ok =
#if defined(__ACPP_ENABLE_HIP_TARGET__) || defined(__ACPP_ENABLE_CUDA_TARGET__)
                    true;
#else
                    false;
#endif
                const bool cpu_ok =
#if defined(__ACPP_ENABLE_OMPHOST_TARGET__)
                    true;
#else
                    false;
#endif
                if (d.is_gpu() && !gpu_ok) continue;
                if (d.is_cpu() && !cpu_ok) continue;
            }
            DeviceEntry e;
            e.dev = d;
            e.alloc_q = std::make_shared<::sycl::queue>(
                d, ::sycl::property::queue::in_order{});
            DeviceInfo info;
            info.id = static_cast<DeviceId>(out.size());
            info.kind = d.is_cpu() ? backend::DeviceKind::host
                                   : backend::DeviceKind::gpu;
            info.name = d.get_info<::sycl::info::device::name>();
            info.total_memory_bytes =
                d.get_info<::sycl::info::device::global_mem_size>();
            info.max_block_dim =
                d.get_info<::sycl::info::device::max_work_group_size>();
            info.supports_unified_memory = d.has(::sycl::aspect::usm_shared_allocations);
            e.info = std::move(info);
            out.push_back(std::move(e));
        }
        return out;
    }();
    return reg;
}

DeviceId g_current = backend::kInvalidDevice;

// Live user queues (weak) so device_synchronize() can drain them the way
// hipDeviceSynchronize() drains all streams of the current device.
std::vector<std::pair<DeviceId, std::weak_ptr<::sycl::queue>>> g_live_queues;

Status entry(DeviceId id, DeviceEntry*& out) {
    auto& reg = registry();
    if (id < 0 || static_cast<std::size_t>(id) >= reg.size())
        return Status::invalid_argument("unknown device id", __FILE__, __LINE__);
    out = &reg[static_cast<std::size_t>(id)];
    return Status::make_ok();
}

Status map_exception(const char* expr, const char* file, int line) noexcept {
    try {
        throw;
    } catch (const ::sycl::exception& e) {
        return Status::device_error(expr, e.what(), file, line);
    } catch (const std::exception& e) {
        return Status::device_error(expr, e.what(), file, line);
    } catch (...) {
        return Status::device_error(expr, "unknown SYCL exception", file, line);
    }
}

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

// Best-effort host drain of every live queue on `device` (used before USM
// frees so destructor semantics match the HIP path).
void drain_device(DeviceId device) {
    std::vector<std::shared_ptr<::sycl::queue>> snapshots;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto& reg = registry();
        DeviceEntry* e = nullptr;
        if (!entry(device, e).ok()) return;
        snapshots.push_back(e->alloc_q);
        for (auto& [id, wq] : g_live_queues) {
            if (id != device) continue;
            if (auto q = wq.lock()) snapshots.push_back(std::move(q));
        }
    }
    for (auto& q : snapshots) { try { q->wait(); } catch (...) {} }
}

} // namespace

// ---- device ---------------------------------------------------------------

Result<std::vector<DeviceInfo>> enumerate_devices() {
    std::vector<DeviceInfo> out;
    for (const auto& e : registry()) out.push_back(e.info);
    return out;
}

Status set_device(DeviceId id) {
    DeviceEntry* e = nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    SHELF_RETURN_IF_ERROR(entry(id, e));
    g_current = id;
    return Status::make_ok();
}

Status get_device(DeviceId& out) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_current == backend::kInvalidDevice) {
        auto& reg = registry();
        if (reg.empty())
            return Status::device_error("no SYCL device available",
                                        "shelf::sycl::get_device", __FILE__, __LINE__);
        // Default device: first GPU, else first device (HIP semantics would
        // have failed here; the SYCL registry makes a defined choice).
        for (const auto& e : reg)
            if (e.info.kind == backend::DeviceKind::gpu) { g_current = e.info.id; break; }
        if (g_current == backend::kInvalidDevice) g_current = reg.front().info.id;
    }
    out = g_current;
    return Status::make_ok();
}

Status device_synchronize() {
    DeviceId current = backend::kInvalidDevice;
    SHELF_RETURN_IF_ERROR(get_device(current));
    std::vector<std::shared_ptr<::sycl::queue>> snapshots;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto& reg = registry();
        DeviceEntry* e = nullptr;
        SHELF_RETURN_IF_ERROR(entry(current, e));
        snapshots.push_back(e->alloc_q);
        for (auto it = g_live_queues.begin(); it != g_live_queues.end();) {
            if (auto q = it->second.lock()) {
                if (it->first == current) snapshots.push_back(std::move(q));
                ++it;
            } else {
                it = g_live_queues.erase(it);
            }
        }
    }
    try {
        for (auto& q : snapshots) q->wait_and_throw();
    } catch (...) {
        return map_exception("device_synchronize", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

// ---- Event ----------------------------------------------------------------

Result<Event> Event::create(bool with_timing) {
    Event ev;
    ev.with_timing_ = with_timing;
    return ev;
}

Event::Event(Event&& other) noexcept
    : handle_(std::move(other.handle_)), with_timing_(other.with_timing_),
      recorded_(other.recorded_), recorded_at_(other.recorded_at_) {
    other.recorded_ = false;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        handle_ = std::move(other.handle_);
        with_timing_ = other.with_timing_;
        recorded_ = other.recorded_;
        recorded_at_ = other.recorded_at_;
        other.recorded_ = false;
    }
    return *this;
}

Event::~Event() = default;

Status Event::record(const Stream& stream) const {
    if (!stream.q_) return Status::invalid_state("Stream has no queue", __FILE__, __LINE__);
    try {
        // Marker command: completes after everything submitted to this
        // (in-order) queue before this submission.
        handle_ = std::make_shared<::sycl::event>(
            stream.q_->submit([](::sycl::handler& cgh) {
                cgh.single_task([]() {});
            }));
    } catch (...) {
        return map_exception("Event::record", __FILE__, __LINE__);
    }
    recorded_ = true;
    recorded_at_ = std::chrono::steady_clock::now();
    return Status::make_ok();
}

Status Event::wait(const Stream& stream) const {
    if (!handle_) return Status::invalid_state("Event has no handle", __FILE__, __LINE__);
    if (!recorded_) {
        return Status::invalid_state("Event::wait before record", __FILE__, __LINE__);
    }
    if (!stream.q_) return Status::invalid_state("Stream has no queue", __FILE__, __LINE__);
    try {
        // SYCL 2020 queue::depends_on is not exposed by this runtime; attach
        // a dependency-carrying marker to the target queue instead. The
        // in-order queue then orders everything submitted afterwards after
        // this event (device-side; host is not blocked).
        const ::sycl::event dep = *handle_;
        stream.q_->submit([dep](::sycl::handler& cgh) {
            cgh.depends_on(dep);
            cgh.single_task([]() {});
        });
    } catch (...) {
        return map_exception("Event::wait", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status Event::synchronize() const {
    if (!handle_) return Status::invalid_state("Event has no handle", __FILE__, __LINE__);
    if (!recorded_) {
        return Status::invalid_state("Event::synchronize before record", __FILE__, __LINE__);
    }
    try {
        handle_->wait_and_throw();
    } catch (...) {
        return map_exception("Event::synchronize", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<double> Event::elapsed_us(const Event& start, const Event& end) {
    if (!start.with_timing_ || !end.with_timing_) {
        return Status::invalid_state("elapsed_us requires timing events", __FILE__, __LINE__);
    }
    if (!start.recorded_ || !end.recorded_) {
        return Status::invalid_state("elapsed_us requires recorded events", __FILE__, __LINE__);
    }
    try {
        if (start.handle_) start.handle_->wait_and_throw();
        if (end.handle_) end.handle_->wait_and_throw();
    } catch (...) {
        return map_exception("Event::elapsed_us", __FILE__, __LINE__);
    }
    const auto d = end.recorded_at_ - start.recorded_at_;
    return std::chrono::duration<double, std::micro>(d).count();
}

// ---- Stream ---------------------------------------------------------------

Result<Stream> Stream::create(bool non_blocking) {
    DeviceId current = backend::kInvalidDevice;
    SHELF_RETURN_IF_ERROR(get_device(current));
    DeviceEntry* e = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        SHELF_RETURN_IF_ERROR(entry(current, e));
    }
    Stream s;
    try {
        // SYCL has no per-queue "non-blocking" flag mirroring
        // hipStreamNonBlocking; all queues are in-order and never block the
        // host on submission, matching the flag's practical effect here.
        (void)non_blocking;
        s.q_ = std::make_shared<::sycl::queue>(e->dev, ::sycl::property::queue::in_order{});
    } catch (...) {
        return map_exception("Stream::create", __FILE__, __LINE__);
    }
    s.device_ = current;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_live_queues.emplace_back(current, std::weak_ptr<::sycl::queue>(s.q_));
    }
    return s;
}

Stream::Stream(Stream&& other) noexcept : q_(std::move(other.q_)), device_(other.device_) {
    other.device_ = backend::kInvalidDevice;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (q_) { try { q_->wait(); } catch (...) {} }
        q_ = std::move(other.q_);
        device_ = other.device_;
        other.device_ = backend::kInvalidDevice;
    }
    return *this;
}

Stream::~Stream() {
    if (q_) {
        // Drain like hipStreamDestroy (in-flight work completes before the
        // queue handle goes away).
        try { q_->wait(); } catch (...) {}
    }
}

Status Stream::synchronize() const {
    if (!q_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    try {
        q_->wait_and_throw();
    } catch (...) {
        return map_exception("Stream::synchronize", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

// ---- DeviceBuffer -----------------------------------------------------------

Result<DeviceBuffer> DeviceBuffer::allocate(std::size_t bytes, DeviceId device) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    DeviceId previous = backend::kInvalidDevice;
    SHELF_RETURN_IF_ERROR(get_device(previous));
    DeviceEntry* e = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        SHELF_RETURN_IF_ERROR(entry(device, e));
    }
    void* ptr = nullptr;
    try {
        ptr = ::sycl::malloc_device(bytes, *e->alloc_q);
    } catch (...) {
        return map_exception("DeviceBuffer::allocate", __FILE__, __LINE__);
    }
    if (ptr == nullptr)
        return Status::insufficient_memory("sycl::malloc_device returned null",
                                           __FILE__, __LINE__);
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
        if (ptr_) {
            DeviceEntry* e = nullptr;
            std::lock_guard<std::mutex> lk(g_mu);
            if (entry(device_, e).ok()) (void)::sycl::free(ptr_, *e->alloc_q);
        }
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
    if (ptr_) {
        // Mirror hipFree semantics: synchronous with respect to in-flight
        // work. SYCL USM free is only safe after all commands referencing
        // the allocation completed, so drain this device's queues first.
        drain_device(device_);
        DeviceEntry* e = nullptr;
        std::lock_guard<std::mutex> lk(g_mu);
        if (entry(device_, e).ok()) {
            try { (void)::sycl::free(ptr_, *e->alloc_q); } catch (...) {}
        }
    }
}

backend::BufferView DeviceBuffer::view() const {
    return backend::BufferView{ptr_, capacity_bytes_, backend::MemoryKind::device, device_};
}

// ---- transfers -------------------------------------------------------------

namespace {
Status enqueue_copy(::sycl::queue& q, const char* what, const char* file, int line,
                    auto&& fn) {
    try {
        fn();
        (void)q;
    } catch (...) {
        return map_exception(what, file, line);
    }
    return Status::make_ok();
}
} // namespace

Status copy_h2d(DeviceBuffer& dst, const void* src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    if (src == nullptr) return Status::invalid_argument("h2d null source", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    if (!stream.q_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    return enqueue_copy(*stream.q_, "copy_h2d", __FILE__, __LINE__, [&] {
        stream.q_->memcpy(dst.ptr_, src, bytes);
    });
}

Status copy_d2h(void* dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    if (dst == nullptr) return Status::invalid_argument("d2h null destination", __FILE__, __LINE__);
    SHELF_RETURN_IF_ERROR(src.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(src));
    if (!stream.q_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    return enqueue_copy(*stream.q_, "copy_d2h", __FILE__, __LINE__, [&] {
        stream.q_->memcpy(dst, src.ptr_, bytes);
    });
}

Status copy_d2d(DeviceBuffer& dst, const DeviceBuffer& src, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(src.view().contains(0, bytes));
    if (dst.device() != src.device()) {
        return Status::unsupported("cross-device d2d not wired yet", __FILE__, __LINE__);
    }
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    if (!stream.q_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    return enqueue_copy(*stream.q_, "copy_d2d", __FILE__, __LINE__, [&] {
        stream.q_->memcpy(dst.ptr_, src.ptr_, bytes);
    });
}

Status fill_zero(DeviceBuffer& dst, std::size_t bytes, const Stream& stream) {
    SHELF_RETURN_IF_ERROR(valid_bytes(bytes));
    SHELF_RETURN_IF_ERROR(dst.view().contains(0, bytes));
    SHELF_RETURN_IF_ERROR(same_device_as(dst));
    if (!stream.q_) return Status::invalid_state("Stream has no handle", __FILE__, __LINE__);
    return enqueue_copy(*stream.q_, "fill_zero", __FILE__, __LINE__, [&] {
        stream.q_->memset(dst.ptr_, 0, bytes);
    });
}

} // namespace shelf::sycl
