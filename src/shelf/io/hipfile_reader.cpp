#include <shelf/io/hipfile_reader.h>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(SHELF_HAS_HIPFILE)
#include <hipfile.h>
#endif

namespace ps::io {

const char* ssd_read_path_name(SsdReadPath path) noexcept {
    switch (path) {
        case SsdReadPath::none: return "none";
        case SsdReadPath::hipfile: return "hipfile";
        case SsdReadPath::posix: return "posix";
    }
    return "unknown";
}

#if defined(SHELF_HAS_HIPFILE)

namespace {

std::mutex g_driver_mutex;
int g_driver_users = 0;

Status driver_acquire() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (g_driver_users > 0) {
        ++g_driver_users;
        return Status::make_ok();
    }
    const hipFileError_t err = hipFileDriverOpen();
    if (err.err != hipFileSuccess) {
        return Status::unsupported(
            (std::string("hipFileDriverOpen: ") + hipFileGetOpErrorString(err.err)).c_str(),
            __FILE__, __LINE__);
    }
    g_driver_users = 1;
    return Status::make_ok();
}

void driver_release() noexcept {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (g_driver_users <= 0) return;
    --g_driver_users;
    if (g_driver_users == 0) {
        const hipFileError_t err = hipFileDriverClose();
        (void)err;
    }
}

Status posix_fill(int fd, void* device_ptr, std::size_t bytes, std::uint64_t file_offset,
                  SsdReadResult& out) {
    std::vector<char> staging(bytes);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, staging.data() + done, bytes - done,
                                  static_cast<off_t>(file_offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return Status::invalid_argument(
                ("pread failed: " + std::string(std::strerror(errno))).c_str(),
                __FILE__, __LINE__);
        }
        if (n == 0) break;
        done += static_cast<std::size_t>(n);
    }
    if (done == 0) {
        out.path = SsdReadPath::posix;
        out.bytes_read = 0;
        out.short_read = true;
        return Status::make_ok();
    }
    const hipError_t he =
        hipMemcpy(device_ptr, staging.data(), done, hipMemcpyHostToDevice);
    if (he != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(he), __FILE__, __LINE__);
    }
    out.path = SsdReadPath::posix;
    out.bytes_read = done;
    out.short_read = done != bytes;
    return Status::make_ok();
}

}  // namespace

struct HipFileReader::Impl {
    int fd = -1;
    hipFileHandle_t handle = nullptr;
    const void* registered_ptr = nullptr;
    std::size_t registered_bytes = 0;
    bool driver_held = false;
};

bool hipfile_available() noexcept { return true; }

HipFileReader::~HipFileReader() noexcept { shutdown(); }

HipFileReader::HipFileReader(HipFileReader&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

HipFileReader& HipFileReader::operator=(HipFileReader&& other) noexcept {
    if (this != &other) {
        shutdown();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void HipFileReader::shutdown() noexcept {
    if (impl_ == nullptr) return;
    if (impl_->registered_ptr != nullptr) {
        const hipFileError_t err = hipFileBufDeregister(impl_->registered_ptr);
        (void)err;
        impl_->registered_ptr = nullptr;
        impl_->registered_bytes = 0;
    }
    if (impl_->handle != nullptr) {
        hipFileHandleDeregister(impl_->handle);
        impl_->handle = nullptr;
    }
    if (impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
    if (impl_->driver_held) {
        driver_release();
        impl_->driver_held = false;
    }
    delete impl_;
    impl_ = nullptr;
}

bool HipFileReader::opened() const noexcept {
    return impl_ != nullptr && impl_->handle != nullptr;
}

bool HipFileReader::buffer_registered() const noexcept {
    return impl_ != nullptr && impl_->registered_ptr != nullptr;
}

Status HipFileReader::open(const std::string& path, bool odirect) {
    shutdown();

    Status st = driver_acquire();
    if (!st.ok()) return st;

    Impl* impl = new Impl();
    impl->driver_held = true;

    impl->fd = ::open(path.c_str(), O_RDONLY | (odirect ? O_DIRECT : 0));
    if (impl->fd < 0) {
        const std::string msg = "open failed: " + path + " (" + std::strerror(errno) + ")";
        delete impl;
        driver_release();
        return Status::invalid_argument(msg.c_str(), __FILE__, __LINE__);
    }

    hipFileDescr_t descr;
    std::memset(&descr, 0, sizeof(descr));
    descr.type = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = impl->fd;
    const hipFileError_t err = hipFileHandleRegister(&impl->handle, &descr);
    if (err.err != hipFileSuccess || impl->handle == nullptr) {
        const std::string msg =
            std::string("hipFileHandleRegister: ") + hipFileGetOpErrorString(err.err);
        ::close(impl->fd);
        delete impl;
        driver_release();
        return Status::unsupported(msg.c_str(), __FILE__, __LINE__);
    }

    impl_ = impl;
    return Status::make_ok();
}

Status HipFileReader::register_buffer(const void* device_ptr, std::size_t bytes) {
    if (impl_ == nullptr || impl_->handle == nullptr) {
        return Status::invalid_state("reader is not open", __FILE__, __LINE__);
    }
    if (device_ptr == nullptr || bytes == 0) {
        return Status::invalid_argument("empty device buffer", __FILE__, __LINE__);
    }
    if (impl_->registered_ptr != nullptr) {
        if (impl_->registered_ptr == device_ptr && impl_->registered_bytes >= bytes) {
            return Status::make_ok();
        }
        return Status::invalid_state("another buffer is still registered", __FILE__, __LINE__);
    }
    const hipFileError_t err = hipFileBufRegister(device_ptr, bytes, 0);
    if (err.err != hipFileSuccess) {
        return Status::unsupported(
            (std::string("hipFileBufRegister: ") + hipFileGetOpErrorString(err.err)).c_str(),
            __FILE__, __LINE__);
    }
    impl_->registered_ptr = device_ptr;
    impl_->registered_bytes = bytes;
    return Status::make_ok();
}

void HipFileReader::deregister_buffer() noexcept {
    if (impl_ == nullptr || impl_->registered_ptr == nullptr) return;
    const hipFileError_t err = hipFileBufDeregister(impl_->registered_ptr);
    (void)err;
    impl_->registered_ptr = nullptr;
    impl_->registered_bytes = 0;
}

Status HipFileReader::read(void* device_ptr, std::size_t bytes, std::uint64_t file_offset,
                           std::size_t buffer_offset, SsdReadResult& out) {
    out.path = SsdReadPath::none;
    out.bytes_read = 0;
    out.short_read = false;

    if (impl_ == nullptr || impl_->handle == nullptr) {
        return Status::invalid_state("reader is not open", __FILE__, __LINE__);
    }
    if (device_ptr == nullptr || bytes == 0) {
        return Status::invalid_argument("empty read request", __FILE__, __LINE__);
    }
    if (impl_->registered_ptr == nullptr) {
        return Status::invalid_state("device buffer is not registered", __FILE__, __LINE__);
    }
    const auto base = reinterpret_cast<std::uintptr_t>(impl_->registered_ptr);
    const auto target = reinterpret_cast<std::uintptr_t>(device_ptr);
    if (target < base || (target - base) > impl_->registered_bytes ||
        bytes > impl_->registered_bytes - (target - base)) {
        return Status::out_of_range("read target outside registered buffer",
                                    __FILE__, __LINE__);
    }

    const ssize_t n = hipFileRead(impl_->handle, device_ptr, bytes,
                                  static_cast<hoff_t>(file_offset),
                                  static_cast<hoff_t>(buffer_offset));
    if (n >= 0) {
        out.path = SsdReadPath::hipfile;
        out.bytes_read = static_cast<std::uint64_t>(n);
        out.short_read = static_cast<std::uint64_t>(n) != bytes;
        return Status::make_ok();
    }
    if (n == -1) {
        return Status::invalid_argument(
            ("hipFileRead posix error: " + std::string(std::strerror(errno))).c_str(),
            __FILE__, __LINE__);
    }
    return posix_fill(impl_->fd, device_ptr, bytes, file_offset, out);
}

#else  // !SHELF_HAS_HIPFILE

struct HipFileReader::Impl {
    int unused = 0;
};

bool hipfile_available() noexcept { return false; }

HipFileReader::~HipFileReader() noexcept { shutdown(); }

HipFileReader::HipFileReader(HipFileReader&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

HipFileReader& HipFileReader::operator=(HipFileReader&& other) noexcept {
    if (this != &other) {
        shutdown();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void HipFileReader::shutdown() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool HipFileReader::opened() const noexcept { return false; }

bool HipFileReader::buffer_registered() const noexcept { return false; }

Status HipFileReader::open(const std::string&, bool) {
    shutdown();
    return Status::unsupported("hipFile is not available in this build", __FILE__, __LINE__);
}

Status HipFileReader::register_buffer(const void*, std::size_t) {
    return Status::unsupported("hipFile is not available in this build", __FILE__, __LINE__);
}

void HipFileReader::deregister_buffer() noexcept {}

Status HipFileReader::read(void*, std::size_t, std::uint64_t, std::size_t, SsdReadResult&) {
    return Status::unsupported("hipFile is not available in this build", __FILE__, __LINE__);
}

#endif

}  // namespace ps::io
