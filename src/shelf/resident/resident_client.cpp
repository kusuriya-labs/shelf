#include <shelf/resident/resident_client.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/resident/resident_protocol.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace ps {
namespace resident {
namespace {

using Clock = std::chrono::steady_clock;

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

Status ipc_status(const char* what, hipError_t error) {
    if (error == hipSuccess) {
        return Status::make_ok();
    }
    return Status::hip_error(what, hipGetErrorString(error), __FILE__, __LINE__);
}

}

bool resident_model_disabled() {
    return env_flag(kDisableEnv) || !resident_socket_env_present();
}

bool resident_socket_env_present() {
    const char* v = std::getenv(kSocketEnv);
    return v != nullptr && v[0] != '\0';
}

Result<std::string> resident_socket_path(const std::string& spec,
                                          std::uint32_t index) {
    if (spec.find('=') == std::string::npos) {
        return spec;
    }
    std::uint32_t selected = 0;
    std::size_t selected_end = std::string::npos;
    bool found = false;
    std::size_t pos = 0;
    while (pos < spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string part =
            spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const std::size_t eq = part.find('=');
        if (eq != std::string::npos) {
            const std::uint32_t rank =
                static_cast<std::uint32_t>(std::strtoul(part.c_str(), nullptr, 10));
            if (rank == index) {
                selected = pos + eq + 1;
                selected_end = comma == std::string::npos ? spec.size() : comma;
                found = true;
                break;
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (!found) {
        return Status::out_of_range("resident socket rank not mapped", __FILE__, __LINE__);
    }
    return spec.substr(selected, selected_end - selected);
}

Result<std::string> resident_socket_path(std::uint32_t tp_rank) {
    const char* v = std::getenv(kSocketEnv);
    if (v == nullptr || v[0] == '\0') {
        return Status::invalid_state("resident model socket env missing", __FILE__, __LINE__);
    }
    return resident_socket_path(std::string(v), tp_rank);
}

Result<std::string> resident_socket_path(const ResidentModelKey& key) {
    const std::uint32_t index =
        key.tp_size > 1 ? key.tp_rank : static_cast<std::uint32_t>(key.device);
    return resident_socket_path(index);
}

Result<ResidentModelAttachment> acquire_resident_model(
    const ResidentModelKey& key, int device) {
    const char* v = std::getenv(kSocketEnv);
    if (v == nullptr || v[0] == '\0') {
        return Status::invalid_state("resident model socket env missing", __FILE__,
                                     __LINE__);
    }
    return acquire_resident_model(key, device, std::string(v));
}

ResidentModelAttachment::ResidentModelAttachment(
    ResidentModelAttachment&& other) noexcept
    : block_(other.block_),
      block_bytes_(other.block_bytes_),
      device_(other.device_),
      host_resident_bytes_(other.host_resident_bytes_),
      key_(std::move(other.key_)),
      header_(other.header_),
      archive_(std::move(other.archive_)) {
    other.block_ = nullptr;
    other.block_bytes_ = 0;
    other.device_ = 0;
}

ResidentModelAttachment& ResidentModelAttachment::operator=(
    ResidentModelAttachment&& other) noexcept {
    if (this != &other) {
        close_mapping();
        block_ = other.block_;
        block_bytes_ = other.block_bytes_;
        device_ = other.device_;
        host_resident_bytes_ = other.host_resident_bytes_;
        key_ = std::move(other.key_);
        header_ = other.header_;
        archive_ = std::move(other.archive_);
        other.block_ = nullptr;
        other.block_bytes_ = 0;
        other.device_ = 0;
    }
    return *this;
}

void ResidentModelAttachment::close_mapping() noexcept {
    if (block_ == nullptr) {
        return;
    }
    auto scope = gpu::ScopedDevice::create(device_);
    if (scope.ok()) {
        const hipError_t error = hipIpcCloseMemHandle(block_);
        if (error != hipSuccess) {
            std::fprintf(stderr, "resident model: hipIpcCloseMemHandle failed: %s\n",
                         hipGetErrorString(error));
        }
    }
    block_ = nullptr;
    block_bytes_ = 0;
}

ResidentModelAttachment::~ResidentModelAttachment() {
    close_mapping();
}

Result<qwen35::Qwen35Model> ResidentModelAttachment::take_qwen35() const {
    if (block_ == nullptr) {
        return Status::invalid_state("resident model not attached", __FILE__, __LINE__);
    }
    if (header_.kind != static_cast<std::uint32_t>(ModelKind::Qwen35Target)) {
        return Status::invalid_argument("resident model kind mismatch", __FILE__, __LINE__);
    }
    if (archive_.size() != header_.archive_bytes) {
        return Status::invalid_state("resident model archive size", __FILE__, __LINE__);
    }
    return build_qwen35_model(archive_.data(), archive_.size(), block_, block_bytes_,
                              device_);
}

Result<qwen35::dflash2::DFlash2Weights> ResidentModelAttachment::take_dflash2(
    qwen35::dflash2::DFlash2Config* config_out) const {
    if (block_ == nullptr) {
        return Status::invalid_state("resident model not attached", __FILE__, __LINE__);
    }
    if (header_.kind != static_cast<std::uint32_t>(ModelKind::DFlash2Draft)) {
        return Status::invalid_argument("resident model kind mismatch", __FILE__, __LINE__);
    }
    if (archive_.size() != header_.archive_bytes) {
        return Status::invalid_state("resident model archive size", __FILE__, __LINE__);
    }
    qwen35::dflash2::DFlash2Config config;
    qwen35::dflash2::DFlash2Weights weights;
    Status st = read_dflash2_archive(archive_.data(), archive_.size(), block_,
                                     block_bytes_, device_, config, weights);
    if (!st.ok()) {
        return st;
    }
    if (config_out != nullptr) {
        *config_out = std::move(config);
    }
    return weights;
}

Result<qwen35::eagle4::Eagle4Weights> ResidentModelAttachment::take_eagle4(
    qwen35::eagle4::Eagle4Config* config_out) const {
    if (block_ == nullptr) {
        return Status::invalid_state("resident model not attached", __FILE__, __LINE__);
    }
    if (header_.kind != static_cast<std::uint32_t>(ModelKind::Eagle4Draft)) {
        return Status::invalid_argument("resident model kind mismatch", __FILE__, __LINE__);
    }
    if (archive_.size() != header_.archive_bytes) {
        return Status::invalid_state("resident model archive size", __FILE__, __LINE__);
    }
    qwen35::eagle4::Eagle4Config config;
    qwen35::eagle4::Eagle4Weights weights;
    Status st = read_eagle4_archive(archive_.data(), archive_.size(), block_,
                                    block_bytes_, device_, config, weights);
    if (!st.ok()) {
        return st;
    }
    if (config_out != nullptr) {
        *config_out = std::move(config);
    }
    return weights;
}

Result<ResidentModelAttachment> acquire_resident_model(
    const ResidentModelKey& key, int device, const std::string& socket_spec) {
    if (env_flag(kDisableEnv)) {
        return Status::invalid_state("resident model disabled", __FILE__, __LINE__);
    }
    const std::uint32_t index =
        key.tp_size > 1 ? key.tp_rank : static_cast<std::uint32_t>(device);
    auto path_result = resident_socket_path(socket_spec, index);
    if (!path_result.ok()) {
        return path_result.status();
    }
    const std::string& path = path_result.value();

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return Status::invalid_state("resident model socket()", __FILE__, __LINE__);
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        return Status::invalid_argument("resident model socket path too long", __FILE__, __LINE__);
    }
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return Status::invalid_state("resident model connect failed", __FILE__, __LINE__);
    }

    auto close_with = [&](const Status& error) {
        ::close(fd);
        return Result<ResidentModelAttachment>(error);
    };

    Status st = write_frame(fd, Command::Hello, StatusCode::Ok, nullptr, 0);
    if (!st.ok()) return close_with(st);
    std::vector<std::byte> payload;
    auto hello = read_frame(fd, payload);
    if (!hello.ok()) return close_with(hello.status());
    if (hello.value().status != static_cast<std::uint32_t>(StatusCode::Ok) ||
        payload.size() < sizeof(HelloInfo)) {
        return close_with(Status::invalid_state("resident host hello rejected", __FILE__, __LINE__));
    }
    HelloInfo hello_info;
    std::memcpy(&hello_info, payload.data(), sizeof(hello_info));

    ResidentModelKey effective = key;
    effective.device = hello_info.device;
    effective.fingerprint = fingerprint_key(effective);

    const auto key_payload = serialize_key(effective);
    st = write_frame(fd, Command::Acquire, StatusCode::Ok, key_payload.data(),
                     key_payload.size());
    if (!st.ok()) return close_with(st);
    auto response = read_frame(fd, payload);
    if (!response.ok()) {
        ::close(fd);
        return response.status();
    }
    if (response.value().status != static_cast<std::uint32_t>(StatusCode::Ok)) {
        const std::string detail(payload.empty() ? std::string("acquire failed")
                                                 : std::string(reinterpret_cast<const char*>(payload.data()),
                                                               payload.size()));
        ::close(fd);
        return Status::invalid_state(detail.c_str(), __FILE__, __LINE__);
    }

    HelloInfo refreshed = hello_info;
    if (write_frame(fd, Command::Hello, StatusCode::Ok, nullptr, 0).ok()) {
        std::vector<std::byte> hello_payload;
        auto second = read_frame(fd, hello_payload);
        if (second.ok() &&
            second.value().status == static_cast<std::uint32_t>(StatusCode::Ok) &&
            hello_payload.size() >= sizeof(HelloInfo)) {
            std::memcpy(&refreshed, hello_payload.data(), sizeof(HelloInfo));
        }
    }
    ::close(fd);

    const std::size_t fixed = sizeof(ManifestHeader) + kIpcHandleBytes;
    if (payload.size() < fixed) {
        return Status::invalid_state("resident manifest too small", __FILE__, __LINE__);
    }
    ResidentModelAttachment attachment;
    attachment.key_ = effective;
    attachment.host_resident_bytes_ = refreshed.resident_bytes_total;
    std::memcpy(&attachment.header_, payload.data(), sizeof(ManifestHeader));
    if (attachment.header_.magic != kManifestMagic ||
        attachment.header_.version != kManifestVersion ||
        attachment.header_.ipc_handle_bytes != kIpcHandleBytes) {
        return Status::invalid_state("resident manifest header", __FILE__, __LINE__);
    }
    const std::size_t expected =
        fixed + static_cast<std::size_t>(attachment.header_.archive_bytes);
    if (payload.size() != expected) {
        return Status::invalid_state("resident manifest payload size", __FILE__, __LINE__);
    }
    attachment.device_ = device;
    attachment.block_bytes_ = static_cast<std::size_t>(attachment.header_.block_bytes);
    attachment.archive_.assign(payload.begin() + static_cast<std::ptrdiff_t>(fixed),
                               payload.end());

    hipIpcMemHandle_t handle;
    std::memcpy(&handle, payload.data() + sizeof(ManifestHeader), sizeof(handle));
    auto scope = gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        return scope.status();
    }
    void* block = nullptr;
    st = ipc_status("hipIpcOpenMemHandle",
                    hipIpcOpenMemHandle(&block, handle, hipIpcMemLazyEnablePeerAccess));
    if (!st.ok()) {
        return st;
    }
    attachment.block_ = block;
    return attachment;
}

namespace {

Result<int> open_control_socket(const std::string& path) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return Status::invalid_state("resident control socket()", __FILE__, __LINE__);
    }
    if (path.size() >= sizeof(sockaddr_un::sun_path)) {
        ::close(fd);
        return Status::invalid_argument("resident control path too long", __FILE__, __LINE__);
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return Status::invalid_state("resident control connect failed", __FILE__, __LINE__);
    }
    return fd;
}

}

Result<std::string> resident_host_stats(const std::string& socket_path) {
    auto fd_result = open_control_socket(socket_path);
    if (!fd_result.ok()) {
        return fd_result.status();
    }
    const int fd = fd_result.value();
    Status st = write_frame(fd, Command::Stats, StatusCode::Ok, nullptr, 0);
    if (st.ok()) {
        std::vector<std::byte> payload;
        auto frame = read_frame(fd, payload);
        ::close(fd);
        if (!frame.ok()) return frame.status();
        if (frame.value().status != static_cast<std::uint32_t>(StatusCode::Ok)) {
            return Status::invalid_state("resident stats rejected", __FILE__, __LINE__);
        }
        return std::string(reinterpret_cast<const char*>(payload.data()), payload.size());
    }
    ::close(fd);
    return st;
}

Status resident_host_release(const std::string& socket_path, const std::string& key_text) {
    auto fd_result = open_control_socket(socket_path);
    if (!fd_result.ok()) {
        return fd_result.status();
    }
    const int fd = fd_result.value();
    Status st = write_frame(fd, Command::Release, StatusCode::Ok, key_text.data(),
                            key_text.size());
    if (!st.ok()) {
        ::close(fd);
        return st;
    }
    std::vector<std::byte> payload;
    auto frame = read_frame(fd, payload);
    ::close(fd);
    if (!frame.ok()) return frame.status();
    if (frame.value().status == static_cast<std::uint32_t>(StatusCode::Ok)) {
        return Status::make_ok();
    }
    if (frame.value().status == static_cast<std::uint32_t>(StatusCode::NotFound)) {
        return Status::out_of_range("resident entry not found", __FILE__, __LINE__);
    }
    return Status::invalid_state("resident release failed", __FILE__, __LINE__);
}

Status resident_health_check(int device, std::uint64_t timeout_ms) {    const auto start = Clock::now();
    auto scope = gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        return scope.status();
    }
    void* probe = nullptr;
    const std::size_t probe_bytes = 1ull * 1024ull * 1024ull;
    hipError_t error = hipMalloc(&probe, probe_bytes);
    if (error != hipSuccess || probe == nullptr) {
        return Status::hip_error("hipMalloc(health)", hipGetErrorString(error), __FILE__, __LINE__);
    }
    error = hipMemset(probe, 0x5a, 64);
    if (error != hipSuccess) {
        gpu::discard_cleanup_result(hipFree(probe));
        return Status::hip_error("hipMemset(health)", hipGetErrorString(error), __FILE__, __LINE__);
    }
    hipStream_t stream = nullptr;
    error = hipStreamCreate(&stream);
    if (error != hipSuccess) {
        gpu::discard_cleanup_result(hipFree(probe));
        return Status::hip_error("hipStreamCreate(health)", hipGetErrorString(error), __FILE__, __LINE__);
    }
    error = hipMemsetAsync(probe, 0xa5, 4096, stream);
    if (error == hipSuccess) {
        error = hipStreamSynchronize(stream);
    }
    const hipError_t stream_error = hipStreamDestroy(stream);
    gpu::discard_cleanup_result(hipFree(probe));
    if (error != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize(health)", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    }
    if (stream_error != hipSuccess) {
        return Status::hip_error("hipStreamDestroy(health)", hipGetErrorString(stream_error),
                                 __FILE__, __LINE__);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - start)
                             .count();
    if (timeout_ms > 0 && static_cast<std::uint64_t>(elapsed) > timeout_ms) {
        return Status::invalid_state("resident health check timed out", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}
}
