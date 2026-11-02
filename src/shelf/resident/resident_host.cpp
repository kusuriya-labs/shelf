#include <shelf/resident/resident_host.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/resident/resident_client.h>
#include <shelf/resident/resident_protocol.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace ps {
namespace resident {
namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) {
    g_stop = 1;
}

void install_signals() {
    struct sigaction stop_action {};
    stop_action.sa_handler = on_signal;
    sigemptyset(&stop_action.sa_mask);
    stop_action.sa_flags = 0;
    sigaction(SIGINT, &stop_action, nullptr);
    sigaction(SIGTERM, &stop_action, nullptr);

    struct sigaction pipe_action {};
    pipe_action.sa_handler = SIG_IGN;
    sigemptyset(&pipe_action.sa_mask);
    pipe_action.sa_flags = 0;
    sigaction(SIGPIPE, &pipe_action, nullptr);
}

std::string& host_error() {
    thread_local std::string buffer;
    return buffer;
}

Status host_errno(const char* what) {
    host_error() = std::string(what) + ": " + std::strerror(errno);
    return Status::invalid_state(host_error().c_str(), __FILE__, __LINE__);
}

constexpr std::size_t kCompactAlignment = 2ull * 1024ull * 1024ull;

std::size_t round_up_compact(std::size_t bytes) {
    const std::size_t rem = bytes % kCompactAlignment;
    return rem == 0 ? bytes : bytes + (kCompactAlignment - rem);
}

Status send_error(int fd, Command cmd, const Status& status) {
    return write_frame(fd, cmd, StatusCode::Error, status.message().data(),
                       status.message().size());
}

struct BounceDeleter {
    void operator()(void* ptr) const noexcept { ::operator delete(ptr); }
};

}

Result<ResidentModelHost> ResidentModelHost::create(
    const ResidentHostOptions& options) {
    if (options.socket_path.empty()) {
        return Status::invalid_argument("resident host socket path is empty", __FILE__, __LINE__);
    }
    if (options.device < 0) {
        return Status::invalid_argument("resident host device is invalid", __FILE__, __LINE__);
    }
    if (options.staging_bytes == 0) {
        return Status::invalid_argument("resident host staging bytes is zero", __FILE__, __LINE__);
    }
    install_signals();
    ResidentModelHost host;
    host.options_ = options;
    if (!options.allow_any_dir) {
        for (const auto& warmup_entry : options.warmup) {
            host.options_.model_dirs.push_back(warmup_entry.first);
        }
    }
    Status st = host.bind_socket();
    if (!st.ok()) {
        return st;
    }
    return host;
}

Status ResidentModelHost::bind_socket() {
    ::unlink(options_.socket_path.c_str());
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return host_errno("resident host socket()");
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (options_.socket_path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        return Status::invalid_argument("resident host socket path too long", __FILE__, __LINE__);
    }
    std::memcpy(addr.sun_path, options_.socket_path.c_str(),
                options_.socket_path.size() + 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        const Status st = host_errno("resident host bind()");
        ::close(fd);
        return st;
    }
    ::chmod(options_.socket_path.c_str(), 0600);
    if (::listen(fd, 8) != 0) {
        const Status st = host_errno("resident host listen()");
        ::close(fd);
        return st;
    }
    listen_fd_ = fd;
    running_ = true;
    return Status::make_ok();
}

ResidentModelHost::ResidentModelHost(ResidentModelHost&& other) noexcept
    : options_(std::move(other.options_)),
      listen_fd_(other.listen_fd_),
      running_(other.running_),
      entries_(std::move(other.entries_)),
      disk_load_count_(other.disk_load_count_),
      attach_count_(other.attach_count_),
      last_health_(std::move(other.last_health_)) {
    other.listen_fd_ = -1;
    other.running_ = false;
}

ResidentModelHost& ResidentModelHost::operator=(ResidentModelHost&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        options_ = std::move(other.options_);
        listen_fd_ = other.listen_fd_;
        running_ = other.running_;
        entries_ = std::move(other.entries_);
        disk_load_count_ = other.disk_load_count_;
        attach_count_ = other.attach_count_;
        last_health_ = std::move(other.last_health_);
        other.listen_fd_ = -1;
        other.running_ = false;
    }
    return *this;
}

ResidentModelHost::~ResidentModelHost() {
    (void)shutdown();
}

bool ResidentModelHost::allowed_dir(const std::string& model_dir) const {
    if (options_.allow_any_dir || options_.model_dirs.empty()) {
        return true;
    }
    const std::string canonical = canonical_model_dir(model_dir);
    for (const auto& dir : options_.model_dirs) {
        if (canonical_model_dir(dir) == canonical) {
            return true;
        }
    }
    return false;
}

void ResidentModelHost::drop_entry(Entry& entry) {
    if (entry.block == nullptr) {
        return;
    }
    auto scope = gpu::ScopedDevice::create(options_.device);
    if (scope.ok()) {
        gpu::discard_cleanup_result(hipFree(entry.block));
    }
    entry.block = nullptr;
    entry.block_bytes = 0;
    entry.archive.clear();
    entry.archive.shrink_to_fit();
}

Status ResidentModelHost::load_entry_once(const ResidentModelKey& key, Entry& out) {
    if (key.device != options_.device) {
        return Status::invalid_argument("resident key device mismatch", __FILE__, __LINE__);
    }
    if (!allowed_dir(key.model_dir)) {
        return Status::unsupported("resident key model dir not allowed", __FILE__, __LINE__);
    }
    auto scope = gpu::ScopedDevice::create(options_.device);
    if (!scope.ok()) {
        return scope.status();
    }

    auto staging_result = gpu::GpuArena::create(options_.device, options_.staging_bytes);
    if (!staging_result.ok()) {
        return staging_result.status();
    }
    gpu::GpuArena staging = staging_result.release();

    hipStream_t stream = nullptr;
    hipError_t stream_error = hipStreamCreate(&stream);
    if (stream_error != hipSuccess) {
        return Status::hip_error("hipStreamCreate", hipGetErrorString(stream_error),
                                 __FILE__, __LINE__);
    }

    std::vector<std::byte> archive;
    const auto t_load_start = std::chrono::steady_clock::now();

    if (key.kind == ModelKind::Qwen35Target) {
        qwen35::Qwen35LoadOptions options;
        options.verify_quantized_payload_crc = (key.load_flags & kLoadFlagVerifyCrc) != 0;
        options.load_mtp_layers = (key.load_flags & kLoadFlagMtpLayers) != 0;
        options.weights.preshuffle = (key.load_flags & kLoadFlagPreshuffle) != 0;
        Result<qwen35::Qwen35Model> model_result =
            key.tp_size > 1
                ? qwen35::Qwen35Model::load_tensor_parallel_from_safetensors(
                      key.model_dir, key.tp_size, key.tp_rank, staging, stream, options)
                : qwen35::Qwen35Model::load_from_safetensors(key.model_dir, staging,
                                                             stream, options);
        if (!model_result.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return model_result.status();
        }
        auto prefix = staging.used_prefix_view();
        if (!prefix.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return prefix.status();
        }
        qwen35::Qwen35Model model = model_result.release();
        Status write_status = write_qwen35_archive(
            model.text_config(), model.weights(), prefix.value().data(), archive);
        if (!write_status.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return write_status;
        }
    } else if (key.kind == ModelKind::DFlash2Draft) {
        auto config_result = qwen35::dflash2::read_dflash2_config(key.model_dir);
        if (!config_result.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return config_result.status();
        }
        qwen35::dflash2::DFlash2Config config = config_result.release();
        ps::weights::WeightLoadOptions options;
        options.preshuffle = (key.load_flags & kLoadFlagPreshuffle) != 0;
        auto weights_result = qwen35::dflash2::load_dflash2_weights(
            key.model_dir, config, staging, stream, options);
        if (!weights_result.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return weights_result.status();
        }
        auto prefix = staging.used_prefix_view();
        if (!prefix.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return prefix.status();
        }
        qwen35::dflash2::DFlash2Weights weights = weights_result.release();
        Status write_status = write_dflash2_archive(
            config, weights, prefix.value().data(), archive);
        if (!write_status.ok()) {
            gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return write_status;
        }
    } else {
        gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return Status::unsupported("resident model kind is not supported", __FILE__,
                                    __LINE__);
    }

    const auto t_load_end = std::chrono::steady_clock::now();
    const double load_ms =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    auto prefix = staging.used_prefix_view();
    if (!prefix.ok()) {
        gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return prefix.status();
    }
    const std::size_t used = prefix.value().bytes();
    const void* base = prefix.value().data();
    const std::size_t block_bytes = round_up_compact(used);

    std::unique_ptr<std::byte, BounceDeleter> bounce(
        static_cast<std::byte*>(::operator new(used, std::nothrow)));
    if (!bounce) {
        gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return Status::insufficient_memory("resident bounce buffer", __FILE__, __LINE__);
    }
    hipError_t copy_error =
        hipMemcpy(bounce.get(), base, used, hipMemcpyDeviceToHost);
    if (copy_error != hipSuccess) {
        gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return Status::hip_error("hipMemcpy(resident bounce)", hipGetErrorString(copy_error),
                                 __FILE__, __LINE__);
    }

    Status staging_status = staging.shutdown();
    gpu::discard_cleanup_result(hipStreamDestroy(stream));
    if (!staging_status.ok()) {
        return staging_status;
    }

    void* block = nullptr;
    hipError_t alloc_error = hipMalloc(&block, block_bytes);
    if (alloc_error != hipSuccess || block == nullptr) {
        return Status::insufficient_memory(
            std::string("resident compact hipMalloc: ") + hipGetErrorString(alloc_error),
            __FILE__, __LINE__);
    }
    copy_error = hipMemcpy(block, bounce.get(), used, hipMemcpyHostToDevice);
    bounce.reset();
    if (copy_error != hipSuccess) {
        gpu::discard_cleanup_result(hipFree(block));
        return Status::hip_error("hipMemcpy(resident compact)", hipGetErrorString(copy_error),
                                 __FILE__, __LINE__);
    }

    std::vector<std::byte> rebuilt_archive;
    Status verify_status = Status::make_ok();
    if (key.kind == ModelKind::Qwen35Target) {
        auto rebuilt = build_qwen35_model(archive.data(), archive.size(), block,
                                          block_bytes, options_.device);
        if (!rebuilt.ok()) {
            verify_status = rebuilt.status();
        } else {
            qwen35::Qwen35Model model = rebuilt.release();
            verify_status = write_qwen35_archive(model.text_config(), model.weights(),
                                                 block, rebuilt_archive);
        }
    } else if (key.kind == ModelKind::DFlash2Draft) {
        qwen35::dflash2::DFlash2Config config;
        qwen35::dflash2::DFlash2Weights weights;
        verify_status = read_dflash2_archive(archive.data(), archive.size(), block,
                                             block_bytes, options_.device, config,
                                             weights);
        if (verify_status.ok()) {
            verify_status = write_dflash2_archive(config, weights, block, rebuilt_archive);
        }
    } else {
        verify_status = Status::unsupported("resident model kind is not supported",
                                            __FILE__, __LINE__);
    }
    if (!verify_status.ok()) {
        gpu::discard_cleanup_result(hipFree(block));
        return verify_status;
    }
    if (rebuilt_archive != archive) {
        gpu::discard_cleanup_result(hipFree(block));
        return Status::invalid_state("resident compact rebuild mismatch", __FILE__, __LINE__);
    }

    hipIpcMemHandle_t handle{};
    hipError_t ipc_error = hipIpcGetMemHandle(&handle, block);
    if (ipc_error != hipSuccess) {
        gpu::discard_cleanup_result(hipFree(block));
        return Status::hip_error("hipIpcGetMemHandle", hipGetErrorString(ipc_error),
                                 __FILE__, __LINE__);
    }

    out.key = key;
    out.block = block;
    out.block_bytes = block_bytes;
    out.resident_bytes = used;
    out.ipc = handle;
    out.archive = std::move(archive);
    out.disk_load_count = 1;

    std::fprintf(stderr,
                 "resident host: loaded %s resident_bytes=%zu block_bytes=%zu load_ms=%.0f\n",
                 key.to_string().c_str(), used, block_bytes, load_ms);
    std::fflush(stderr);
    return Status::make_ok();
}

Status ResidentModelHost::load_entry(const ResidentModelKey& key, Entry& out) {
    Status first = load_entry_once(key, out);
    if (first.ok() || entries_.empty()) {
        return first;
    }
    std::fprintf(stderr,
                 "resident host: load failed (%s); releasing %zu resident entries "
                 "and retrying\n",
                 first.message().c_str(), entries_.size());
    std::fflush(stderr);
    for (auto& item : entries_) {
        drop_entry(item.second);
    }
    entries_.clear();
    return load_entry_once(key, out);
}

Status ResidentModelHost::send_entry(int fd, Entry& entry) {
    std::vector<std::byte> payload(sizeof(ManifestHeader) + kIpcHandleBytes +
                                   entry.archive.size());
    ManifestHeader header;
    header.kind = static_cast<std::uint32_t>(entry.key.kind);
    header.device = options_.device;
    header.resident_bytes = entry.resident_bytes;
    header.block_bytes = entry.block_bytes;
    header.disk_load_count = entry.disk_load_count;
    header.attach_count = entry.attach_count;
    header.archive_bytes = entry.archive.size();
    std::memcpy(payload.data(), &header, sizeof(header));
    std::memcpy(payload.data() + sizeof(header), &entry.ipc, sizeof(entry.ipc));
    std::memcpy(payload.data() + sizeof(header) + kIpcHandleBytes, entry.archive.data(),
                entry.archive.size());
    ++entry.attach_count;
    ++attach_count_;
    return write_frame(fd, Command::Acquire, StatusCode::Ok, payload.data(),
                       payload.size());
}

void ResidentModelHost::handle_connection(int fd) {
    for (;;) {
        std::vector<std::byte> payload;
        auto frame = read_frame(fd, payload);
        if (!frame.ok()) {
            return;
        }
        const auto cmd = static_cast<Command>(frame.value().cmd);
        switch (cmd) {
            case Command::Hello: {
                HelloInfo info;
                info.device = options_.device;
                info.resident_bytes_total = 0;
                for (const auto& item : entries_) {
                    const Entry& entry = item.second;
                    info.resident_bytes_total += entry.resident_bytes;
                }
                (void)write_frame(fd, Command::Hello, StatusCode::Ok, &info, sizeof(info));
                break;
            }
            case Command::Acquire: {
                auto key_result = parse_key(payload.data(), payload.size());
                if (!key_result.ok()) {
                    (void)send_error(fd, Command::Acquire, key_result.status());
                    break;
                }
                const ResidentModelKey& key = key_result.value();
                const std::string map_key = key.to_string();
                auto it = entries_.find(map_key);
                if (it == entries_.end()) {
                    Entry entry;
                    Status st = load_entry(key, entry);
                    if (!st.ok()) {
                        (void)send_error(fd, Command::Acquire, st);
                        break;
                    }
                    ++disk_load_count_;
                    it = entries_.emplace(map_key, std::move(entry)).first;
                }
                Status send_status = send_entry(fd, it->second);
                if (!send_status.ok()) {
                    return;
                }
                break;
            }
            case Command::Release: {
                std::string map_key;
                auto key_result = parse_key(payload.data(), payload.size());
                if (key_result.ok()) {
                    map_key = key_result.value().to_string();
                } else if (!payload.empty()) {
                    map_key.assign(reinterpret_cast<const char*>(payload.data()),
                                   payload.size());
                }
                auto it = entries_.find(map_key);
                if (it != entries_.end()) {
                    drop_entry(it->second);
                    entries_.erase(it);
                    (void)write_frame(fd, Command::Release, StatusCode::Ok, nullptr, 0);
                } else {
                    (void)write_frame(fd, Command::Release, StatusCode::NotFound,
                                      map_key.data(), map_key.size());
                }
                break;
            }
            case Command::Health: {
                Status st = health(options_.health_timeout_ms);
                HealthInfo info;
                info.healthy = st.ok() ? 1u : 0u;
                info.device = options_.device;
                info.elapsed_us = 0;
                (void)write_frame(fd, Command::Health,
                                  st.ok() ? StatusCode::Ok : StatusCode::Unhealthy,
                                  &info, sizeof(info));
                break;
            }
            case Command::Stats: {
                const std::string text = stats_text();
                (void)write_frame(fd, Command::Stats, StatusCode::Ok, text.data(),
                                  text.size());
                break;
            }
            case Command::Shutdown: {
                (void)write_frame(fd, Command::Shutdown, StatusCode::Ok, nullptr, 0);
                running_ = false;
                return;
            }
            default: {
                const std::string message = "unknown resident command";
                (void)write_frame(fd, cmd, StatusCode::Error, message.data(),
                                  message.size());
                break;
            }
        }
    }
}

Status ResidentModelHost::warmup() {
    for (const auto& [dir, kind] : options_.warmup) {
        const std::uint32_t flags =
            kind == ModelKind::Qwen35Target
                ? resident_load_flags(qwen35::Qwen35LoadOptions{})
                : resident_load_flags(weights::WeightLoadOptions{});
        ResidentModelKey key = ResidentModelKey::make(dir, kind, flags,
                                                      options_.warmup_tp_size,
                                                      options_.warmup_tp_rank,
                                                      options_.device);
        const std::string map_key = key.to_string();
        if (entries_.find(map_key) != entries_.end()) {
            continue;
        }
        Entry entry;
        Status st = load_entry(key, entry);
        if (!st.ok()) {
            return st;
        }
        ++disk_load_count_;
        entries_.emplace(map_key, std::move(entry));
    }
    return Status::make_ok();
}

Status ResidentModelHost::serve() {
    while (running_ && g_stop == 0) {
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            return host_errno("resident host accept()");
        }
        handle_connection(fd);
        ::close(fd);
    }
    return Status::make_ok();
}

Status ResidentModelHost::health(std::uint64_t timeout_ms) {
    Status st = resident_health_check(options_.device, timeout_ms);
    if (st.ok()) {
        for (auto& item : entries_) {
        Entry& entry = item.second;
            if (entry.block == nullptr) {
                st = Status::invalid_state("resident block missing", __FILE__, __LINE__);
                break;
            }
        }
    }
    last_health_ = st.ok() ? "ok" : st.message();
    return st;
}

Status ResidentModelHost::shutdown() {
    Status first_error = Status::make_ok();
    running_ = false;
    for (auto& item : entries_) {
        Entry& entry = item.second;
        drop_entry(entry);
    }
    entries_.clear();
    if (listen_fd_ >= 0) {
        if (::close(listen_fd_) != 0 && first_error.ok()) {
            first_error = host_errno("resident host close()");
        }
        listen_fd_ = -1;
    }
    if (!options_.socket_path.empty()) {
        ::unlink(options_.socket_path.c_str());
    }
    return first_error;
}

ResidentStats ResidentModelHost::stats() const {
    ResidentStats out;
    out.disk_load_count = disk_load_count_;
    out.attach_count = attach_count_;
    out.entries = entries_.size();
    for (const auto& item : entries_) {
                    const Entry& entry = item.second;
        out.resident_bytes += entry.resident_bytes;
    }
    out.healthy = last_health_ == "ok";
    return out;
}

std::string ResidentModelHost::stats_text() const {
    std::string text;
    text += "proto_version=" + std::to_string(kWireVersion) + "\n";
    text += "device=" + std::to_string(options_.device) + "\n";
    text += "last_health=" + last_health_ + "\n";
    text += "disk_load_count=" + std::to_string(disk_load_count_) + "\n";
    text += "attach_count=" + std::to_string(attach_count_) + "\n";
    text += "entries=" + std::to_string(entries_.size()) + "\n";
    text += "resident_bytes=" + std::to_string(stats().resident_bytes) + "\n";
    text += "socket=" + options_.socket_path + "\n";
    for (const auto& item : entries_) {
                    const Entry& entry = item.second;
        text += "entry key=" + entry.key.to_string();
        text += " resident_bytes=" + std::to_string(entry.resident_bytes);
        text += " block_bytes=" + std::to_string(entry.block_bytes);
        text += " disk_load_count=" + std::to_string(entry.disk_load_count);
        text += " attach_count=" + std::to_string(entry.attach_count);
        text += "\n";
    }
    return text;
}

}
}
