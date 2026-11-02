#include <shelf/resident/resident_client.h>
#include <shelf/resident/resident_host.h>
#include <shelf/resident/resident_protocol.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

namespace r = ps::resident;

using ps::Status;
using r::Command;
using r::ModelKind;
using r::ResidentHostOptions;
using r::ResidentModelHost;
using r::StatusCode;

void usage() {
    std::fprintf(stderr,
        "usage: shelf-model-host --socket PATH [--device N] "
        "[--staging-gib G] [--warmup-qwen35 DIR]... [--warmup-dflash2 DIR]... "
        "[--warmup-eagle4 DIR]... "
        "[--warmup-tp-size N] [--warmup-tp-rank N] "
        "[--model-dir DIR]... [--allow-any-dir] [--health-timeout-ms N] [--no-vmm]\n"
        "       shelf-model-host --socket PATH --ctl "
        "{ping|health|stats|shutdown|release KEY}\n");
}

bool take_arg(int argc, char** argv, int* index, const char* flag, std::string* out) {
    if (std::strcmp(argv[*index], flag) != 0) return false;
    if (*index + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
    }
    *out = argv[*index + 1];
    *index += 2;
    return true;
}

int control(const std::string& socket_path, const std::string& command,
            const std::string& argument) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::fprintf(stderr, "socket() failed\n");
        return 1;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "connect(%s) failed\n", socket_path.c_str());
        ::close(fd);
        return 1;
    }

    Command cmd;
    std::string payload;
    if (command == "ping") {
        cmd = Command::Hello;
    } else if (command == "health") {
        cmd = Command::Health;
    } else if (command == "stats") {
        cmd = Command::Stats;
    } else if (command == "shutdown") {
        cmd = Command::Shutdown;
    } else if (command == "release") {
        cmd = Command::Release;
        payload = argument;
    } else if (command == "acquire") {
        cmd = Command::Acquire;
        payload = argument;
    } else {
        std::fprintf(stderr, "unknown control command: %s\n", command.c_str());
        ::close(fd);
        return 2;
    }

    Status st = r::write_frame(fd, cmd, StatusCode::Ok, payload.data(), payload.size());
    if (!st.ok()) {
        std::fprintf(stderr, "write failed: %s\n", st.message().c_str());
        ::close(fd);
        return 1;
    }
    std::vector<std::byte> response_payload;
    auto frame = r::read_frame(fd, response_payload);
    ::close(fd);
    if (!frame.ok()) {
        std::fprintf(stderr, "read failed: %s\n", frame.status().message().c_str());
        return 1;
    }
    const std::uint32_t status = frame.value().status;
    if (cmd == Command::Health && response_payload.size() == sizeof(r::HealthInfo)) {
        r::HealthInfo info;
        std::memcpy(&info, response_payload.data(), sizeof(info));
        std::printf("healthy=%u device=%d elapsed_us=%llu\n", info.healthy, info.device,
                    static_cast<unsigned long long>(info.elapsed_us));
    } else if (cmd == Command::Hello &&
               response_payload.size() == sizeof(r::HelloInfo)) {
        r::HelloInfo info;
        std::memcpy(&info, response_payload.data(), sizeof(info));
        std::printf("proto_version=%u device=%d resident_bytes=%llu\n",
                    info.proto_version, info.device,
                    static_cast<unsigned long long>(info.resident_bytes_total));
    } else if (!response_payload.empty()) {
        std::fwrite(response_payload.data(), 1, response_payload.size(), stdout);
        if (response_payload.back() != std::byte{'\n'}) {
            std::fputc('\n', stdout);
        }
    }
    std::fflush(stdout);
    if (status != static_cast<std::uint32_t>(StatusCode::Ok)) {
        return status == static_cast<std::uint32_t>(StatusCode::NotFound) ? 3 : 1;
    }
    return 0;
}

}

int main(int argc, char** argv) {
    ResidentHostOptions options;
    std::string control_command;
    std::string control_argument;
    bool vmm = true;

    for (int i = 1; i < argc;) {
        std::string value;
        if (take_arg(argc, argv, &i, "--socket", &value)) {
            options.socket_path = value;
        } else if (take_arg(argc, argv, &i, "--device", &value)) {
            options.device = std::atoi(value.c_str());
        } else if (take_arg(argc, argv, &i, "--staging-gib", &value)) {
            options.staging_bytes =
                static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10)) *
                1024ull * 1024ull * 1024ull;
        } else if (take_arg(argc, argv, &i, "--health-timeout-ms", &value)) {
            options.health_timeout_ms = std::strtoull(value.c_str(), nullptr, 10);
        } else if (take_arg(argc, argv, &i, "--model-dir", &value)) {
            options.model_dirs.push_back(value);
        } else if (take_arg(argc, argv, &i, "--warmup-qwen35", &value)) {
            options.warmup.emplace_back(value, ModelKind::Qwen35Target);
        } else if (take_arg(argc, argv, &i, "--warmup-dflash2", &value)) {
            options.warmup.emplace_back(value, ModelKind::DFlash2Draft);
        } else if (take_arg(argc, argv, &i, "--warmup-eagle4", &value)) {
            options.warmup.emplace_back(value, ModelKind::Eagle4Draft);
        } else if (take_arg(argc, argv, &i, "--warmup-tp-size", &value)) {
            options.warmup_tp_size =
                static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (take_arg(argc, argv, &i, "--warmup-tp-rank", &value)) {
            options.warmup_tp_rank =
                static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (take_arg(argc, argv, &i, "--ctl-arg", &value)) {
            control_argument = value;
        } else if (std::strcmp(argv[i], "--ctl") == 0) {
            if (i + 1 >= argc) {
                usage();
                return 2;
            }
            control_command = argv[i + 1];
            i += 2;
        } else if (std::strcmp(argv[i], "--allow-any-dir") == 0) {
            options.allow_any_dir = true;
            ++i;
        } else if (std::strcmp(argv[i], "--no-vmm") == 0) {
            vmm = false;
            ++i;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage();
            return 2;
        }
    }

    if (options.socket_path.empty()) {
        usage();
        return 2;
    }

    if (!control_command.empty()) {
        if (control_argument.empty() && control_command == "acquire") {
            std::fprintf(stderr, "acquire requires --ctl-arg\n");
            return 2;
        }
        return control(options.socket_path, control_command, control_argument);
    }

    if (options.staging_bytes == 0) {
        options.staging_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    }
    if (vmm) {
        ::setenv("SHELF_ARENA_VMM", "1", 0);
    }

    auto host_result = ResidentModelHost::create(options);
    if (!host_result.ok()) {
        std::fprintf(stderr, "resident host start failed: %s\n",
                     host_result.status().message().c_str());
        return 1;
    }
    ResidentModelHost host = host_result.release();
    Status warmup_status = host.warmup();
    if (!warmup_status.ok()) {
        std::fprintf(stderr, "resident host warmup failed: %s\n",
                     warmup_status.message().c_str());
        return 1;
    }
    const auto stats = host.stats();
    std::fprintf(stderr, "SHELF_RESIDENT_HOST_READY socket=%s device=%d "
                 "entries=%zu resident_bytes=%zu disk_load_count=%llu\n",
                 options.socket_path.c_str(), options.device, stats.entries,
                 stats.resident_bytes,
                 static_cast<unsigned long long>(stats.disk_load_count));
    std::fflush(stderr);

    Status serve_status = host.serve();
    if (!serve_status.ok()) {
        std::fprintf(stderr, "resident host serve failed: %s\n",
                     serve_status.message().c_str());
    }
    Status shutdown_status = host.shutdown();
    if (!shutdown_status.ok()) {
        std::fprintf(stderr, "resident host shutdown failed: %s\n",
                     shutdown_status.message().c_str());
        return 1;
    }
    return serve_status.ok() ? 0 : 1;
}
