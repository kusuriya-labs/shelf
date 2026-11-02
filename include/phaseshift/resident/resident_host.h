#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/resident/resident_format.h>
#include <phaseshift/resident/resident_model_key.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps {
namespace resident {

struct ResidentHostOptions {
    std::string socket_path;
    int device = 0;
    std::size_t staging_bytes = 0;
    std::uint64_t health_timeout_ms = 5000;
    bool allow_any_dir = false;
    std::vector<std::string> model_dirs;
    std::vector<std::pair<std::string, ModelKind>> warmup;
    std::uint32_t warmup_tp_size = 1;
    std::uint32_t warmup_tp_rank = 0;
};

struct ResidentStats {
    std::uint64_t disk_load_count = 0;
    std::uint64_t attach_count = 0;
    std::uint64_t resident_bytes = 0;
    std::size_t entries = 0;
    bool healthy = false;
};

class ResidentModelHost {
 public:
    static Result<ResidentModelHost> create(const ResidentHostOptions& options);

    ResidentModelHost(const ResidentModelHost&) = delete;
    ResidentModelHost& operator=(const ResidentModelHost&) = delete;
    ResidentModelHost(ResidentModelHost&& other) noexcept;
    ResidentModelHost& operator=(ResidentModelHost&& other) noexcept;
    ~ResidentModelHost();

    Status warmup();
    Status serve();
    Status shutdown();

    ResidentStats stats() const;
    std::string stats_text() const;
    Status health(std::uint64_t timeout_ms);

 private:
    struct Entry {
        ResidentModelKey key;
        void* block = nullptr;
        std::size_t block_bytes = 0;
        std::size_t resident_bytes = 0;
        hipIpcMemHandle_t ipc{};
        std::vector<std::byte> archive;
        std::uint64_t disk_load_count = 0;
        std::uint64_t attach_count = 0;
    };

    ResidentModelHost() = default;

    Status bind_socket();
    Status load_entry(const ResidentModelKey& key, Entry& out);
    Status load_entry_once(const ResidentModelKey& key, Entry& out);
    bool allowed_dir(const std::string& model_dir) const;
    void drop_entry(Entry& entry);
    void handle_connection(int fd);
    Status send_entry(int fd, Entry& entry);

    ResidentHostOptions options_;
    int listen_fd_ = -1;
    bool running_ = false;
    std::map<std::string, Entry> entries_;
    std::uint64_t disk_load_count_ = 0;
    std::uint64_t attach_count_ = 0;
    std::string last_health_ = "unknown";
};

}
}
