#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/eagle4/config.h>
#include <shelf/models/qwen35/eagle4/weights.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/resident/resident_format.h>
#include <shelf/resident/resident_model_key.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ps {
namespace resident {

constexpr const char* kSocketEnv = "SHELF_MODEL_HOST_SOCKET";
constexpr const char* kDisableEnv = "SHELF_DISABLE_RESIDENT_MODEL";

bool resident_model_disabled();
bool resident_socket_env_present();
Result<std::string> resident_socket_path(const std::string& spec, std::uint32_t index);
Result<std::string> resident_socket_path(std::uint32_t tp_rank);
Result<std::string> resident_socket_path(const ResidentModelKey& key);

class ResidentModelAttachment {
 public:
    ResidentModelAttachment() = default;
    ResidentModelAttachment(const ResidentModelAttachment&) = delete;
    ResidentModelAttachment& operator=(const ResidentModelAttachment&) = delete;
    ResidentModelAttachment(ResidentModelAttachment&& other) noexcept;
    ResidentModelAttachment& operator=(ResidentModelAttachment&& other) noexcept;
    ~ResidentModelAttachment();

    bool valid() const noexcept { return block_ != nullptr; }
    void* block() const noexcept { return block_; }
    std::size_t block_bytes() const noexcept { return block_bytes_; }
    std::size_t resident_bytes() const noexcept
    {
        return static_cast<std::size_t>(header_.resident_bytes);
    }

    std::size_t host_resident_bytes() const noexcept { return host_resident_bytes_; }

    std::size_t persistent_bytes() const noexcept { return resident_bytes(); }
    int device() const noexcept { return device_; }
    const ManifestHeader& header() const noexcept { return header_; }
    const ResidentModelKey& key() const noexcept { return key_; }

    Result<qwen35::Qwen35Model> take_qwen35() const;
    Result<qwen35::dflash2::DFlash2Weights> take_dflash2(
        qwen35::dflash2::DFlash2Config* config_out) const;
    Result<qwen35::eagle4::Eagle4Weights> take_eagle4(
        qwen35::eagle4::Eagle4Config* config_out) const;

 private:
    friend Result<ResidentModelAttachment> acquire_resident_model(
        const ResidentModelKey& key, int device, const std::string& socket_spec);

    void close_mapping() noexcept;

    void* block_ = nullptr;
    std::size_t block_bytes_ = 0;
    int device_ = 0;
    std::size_t host_resident_bytes_ = 0;
    ResidentModelKey key_{};
    ManifestHeader header_{};
    std::vector<std::byte> archive_;
};

Result<ResidentModelAttachment> acquire_resident_model(
    const ResidentModelKey& key, int device, const std::string& socket_spec);

Result<ResidentModelAttachment> acquire_resident_model(
    const ResidentModelKey& key, int device);

Result<std::string> resident_host_stats(const std::string& socket_path);
Status resident_host_release(const std::string& socket_path, const std::string& key_text);

Status resident_health_check(int device, std::uint64_t timeout_ms);

}
}
