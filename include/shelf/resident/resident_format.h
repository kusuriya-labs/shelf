#pragma once

#include <shelf/core/status.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/weights/matrix_weight.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ps {
namespace resident {

constexpr std::uint32_t kManifestMagic = 0x31525350u;
constexpr std::uint32_t kManifestVersion = 1u;
constexpr std::uint32_t kIpcHandleBytes = 64u;
constexpr std::size_t kMaxTensorDimsWire = 8;

#pragma pack(push, 1)
struct ManifestHeader {
    std::uint32_t magic = kManifestMagic;
    std::uint32_t version = kManifestVersion;
    std::uint32_t kind = 0;
    std::uint32_t ipc_handle_bytes = kIpcHandleBytes;
    std::int32_t device = 0;
    std::uint32_t reserved = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t block_bytes = 0;
    std::uint64_t disk_load_count = 0;
    std::uint64_t attach_count = 0;
    std::uint64_t archive_bytes = 0;
};
#pragma pack(pop)

static_assert(sizeof(ManifestHeader) == 64, "ManifestHeader wire layout");

#pragma pack(push, 1)
struct TensorRecord {
    std::uint64_t alloc_offset = 0;
    std::uint64_t alloc_bytes = 0;
    std::uint64_t view_offset = 0;
    std::uint64_t shape[kMaxTensorDimsWire] = {};
    std::uint64_t strides[kMaxTensorDimsWire] = {};
    std::uint32_t ndim = 0;
    std::uint32_t elem_size = 0;
    std::int32_t device = -1;
    std::uint32_t present = 0;
};
#pragma pack(pop)

static_assert(sizeof(TensorRecord) == 168, "TensorRecord wire layout");

class ArchiveWriter {
 public:
    explicit ArchiveWriter(const void* base);

    void u8(std::uint8_t value);
    void u32(std::uint32_t value);
    void i32(std::int32_t value);
    void u64(std::uint64_t value);
    void i64(std::int64_t value);
    void f32(float value);
    void str(const std::string& value);
    void i32_vec(const std::vector<std::int32_t>& values);
    void int_vec(const std::vector<int>& values);
    void i64_vec(const std::vector<std::int64_t>& values);
    void tensor(const gpu::Tensor& tensor);
    void matrix(const weights::MatrixWeight& weight);
    void partition(const std::optional<weights::TensorPartitionDesc>& partition);
    void raw(const void* data, std::size_t bytes);

    const std::vector<std::byte>& bytes() const noexcept { return buf_; }
    std::vector<std::byte> release() { return std::move(buf_); }

 private:
    const void* base_;
    std::vector<std::byte> buf_;
};

class ArchiveReader {
 public:
    ArchiveReader(const std::byte* data, std::size_t size, void* block,
                  std::size_t block_bytes, int device);

    std::uint8_t u8();
    std::uint32_t u32();
    std::int32_t i32();
    std::uint64_t u64();
    std::int64_t i64();
    float f32();
    std::string str();
    std::vector<std::int32_t> i32_vec();
    std::vector<int> int_vec();
    std::vector<std::int64_t> i64_vec();
    gpu::Tensor tensor();
    weights::MatrixWeight matrix();
    std::optional<weights::TensorPartitionDesc> partition();
    void raw(void* out, std::size_t bytes);

    const Status& status() const noexcept { return status_; }
    bool ok() const noexcept { return status_.ok(); }
    void fail(const std::string& message);
    std::size_t remaining() const noexcept;
    std::size_t offset() const noexcept { return offset_; }

 private:
    const std::byte* cur_;
    std::size_t left_;
    void* block_;
    std::size_t block_bytes_;
    int device_;
    Status status_;
    std::size_t offset_ = 0;
};

Status write_qwen35_archive(
    const qwen35::Qwen35TextConfig& text_config,
    const qwen35::Qwen35ModelWeights& weights,
    const void* base,
    std::vector<std::byte>& out);

Result<qwen35::Qwen35Model> build_qwen35_model(
    const std::byte* data,
    std::size_t size,
    void* block,
    std::size_t block_bytes,
    int device);

Status write_dflash2_archive(
    const qwen35::dflash2::DFlash2Config& config,
    const qwen35::dflash2::DFlash2Weights& weights,
    const void* base,
    std::vector<std::byte>& out);

Status read_dflash2_archive(
    const std::byte* data,
    std::size_t size,
    void* block,
    std::size_t block_bytes,
    int device,
    qwen35::dflash2::DFlash2Config& config_out,
    qwen35::dflash2::DFlash2Weights& weights_out);

}
}
