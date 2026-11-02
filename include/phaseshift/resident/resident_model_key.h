#pragma once

#include <cstdint>
#include <string>

#include <phaseshift/models/qwen35/weights/model_weights.h>

namespace ps {
namespace resident {

enum class ModelKind : std::uint32_t {
    Qwen35Target = 1,
    DFlash2Draft = 2,
    Eagle4Draft = 3,
};

constexpr std::uint32_t kLoadFlagVerifyCrc = 1u << 0;
constexpr std::uint32_t kLoadFlagMtpLayers = 1u << 1;
constexpr std::uint32_t kLoadFlagPreshuffle = 1u << 2;
constexpr std::uint32_t kLoadFlagQuantized = 1u << 3;
constexpr std::uint32_t kLoadFlagVocabParallelLmHead = 1u << 4;

struct ResidentModelKey {
    std::string model_dir;
    ModelKind kind = ModelKind::Qwen35Target;
    std::uint32_t load_flags = kLoadFlagPreshuffle;
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    std::int32_t device = 0;
    std::uint64_t fingerprint = 0;

    static ResidentModelKey make(
        std::string model_dir,
        ModelKind kind,
        std::uint32_t load_flags,
        std::uint32_t tp_size,
        std::uint32_t tp_rank,
        std::int32_t device);

    bool operator==(const ResidentModelKey& other) const noexcept;
    std::string to_string() const;
};

std::string canonical_model_dir(const std::string& model_dir);
std::uint64_t fingerprint_key(const ResidentModelKey& key);

const char* model_kind_name(ModelKind kind);

std::uint32_t resident_load_flags(const qwen35::Qwen35LoadOptions& options);
std::uint32_t resident_load_flags(const weights::WeightLoadOptions& options);

}
}
