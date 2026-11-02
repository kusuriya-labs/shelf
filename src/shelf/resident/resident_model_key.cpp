#include <shelf/resident/resident_model_key.h>

#include <shelf/weights/weight_loader.h>

#include <filesystem>

namespace ps {
namespace resident {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void hash_bytes(std::uint64_t& h, const void* data, std::size_t size) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= kFnvPrime;
    }
}

void hash_string(std::uint64_t& h, const std::string& s) {
    hash_bytes(h, s.data(), s.size());
    const std::uint64_t n = static_cast<std::uint64_t>(s.size());
    hash_bytes(h, &n, sizeof(n));
}

}

std::string canonical_model_dir(const std::string& model_dir) {
    if (model_dir.empty()) {
        return model_dir;
    }
    std::error_code ec;
    const std::filesystem::path p(model_dir);
    const std::filesystem::path abs = std::filesystem::absolute(p, ec);
    if (ec) {
        return model_dir;
    }
    const std::filesystem::path canon = std::filesystem::weakly_canonical(abs, ec);
    if (ec) {
        return abs.lexically_normal().string();
    }
    return canon.string();
}

std::uint64_t fingerprint_key(const ResidentModelKey& key) {
    std::uint64_t h = kFnvOffset;
    hash_string(h, key.model_dir);
    const std::uint32_t kind = static_cast<std::uint32_t>(key.kind);
    hash_bytes(h, &kind, sizeof(kind));
    hash_bytes(h, &key.load_flags, sizeof(key.load_flags));
    hash_bytes(h, &key.tp_size, sizeof(key.tp_size));
    hash_bytes(h, &key.tp_rank, sizeof(key.tp_rank));
    hash_bytes(h, &key.device, sizeof(key.device));
    return h;
}

ResidentModelKey ResidentModelKey::make(
    std::string model_dir,
    ModelKind kind,
    std::uint32_t load_flags,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    std::int32_t device) {
    ResidentModelKey key;
    key.model_dir = canonical_model_dir(model_dir);
    key.kind = kind;
    key.load_flags = load_flags;
    key.tp_size = tp_size == 0 ? 1u : tp_size;
    key.tp_rank = tp_rank;
    key.device = device;
    if (!key.model_dir.empty() && weights::is_quantized_model_dir(key.model_dir)) {
        key.load_flags |= kLoadFlagQuantized;
    }
    key.fingerprint = fingerprint_key(key);
    return key;
}

bool ResidentModelKey::operator==(const ResidentModelKey& other) const noexcept {
    return fingerprint == other.fingerprint && model_dir == other.model_dir &&
           kind == other.kind && load_flags == other.load_flags &&
           tp_size == other.tp_size && tp_rank == other.tp_rank &&
           device == other.device;
}

std::string ResidentModelKey::to_string() const {
    std::string out = model_kind_name(kind);
    out += ":";
    out += model_dir;
    out += ":tp";
    out += std::to_string(tp_size);
    out += "x";
    out += std::to_string(tp_rank);
    out += ":f";
    out += std::to_string(load_flags);
    out += ":d";
    out += std::to_string(device);
    out += ":";
    out += std::to_string(fingerprint);
    return out;
}

const char* model_kind_name(ModelKind kind) {
    switch (kind) {
        case ModelKind::Qwen35Target: return "qwen35";
        case ModelKind::DFlash2Draft: return "dflash2";
        case ModelKind::Eagle4Draft: return "eagle4";
    }
    return "unknown";
}

std::uint32_t resident_load_flags(const weights::WeightLoadOptions& options) {
    return options.preshuffle ? kLoadFlagPreshuffle : 0u;
}

std::uint32_t resident_load_flags(const qwen35::Qwen35LoadOptions& options) {
    std::uint32_t flags = 0;
    if (options.verify_quantized_payload_crc) flags |= kLoadFlagVerifyCrc;
    if (options.load_mtp_layers) flags |= kLoadFlagMtpLayers;
    if (options.vocab_parallel_lm_head) flags |= kLoadFlagVocabParallelLmHead;
    return flags | resident_load_flags(options.weights);
}

}
}
