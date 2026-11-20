#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <shelf/quantization/exl3/exl3.h>
#include <shelf/quantization/int6/int6_w32.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <queue>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <map>
#include <mutex>
#include <thread>
#include <nlohmann/json.hpp>
#include <shelf/io/safetensors_reader.h>

namespace shelf {
namespace qwen4exp {

namespace {
float e4m3_to_f32(uint8_t x) {
    const int sign = (x & 0x80u) ? -1 : 1;
    const int exp = (x >> 3) & 0x0Fu;
    const int man = x & 0x07u;
    if (exp == 0x0F) {
        if (man == 0x07) return std::numeric_limits<float>::quiet_NaN();
        return sign * 448.0f;
    }
    const float value = (exp == 0)
        ? (static_cast<float>(man) * 0.125f) * std::ldexp(1.0f, -6)
        : (1.0f + static_cast<float>(man) * 0.125f) * std::ldexp(1.0f, exp - 7);
    return sign * value;
}

float bf16_bits_to_f32(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}
}  // namespace
namespace {

constexpr std::size_t kReadChunk = 64ull * 1024ull * 1024ull;
constexpr std::uint32_t kExpectedRowWidth = 160u;

struct TensorRecord {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

Status read_exact_at(int fd, std::uint64_t offset, void* dst, std::size_t bytes) {
    std::uint8_t* out = static_cast<std::uint8_t*>(dst);
    std::size_t done = 0;
    while (done < bytes) {
        const std::size_t want = std::min<std::size_t>(kReadChunk, bytes - done);
        const ssize_t got = ::pread(fd, out + done, want, static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return Status::invalid_argument(
                (std::string("PLE read failed: ") + std::strerror(errno)).c_str(), __FILE__,
                __LINE__);
        }
        if (got == 0) {
            return Status::invalid_argument("PLE file ended before the tensor payload",
                                            __FILE__, __LINE__);
        }
        done += static_cast<std::size_t>(got);
    }
    return Status::make_ok();
}

Result<bool> read_json_header(int fd, std::uint64_t file_size, nlohmann::json& header,
                              std::uint64_t& header_len) {
    std::uint64_t raw_len = 0;
    Status st = read_exact_at(fd, 0, &raw_len, sizeof(raw_len));
    if (!st.ok()) return st;
    if (raw_len == 0 || raw_len > file_size - 8u || raw_len > (64ull << 20)) {
        return Status::invalid_argument("PLE safetensors header length is invalid", __FILE__,
                                        __LINE__);
    }
    std::vector<char> bytes(static_cast<std::size_t>(raw_len));
    st = read_exact_at(fd, 8, bytes.data(), bytes.size());
    if (!st.ok()) return st;
    try {
        header = nlohmann::json::parse(bytes.begin(), bytes.end());
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string("PLE safetensors header is not JSON: ") + e.what()).c_str(),
            __FILE__, __LINE__);
    }
    if (!header.is_object()) {
        return Status::invalid_argument("PLE safetensors header is not an object", __FILE__,
                                        __LINE__);
    }
    header_len = raw_len;
    return true;
}

Result<std::uint64_t> json_u64(const nlohmann::json& object, const char* key) {
    if (!object.contains(key)) {
        return Status::invalid_argument(
            (std::string("PLE metadata is missing ") + key).c_str(), __FILE__, __LINE__);
    }
    const nlohmann::json& value = object[key];
    if (value.is_number_unsigned()) return value.get<std::uint64_t>();
    if (value.is_number_integer()) {
        const std::int64_t signed_value = value.get<std::int64_t>();
        if (signed_value >= 0) return static_cast<std::uint64_t>(signed_value);
    }
    if (value.is_string()) {
        try {
            return std::stoull(value.get<std::string>());
        } catch (const std::exception&) {
        }
    }
    return Status::invalid_argument(
        (std::string("PLE metadata is not an integer: ") + key).c_str(), __FILE__, __LINE__);
}

Result<std::string> json_string(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || !object[key].is_string()) {
        return Status::invalid_argument(
            (std::string("PLE metadata is missing string ") + key).c_str(), __FILE__, __LINE__);
    }
    return object[key].get<std::string>();
}

Result<TensorRecord> parse_tensor(const nlohmann::json& entry) {
    if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets")) {
        return Status::invalid_argument("PLE tensor entry is incomplete", __FILE__, __LINE__);
    }
    TensorRecord record;
    record.dtype = entry["dtype"].get<std::string>();
    if (!entry["shape"].is_array() || entry["shape"].size() == 0 ||
        entry["shape"].size() > 8) {
        return Status::invalid_argument("PLE tensor shape is invalid", __FILE__, __LINE__);
    }
    for (const auto& dim : entry["shape"]) {
        if (dim.is_number_unsigned()) {
            record.shape.push_back(dim.get<std::uint64_t>());
        } else if (dim.is_number_integer()) {
            const std::int64_t value = dim.get<std::int64_t>();
            if (value < 0) {
                return Status::invalid_argument("PLE tensor shape is negative", __FILE__,
                                                __LINE__);
            }
            record.shape.push_back(static_cast<std::uint64_t>(value));
        } else {
            return Status::invalid_argument("PLE tensor shape is not an integer", __FILE__,
                                            __LINE__);
        }
    }
    if (!entry["data_offsets"].is_array() || entry["data_offsets"].size() != 2) {
        return Status::invalid_argument("PLE tensor data_offsets is invalid", __FILE__,
                                        __LINE__);
    }
    record.begin = entry["data_offsets"][0].get<std::uint64_t>();
    record.end = entry["data_offsets"][1].get<std::uint64_t>();
    if (record.end < record.begin) {
        return Status::invalid_argument("PLE tensor data_offsets is reversed", __FILE__,
                                        __LINE__);
    }
    return record;
}

Status expect_tensor_bytes(const TensorRecord& record, std::uint64_t expected) {
    if (record.end - record.begin != expected) {
        return Status::invalid_argument("PLE tensor byte count mismatch", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

namespace {

constexpr const char* kPleArtifactName = "quanted_ple.bin";
constexpr std::uint32_t kPleArtifactVersion = 1u;
constexpr std::uint32_t kPleArtifactHeaderBytes = 8192u;
constexpr std::uint32_t kPleArtifactFormatInt6 = 1u;
constexpr std::uint32_t kPleArtifactSourceBf16 = 1u;
constexpr std::uint32_t kPleArtifactSourceBytes = 32u;

bool ple_artifact_enabled() noexcept {
    const char* value = std::getenv("SHELF_PLE_ARTIFACT");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

std::uint64_t ple_hot_cache_bytes() noexcept {
    const char* value = std::getenv("SHELF_PLE_HOT_CACHE_GIB");
    if (value == nullptr || *value == '\0') return 0u;
    const double gib = std::strtod(value, nullptr);
    if (gib <= 0.0) return 0u;
    return static_cast<std::uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

constexpr std::uint32_t kHotBlockRows = 2048u;
constexpr std::uint64_t kHotRebalanceAccesses = 8192u;
constexpr std::uint64_t kHotSwapBudget = 64u;
constexpr std::uint32_t kHotMinGain = 150u;
constexpr std::uint32_t kHotDecayAt = 4096u;

std::string ple_artifact_path(const std::string& model_dir) {
    return model_dir + "/" + kPleArtifactName;
}

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void append_u32(std::vector<std::uint8_t>& out, std::size_t& at, std::uint32_t value) {
    std::memcpy(out.data() + at, &value, sizeof(value));
    at += sizeof(value);
}

void append_u64(std::vector<std::uint8_t>& out, std::size_t& at, std::uint64_t value) {
    std::memcpy(out.data() + at, &value, sizeof(value));
    at += sizeof(value);
}

void append_i64(std::vector<std::uint8_t>& out, std::size_t& at, std::int64_t value) {
    std::memcpy(out.data() + at, &value, sizeof(value));
    at += sizeof(value);
}

bool take_u32(const std::uint8_t*& at, const std::uint8_t* end, std::uint32_t& value) {
    if (static_cast<std::size_t>(end - at) < sizeof(value)) return false;
    std::memcpy(&value, at, sizeof(value));
    at += sizeof(value);
    return true;
}

bool take_u64(const std::uint8_t*& at, const std::uint8_t* end, std::uint64_t& value) {
    if (static_cast<std::size_t>(end - at) < sizeof(value)) return false;
    std::memcpy(&value, at, sizeof(value));
    at += sizeof(value);
    return true;
}

bool take_i64(const std::uint8_t*& at, const std::uint8_t* end, std::int64_t& value) {
    if (static_cast<std::size_t>(end - at) < sizeof(value)) return false;
    std::memcpy(&value, at, sizeof(value));
    at += sizeof(value);
    return true;
}

Status pwrite_exact(int fd, const void* src, std::size_t bytes, std::uint64_t offset) {
    const auto* bytes_ptr = static_cast<const std::uint8_t*>(src);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t got = ::pwrite(fd, bytes_ptr + done, bytes - done,
                                     static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return Status::invalid_argument(
                (std::string("PLE artifact write failed: ") + std::strerror(errno)).c_str(),
                __FILE__, __LINE__);
        }
        if (got == 0) {
            return Status::invalid_argument("PLE artifact write made no progress",
                                            __FILE__, __LINE__);
        }
        done += static_cast<std::size_t>(got);
    }
    return Status::make_ok();
}

}  // namespace

void PleTable::FreeDeleter::operator()(void* p) const noexcept {
    std::free(p);
}

void PleTable::MappedDeleter::operator()(const std::uint8_t* p) const noexcept {
    if (p != nullptr) (void)::munmap(const_cast<std::uint8_t*>(p), bytes);
}

void PleTable::PinnedDeleter::operator()(std::uint8_t* p) const noexcept {
    if (p != nullptr) (void)::hipHostFree(p);
}

void PleTable::FdDeleter::operator()(int* p) const noexcept {
    if (p != nullptr && *p >= 0) (void)::close(*p);
    delete p;
}


Result<PleTable> PleTable::load_index_shards(const std::string& model_dir,
                                             const Qwen4ExpTextConfig& config,
                                             bool fadvise_dontneed) {
    const std::vector<std::size_t> ple_layers = config.ple_layers();
    if (ple_layers.size() != 1u) {
        return Status::unsupported(
            "PLE table load expects exactly one ple_layer_ids entry", __FILE__, __LINE__);
    }
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(ple_layers[0]) +
                               ".ple.ple_embedding.";
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream index_file(index_path);
    if (!index_file.is_open()) {
        return Status::invalid_argument("PLE table index is missing", __FILE__, __LINE__);
    }
    nlohmann::json index;
    try {
        index_file >> index;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string("bad PLE index: ") + e.what()).c_str(), __FILE__, __LINE__);
    }
    if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
        return Status::invalid_argument("PLE index has no weight_map", __FILE__, __LINE__);
    }
    const auto& weight_map = index["weight_map"];
    const std::string shard_prefix = prefix + "ngram_embedding.shard_";
    std::map<std::uint32_t, std::string> shards;
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
        const std::string name = it.key();
        if (name.size() < shard_prefix.size() + 7u ||
            name.compare(0, shard_prefix.size(), shard_prefix) != 0 ||
            name.compare(name.size() - 7u, 7u, ".weight") != 0) {
            continue;
        }
        const std::string digits =
            name.substr(shard_prefix.size(), name.size() - shard_prefix.size() - 7u);
        if (digits.empty()) continue;
        const std::uint32_t id = static_cast<std::uint32_t>(std::stoul(digits));
        shards.emplace(id, name);
    }
    if (shards.size() != config.ple.split_ngram_parts) {
        return Status::invalid_argument("PLE shard count differs from split_ngram_parts",
                                        __FILE__, __LINE__);
    }
    const auto head_sizes = qwen4exp_ple_head_vocab_sizes(config, 0u);
    const auto head_offsets = qwen4exp_ple_head_offsets(config, 0u);
    const auto multipliers = qwen4exp_ple_layer_multipliers(config, 0u);
    if (!head_sizes.ok() || !head_offsets.ok() || !multipliers.ok()) {
        return Status::invalid_argument("cannot derive PLE hash metadata from config",
                                        __FILE__, __LINE__);
    }
    PleTable table;
    table.file_path_ = index_path;
    auto padded_rows = qwen4exp_ple_padded_rows(config, 0u);
    if (!padded_rows.ok()) return padded_rows.status();
    table.rows_ = padded_rows.value();
    table.row_width_ = kExpectedRowWidth;
    table.shard_count_ = static_cast<std::uint32_t>(shards.size());
    table.ngram_size_ = static_cast<std::uint32_t>(config.ple.ngram_size);
    table.heads_per_ngram_ = static_cast<std::uint32_t>(config.ple.heads_per_ngram);
    table.eos_token_id_ = config.ple.eos_token_id;
    table.head_offsets_.assign(head_offsets.value().begin(), head_offsets.value().end());
    table.head_vocab_sizes_.assign(head_sizes.value().begin(), head_sizes.value().end());
    table.layer_multipliers_.assign(multipliers.value().begin(), multipliers.value().end());

    std::map<std::string, std::unique_ptr<shelf::io::SafetensorsReader>> readers;
    std::mutex readers_mutex;
    auto reader_for = [&](const std::string& shard)
        -> Result<const shelf::io::SafetensorsReader*> {
        std::lock_guard<std::mutex> lock(readers_mutex);
        auto it = readers.find(shard);
        if (it != readers.end()) return it->second.get();
        auto opened = shelf::io::SafetensorsReader::open(model_dir + "/" + shard);
        if (!opened.ok()) return opened.status();
        auto inserted = readers.emplace(
            shard, std::make_unique<shelf::io::SafetensorsReader>(opened.release()));
        return inserted.first->second.get();
    };

    auto first_shard_it = weight_map.find(shards.begin()->second);
    if (first_shard_it == weight_map.end()) {
        return Status::invalid_argument("PLE shard is missing from the index", __FILE__,
                                        __LINE__);
    }
    auto first_reader = reader_for(first_shard_it.value().get<std::string>());
    if (!first_reader.ok()) return first_reader.status();
    auto first_spec = first_reader.value()->tensor_spec(shards.begin()->second);
    if (!first_spec.ok() || first_spec.value().shape.size() != 2u ||
        first_spec.value().shape[1] != kExpectedRowWidth) {
        return Status::invalid_argument("PLE shard shape is invalid", __FILE__, __LINE__);
    }
    const shelf::io::SType source_dtype = first_spec.value().dtype;
    table.shard_rows_ = static_cast<std::uint32_t>(first_spec.value().shape[0]);
    if (table.shard_rows_ == 0u ||
        static_cast<std::uint64_t>(table.shard_rows_) * table.shard_count_ != table.rows_) {
        return Status::invalid_argument(
            "PLE shard rows do not cover the padded row count", __FILE__, __LINE__);
    }
    const std::uint32_t groups = table.row_width_ / 32u;
    if (groups == 0u || table.row_width_ % 32u != 0u) {
        return Status::invalid_argument("PLE row width is not a W32 multiple", __FILE__,
                                        __LINE__);
    }

    const bool artifact_enabled = ple_artifact_enabled();
    std::vector<ArtifactSource> artifact_sources;
    if (artifact_enabled) {
        artifact_sources.reserve(table.shard_count_);
        for (std::uint32_t id = 0; id < table.shard_count_; ++id) {
            auto name_it = shards.find(id);
            if (name_it == shards.end()) {
                artifact_sources.clear();
                break;
            }
            auto shard_it = weight_map.find(name_it->second);
            if (shard_it == weight_map.end()) {
                artifact_sources.clear();
                break;
            }
            struct stat st {};
            const std::string shard_path =
                model_dir + "/" + shard_it.value().get<std::string>();
            if (::stat(shard_path.c_str(), &st) != 0) {
                artifact_sources.clear();
                break;
            }
            ArtifactSource source;
            source.id = id;
            source.size = static_cast<std::uint64_t>(st.st_size);
            source.mtime_sec = static_cast<std::int64_t>(st.st_mtim.tv_sec);
            source.mtime_nsec = static_cast<std::int64_t>(st.st_mtim.tv_nsec);
            artifact_sources.push_back(source);
        }
        if (artifact_sources.size() != table.shard_count_) artifact_sources.clear();
    }
    if (source_dtype == shelf::io::SType::BF16 && !artifact_sources.empty()) {
        PleTable cached;
        auto loaded = load_int6_artifact(model_dir, config, artifact_sources, cached,
                                         ple_hot_cache_bytes());
        if (loaded.ok() && loaded.value()) return cached;
    }

    if (source_dtype == shelf::io::SType::BF16) {
        table.format_ = TableFormat::Int6W32Fp16;
        table.row_bytes_ = groups * shelf::quantization::int6::kGroupBytes;
        table.k_bits_ = 0u;
    } else if (source_dtype == shelf::io::SType::F8_E4M3) {
        table.format_ = TableFormat::Fp8E4M3;
        table.row_bytes_ = table.row_width_;
        table.k_bits_ = 0u;
    } else {
        return Status::unsupported(
            "PLE shards are neither BF16 nor FP8 E4M3", __FILE__, __LINE__);
    }

    if (table.format_ == TableFormat::Fp8E4M3) {
        auto scale_name = prefix + "ngram_embedding.weight_scale";
        auto scale_it = weight_map.find(scale_name);
        if (scale_it == weight_map.end()) {
            return Status::invalid_argument("PLE FP8 weight_scale is missing", __FILE__,
                                            __LINE__);
        }
        auto scale_reader = reader_for(scale_it.value().get<std::string>());
        if (!scale_reader.ok()) return scale_reader.status();
        auto scale_spec = scale_reader.value()->tensor_spec(scale_name);
        if (!scale_spec.ok() || scale_spec.value().dtype != shelf::io::SType::BF16) {
            return Status::invalid_argument("PLE FP8 weight_scale is not BF16", __FILE__,
                                            __LINE__);
        }
        std::size_t scale_bytes = 0;
        auto scale_data = scale_reader.value()->tensor_data(scale_name, scale_bytes);
        if (!scale_data.ok() || scale_bytes != 2u) {
            return Status::invalid_argument("PLE FP8 weight_scale size is invalid", __FILE__,
                                            __LINE__);
        }
        const std::uint16_t scale_bits = *static_cast<const std::uint16_t*>(scale_data.value());
        table.fp8_scale_ = bf16_bits_to_f32(scale_bits);
    }

    table.shards_.resize(table.shard_count_);
    auto load_shard = [&](std::uint32_t id) -> Status {
        auto name_it = shards.find(id);
        if (name_it == shards.end()) {
            return Status::invalid_argument("PLE shard index is not contiguous",
                                            __FILE__, __LINE__);
        }
        auto shard_it = weight_map.find(name_it->second);
        if (shard_it == weight_map.end()) {
            return Status::invalid_argument("PLE shard is missing from the index",
                                            __FILE__, __LINE__);
        }
        const std::string shard_file = shard_it.value().get<std::string>();
        const std::string shard_path = model_dir + "/" + shard_file;
        const std::size_t row_bytes_in =
            static_cast<std::size_t>(table.row_width_) *
            (source_dtype == shelf::io::SType::BF16 ? 2u : 1u);
        const std::size_t out_bytes =
            static_cast<std::size_t>(table.shard_rows_) * table.row_bytes_;

        if (table.format_ == TableFormat::Int6W32Fp16) {
            std::uint64_t tensor_offset = 0;
            std::uint64_t tensor_bytes = 0;
            {
                auto local = shelf::io::SafetensorsReader::open(shard_path);
                if (!local.ok()) return local.status();
                auto spec = local.value().tensor_spec(name_it->second);
                if (!spec.ok() || spec.value().dtype != source_dtype ||
                    spec.value().shape.size() != 2u ||
                    spec.value().shape[0] != table.shard_rows_ ||
                    spec.value().shape[1] != table.row_width_) {
                    return Status::invalid_argument("PLE shard shape is invalid", __FILE__,
                                                    __LINE__);
                }
                auto span = local.value().tensor_span(name_it->second);
                if (!span.ok()) return span.status();
                tensor_offset = span.value().absolute_offset;
                tensor_bytes = span.value().byte_length;
            }
            if (tensor_bytes != static_cast<std::uint64_t>(table.shard_rows_) * row_bytes_in) {
                return Status::invalid_argument("PLE shard byte count is invalid", __FILE__,
                                                __LINE__);
            }
            enum class StreamIo { Pread, Mmap, Direct };
            StreamIo io_mode = StreamIo::Mmap;
            if (const char* value = std::getenv("SHELF_PLE_STREAM_IO")) {
                const std::string mode(value);
                if (mode == "pread") io_mode = StreamIo::Pread;
                else if (mode == "mmap") io_mode = StreamIo::Mmap;
                else if (mode == "direct") io_mode = StreamIo::Direct;
            }
            int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
            if (io_mode == StreamIo::Direct) flags |= O_DIRECT;
#endif
            const int fd = ::open(shard_path.c_str(), flags);
            if (fd < 0) {
                if (io_mode == StreamIo::Direct) {
                    return Status::unsupported(
                        (std::string("cannot open PLE shard with O_DIRECT: ") +
                         std::strerror(errno)).c_str(), __FILE__, __LINE__);
                }
                return Status::invalid_argument(
                    (std::string("cannot open PLE shard: ") + std::strerror(errno)).c_str(),
                    __FILE__, __LINE__);
            }
            void* memory = nullptr;
            if (::posix_memalign(&memory, 64u, out_bytes) != 0) {
                ::close(fd);
                return Status::insufficient_memory("cannot allocate PLE shard", __FILE__,
                                                   __LINE__);
            }
            auto* out = static_cast<std::uint8_t*>(memory);
            constexpr std::uint32_t kChunkRows = 65536u;
            std::vector<std::uint8_t> chunk(
                static_cast<std::size_t>(kChunkRows) * row_bytes_in);
            const long page_size = ::sysconf(_SC_PAGESIZE);
            const std::uint64_t map_page =
                page_size > 0 ? static_cast<std::uint64_t>(page_size) : 4096u;
            constexpr std::uint64_t kDirectAlign = 4096u;
            std::uint8_t* direct_buffer = nullptr;
            const std::size_t direct_capacity =
                static_cast<std::size_t>(kChunkRows) * row_bytes_in + kDirectAlign;
            if (io_mode == StreamIo::Direct) {
                void* aligned = nullptr;
                if (::posix_memalign(&aligned, kDirectAlign, direct_capacity) != 0) {
                    std::free(memory);
                    ::close(fd);
                    return Status::insufficient_memory(
                        "cannot allocate PLE direct IO buffer", __FILE__, __LINE__);
                }
                direct_buffer = static_cast<std::uint8_t*>(aligned);
            }
            auto cleanup = [&]() {
                std::free(memory);
                std::free(direct_buffer);
                ::close(fd);
            };
            for (std::uint32_t row = 0; row < table.shard_rows_; row += kChunkRows) {
                const std::uint32_t rows =
                    std::min(kChunkRows, table.shard_rows_ - row);
                const std::size_t chunk_bytes = static_cast<std::size_t>(rows) * row_bytes_in;
                const std::uint64_t chunk_offset =
                    tensor_offset + static_cast<std::uint64_t>(row) * row_bytes_in;
                const shelf::bf16_t* source = nullptr;
                void* mapped = nullptr;
                std::size_t mapped_bytes = 0;

                if (io_mode == StreamIo::Mmap) {
                    const std::uint64_t aligned_offset = chunk_offset & ~(map_page - 1u);
                    const std::uint64_t delta = chunk_offset - aligned_offset;
                    const std::uint64_t length =
                        (delta + chunk_bytes + map_page - 1u) & ~(map_page - 1u);
                    void* candidate = ::mmap(nullptr, static_cast<std::size_t>(length),
                                             PROT_READ, MAP_PRIVATE, fd,
                                             static_cast<off_t>(aligned_offset));
                    if (candidate != MAP_FAILED) {
                        mapped = candidate;
                        mapped_bytes = static_cast<std::size_t>(length);
                        source = reinterpret_cast<const shelf::bf16_t*>(
                            static_cast<const std::uint8_t*>(candidate) + delta);
                    }
                }

                if (source == nullptr && io_mode == StreamIo::Direct) {
                    const std::uint64_t aligned_offset = chunk_offset & ~(kDirectAlign - 1u);
                    const std::uint64_t delta = chunk_offset - aligned_offset;
                    const std::uint64_t length =
                        (delta + chunk_bytes + kDirectAlign - 1u) & ~(kDirectAlign - 1u);
                    if (length > direct_capacity) {
                        cleanup();
                        return Status::invalid_state(
                            "PLE direct IO buffer is too small", __FILE__, __LINE__);
                    }
                    const ssize_t got = ::pread(fd, direct_buffer,
                                                static_cast<std::size_t>(length),
                                                static_cast<off_t>(aligned_offset));
                    if (got < 0 || static_cast<std::uint64_t>(got) < delta + chunk_bytes) {
                        cleanup();
                        return Status::invalid_argument(
                            (std::string("cannot read PLE shard with O_DIRECT: ") +
                             std::strerror(errno)).c_str(), __FILE__, __LINE__);
                    }
                    source = reinterpret_cast<const shelf::bf16_t*>(direct_buffer + delta);
                }

                if (source == nullptr) {
                    Status read = read_exact_at(fd, chunk_offset, chunk.data(), chunk_bytes);
                    if (!read.ok()) {
                        cleanup();
                        return read;
                    }
                    source = reinterpret_cast<const shelf::bf16_t*>(chunk.data());
                }

                for (std::uint32_t local_row = 0; local_row < rows; ++local_row) {
                    shelf::quantization::int6::quantize_bf16_row(
                        source + static_cast<std::size_t>(local_row) * table.row_width_, groups,
                        out + static_cast<std::size_t>(row + local_row) * table.row_bytes_);
                }

                if (mapped != nullptr) {
                    (void)::madvise(mapped, mapped_bytes, MADV_DONTNEED);
                    (void)::munmap(mapped, mapped_bytes);
                }
                if (fadvise_dontneed && io_mode != StreamIo::Direct && page_size > 0) {
                    const std::uint64_t begin = chunk_offset & ~(map_page - 1u);
                    const std::uint64_t end =
                        (chunk_offset + chunk_bytes + map_page - 1u) & ~(map_page - 1u);
                    (void)::posix_fadvise(fd, static_cast<off_t>(begin),
                                          static_cast<off_t>(end - begin),
                                          POSIX_FADV_DONTNEED);
                }
            }
            std::free(direct_buffer);
            ::close(fd);
            table.shards_[id].data.reset(out);
            table.shards_[id].bytes = out_bytes;
            return Status::make_ok();
        }

        auto reader = reader_for(shard_file);
        if (!reader.ok()) return reader.status();
        auto spec = reader.value()->tensor_spec(name_it->second);
        if (!spec.ok() || spec.value().dtype != source_dtype ||
            spec.value().shape.size() != 2u ||
            spec.value().shape[0] != table.shard_rows_ ||
            spec.value().shape[1] != table.row_width_) {
            return Status::invalid_argument("PLE shard shape is invalid", __FILE__,
                                            __LINE__);
        }
        std::size_t bytes = 0;
        auto data = reader.value()->tensor_data(name_it->second, bytes);
        if (!data.ok()) return data.status();
        if (bytes != static_cast<std::size_t>(table.shard_rows_) * row_bytes_in) {
            return Status::invalid_argument("PLE shard byte count is invalid", __FILE__,
                                            __LINE__);
        }
        void* memory = nullptr;
        if (::posix_memalign(&memory, 64u, out_bytes) != 0) {
            return Status::insufficient_memory("cannot allocate PLE shard", __FILE__,
                                               __LINE__);
        }
        std::memcpy(memory, data.value(), bytes);
        table.shards_[id].data.reset(static_cast<std::uint8_t*>(memory));
        table.shards_[id].bytes = out_bytes;
        return Status::make_ok();
    };
    const std::uint32_t worker_count =
        std::min<std::uint32_t>(16u, table.shard_count_);
    std::vector<Status> errors(worker_count, Status::make_ok());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            for (std::uint32_t id = worker; id < table.shard_count_; id += worker_count) {
                Status status = load_shard(id);
                if (!status.ok()) {
                    errors[worker] = status;
                    return;
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    for (const Status& error : errors) {
        if (!error.ok()) return error;
    }
    table.payload_bytes_ = static_cast<std::uint64_t>(table.shard_count_) *
                           table.shard_rows_ * table.row_bytes_;
    if (table.format_ == TableFormat::Int6W32Fp16 && artifact_enabled &&
        !artifact_sources.empty()) {
        Status written = write_int6_artifact(model_dir, artifact_sources, table);
        if (!written.ok()) {
            std::fprintf(stderr, "PLE artifact write skipped: %s\n",
                         written.message().c_str());
        }
    }
    return table;
}

Result<bool> PleTable::load_int6_artifact(
    const std::string& model_dir, const Qwen4ExpTextConfig& config,
    const std::vector<ArtifactSource>& sources, PleTable& out,
    std::uint64_t hot_cache_bytes) {
    if (sources.empty()) return false;
    const std::string path = ple_artifact_path(model_dir);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st {};
    if (::fstat(fd, &st) != 0 ||
        st.st_size < static_cast<off_t>(kPleArtifactHeaderBytes)) {
        ::close(fd);
        return false;
    }
    std::vector<std::uint8_t> header(kPleArtifactHeaderBytes);
    Status read = read_exact_at(fd, 0, header.data(), header.size());
    if (!read.ok()) {
        ::close(fd);
        return false;
    }
    static constexpr char kMagic[8] = {'S', 'H', 'E', 'L', 'Q', 'P', 'L', '1'};
    const std::uint8_t* at = header.data();
    const std::uint8_t* end = header.data() + header.size();
    if (std::memcmp(at, kMagic, sizeof(kMagic)) != 0) {
        ::close(fd);
        return false;
    }
    at += sizeof(kMagic);
    std::uint32_t version = 0;
    std::uint32_t header_bytes = 0;
    std::uint32_t format = 0;
    std::uint32_t source_format = 0;
    std::uint64_t rows = 0;
    std::uint32_t row_width = 0;
    std::uint32_t row_bytes = 0;
    std::uint32_t shard_rows = 0;
    std::uint32_t shard_count = 0;
    std::uint32_t source_count = 0;
    std::uint64_t payload_stride = 0;
    std::uint64_t payload_bytes = 0;
    std::uint32_t source_bytes = 0;
    std::uint32_t reserved = 0;
    const bool parsed =
        take_u32(at, end, version) && take_u32(at, end, header_bytes) &&
        take_u32(at, end, format) && take_u32(at, end, source_format) &&
        take_u64(at, end, rows) && take_u32(at, end, row_width) &&
        take_u32(at, end, row_bytes) && take_u32(at, end, shard_rows) &&
        take_u32(at, end, shard_count) && take_u32(at, end, source_count) &&
        take_u64(at, end, payload_stride) && take_u64(at, end, payload_bytes) &&
        take_u32(at, end, source_bytes) && take_u32(at, end, reserved);
    auto reject = [&]() -> Result<bool> {
        std::fprintf(stderr, "PLE artifact is stale or invalid; rebuilding\n");
        ::close(fd);
        return false;
    };
    if (!parsed || version != kPleArtifactVersion ||
        header_bytes != kPleArtifactHeaderBytes || format != kPleArtifactFormatInt6 ||
        source_format != kPleArtifactSourceBf16 || source_bytes != kPleArtifactSourceBytes ||
        row_width != kExpectedRowWidth || row_bytes != 130u ||
        shard_count != config.ple.split_ngram_parts || source_count != sources.size() ||
        shard_rows == 0u || shard_count == 0u ||
        static_cast<std::uint64_t>(shard_rows) * shard_count != rows ||
        payload_bytes != static_cast<std::uint64_t>(shard_rows) * row_bytes ||
        payload_stride < payload_bytes || (payload_stride % 4096u) != 0u) {
        return reject();
    }
    auto padded_rows = qwen4exp_ple_padded_rows(config, 0u);
    if (!padded_rows.ok() || padded_rows.value() != rows) return reject();
    const std::uint64_t required = static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
                                   static_cast<std::uint64_t>(shard_count - 1u) *
                                       payload_stride +
                                   payload_bytes;
    if (static_cast<std::uint64_t>(st.st_size) < required) return reject();
    for (std::size_t index = 0; index < sources.size(); ++index) {
        std::uint32_t id = 0;
        std::uint32_t pad = 0;
        std::uint64_t size = 0;
        std::int64_t sec = 0;
        std::int64_t nsec = 0;
        if (!take_u32(at, end, id) || !take_u32(at, end, pad) ||
            !take_u64(at, end, size) || !take_i64(at, end, sec) ||
            !take_i64(at, end, nsec) ||
            id != sources[index].id || size != sources[index].size ||
            sec != sources[index].mtime_sec || nsec != sources[index].mtime_nsec) {
            return reject();
        }
    }
    const auto head_sizes = qwen4exp_ple_head_vocab_sizes(config, 0u);
    const auto head_offsets = qwen4exp_ple_head_offsets(config, 0u);
    const auto multipliers = qwen4exp_ple_layer_multipliers(config, 0u);
    if (!head_sizes.ok() || !head_offsets.ok() || !multipliers.ok()) return reject();

    PleTable loaded;
    loaded.file_path_ = path;
    loaded.format_ = TableFormat::Int6W32Fp16;
    loaded.rows_ = rows;
    loaded.row_width_ = row_width;
    loaded.row_bytes_ = row_bytes;
    loaded.k_bits_ = 0u;
    loaded.shard_count_ = shard_count;
    loaded.shard_rows_ = shard_rows;
    loaded.ngram_size_ = static_cast<std::uint32_t>(config.ple.ngram_size);
    loaded.heads_per_ngram_ = static_cast<std::uint32_t>(config.ple.heads_per_ngram);
    loaded.eos_token_id_ = config.ple.eos_token_id;
    loaded.head_offsets_.assign(head_offsets.value().begin(), head_offsets.value().end());
    loaded.head_vocab_sizes_.assign(head_sizes.value().begin(), head_sizes.value().end());
    loaded.layer_multipliers_.assign(multipliers.value().begin(), multipliers.value().end());
    loaded.payload_bytes_ = static_cast<std::uint64_t>(shard_count) * payload_bytes;
    loaded.shards_.resize(shard_count);

    if (hot_cache_bytes != 0u) {
        std::vector<std::uint64_t> shard_offsets(shard_count);
        for (std::uint32_t id = 0; id < shard_count; ++id) {
            shard_offsets[id] = static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
                                static_cast<std::uint64_t>(id) * payload_stride;
        }
        Status tiered = setup_tiered(loaded, fd, std::move(shard_offsets), shard_rows,
                                     static_cast<std::size_t>(hot_cache_bytes));
        ::close(fd);
        if (!tiered.ok()) {
            std::fprintf(stderr, "PLE tiered cache setup failed: %s\n",
                         tiered.message().c_str());
            return false;
        }
        out = std::move(loaded);
        return true;
    }

    const std::uint32_t worker_count = std::min<std::uint32_t>(16u, shard_count);
    std::vector<Status> errors(worker_count, Status::make_ok());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            for (std::uint32_t id = worker; id < shard_count; id += worker_count) {
                void* memory = nullptr;
                if (::posix_memalign(&memory, 64u, static_cast<std::size_t>(payload_bytes)) != 0) {
                    errors[worker] = Status::insufficient_memory(
                        "cannot allocate PLE artifact shard", __FILE__, __LINE__);
                    return;
                }
                auto* dst = static_cast<std::uint8_t*>(memory);
                const std::uint64_t offset =
                    static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
                    static_cast<std::uint64_t>(id) * payload_stride;
                Status st = read_exact_at(fd, offset, dst,
                                          static_cast<std::size_t>(payload_bytes));
                if (!st.ok()) {
                    std::free(memory);
                    errors[worker] = st;
                    return;
                }
                (void)::posix_fadvise(fd, static_cast<off_t>(offset),
                                      static_cast<off_t>(payload_bytes),
                                      POSIX_FADV_DONTNEED);
                loaded.shards_[id].data.reset(dst);
                loaded.shards_[id].bytes = static_cast<std::size_t>(payload_bytes);
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    ::close(fd);
    for (const Status& error : errors) {
        if (!error.ok()) return false;
    }
    out = std::move(loaded);
    return true;
}

Status PleTable::write_int6_artifact(const std::string& model_dir,
                                     const std::vector<ArtifactSource>& sources,
                                     const PleTable& table) {
    if (table.format_ != TableFormat::Int6W32Fp16 || sources.empty() ||
        table.shard_count_ != sources.size() || table.shard_rows_ == 0u) {
        return Status::make_ok();
    }
    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(table.shard_rows_) * table.row_bytes_;
    const std::uint64_t payload_stride = align_up_u64(payload_bytes, 4096u);
    const std::uint64_t required = static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
                                   static_cast<std::uint64_t>(table.shard_count_ - 1u) *
                                       payload_stride +
                                   payload_bytes;
    struct statvfs vfs {};
    if (::statvfs(model_dir.c_str(), &vfs) == 0) {
        const std::uint64_t free_bytes =
            static_cast<std::uint64_t>(vfs.f_bavail) * vfs.f_frsize;
        constexpr std::uint64_t kMargin = 64ull * 1024ull * 1024ull;
        if (free_bytes < required + kMargin) {
            return Status::insufficient_memory(
                "not enough free space for PLE artifact", __FILE__, __LINE__);
        }
    }
    std::vector<std::uint8_t> header(kPleArtifactHeaderBytes, 0u);
    static constexpr char kMagic[8] = {'S', 'H', 'E', 'L', 'Q', 'P', 'L', '1'};
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    std::size_t at = sizeof(kMagic);
    append_u32(header, at, kPleArtifactVersion);
    append_u32(header, at, kPleArtifactHeaderBytes);
    append_u32(header, at, kPleArtifactFormatInt6);
    append_u32(header, at, kPleArtifactSourceBf16);
    append_u64(header, at, table.rows_);
    append_u32(header, at, table.row_width_);
    append_u32(header, at, table.row_bytes_);
    append_u32(header, at, table.shard_rows_);
    append_u32(header, at, table.shard_count_);
    append_u32(header, at, static_cast<std::uint32_t>(sources.size()));
    append_u64(header, at, payload_stride);
    append_u64(header, at, payload_bytes);
    append_u32(header, at, kPleArtifactSourceBytes);
    append_u32(header, at, 0u);
    for (const ArtifactSource& source : sources) {
        append_u32(header, at, source.id);
        append_u32(header, at, 0u);
        append_u64(header, at, source.size);
        append_i64(header, at, source.mtime_sec);
        append_i64(header, at, source.mtime_nsec);
    }
    const std::string path = ple_artifact_path(model_dir);
    std::string temp = path + ".tmp.XXXXXX";
    std::vector<char> temp_name(temp.begin(), temp.end());
    temp_name.push_back('\0');
    const int fd = ::mkstemp(temp_name.data());
    if (fd < 0) {
        return Status::invalid_argument(
            (std::string("cannot create PLE artifact: ") + std::strerror(errno)).c_str(),
            __FILE__, __LINE__);
    }
    (void)::fchmod(fd, 0644);
    auto cleanup = [&]() {
        ::close(fd);
        (void)::unlink(temp_name.data());
    };
    Status st = pwrite_exact(fd, header.data(), header.size(), 0);
    if (!st.ok()) {
        cleanup();
        return st;
    }
    for (std::uint32_t id = 0; id < table.shard_count_; ++id) {
        const Shard& shard = table.shards_[id];
        if (shard.data == nullptr || shard.bytes != payload_bytes) {
            cleanup();
            return Status::invalid_state("PLE artifact shard is incomplete", __FILE__,
                                         __LINE__);
        }
        const std::uint64_t offset =
            static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
            static_cast<std::uint64_t>(id) * payload_stride;
        st = pwrite_exact(fd, shard.data.get(), shard.bytes, offset);
        if (!st.ok()) {
            cleanup();
            return st;
        }
        if (((id + 1u) % 8u) == 0u || (id + 1u) == table.shard_count_) {
            if (::fdatasync(fd) != 0) {
                const Status sync_error = Status::invalid_argument(
                    (std::string("PLE artifact fdatasync failed: ") +
                     std::strerror(errno)).c_str(),
                    __FILE__, __LINE__);
                cleanup();
                return sync_error;
            }
            const std::uint32_t begin_id = id - (id % 8u);
            const std::uint64_t begin =
                static_cast<std::uint64_t>(kPleArtifactHeaderBytes) +
                static_cast<std::uint64_t>(begin_id) * payload_stride;
            const std::uint64_t end = offset + static_cast<std::uint64_t>(shard.bytes);
            (void)::posix_fadvise(fd, static_cast<off_t>(begin),
                                  static_cast<off_t>(end - begin),
                                  POSIX_FADV_DONTNEED);
        }
    }
    if (::fsync(fd) != 0) {
        const Status sync_error = Status::invalid_argument(
            (std::string("PLE artifact fsync failed: ") + std::strerror(errno)).c_str(),
            __FILE__, __LINE__);
        cleanup();
        return sync_error;
    }
    if (::close(fd) != 0) {
        (void)::unlink(temp_name.data());
        return Status::invalid_argument("PLE artifact close failed", __FILE__, __LINE__);
    }
    if (::rename(temp_name.data(), path.c_str()) != 0) {
        (void)::unlink(temp_name.data());
        return Status::invalid_argument(
            (std::string("PLE artifact rename failed: ") + std::strerror(errno)).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<PleTable> PleTable::load(const std::string& model_dir,
                                const Qwen4ExpTextConfig& config,
                                bool fadvise_dontneed) {
    const std::uint64_t hot_cache_bytes = ple_hot_cache_bytes();
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return load_index_shards(model_dir, config, fadvise_dontneed);
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const std::string message = std::strerror(errno);
        ::close(fd);
        return Status::invalid_argument(
            (std::string("cannot stat PLE table: ") + message).c_str(), __FILE__, __LINE__);
    }
    const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);

    nlohmann::json header;
    std::uint64_t header_len = 0;
    auto header_result = read_json_header(fd, file_size, header, header_len);
    if (!header_result.ok()) {
        ::close(fd);
        return header_result.status();
    }
    if (!header.contains("__metadata__") || !header["__metadata__"].is_object()) {
        ::close(fd);
        return Status::invalid_argument("PLE table has no metadata", __FILE__, __LINE__);
    }
    const nlohmann::json metadata = header["__metadata__"];
    header.erase("__metadata__");

    auto format = json_string(metadata, "format");
    if (!format.ok()) {
        ::close(fd);
        return format.status();
    }
    if (format.value() != "exl3_ngram_trellis") {
        ::close(fd);
        return Status::unsupported("PLE table is not exl3_ngram_trellis", __FILE__, __LINE__);
    }
    auto codebook = json_string(metadata, "codebook");
    if (!codebook.ok()) {
        ::close(fd);
        return codebook.status();
    }
    if (codebook.value() != "mul1") {
        ::close(fd);
        return Status::unsupported("PLE table codebook is not mul1", __FILE__, __LINE__);
    }

    auto k_result = json_u64(metadata, "K");
    auto row_dim_result = json_u64(metadata, "row_dim");
    auto rows_result = json_u64(metadata, "rows");
    auto shard_rows_result = json_u64(metadata, "shard_rows");
    if (!k_result.ok() || !row_dim_result.ok() || !rows_result.ok() ||
        !shard_rows_result.ok()) {
        ::close(fd);
        return Status::invalid_argument("PLE table metadata is incomplete", __FILE__, __LINE__);
    }
    const std::uint32_t k_bits = static_cast<std::uint32_t>(k_result.value());
    const std::uint32_t row_width = static_cast<std::uint32_t>(row_dim_result.value());
    const std::uint64_t rows = rows_result.value();
    const std::uint32_t shard_rows = static_cast<std::uint32_t>(shard_rows_result.value());
    if (k_bits < 1u || k_bits > 16u || row_width != kExpectedRowWidth) {
        ::close(fd);
        return Status::unsupported("PLE table has an unsupported trellis geometry", __FILE__,
                                   __LINE__);
    }
    if ((row_width * k_bits) % 16u != 0u) {
        ::close(fd);
        return Status::invalid_argument("PLE table row is not word aligned", __FILE__,
                                        __LINE__);
    }
    const std::uint32_t row_words = 1u + row_width * k_bits / 16u;
    const std::uint32_t row_bytes = row_words * 2u;

    auto padded_rows = qwen4exp_ple_padded_rows(config, 0u);
    if (!padded_rows.ok()) {
        ::close(fd);
        return padded_rows.status();
    }
    if (rows != padded_rows.value()) {
        ::close(fd);
        return Status::invalid_argument("PLE table row count differs from the config",
                                        __FILE__, __LINE__);
    }
    if (shard_rows == 0u || (rows % shard_rows) != 0u) {
        ::close(fd);
        return Status::invalid_argument("PLE table shard row count is invalid", __FILE__,
                                        __LINE__);
    }
    const std::uint32_t shard_count = static_cast<std::uint32_t>(rows / shard_rows);
    if (shard_count != config.ple.split_ngram_parts) {
        ::close(fd);
        return Status::invalid_argument(
            "PLE table shard count differs from split_ngram_parts", __FILE__, __LINE__);
    }

    auto head_sizes_result = qwen4exp_ple_head_vocab_sizes(config, 0u);
    auto head_offsets_result = qwen4exp_ple_head_offsets(config, 0u);
    auto multipliers_result = qwen4exp_ple_layer_multipliers(config, 0u);
    if (!head_sizes_result.ok() || !head_offsets_result.ok() || !multipliers_result.ok()) {
        ::close(fd);
        return Status::invalid_argument("cannot derive PLE hash metadata from config", __FILE__,
                                        __LINE__);
    }
    const std::vector<std::size_t>& expected_sizes = head_sizes_result.value();
    const std::vector<std::size_t>& expected_offsets = head_offsets_result.value();
    const std::vector<std::int64_t>& expected_multipliers = multipliers_result.value();

    const std::vector<std::size_t> ple_layers = config.ple_layers();
    if (ple_layers.size() != 1u) {
        ::close(fd);
        return Status::unsupported(
            "PLE table load expects exactly one ple_layer_ids entry", __FILE__, __LINE__);
    }
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(ple_layers[0]) +
                               ".ple.ple_embedding.ngram_embedding.";
    auto find_tensor = [&](const std::string& name, TensorRecord& out) -> Status {
        auto it = header.find(name);
        if (it == header.end()) {
            return Status::invalid_argument(
                (std::string("PLE tensor is missing: ") + name).c_str(), __FILE__, __LINE__);
        }
        auto parsed = parse_tensor(it.value());
        if (!parsed.ok()) return parsed.status();
        out = parsed.release();
        return Status::make_ok();
    };

    TensorRecord head_bias_record;
    TensorRecord head_offsets_record;
    TensorRecord head_sizes_record;
    TensorRecord multipliers_record;
    Status ts = find_tensor(prefix + "head_bias", head_bias_record);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = find_tensor(prefix + "head_offsets", head_offsets_record);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = find_tensor(prefix + "head_vocab_sizes", head_sizes_record);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = find_tensor(prefix + "layer_multipliers", multipliers_record);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }

    const std::size_t head_count = expected_sizes.size();
    if (head_bias_record.dtype != "F16" || head_bias_record.shape.size() != 2 ||
        head_bias_record.shape[0] != head_count ||
        head_bias_record.shape[1] != row_width) {
        ::close(fd);
        return Status::invalid_argument("PLE head_bias shape is invalid", __FILE__, __LINE__);
    }
    if (head_offsets_record.dtype != "I64" || head_offsets_record.shape.size() != 1 ||
        head_offsets_record.shape[0] != head_count ||
        head_sizes_record.dtype != "I64" || head_sizes_record.shape.size() != 1 ||
        head_sizes_record.shape[0] != head_count) {
        ::close(fd);
        return Status::invalid_argument("PLE head metadata shape is invalid", __FILE__,
                                        __LINE__);
    }
    if (multipliers_record.dtype != "I64" || multipliers_record.shape.size() != 1 ||
        multipliers_record.shape[0] != config.ple.ngram_size) {
        ::close(fd);
        return Status::invalid_argument("PLE layer_multipliers shape is invalid", __FILE__,
                                        __LINE__);
    }
    ts = expect_tensor_bytes(head_bias_record,
                             static_cast<std::uint64_t>(head_count) * row_width * 2u);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = expect_tensor_bytes(head_offsets_record,
                             static_cast<std::uint64_t>(head_count) * 8u);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = expect_tensor_bytes(head_sizes_record,
                             static_cast<std::uint64_t>(head_count) * 8u);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    ts = expect_tensor_bytes(multipliers_record,
                             static_cast<std::uint64_t>(config.ple.ngram_size) * 8u);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    const std::uint64_t data_base = 8u + header_len;
    if (head_bias_record.end > file_size - data_base ||
        head_offsets_record.end > file_size - data_base ||
        head_sizes_record.end > file_size - data_base ||
        multipliers_record.end > file_size - data_base) {
        ::close(fd);
        return Status::invalid_argument("PLE metadata is outside the file", __FILE__,
                                        __LINE__);
    }

    PleTable table;
    table.file_path_ = path;
    table.data_base_ = 8u + header_len;
    table.rows_ = rows;
    table.row_width_ = row_width;
    table.row_bytes_ = row_bytes;
    table.k_bits_ = k_bits;
    table.shard_count_ = shard_count;
    table.shard_rows_ = shard_rows;
    table.ngram_size_ = static_cast<std::uint32_t>(config.ple.ngram_size);
    table.heads_per_ngram_ = static_cast<std::uint32_t>(config.ple.heads_per_ngram);
    table.eos_token_id_ = config.ple.eos_token_id;
    table.payload_bytes_ = static_cast<std::uint64_t>(shard_count) * shard_rows * row_bytes;

    table.head_bias_f16_.resize(head_count * row_width);
    ts = read_exact_at(fd, table.data_base_ + head_bias_record.begin,
                       table.head_bias_f16_.data(), table.head_bias_f16_.size() * 2u);
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    table.head_offsets_.resize(head_count);
    ts = read_exact_at(fd, table.data_base_ + head_offsets_record.begin,
                       table.head_offsets_.data(), head_count * sizeof(std::int64_t));
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    table.head_vocab_sizes_.resize(head_count);
    ts = read_exact_at(fd, table.data_base_ + head_sizes_record.begin,
                       table.head_vocab_sizes_.data(), head_count * sizeof(std::int64_t));
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }
    table.layer_multipliers_.resize(config.ple.ngram_size);
    ts = read_exact_at(fd, table.data_base_ + multipliers_record.begin,
                       table.layer_multipliers_.data(),
                       table.layer_multipliers_.size() * sizeof(std::int64_t));
    if (!ts.ok()) {
        ::close(fd);
        return ts;
    }

    for (std::size_t i = 0; i < head_count; ++i) {
        if (static_cast<std::size_t>(table.head_vocab_sizes_[i]) != expected_sizes[i] ||
            static_cast<std::size_t>(table.head_offsets_[i]) != expected_offsets[i]) {
            ::close(fd);
            return Status::invalid_argument(
                "PLE head vocab sizes/offsets differ from the config", __FILE__, __LINE__);
        }
    }
    if (table.layer_multipliers_ != expected_multipliers) {
        ::close(fd);
        return Status::invalid_argument(
            "PLE layer multipliers differ from the config-derived values", __FILE__,
            __LINE__);
    }

    const std::string shard_regex =
        "^model\\.language_model\\.layers\\." + std::to_string(ple_layers[0]) +
        "\\.ple\\.ple_embedding\\.ngram_embedding\\.shard_([0-9]+)\\.trellis$";
    const std::regex shard_pattern(shard_regex);
    std::vector<std::pair<std::uint32_t, std::string>> shard_names;
    for (auto it = header.begin(); it != header.end(); ++it) {
        std::smatch match;
        if (std::regex_match(it.key(), match, shard_pattern)) {
            shard_names.emplace_back(static_cast<std::uint32_t>(std::stoul(match[1].str())),
                                     it.key());
        }
    }
    std::sort(shard_names.begin(), shard_names.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    if (shard_names.size() != shard_count) {
        ::close(fd);
        return Status::invalid_argument("PLE shard tensor count differs from the table",
                                        __FILE__, __LINE__);
    }
    for (std::uint32_t index = 0; index < shard_count; ++index) {
        if (shard_names[index].first != index) {
            ::close(fd);
            return Status::invalid_argument("PLE shard indices are not contiguous", __FILE__,
                                            __LINE__);
        }
    }

    table.shards_.resize(shard_count);
    std::vector<std::uint64_t> shard_offsets(shard_count);
    for (std::uint32_t index = 0; index < shard_count; ++index) {
        auto it = header.find(shard_names[index].second);
        auto parsed = parse_tensor(it.value());
        if (!parsed.ok()) {
            ::close(fd);
            return parsed.status();
        }
        const TensorRecord& record = parsed.value();
        if (record.dtype != "I16" || record.shape.size() != 2 ||
            record.shape[0] != shard_rows || record.shape[1] != row_words) {
            ::close(fd);
            return Status::invalid_argument("PLE shard tensor shape is invalid", __FILE__,
                                            __LINE__);
        }
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(shard_rows) * row_bytes;
        ts = expect_tensor_bytes(record, bytes);
        if (!ts.ok()) {
            ::close(fd);
            return ts;
        }
        if (record.end > file_size - table.data_base_) {
            ::close(fd);
            return Status::invalid_argument("PLE shard is outside the file", __FILE__,
                                            __LINE__);
        }
        shard_offsets[index] = table.data_base_ + record.begin;
        if (hot_cache_bytes != 0u) continue;

        void* memory = nullptr;
        if (::posix_memalign(&memory, 64u, static_cast<std::size_t>(bytes)) != 0) {
            ::close(fd);
            return Status::insufficient_memory(
                "cannot allocate PLE shard in host RAM", __FILE__, __LINE__);
        }
        table.shards_[index].data.reset(static_cast<std::uint8_t*>(memory));
        table.shards_[index].bytes = static_cast<std::size_t>(bytes);
        table.shards_[index].file_offset = shard_offsets[index];
        ts = read_exact_at(fd, table.shards_[index].file_offset,
                           table.shards_[index].data.get(),
                           table.shards_[index].bytes);
        if (!ts.ok()) {
            ::close(fd);
            return ts;
        }
        if (fadvise_dontneed) {
            (void)::posix_fadvise(fd, static_cast<off_t>(table.shards_[index].file_offset),
                                  static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
        }
    }

    if (hot_cache_bytes != 0u) {
        Status tiered = setup_tiered(table, fd, std::move(shard_offsets), shard_rows,
                                     static_cast<std::size_t>(hot_cache_bytes));
        ::close(fd);
        if (!tiered.ok()) return tiered;
        return table;
    }

    ::close(fd);
    return table;
}

std::uint64_t PleTable::host_bytes() const noexcept {
    return tiered_ != nullptr ? tiered_->hot_bytes : payload_bytes_;
}

std::size_t PleTable::cached_blocks() const noexcept {
    if (tiered_ == nullptr) return 0u;
    TieredState& state = *tiered_;
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.slot_block.size() - static_cast<std::size_t>(
        std::count(state.slot_block.begin(), state.slot_block.end(), -1));
}

Status PleTable::setup_tiered(PleTable& table, int fd,
                              std::vector<std::uint64_t> shard_offsets,
                              std::uint32_t shard_rows, std::size_t hot_bytes) {
    struct stat mapped_stat {};
    if (::fstat(fd, &mapped_stat) != 0) {
        return Status::invalid_state("PLE artifact stat failed", __FILE__, __LINE__);
    }
    auto state = std::make_shared<TieredState>();
    state->mapped_bytes = static_cast<std::size_t>(mapped_stat.st_size);
    void* base = ::mmap(nullptr, state->mapped_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        return Status::invalid_state("PLE artifact mmap failed", __FILE__, __LINE__);
    }
    state->mapped = std::shared_ptr<const std::uint8_t>(
        static_cast<const std::uint8_t*>(base), MappedDeleter{state->mapped_bytes});

    const int direct = ::open(table.file_path_.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    const int fill = direct >= 0 ? direct : ::open(table.file_path_.c_str(),
                                                   O_RDONLY | O_CLOEXEC);
    if (fill < 0) {
        return Status::invalid_state("PLE artifact reopen failed", __FILE__, __LINE__);
    }
    state->direct_fd = std::shared_ptr<int>(new int(fill), FdDeleter{});

    const std::uint64_t block_bytes =
        static_cast<std::uint64_t>(kHotBlockRows) * table.row_bytes_;
    const std::size_t slots = hot_bytes / static_cast<std::size_t>(block_bytes);
    if (slots == 0u) {
        return Status::invalid_argument("PLE hot cache is smaller than one block", __FILE__,
                                        __LINE__);
    }
    void* pinned = nullptr;
    const hipError_t error =
        ::hipHostMalloc(&pinned, slots * static_cast<std::size_t>(block_bytes),
                        hipHostMallocMapped);
    if (error != hipSuccess || pinned == nullptr) {
        return Status::hip_error("hipHostMalloc PLE hot cache", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    }
    if ((reinterpret_cast<std::uintptr_t>(pinned) & 4095u) != 0u) {
        (void)::hipHostFree(pinned);
        return Status::invalid_state("PLE hot cache is not page aligned", __FILE__, __LINE__);
    }
    state->hot = std::shared_ptr<std::uint8_t>(static_cast<std::uint8_t*>(pinned),
                                               PinnedDeleter{});
    void* scratch = nullptr;
    const std::size_t scratch_bytes = static_cast<std::size_t>(
        align_up_u64(4096u + block_bytes, 4096u));
    if (::posix_memalign(&scratch, 4096u, scratch_bytes) != 0) {
        return Status::insufficient_memory("cannot allocate PLE block scratch", __FILE__,
                                           __LINE__);
    }
    state->scratch.reset(static_cast<std::uint8_t*>(scratch));
    state->hot_bytes = static_cast<std::uint64_t>(slots) * block_bytes;
    state->hot_slots = slots;
    state->shard_offsets = std::move(shard_offsets);
    state->block_rows = kHotBlockRows;
    state->block_bytes = block_bytes;
    state->blocks_per_shard = (shard_rows + kHotBlockRows - 1u) / kHotBlockRows;
    state->slot_block.assign(slots, -1);
    const std::uint64_t total_blocks =
        static_cast<std::uint64_t>(table.shard_count_) * state->blocks_per_shard;
    state->freq.assign(static_cast<std::size_t>(total_blocks), 0u);
    if (const char* interval = std::getenv("SHELF_PLE_CACHE_LOG_INTERVAL")) {
        state->log_interval = std::max<std::uint32_t>(
            static_cast<std::uint32_t>(std::strtoul(interval, nullptr, 10)), 1u);
    }
    const std::uint64_t resident_bytes = state->hot_bytes;
    const std::size_t hot_slots = state->hot_slots;
    table.tiered_ = std::move(state);
    std::fprintf(stderr, "PLE tiered cache: hot %.2f GiB (%zu blocks, %llu KiB each)\n",
                 static_cast<double>(resident_bytes) / 1073741824.0, hot_slots,
                 static_cast<unsigned long long>(block_bytes / 1024u));
    return Status::make_ok();
}

std::vector<std::uint64_t> PleTable::tiered_candidates(const TieredState& state,
                                                       std::size_t count,
                                                       std::uint32_t shard_rows) {
    std::vector<std::uint64_t> result;
    if (count == 0u) return result;
    using Entry = std::pair<std::uint32_t, std::uint64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    const std::uint32_t last = state.blocks_per_shard - 1u;
    const bool partial = (shard_rows % state.block_rows) != 0u;
    for (std::uint64_t block = 0; block < state.freq.size(); ++block) {
        const std::uint32_t frequency = state.freq[static_cast<std::size_t>(block)];
        if (frequency == 0u || state.block_slot.count(block) != 0u) continue;
        if (partial &&
            static_cast<std::uint32_t>(block % state.blocks_per_shard) == last) {
            continue;
        }
        const std::uint64_t shard = block / state.blocks_per_shard;
        const std::uint64_t index = block % state.blocks_per_shard;
        const std::uint64_t offset =
            state.shard_offsets[static_cast<std::size_t>(shard)] +
            index * state.block_bytes;
        const std::uint64_t aligned = offset & ~std::uint64_t(4095u);
        const std::uint64_t length =
            align_up_u64((offset - aligned) + state.block_bytes, 4096u);
        if (aligned + length > state.mapped_bytes) continue;
        if (heap.size() < count) {
            heap.push({frequency, block});
        } else if (frequency > heap.top().first) {
            heap.pop();
            heap.push({frequency, block});
        }
    }
    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }
    return result;
}

void PleTable::fill_tiered_block(TieredState& state, std::size_t slot,
                                 std::uint64_t block) {
    const std::uint64_t shard = block / state.blocks_per_shard;
    const std::uint64_t index = block % state.blocks_per_shard;
    const std::uint64_t offset =
        state.shard_offsets[static_cast<std::size_t>(shard)] +
        index * state.block_bytes;
    const std::uint64_t aligned = offset & ~std::uint64_t(4095u);
    const std::size_t delta = static_cast<std::size_t>(offset - aligned);
    const std::int64_t old = state.slot_block[slot];
    if (old >= 0) state.block_slot.erase(static_cast<std::uint64_t>(old));
    state.slot_block[slot] = -1;
    void* dst = state.hot.get() + slot * static_cast<std::size_t>(state.block_bytes);
    Status st = Status::make_ok();
    if (delta == 0u) {
        st = read_exact_at(*state.direct_fd, aligned, dst,
                           static_cast<std::size_t>(state.block_bytes));
    } else {
        const std::size_t length = static_cast<std::size_t>(
            align_up_u64(delta + state.block_bytes, 4096u));
        st = read_exact_at(*state.direct_fd, aligned, state.scratch.get(), length);
        if (st.ok()) {
            std::memcpy(dst, state.scratch.get() + delta,
                        static_cast<std::size_t>(state.block_bytes));
        }
    }
    if (!st.ok()) return;
    state.slot_block[slot] = static_cast<std::int64_t>(block);
    state.block_slot[block] = static_cast<std::uint32_t>(slot);
}

void PleTable::rebalance_tiered(TieredState& state, std::uint32_t shard_rows) {
    if (state.log_interval != 0u && state.accesses != 0u &&
        (state.accesses / kHotRebalanceAccesses) % state.log_interval == 0u) {
        const std::size_t occupied = state.slot_block.size() - static_cast<std::size_t>(
            std::count(state.slot_block.begin(), state.slot_block.end(), -1));
        std::fprintf(stderr,
                     "PLE tiered cache: hot blocks %zu/%zu hits=%llu misses=%llu\n",
                     occupied, state.hot_slots,
                     static_cast<unsigned long long>(state.hot_hits),
                     static_cast<unsigned long long>(state.cold_reads));
    }
    std::uint32_t peak = 0u;
    for (std::uint32_t frequency : state.freq) peak = std::max(peak, frequency);
    if (peak >= kHotDecayAt) {
        for (std::uint32_t& frequency : state.freq) {
            frequency = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(frequency) * 7u / 10u);
        }
    }
    struct Cached {
        std::size_t slot = 0;
        std::uint64_t block = 0;
        std::uint32_t frequency = 0;
    };
    std::vector<Cached> cached;
    cached.reserve(state.hot_slots);
    for (std::size_t slot = 0; slot < state.slot_block.size(); ++slot) {
        const std::int64_t block = state.slot_block[slot];
        if (block >= 0) {
            cached.push_back({slot, static_cast<std::uint64_t>(block),
                              state.freq[static_cast<std::size_t>(block)]});
        }
    }
    if (cached.size() < state.hot_slots) {
        const std::size_t fill_count = std::min<std::size_t>(
            static_cast<std::size_t>(kHotSwapBudget), state.hot_slots - cached.size());
        const auto candidates = tiered_candidates(state, fill_count, shard_rows);
        for (std::uint64_t block : candidates) {
            const auto free_slot =
                std::find(state.slot_block.begin(), state.slot_block.end(), -1);
            if (free_slot == state.slot_block.end()) break;
            fill_tiered_block(state,
                              static_cast<std::size_t>(free_slot - state.slot_block.begin()),
                              block);
        }
        return;
    }
    std::sort(cached.begin(), cached.end(),
              [](const Cached& a, const Cached& b) { return a.frequency < b.frequency; });
    const std::size_t swaps = std::min<std::size_t>(kHotSwapBudget, cached.size());
    for (std::size_t index = 0; index < swaps; ++index) {
        const auto candidates = tiered_candidates(state, 1u, shard_rows);
        if (candidates.empty()) break;
        const std::uint64_t block = candidates.front();
        if (state.freq[static_cast<std::size_t>(block)] <=
            cached[index].frequency + kHotMinGain) {
            break;
        }
        fill_tiered_block(state, cached[index].slot, block);
    }
}

const std::uint8_t* PleTable::row_ptr(std::uint64_t row) const noexcept {
    if (row >= rows_ || shard_rows_ == 0u) return nullptr;
    const std::uint64_t shard = row / shard_rows_;
    const std::uint64_t local = row % shard_rows_;
    if (tiered_ == nullptr) {
        if (shard >= shards_.size()) return nullptr;
        return shards_[static_cast<std::size_t>(shard)].data.get() +
               static_cast<std::size_t>(local) * row_bytes_;
    }
    if (row_bytes_ > 4096u) return nullptr;
    TieredState& state = *tiered_;
    std::lock_guard<std::mutex> lock(state.mutex);
    const std::uint64_t block_in_shard = local / state.block_rows;
    const std::uint64_t row_in_block = local % state.block_rows;
    const std::uint64_t block = shard * state.blocks_per_shard + block_in_shard;
    if (block >= state.freq.size()) return nullptr;
    std::uint32_t& frequency = state.freq[static_cast<std::size_t>(block)];
    if (frequency != UINT32_MAX) ++frequency;
    ++state.accesses;
    if (state.accesses % kHotRebalanceAccesses == 0u) {
        rebalance_tiered(state, shard_rows_);
    }
    const std::uint8_t* source = nullptr;
    const auto found = state.block_slot.find(block);
    if (found != state.block_slot.end()) {
        ++state.hot_hits;
        source = state.hot.get() +
                 static_cast<std::size_t>(found->second) *
                     static_cast<std::size_t>(state.block_bytes) +
                 static_cast<std::size_t>(row_in_block) * row_bytes_;
    } else {
        ++state.cold_reads;
        const std::uint64_t offset =
            state.shard_offsets[static_cast<std::size_t>(shard)] +
            block_in_shard * state.block_bytes +
            static_cast<std::uint64_t>(row_in_block) * row_bytes_;
        if (offset + row_bytes_ > state.mapped_bytes) return nullptr;
        source = state.mapped.get() + static_cast<std::size_t>(offset);
    }
    static thread_local std::uint8_t staging[4096];
    std::memcpy(staging, source, row_bytes_);
    return staging;
}

std::size_t PleTable::head_for_row(std::uint64_t row) const noexcept {
    const std::size_t heads = head_offsets_.size();
    if (heads == 0u) return 0u;
    auto upper = std::upper_bound(head_offsets_.begin(), head_offsets_.end(),
                                  static_cast<std::int64_t>(row));
    std::size_t head = 0u;
    if (upper != head_offsets_.begin()) {
        head = static_cast<std::size_t>(upper - head_offsets_.begin() - 1);
    }
    if (head >= heads) head = heads - 1u;
    return head;
}

Result<std::vector<float>> PleTable::decode_row(std::uint64_t row,
                                                bool include_head_bias) const {
    const std::uint8_t* packed = row_ptr(row);
    if (packed == nullptr) {
        return Status::out_of_range("PLE row is out of range", __FILE__, __LINE__);
    }
    if (format_ == TableFormat::Fp8E4M3) {
        std::vector<float> out(row_width_);
        const std::size_t head = head_for_row(row);
        for (std::uint32_t i = 0; i < row_width_; ++i) {
            float value = e4m3_to_f32(packed[i]) * fp8_scale_;
            if (include_head_bias && !head_bias_f16_.empty())
                value += shelf::quantization::exl3::f32_from_f16_bits(
                    head_bias_f16_[head * row_width_ + i]);
            out[i] = value;
        }
        return out;
    }
    if (format_ == TableFormat::Int6W32Fp16) {
        std::vector<float> out(row_width_);
        shelf::quantization::int6::dequantize_row(packed, row_width_ / 32u, out.data());
        if (include_head_bias && !head_bias_f16_.empty()) {
            const std::size_t head = head_for_row(row);
            for (std::uint32_t i = 0; i < row_width_; ++i) {
                out[i] += shelf::quantization::exl3::f32_from_f16_bits(
                    head_bias_f16_[head * row_width_ + i]);
            }
        }
        return out;
    }
    const std::size_t head = head_for_row(row);

    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, packed, sizeof(scale_bits));
    const float scale = shelf::quantization::exl3::f32_from_f16_bits(scale_bits);

    std::vector<std::uint16_t> words(row_bytes_ / 2u - 1u);
    std::memcpy(words.data(), packed + 2u, words.size() * sizeof(std::uint16_t));
    const std::uint32_t stream_words = static_cast<std::uint32_t>(words.size());
    const std::uint32_t symbol_mask = (1u << k_bits_) - 1u;
    const std::uint32_t state_steps = (15u + k_bits_) / k_bits_;

    std::vector<float> out(row_width_);
    for (std::uint32_t i = 0; i < row_width_; ++i) {
        const std::uint32_t bit = i * k_bits_;
        const std::uint32_t word = bit >> 4u;
        const std::uint32_t window = static_cast<std::uint32_t>(words[word]) |
                                     (static_cast<std::uint32_t>(
                                          words[(word + 1u) % stream_words])
                                      << 16);
        const std::uint32_t symbol = (window >> (bit & 15u)) & symbol_mask;
        std::uint32_t state = symbol;
        for (std::uint32_t j = 1; j < state_steps; ++j) {
            const std::uint32_t previous = (i + row_width_ - j) % row_width_;
            const std::uint32_t previous_bit = previous * k_bits_;
            const std::uint32_t previous_word = previous_bit >> 4u;
            const std::uint32_t previous_window =
                static_cast<std::uint32_t>(words[previous_word]) |
                (static_cast<std::uint32_t>(
                     words[(previous_word + 1u) % stream_words])
                 << 16);
            const std::uint32_t previous_symbol =
                (previous_window >> (previous_bit & 15u)) & symbol_mask;
            state |= previous_symbol << (j * k_bits_);
        }
        const std::uint16_t codebook_bits =
            shelf::quantization::exl3::codebook_f16_bits(
                static_cast<std::uint16_t>(state & 0xFFFFu));
        const float value =
            shelf::quantization::exl3::f32_from_f16_bits(codebook_bits) * scale;
        const float bias = head_bias_f16_.empty()
            ? 0.0f
            : shelf::quantization::exl3::f32_from_f16_bits(
                  head_bias_f16_[head * row_width_ + i]);
        out[i] = value + (include_head_bias ? bias : 0.0f);
    }
    return out;
}

Result<std::vector<std::uint64_t>> PleTable::ngram_rows(
    const std::vector<std::int32_t>& history, std::size_t out_len) const {
    if (out_len == 0u || out_len > history.size()) {
        return Status::invalid_argument("PLE hash output length is invalid", __FILE__,
                                        __LINE__);
    }
    if (ngram_size_ < 2u || heads_per_ngram_ == 0u) {
        return Status::invalid_argument("PLE hash geometry is invalid", __FILE__, __LINE__);
    }
    const std::size_t head_count = head_offsets_.size();
    if (head_count != (ngram_size_ - 1u) * heads_per_ngram_) {
        return Status::invalid_argument("PLE head count differs from the hash geometry",
                                        __FILE__, __LINE__);
    }

    const std::size_t n = history.size();
    std::vector<std::uint64_t> rows;
    rows.reserve(out_len * head_count);
    const std::size_t first = n - out_len;
    for (std::uint32_t ngram = 2; ngram <= ngram_size_; ++ngram) {
        const std::size_t head_begin = (ngram - 2u) * heads_per_ngram_;
        const std::size_t head_end = head_begin + heads_per_ngram_;
        for (std::size_t position = first; position < n; ++position) {
            std::uint64_t mixed = 0u;
            for (std::uint32_t shift = 0; shift < ngram; ++shift) {
                std::int32_t token = eos_token_id_;
                if (position >= shift) {
                    bool in_segment = true;
                    for (std::size_t i = position - shift; i < position; ++i) {
                        if (history[i] == eos_token_id_) {
                            in_segment = false;
                            break;
                        }
                    }
                    if (in_segment) token = history[position - shift];
                }
                const std::uint64_t weighted =
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(token)) *
                    static_cast<std::uint64_t>(layer_multipliers_[shift]);
                if (shift == 0u) mixed = weighted;
                else mixed ^= weighted;
            }
            for (std::size_t head = head_begin; head < head_end; ++head) {
                const std::int64_t size = head_vocab_sizes_[head];
                if (size <= 0) {
                    return Status::invalid_argument(
                        "PLE head vocab size is not positive", __FILE__, __LINE__);
                }
                std::int64_t remainder = static_cast<std::int64_t>(mixed) % size;
                if (remainder < 0) remainder += size;
                rows.push_back(static_cast<std::uint64_t>(head_offsets_[head]) +
                               static_cast<std::uint64_t>(remainder));
            }
        }
    }
    return rows;
}

Result<std::vector<float>> PleTable::gather(const std::vector<std::int32_t>& history,
                                            std::size_t out_len) const {
    auto rows = ngram_rows(history, out_len);
    if (!rows.ok()) return rows.status();
    const std::size_t head_count = head_offsets_.size();
    std::vector<float> out;
    out.reserve(rows.value().size() * row_width_);
    for (const std::uint64_t row : rows.value()) {
        auto decoded = decode_row(row);
        if (!decoded.ok()) return decoded.status();
        out.insert(out.end(), decoded.value().begin(), decoded.value().end());
    }
    if (out.size() != out_len * head_count * row_width_) {
        return Status::invalid_argument("PLE gather size is inconsistent", __FILE__, __LINE__);
    }
    return out;
}

Result<std::vector<bf16_t>> PleTable::gather_bf16(
    const std::vector<std::int32_t>& history, std::size_t out_len) const {
    auto floats = gather(history, out_len);
    if (!floats.ok()) return floats.status();
    std::vector<bf16_t> out(floats.value().size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i].data = shelf::f32_to_bf16_rne(floats.value()[i]);
    return out;
}

Result<std::vector<bf16_t>> PleTable::gather_bf16_stateless(
    const std::vector<std::int32_t>& prefix_tail,
    std::span<const std::int32_t> tokens) const {
    const std::size_t head_count = head_offsets_.size();
    std::vector<std::int32_t> history(prefix_tail.begin(), prefix_tail.end());
    std::vector<bf16_t> out;
    out.reserve(tokens.size() * head_count * row_width_);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        history.push_back(tokens[t]);
        auto rows = gather_bf16(history, 1u);
        if (!rows.ok()) return rows.status();
        if (rows.value().size() != head_count * row_width_) {
            return Status::invalid_state("PLE stateless gather width mismatch", __FILE__,
                                         __LINE__);
        }
        out.insert(out.end(), rows.value().begin(), rows.value().end());
    }
    return out;
}

Result<std::vector<std::uint8_t>> PleTable::read_source_row(const std::string& model_dir,
                                                            std::uint64_t row) const {
    if (format_ != TableFormat::Exl3Trellis) {
        return Status::unsupported(
            "read_source_row needs the EXL3 packaged table", __FILE__, __LINE__);
    }
    if (row >= rows_ || shard_rows_ == 0u) {
        return Status::out_of_range("PLE row is out of range", __FILE__, __LINE__);
    }
    const std::uint64_t shard = row / shard_rows_;
    std::uint64_t shard_offset = 0u;
    if (tiered_ != nullptr) {
        TieredState& state = *tiered_;
        std::lock_guard<std::mutex> lock(state.mutex);
        if (shard >= state.shard_offsets.size()) {
            return Status::out_of_range("PLE shard is out of range", __FILE__, __LINE__);
        }
        shard_offset = state.shard_offsets[static_cast<std::size_t>(shard)];
    } else {
        if (shard >= shards_.size()) {
            return Status::out_of_range("PLE shard is out of range", __FILE__, __LINE__);
        }
        shard_offset = shards_[static_cast<std::size_t>(shard)].file_offset;
    }
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Status::invalid_argument("cannot open PLE table for validation", __FILE__,
                                        __LINE__);
    }
    std::vector<std::uint8_t> out(row_bytes_);
    const std::uint64_t offset = shard_offset + (row % shard_rows_) * row_bytes_;
    Status st = read_exact_at(fd, offset, out.data(), out.size());
    ::close(fd);
    if (!st.ok()) return st;
    return out;
}

}  // namespace qwen4exp
}  // namespace shelf
