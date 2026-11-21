#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <shelf/quantization/exl3/exl3.h>
#include <shelf/quantization/int6/int6_w32.h>
#include <shelf/quantization/ngram/ngram.h>

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


Result<PleTable> PleTable::load_ngram(const std::string& root,
                                      const Qwen4ExpTextConfig& config) {
    namespace nq = shelf::quantization::ngram;
    std::ifstream input(root + "/ngram_manifest.json");
    if (!input) return Status::invalid_argument("cannot open ngram manifest", __FILE__, __LINE__);
    std::string json((std::istreambuf_iterator<char>(input)), {});
    auto manifest = nq::parse_manifest(json);
    if (!manifest.ok()) return manifest.status();
    auto valid = nq::validate_bundle(root, manifest.value());
    if (!valid.ok()) return valid;
    auto sizes = qwen4exp_ple_head_vocab_sizes(config, 0u);
    auto offsets = qwen4exp_ple_head_offsets(config, 0u);
    auto multipliers = qwen4exp_ple_layer_multipliers(config, 0u);
    const auto layers = config.ple_layers();
    if (!sizes.ok() || !offsets.ok() || !multipliers.ok() || layers.size() != 1u ||
        sizes.value().size() != 16u || config.ple.ngram_size != 3u ||
        config.ple.heads_per_ngram != 8u || config.ple.embed_dim != 2560u ||
        manifest.value().tables.size() != 16u) {
        return Status::invalid_argument("ngram PLE geometry differs from config", __FILE__, __LINE__);
    }
    PleTable result;
    result.format_ = TableFormat::Ngram;
    result.row_width_ = kExpectedRowWidth;
    result.ngram_size_ = static_cast<std::uint32_t>(config.ple.ngram_size);
    result.heads_per_ngram_ = static_cast<std::uint32_t>(config.ple.heads_per_ngram);
    result.eos_token_id_ = config.ple.eos_token_id;
    result.head_offsets_.assign(offsets.value().begin(), offsets.value().end());
    result.head_vocab_sizes_.assign(sizes.value().begin(), sizes.value().end());
    result.layer_multipliers_ = multipliers.value();
    result.rows_ = offsets.value().back() + sizes.value().back();
    result.shard_count_ = 16u;
    result.ngram_ = std::make_shared<NgramState>();
    result.ngram_->heads.resize(16u);
    result.ngram_->capacity = static_cast<std::size_t>(ple_hot_cache_bytes() / (kHotBlockRows * 102u));
    for (const auto& t : manifest.value().tables) {
        if (t.head >= 16u || t.layer != layers[0] || t.ngram_order != 2u + t.head / 8u ||
            t.dim != kExpectedRowWidth || t.rows != sizes.value()[t.head] ||
            result.ngram_->heads[t.head].reader) {
            return Status::invalid_argument("ngram PLE table differs from config", __FILE__, __LINE__);
        }
        auto opened = io::SafetensorsReader::open((std::filesystem::path(root) / t.shard).string());
        if (!opened.ok()) return opened.status();
        auto& head = result.ngram_->heads[t.head];
        head.reader = std::make_unique<io::SafetensorsReader>(opened.release());
        std::size_t bytes = 0;
        auto codes = head.reader->tensor_data(t.codes_tensor, bytes);
        if (!codes.ok()) return codes.status();
        head.codes = static_cast<const std::uint8_t*>(codes.value());
        auto scales = head.reader->tensor_data(t.scale_tensor, bytes);
        if (!scales.ok()) return scales.status();
        head.scales = static_cast<const std::uint8_t*>(scales.value());
        auto book = head.reader->tensor_data(t.codebook_tensor, bytes);
        if (!book.ok()) return book.status();
        head.codebook.resize(1u << t.bits);
        std::memcpy(head.codebook.data(), book.value(), bytes);
        head.rows = t.rows;
        head.bits = t.bits;
        head.width = nq::row_bytes(t.dim, t.bits).value();
        result.payload_bytes_ += t.rows * (head.width + 2u) + bytes;
    }
    return result;
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
    } else if (source_dtype == shelf::io::SType::F8_E4M3) {
        table.format_ = TableFormat::Fp8E4M3;
        table.row_bytes_ = table.row_width_;
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
    std::error_code manifest_error;
    const auto manifest_status = std::filesystem::symlink_status(
        std::filesystem::path(model_dir) / "ngram_manifest.json", manifest_error);
    if (manifest_error && manifest_error != std::errc::no_such_file_or_directory)
        return Status::invalid_argument("cannot inspect ngram manifest", __FILE__, __LINE__);
    if (manifest_status.type() != std::filesystem::file_type::not_found)
        return load_ngram(model_dir, config);
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return load_index_shards(model_dir, config, fadvise_dontneed);
    ::close(fd);
    return Status::unsupported("EXL3 PLE tables are not supported", __FILE__, __LINE__);
}

std::uint64_t PleTable::host_bytes() const noexcept {
    if (ngram_) {
        std::lock_guard<std::mutex> lock(ngram_->mutex);
        std::uint64_t bytes = 0;
        for (const auto& head : ngram_->heads) bytes += head.codebook.size() * 2u;
        for (const auto& block : ngram_->blocks) bytes += block.data.capacity();
        return bytes;
    }
    return tiered_ != nullptr ? tiered_->hot_bytes : payload_bytes_;
}

std::size_t PleTable::cached_blocks() const noexcept {
    if (ngram_) {
        std::lock_guard<std::mutex> lock(ngram_->mutex);
        return ngram_->blocks.size();
    }
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
    if (ngram_) {
        if (row >= rows_) return Status::out_of_range("PLE row is out of range", __FILE__, __LINE__);
        auto& state = *ngram_;
        std::lock_guard<std::mutex> lock(state.mutex);
        const std::size_t h = head_for_row(row);
        const auto& head = state.heads[h];
        const std::size_t local = static_cast<std::size_t>(row - head_offsets_[h]);
        const std::size_t block = local / kHotBlockRows;
        const std::uint64_t key = (static_cast<std::uint64_t>(h) << 48u) | block;
        const std::uint8_t* codes = head.codes + local * head.width;
        const std::uint8_t* scale = head.scales + local * 2u;
        if (state.capacity != 0u) {
            auto found = state.index.find(key);
            if (found == state.index.end()) {
                std::size_t slot;
                if (state.blocks.size() < state.capacity) {
                    slot = state.blocks.size();
                    state.blocks.emplace_back();
                } else {
                    slot = state.next++ % state.capacity;
                    state.index.erase(state.blocks[slot].key);
                }
                auto& entry = state.blocks[slot];
                entry.key = key;
                const std::size_t first = block * kHotBlockRows;
                const std::size_t count = std::min<std::size_t>(kHotBlockRows, head.rows - first);
                entry.data.resize(count * (head.width + 2u));
                std::memcpy(entry.data.data(), head.codes + first * head.width, count * head.width);
                std::memcpy(entry.data.data() + count * head.width, head.scales + first * 2u, count * 2u);
                state.index[key] = slot;
                found = state.index.find(key);
            }
            const auto& entry = state.blocks[found->second];
            const std::size_t first = block * kHotBlockRows;
            const std::size_t count = std::min<std::size_t>(kHotBlockRows, head.rows - first);
            const std::size_t within = local - first;
            codes = entry.data.data() + within * head.width;
            scale = entry.data.data() + count * head.width + within * 2u;
        }
        std::uint16_t scale_bits;
        std::memcpy(&scale_bits, scale, sizeof(scale_bits));
        std::vector<float> output(row_width_);
        auto status = shelf::quantization::ngram::decode(
            {codes, head.width}, scale_bits, head.codebook, head.bits, output);
        if (!status.ok()) return status;
        return output;
    }
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
    return Status::unsupported("PLE table format is not supported", __FILE__, __LINE__);
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

}  // namespace qwen4exp
}  // namespace shelf
