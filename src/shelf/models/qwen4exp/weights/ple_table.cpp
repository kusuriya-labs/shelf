#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <shelf/quantization/exl3/exl3.h>
#include <shelf/quantization/ngram/ngram.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <fcntl.h>
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

constexpr std::uint32_t kExpectedRowWidth = 160u;

std::uint64_t ple_hot_cache_bytes() noexcept {
    const char* value = std::getenv("SHELF_PLE_HOT_CACHE_GIB");
    if (value == nullptr || *value == '\0') return 0u;
    const double gib = std::strtod(value, nullptr);
    if (gib <= 0.0) return 0u;
    return static_cast<std::uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

constexpr std::uint32_t kHotBlockRows = 2048u;

}  // namespace

void PleTable::FreeDeleter::operator()(void* p) const noexcept {
    std::free(p);
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
                                             const Qwen4ExpTextConfig& config) {
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
    if (source_dtype == shelf::io::SType::BF16) {
        return Status::unsupported(
            "BF16 PLE tables require online INT6 quantization, which is disabled",
            __FILE__, __LINE__);
    }
    if (source_dtype != shelf::io::SType::F8_E4M3) {
        return Status::unsupported("PLE shards are not FP8 E4M3", __FILE__, __LINE__);
    }
    table.format_ = TableFormat::Fp8E4M3;
    table.row_bytes_ = table.row_width_;

    {
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
        const std::size_t row_bytes_in = static_cast<std::size_t>(table.row_width_);
        const std::size_t out_bytes =
            static_cast<std::size_t>(table.shard_rows_) * table.row_bytes_;

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
    return table;
}

Result<PleTable> PleTable::load(const std::string& model_dir,
                                const Qwen4ExpTextConfig& config) {
    const std::string manifest_path = model_dir + "/ngram_manifest.json";
    std::error_code manifest_error;
    const auto manifest_status = std::filesystem::symlink_status(manifest_path, manifest_error);
    if (manifest_error && manifest_error != std::errc::no_such_file_or_directory) {
        return Status::invalid_argument(manifest_error.message().c_str(), __FILE__, __LINE__)
            .with_context(manifest_path);
    }
    if (manifest_status.type() != std::filesystem::file_type::not_found) {
        auto loaded = load_ngram(model_dir, config);
        if (!loaded.ok()) return loaded.status().with_context(manifest_path);
        return loaded;
    }
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (errno != ENOENT) {
            return Status::invalid_argument(std::strerror(errno), __FILE__, __LINE__)
                .with_context(path);
        }
        auto loaded = load_index_shards(model_dir, config);
        if (!loaded.ok()) {
            return loaded.status().with_context(model_dir + "/model.safetensors.index.json");
        }
        return loaded;
    }
    ::close(fd);
    return Status::unsupported(
        "EXL3 PLE tables are not supported. Install Ngram4 or Ngram5 table shards "
        "and ngram_manifest.json in the model directory.", __FILE__, __LINE__)
        .with_context(path);
}

Result<std::shared_ptr<const PleTable>> PleTable::load_shared(
    const std::string& model_dir, const Qwen4ExpTextConfig& config) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<const PleTable>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = cache[model_dir];
    if (auto existing = entry.lock()) return existing;
    auto loaded = load(model_dir, config);
    if (!loaded.ok()) return loaded.status();
    auto table = std::make_shared<const PleTable>(loaded.release());
    entry = table;
    return table;
}

std::uint64_t PleTable::host_bytes() const noexcept {
    if (ngram_) {
        std::lock_guard<std::mutex> lock(ngram_->mutex);
        std::uint64_t bytes = 0;
        for (const auto& head : ngram_->heads) bytes += head.codebook.size() * 2u;
        for (const auto& block : ngram_->blocks) bytes += block.data.capacity();
        return bytes;
    }
    return payload_bytes_;
}

std::size_t PleTable::cached_blocks() const noexcept {
    if (!ngram_) return 0u;
    std::lock_guard<std::mutex> lock(ngram_->mutex);
    return ngram_->blocks.size();
}

const std::uint8_t* PleTable::row_ptr(std::uint64_t row) const noexcept {
    if (row >= rows_ || shard_rows_ == 0u) return nullptr;
    const std::uint64_t shard = row / shard_rows_;
    const std::uint64_t local = row % shard_rows_;
    if (shard >= shards_.size()) return nullptr;
    return shards_[static_cast<std::size_t>(shard)].data.get() +
           static_cast<std::size_t>(local) * row_bytes_;
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
    for (std::size_t position = first; position < n; ++position) {
        for (std::uint32_t ngram = 2; ngram <= ngram_size_; ++ngram) {
            const std::size_t head_begin = (ngram - 2u) * heads_per_ngram_;
            const std::size_t head_end = head_begin + heads_per_ngram_;
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
