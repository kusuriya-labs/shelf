#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <shelf/quantization/exl3/exl3.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
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

void PleTable::FreeDeleter::operator()(void* p) const noexcept {
    std::free(p);
}


Result<PleTable> PleTable::load_fp8(const std::string& model_dir,
                                    const Qwen4ExpTextConfig& config,
                                    bool fadvise_dontneed) {
    (void)fadvise_dontneed;
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
        return Status::invalid_argument("PLE FP8 shard count differs from split_ngram_parts",
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
    table.mode_ = TableMode::Fp8E4M3;
    table.file_path_ = index_path;
    auto padded_rows = qwen4exp_ple_padded_rows(config, 0u);
    if (!padded_rows.ok()) return padded_rows.status();
    table.rows_ = padded_rows.value();
    table.row_width_ = kExpectedRowWidth;
    table.row_bytes_ = kExpectedRowWidth;
    table.shard_count_ = static_cast<std::uint32_t>(shards.size());
    table.shard_rows_ = 2500012u;
    table.ngram_size_ = static_cast<std::uint32_t>(config.ple.ngram_size);
    table.heads_per_ngram_ = static_cast<std::uint32_t>(config.ple.heads_per_ngram);
    table.eos_token_id_ = config.ple.eos_token_id;
    table.head_offsets_.assign(head_offsets.value().begin(), head_offsets.value().end());
    table.head_vocab_sizes_.assign(head_sizes.value().begin(), head_sizes.value().end());
    table.layer_multipliers_.assign(multipliers.value().begin(), multipliers.value().end());

    std::map<std::string, std::unique_ptr<shelf::io::SafetensorsReader>> readers;
    auto reader_for = [&](const std::string& shard)
        -> Result<const shelf::io::SafetensorsReader*> {
        auto it = readers.find(shard);
        if (it != readers.end()) return it->second.get();
        auto opened = shelf::io::SafetensorsReader::open(model_dir + "/" + shard);
        if (!opened.ok()) return opened.status();
        auto inserted = readers.emplace(
            shard, std::make_unique<shelf::io::SafetensorsReader>(opened.release()));
        return inserted.first->second.get();
    };

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

    table.shards_.resize(table.shard_count_);
    for (std::uint32_t id = 0; id < table.shard_count_; ++id) {
        auto name_it = shards.find(id);
        if (name_it == shards.end()) {
            return Status::invalid_argument("PLE FP8 shard index is not contiguous",
                                            __FILE__, __LINE__);
        }
        auto shard_it = weight_map.find(name_it->second);
        if (shard_it == weight_map.end()) {
            return Status::invalid_argument("PLE FP8 shard is missing from the index",
                                            __FILE__, __LINE__);
        }
        auto reader = reader_for(shard_it.value().get<std::string>());
        if (!reader.ok()) return reader.status();
        auto spec = reader.value()->tensor_spec(name_it->second);
        if (!spec.ok() || spec.value().dtype != shelf::io::SType::F8_E4M3 ||
            spec.value().shape.size() != 2u ||
            spec.value().shape[0] != table.shard_rows_ ||
            spec.value().shape[1] != table.row_width_) {
            return Status::invalid_argument("PLE FP8 shard shape is invalid", __FILE__,
                                            __LINE__);
        }
        std::size_t bytes = 0;
        auto data = reader.value()->tensor_data(name_it->second, bytes);
        if (!data.ok() || bytes != static_cast<std::size_t>(table.shard_rows_) *
                                     table.row_width_) {
            return Status::invalid_argument("PLE FP8 shard byte count is invalid", __FILE__,
                                            __LINE__);
        }
        void* memory = nullptr;
        if (::posix_memalign(&memory, 64u, bytes) != 0) {
            return Status::insufficient_memory("cannot allocate PLE FP8 shard", __FILE__,
                                               __LINE__);
        }
        std::memcpy(memory, data.value(), bytes);
        table.shards_[id].data.reset(static_cast<std::uint8_t*>(memory));
        table.shards_[id].bytes = bytes;
    }
    table.payload_bytes_ = static_cast<std::uint64_t>(table.shard_count_) *
                           table.shard_rows_ * table.row_bytes_;
    return table;
}

Result<PleTable> PleTable::load(const std::string& model_dir,
                                const Qwen4ExpTextConfig& config,
                                bool fadvise_dontneed) {
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return load_fp8(model_dir, config, fadvise_dontneed);
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

        void* memory = nullptr;
        if (::posix_memalign(&memory, 64u, static_cast<std::size_t>(bytes)) != 0) {
            ::close(fd);
            return Status::insufficient_memory(
                "cannot allocate PLE shard in host RAM", __FILE__, __LINE__);
        }
        table.shards_[index].data.reset(static_cast<std::uint8_t*>(memory));
        table.shards_[index].bytes = static_cast<std::size_t>(bytes);
        table.shards_[index].file_offset = table.data_base_ + record.begin;
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

    ::close(fd);
    return table;
}

const std::uint8_t* PleTable::row_ptr(std::uint64_t row) const noexcept {
    if (row >= rows_ || shard_rows_ == 0u) return nullptr;
    const std::uint64_t shard = row / shard_rows_;
    if (shard >= shards_.size()) return nullptr;
    const std::uint64_t local = row % shard_rows_;
    return shards_[static_cast<std::size_t>(shard)].data.get() +
           static_cast<std::size_t>(local) * row_bytes_;
}

Result<std::vector<float>> PleTable::decode_row(std::uint64_t row) const {
    const std::uint8_t* packed = row_ptr(row);
    if (packed == nullptr) {
        return Status::out_of_range("PLE row is out of range", __FILE__, __LINE__);
    }
    if (mode_ == TableMode::Fp8E4M3) {
        std::vector<float> out(row_width_);
        const std::uint32_t heads = static_cast<std::uint32_t>(head_offsets_.size());
        std::uint32_t head = 0;
        if (heads != 0u) {
            auto upper = std::upper_bound(head_offsets_.begin(), head_offsets_.end(),
                                          static_cast<std::int64_t>(row));
            if (upper != head_offsets_.begin())
                head = static_cast<std::uint32_t>(upper - head_offsets_.begin() - 1);
            if (head >= heads) head = heads - 1;
        }
        for (std::uint32_t i = 0; i < row_width_; ++i) {
            float value = e4m3_to_f32(packed[i]) * fp8_scale_;
            if (!head_bias_f16_.empty())
                value += shelf::quantization::exl3::f32_from_f16_bits(
                    head_bias_f16_[static_cast<std::size_t>(head) * row_width_ + i]);
            out[i] = value;
        }
        return out;
    }
    auto upper = std::upper_bound(head_offsets_.begin(), head_offsets_.end(),
                                  static_cast<std::int64_t>(row));
    std::size_t head = 0;
    if (upper != head_offsets_.begin()) {
        head = static_cast<std::size_t>(upper - head_offsets_.begin() - 1);
    }
    if (head >= head_offsets_.size()) head = head_offsets_.size() - 1;

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
        const std::uint16_t bias_bits =
            head_bias_f16_[head * row_width_ + i];
        out[i] = value + shelf::quantization::exl3::f32_from_f16_bits(bias_bits);
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
    std::vector<std::int32_t> previous_eos_inclusive(n, -1);
    std::int32_t last_eos = -1;
    for (std::size_t i = 0; i < n; ++i) {
        if (history[i] == eos_token_id_) last_eos = static_cast<std::int32_t>(i);
        previous_eos_inclusive[i] = last_eos;
    }

    std::vector<std::vector<std::int32_t>> shifted(ngram_size_,
                                                   std::vector<std::int32_t>(n, eos_token_id_));
    for (std::uint32_t shift = 0; shift < ngram_size_; ++shift) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::int32_t previous_eos =
                i == 0u ? -1 : previous_eos_inclusive[i - 1u];
            const std::int32_t segment_start = previous_eos + 1;
            const std::int64_t source = static_cast<std::int64_t>(i) - shift;
            const bool valid = (static_cast<std::int64_t>(i) - segment_start >= shift) &&
                               source >= 0;
            shifted[shift][i] =
                valid ? history[static_cast<std::size_t>(source)] : eos_token_id_;
        }
    }

    std::vector<std::uint64_t> rows;
    rows.reserve(out_len * head_count);
    const std::size_t first = history.size() - out_len;
    for (std::uint32_t ngram = 2; ngram <= ngram_size_; ++ngram) {
        const std::size_t head_begin = (ngram - 2u) * heads_per_ngram_;
        const std::size_t head_end = head_begin + heads_per_ngram_;
        for (std::size_t position = first; position < n; ++position) {
            std::uint64_t mixed =
                static_cast<std::uint64_t>(static_cast<std::int64_t>(shifted[0][position])) *
                static_cast<std::uint64_t>(layer_multipliers_[0]);
            for (std::uint32_t p = 1; p < ngram; ++p) {
                mixed ^= static_cast<std::uint64_t>(
                             static_cast<std::int64_t>(shifted[p][position])) *
                         static_cast<std::uint64_t>(layer_multipliers_[p]);
            }
            for (std::size_t head = head_begin; head < head_end; ++head) {
                const std::int64_t size = head_vocab_sizes_[head];
                if (size <= 0) {
                    return Status::invalid_argument("PLE head vocab size is not positive",
                                                    __FILE__, __LINE__);
                }
                std::int64_t remainder =
                    static_cast<std::int64_t>(mixed) % size;
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

Result<std::vector<std::uint8_t>> PleTable::read_source_row(const std::string& model_dir,
                                                            std::uint64_t row) const {
    if (row >= rows_ || shard_rows_ == 0u) {
        return Status::out_of_range("PLE row is out of range", __FILE__, __LINE__);
    }
    const std::uint64_t shard = row / shard_rows_;
    if (shard >= shards_.size()) {
        return Status::out_of_range("PLE shard is out of range", __FILE__, __LINE__);
    }
    const std::string path = model_dir + "/ngram_embedding.safetensors";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Status::invalid_argument("cannot open PLE table for validation", __FILE__,
                                        __LINE__);
    }
    std::vector<std::uint8_t> out(row_bytes_);
    const std::uint64_t offset = shards_[static_cast<std::size_t>(shard)].file_offset +
                                 (row % shard_rows_) * row_bytes_;
    Status st = read_exact_at(fd, offset, out.data(), out.size());
    ::close(fd);
    if (!st.ok()) return st;
    return out;
}

}  // namespace qwen4exp
}  // namespace shelf
