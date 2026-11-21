#pragma once
#include <shelf/core/status.h>
#include <shelf/io/safetensors_reader.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace shelf::quantization::ngram {

struct Table {
    std::string id, name, shard, codes_tensor, scale_tensor, codebook_tensor;
    std::size_t layer = 0, ngram_order = 0, head = 0, rows = 0, dim = 0;
    unsigned bits = 0;
};
struct Manifest {
    std::string format = "ngram-q", profile, codec = "lloyd-row-f16", source_model;
    unsigned version = 1;
    std::vector<Table> tables;
};

Result<std::size_t> row_bytes(std::size_t dim, unsigned bits);
Status pack(std::span<const uint8_t> indices, unsigned bits, std::span<uint8_t> codes);
Status unpack(std::span<const uint8_t> codes, unsigned bits, std::span<uint8_t> indices);
Status decode(std::span<const uint8_t> codes, uint16_t scale, std::span<const uint16_t> codebook, unsigned bits, std::span<float> output);
Status decode_bf16(std::span<const uint8_t> codes, uint16_t scale, std::span<const uint16_t> codebook, unsigned bits, std::span<uint16_t> output);
Status quantize(std::span<const float> row, std::span<const uint16_t> codebook, unsigned bits, uint16_t& scale, std::span<uint8_t> codes);
Result<std::vector<uint16_t>> train(std::span<const float> samples, unsigned bits);
Result<Manifest> parse_manifest(const std::string& json);
Result<std::string> serialize_manifest(const Manifest& manifest);
Status validate_manifest(const Manifest& manifest);
Status validate_bundle(const std::filesystem::path& root, const Manifest& manifest);

class TableReader {
public:
    static Result<TableReader> open(const std::filesystem::path& root, const Table& table);
    Status validate() const;
    Status decode_row(std::size_t row, std::span<float> output) const;
private:
    TableReader(Table table, io::SafetensorsReader reader) : table_(std::move(table)), reader_(std::move(reader)) {}
    Table table_;
    io::SafetensorsReader reader_;
};
}
