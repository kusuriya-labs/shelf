// tests support — minimal safetensors fixture writer (no deps, hand-built).
// Malformed headers are written verbatim by tests; this builder produces
// spec-conforming files and lets tests tamper with offsets for error probes.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace st_fixture {

struct Tensor {
    std::string name;
    std::string dtype;                       // "BF16", "F32", ...
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> payload;
};

inline void put_u64_le(std::string& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

// Spec-conforming header with contiguous offsets, plus payload concatenation.
inline bool write_file(const std::string& path, const std::vector<Tensor>& tensors,
                       const std::string& metadata_json = "") {
    std::string header = "{";
    if (!metadata_json.empty()) {
        header += "\"__metadata__\":" + metadata_json + ",";
    }
    std::uint64_t offset = 0;
    std::string payload;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (i != 0) header += ",";
        header += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (std::size_t d = 0; d < t.shape.size(); ++d) {
            if (d != 0) header += ",";
            header += std::to_string(t.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + t.payload.size()) + "]}";
        offset += t.payload.size();
        payload.append(reinterpret_cast<const char*>(t.payload.data()), t.payload.size());
    }
    header += "}";

    std::string blob;
    put_u64_le(blob, header.size());
    blob += header;
    blob += payload;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    return f.good();
}

// Raw variant: caller-provided header JSON and payload (for error probes).
inline bool write_raw(const std::string& path, const std::string& header_json,
                      const std::string& payload) {
    std::string blob;
    put_u64_le(blob, header_json.size());
    blob += header_json;
    blob += payload;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    return f.good();
}

inline std::string tmp_path(const char* tag) {
    static int counter = 0;
    return std::string("/tmp/shelf_st_") + tag + "_" + std::to_string(counter++) + ".safetensors";
}

} // namespace st_fixture
