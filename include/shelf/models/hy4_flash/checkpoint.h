#pragma once

#include <shelf/io/safetensors_reader.h>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace shelf::hy4 {
using Path = std::filesystem::path;
using Owners = std::array<std::array<int, 512>, 41>;

struct Weight {
    io::StTensorSpec spec;
    const void *host;
    size_t bytes;
    int rank;
    size_t offset;
};

struct Checkpoint {
    Checkpoint() = default;
    Checkpoint(const Path &model, const Path &engram, const Path &draft, const Owners &owners);
    const Weight &at(const std::string &key) const;
    void relocate(const Owners &layout);
    void populate_engram(bool lock) const;
    void report_engram_residency(const char *phase) const;
    void load(const Path &directory, bool draft);
    std::vector<io::SafetensorsReader> files;
    std::array<size_t, 4> engram_files{};
    std::map<std::string, Weight> weights;
    std::vector<Weight *> upload_order;
    std::array<size_t, 3> bytes{};
    std::map<std::string, std::array<int, 2>> shape;
    std::array<std::array<int, 16>, 4> moduli{};
    std::array<std::array<uint8_t, 512>, 41> expert_bits{};
    bool mixed = false;
    Owners owners{};
    bool has_draft = false;
};

Owners read_owners(const Path &file);
}
