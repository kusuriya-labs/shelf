#pragma once

#include <shelf/io/safetensors_reader.h>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
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
    Checkpoint(const Path &model, const Path &engram, const Path &draft, const Owners &owners, int ep_size,
               int tp_size);
    const Weight &at(const std::string &key, int rank) const;
    const Weight *find(const std::string &key, int rank) const;
    const std::array<int, 2> *shape_of(const std::string &key, int rank) const;
    bool row_parallel(const std::string &key, int rank) const;
    void populate_engram(bool lock) const;
    void report_engram_residency(const char *phase) const;
    void load(const Path &directory, bool draft);
    std::vector<io::SafetensorsReader> files;
    std::array<size_t, 4> engram_files{};
    std::map<std::string, Weight> source;
    std::array<std::map<std::string, Weight>, 4> local;
    std::array<std::vector<Weight *>, 4> upload_order;
    std::array<size_t, 4> bytes{};
    std::map<std::string, std::array<int, 2>> shape;
    std::array<std::map<std::string, std::array<int, 2>>, 4> local_shape;
    std::array<std::set<std::string>, 4> row_split;
    std::vector<std::unique_ptr<uint8_t[]>> slices;
    std::array<std::array<int, 16>, 4> moduli{};
    std::array<std::array<uint8_t, 512>, 41> expert_bits{};
    bool mixed = false;
    Owners owners{};
    bool has_draft = false;
    int ep = 1;
    int tp = 1;
    int ranks = 1;
};

Owners read_owners(const Path &file);
}
