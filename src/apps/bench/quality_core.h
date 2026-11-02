#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace shelf::bench::quality {

inline constexpr char kDumpMagic[8] = {'P', 'S', 'Q', 'U', 'A', 'L', '0', '1'};
inline constexpr uint32_t kDumpVersion = 1u;

enum class KvDtype : uint32_t {
    Bf16 = 0u,
    Fp8E4M3 = 1u,
};

inline const char* kv_dtype_name(KvDtype dtype) {
    return dtype == KvDtype::Fp8E4M3 ? "fp8" : "bf16";
}

struct DumpHeader {
    char magic[8];
    uint32_t version;
    uint32_t vocab;
    uint64_t positions;
    uint32_t kv_dtype;
    uint32_t reserved;
};

static_assert(sizeof(DumpHeader) == 32u, "dump header must be 32 bytes");

inline double log_sum_exp(const float* logits, std::size_t count) {
    double max_value = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(logits[i]);
        if (value > max_value) max_value = value;
    }
    if (!std::isfinite(max_value)) return max_value;
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - max_value);
    }
    return max_value + std::log(sum);
}

inline double row_nll(const float* logits, std::size_t vocab, int32_t target) {
    const double log_z = log_sum_exp(logits, vocab);
    return -(static_cast<double>(logits[target]) - log_z);
}

inline uint32_t argmax(const float* logits, std::size_t vocab) {
    uint32_t best = 0;
    float best_value = logits[0];
    for (std::size_t v = 1; v < vocab; ++v) {
        if (logits[v] > best_value) {
            best_value = logits[v];
            best = static_cast<uint32_t>(v);
        }
    }
    return best;
}

inline void topk_ids(const float* logits, std::size_t vocab, uint32_t k,
                     std::vector<uint32_t>& out) {
    out.clear();
    for (std::size_t v = 0; v < vocab && out.size() < k; ++v) {
        out.push_back(static_cast<uint32_t>(v));
    }
    std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    for (std::size_t v = out.size(); v < vocab; ++v) {
        const uint32_t candidate = static_cast<uint32_t>(v);
        const float candidate_value = logits[candidate];
        if (candidate_value <= logits[out.back()]) continue;
        std::size_t pos = out.size();
        while (pos > 0 && logits[out[pos - 1]] < candidate_value) --pos;
        out.insert(out.begin() + static_cast<std::ptrdiff_t>(pos), candidate);
        out.pop_back();
    }
}

struct Metrics {
    uint64_t positions = 0;
    double nll_base = 0.0;
    double nll_candidate = 0.0;
    double kl = 0.0;
    uint64_t top1_match = 0;
    uint64_t top5_intersection = 0;
    double max_abs_delta = 0.0;
};

inline double perplexity(double nll_sum, uint64_t positions) {
    if (positions == 0) return 0.0;
    return std::exp(nll_sum / static_cast<double>(positions));
}

inline void accumulate_row(const float* base, const float* candidate, std::size_t vocab,
                           int32_t target, Metrics& metrics) {
    const double base_lse = log_sum_exp(base, vocab);
    const double candidate_lse = log_sum_exp(candidate, vocab);
    const double target_value = static_cast<double>(base[static_cast<std::size_t>(target)]);
    const double candidate_target =
        static_cast<double>(candidate[static_cast<std::size_t>(target)]);
    metrics.nll_base += -(target_value - base_lse);
    metrics.nll_candidate += -(candidate_target - candidate_lse);
    double kl = 0.0;
    double max_delta = 0.0;
    for (std::size_t v = 0; v < vocab; ++v) {
        const double base_logp = static_cast<double>(base[v]) - base_lse;
        kl += std::exp(base_logp) * (base_logp - (static_cast<double>(candidate[v]) - candidate_lse));
        const double delta = std::fabs(static_cast<double>(base[v]) - static_cast<double>(candidate[v]));
        if (delta > max_delta) max_delta = delta;
    }
    metrics.kl += kl;
    if (argmax(base, vocab) == argmax(candidate, vocab)) ++metrics.top1_match;
    std::vector<uint32_t> base_top;
    std::vector<uint32_t> candidate_top;
    topk_ids(base, vocab, 5u, base_top);
    topk_ids(candidate, vocab, 5u, candidate_top);
    uint64_t intersection = 0;
    for (uint32_t a : base_top) {
        for (uint32_t b : candidate_top) {
            if (a == b) {
                ++intersection;
                break;
            }
        }
    }
    metrics.top5_intersection += intersection;
    if (max_delta > metrics.max_abs_delta) metrics.max_abs_delta = max_delta;
    ++metrics.positions;
}

inline double mean_kl(const Metrics& metrics) {
    if (metrics.positions == 0) return 0.0;
    return metrics.kl / static_cast<double>(metrics.positions);
}

inline double top1_agreement(const Metrics& metrics) {
    if (metrics.positions == 0) return 0.0;
    return static_cast<double>(metrics.top1_match) / static_cast<double>(metrics.positions);
}

inline double top5_overlap(const Metrics& metrics) {
    if (metrics.positions == 0) return 0.0;
    return static_cast<double>(metrics.top5_intersection) /
           (5.0 * static_cast<double>(metrics.positions));
}

class DumpWriter {
public:
    DumpWriter() = default;
    DumpWriter(const DumpWriter&) = delete;
    DumpWriter& operator=(const DumpWriter&) = delete;
    ~DumpWriter() noexcept {
        if (file_ != nullptr) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    bool open(const std::string& path, uint32_t vocab, KvDtype dtype,
              uint32_t score_start = 0u) {
        file_ = std::fopen(path.c_str(), "wb");
        if (file_ == nullptr) return false;
        header_ = DumpHeader{};
        std::memcpy(header_.magic, kDumpMagic, sizeof(kDumpMagic));
        header_.version = kDumpVersion;
        header_.vocab = vocab;
        header_.kv_dtype = static_cast<uint32_t>(dtype);
        header_.reserved = score_start;
        if (std::fwrite(&header_, sizeof(header_), 1u, file_) != 1u) {
            std::fclose(file_);
            file_ = nullptr;
            return false;
        }
        return true;
    }

    bool write(int32_t target, uint32_t position, const float* logits) {
        if (file_ == nullptr) return false;
        if (std::fwrite(&target, sizeof(target), 1u, file_) != 1u) return false;
        if (std::fwrite(&position, sizeof(position), 1u, file_) != 1u) return false;
        if (std::fwrite(logits, sizeof(float), header_.vocab, file_) != header_.vocab) {
            return false;
        }
        ++header_.positions;
        return true;
    }

    bool close() {
        if (file_ == nullptr) return true;
        bool ok = true;
        if (std::fflush(file_) != 0) ok = false;
        if (std::fseek(file_, 0, SEEK_SET) != 0) ok = false;
        if (std::fwrite(&header_, sizeof(header_), 1u, file_) != 1u) ok = false;
        if (std::fclose(file_) != 0) ok = false;
        file_ = nullptr;
        return ok;
    }

    uint64_t positions() const noexcept { return header_.positions; }

private:
    FILE* file_ = nullptr;
    DumpHeader header_{};
};

class DumpReader {
public:
    DumpReader() = default;
    DumpReader(const DumpReader&) = delete;
    DumpReader& operator=(const DumpReader&) = delete;
    ~DumpReader() noexcept {
        if (file_ != nullptr) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    bool open(const std::string& path, std::string& error) {
        file_ = std::fopen(path.c_str(), "rb");
        if (file_ == nullptr) {
            error = "cannot open " + path;
            return false;
        }
        if (std::fread(&header_, sizeof(header_), 1u, file_) != 1u) {
            error = "truncated header";
            return false;
        }
        if (std::memcmp(header_.magic, kDumpMagic, sizeof(kDumpMagic)) != 0) {
            error = "bad magic";
            return false;
        }
        if (header_.version != kDumpVersion) {
            error = "unsupported version";
            return false;
        }
        if (header_.vocab == 0u) {
            error = "zero vocab";
            return false;
        }
        read_ = 0u;
        return true;
    }

    bool read(std::vector<float>& logits, int32_t& target, uint32_t& position,
              std::string& error) {
        if (file_ == nullptr) {
            error = "not open";
            return false;
        }
        if (read_ == header_.positions) return false;
        const std::size_t vocab = header_.vocab;
        if (std::fread(&target, sizeof(target), 1u, file_) != 1u ||
            std::fread(&position, sizeof(position), 1u, file_) != 1u) {
            error = "truncated row header";
            return false;
        }
        logits.resize(vocab);
        if (std::fread(logits.data(), sizeof(float), vocab, file_) != vocab) {
            error = "truncated logits";
            return false;
        }
        ++read_;
        return true;
    }

    bool done() const noexcept { return file_ != nullptr && read_ == header_.positions; }

    const DumpHeader& header() const noexcept { return header_; }

    bool close() {
        if (file_ == nullptr) return true;
        const bool ok = std::fclose(file_) == 0;
        file_ = nullptr;
        return ok;
    }

private:
    FILE* file_ = nullptr;
    DumpHeader header_{};
    uint64_t read_ = 0u;
};

}
