// shelf bench — 共通ヘルパの実装。
#include "bench_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>

namespace shelf::bench {

// ---- Args ------------------------------------------------------------------

bool Args::parse(int argc, char** argv, std::string* error) {
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i] == nullptr ? std::string() : std::string(argv[i]);
        if (a.rfind("--", 0) == 0) {
            const auto eq = a.find('=');
            if (eq != std::string::npos) {
                const std::string key = a.substr(2, eq - 2);
                keys_.push_back(key);
                kv_.emplace_back(key, a.substr(eq + 1));
                continue;
            }
            const std::string key = a.substr(2);
            if (key == "help" || key == "h") {  // フラグ（値を取らない）
                keys_.push_back(key);
                kv_.emplace_back(key, "");
                continue;
            }
            if (i + 1 >= argc) {
                if (error) *error = "missing value for " + a;
                return false;
            }
            keys_.push_back(key);
            kv_.emplace_back(key, argv[i + 1]);
            ++i;
            continue;
        }
        if (a.rfind("-", 0) == 0 && a.size() > 1) {  // -h のみ
            if (a == "-h") { keys_.push_back("h"); kv_.emplace_back("h", ""); continue; }
            if (error) *error = "unknown option " + a;
            return false;
        }
        pos_.push_back(a);
    }
    return true;
}

bool only_known_keys(const Args& a, const std::vector<std::string>& known, std::string* bad) {
    for (const auto& k : a.keys()) {
        bool hit = false;
        for (const auto& n : known)
            if (k == n) { hit = true; break; }
        if (!hit) {
            if (bad) *bad = k;
            return false;
        }
    }
    return true;
}

const std::string* Args::find(const std::string& key) const {
    for (const auto& kv : kv_)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

bool Args::has(const std::string& key) const { return find(key) != nullptr; }

std::string Args::str(const std::string& key, const std::string& def) const {
    const auto* v = find(key);
    return v == nullptr ? def : *v;
}

std::int64_t Args::i64(const std::string& key, std::int64_t def, bool* ok) const {
    const auto* v = find(key);
    if (v == nullptr) { *ok = true; return def; }
    if (v->empty()) { *ok = false; return def; }
    char* end = nullptr;
    const long long parsed = std::strtoll(v->c_str(), &end, 10);
    if (end == nullptr || *end != '\0') { *ok = false; return def; }
    *ok = true;
    return static_cast<std::int64_t>(parsed);
}

std::vector<std::int64_t> Args::i64_list(const std::string& key,
                                         const std::vector<std::int64_t>& def,
                                         bool* ok) const {
    const auto* v = find(key);
    if (v == nullptr) { *ok = true; return def; }
    std::vector<std::int64_t> out;
    std::size_t begin = 0;
    while (begin <= v->size()) {
        const std::size_t comma = v->find(',', begin);
        const std::string tok =
            v->substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (tok.empty()) { *ok = false; return def; }
        char* end = nullptr;
        const long long parsed = std::strtoll(tok.c_str(), &end, 10);
        if (end == nullptr || *end != '\0') { *ok = false; return def; }
        out.push_back(static_cast<std::int64_t>(parsed));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    *ok = true;
    return out;
}

// ---- stats -----------------------------------------------------------------

Stats summarize(std::vector<double> xs) {
    Stats s;
    if (xs.empty()) return s;
    std::sort(xs.begin(), xs.end());
    const std::size_t n = xs.size();
    const auto at = [&](double q) {
        const auto idx = static_cast<std::size_t>(std::llround(q * static_cast<double>(n - 1)));
        return xs[idx];
    };
    s.min = xs.front();
    s.max = xs.back();
    s.p10 = at(0.10);
    s.p50 = at(0.50);
    s.p90 = at(0.90);
    s.p95 = at(0.95);
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    s.mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double x : xs) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / static_cast<double>(n));  // 母標準偏差
    return s;
}

// ---- output ----------------------------------------------------------------

void kv_print(const char* key, double value, int decimals) {
    std::printf("%s=%.*f\n", key, decimals, value);
}

void kv_print(const char* key, std::int64_t value) {
    std::printf("%s=%lld\n", key, static_cast<long long>(value));
}

void kv_print(const char* key, const std::string& value) {
    std::printf("%s=%s\n", key, value.c_str());
}

std::string csv_join(const std::vector<std::string>& row) {
    std::string out;
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i) out += ',';
        out += row[i];
    }
    return out;
}

bool append_csv(const std::string& path, const std::string& header,
                const std::vector<std::string>& row, std::string* error) {
    std::error_code ec;
    const bool existed = std::filesystem::exists(path, ec);
    const bool empty = !existed || std::filesystem::file_size(path, ec) == 0 || ec;
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (f == nullptr) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    if (empty) std::fprintf(f, "%s\n", header.c_str());
    std::fprintf(f, "%s\n", csv_join(row).c_str());
    std::fclose(f);
    return true;
}

// ---- HIP timer -------------------------------------------------------------

#ifdef SHELF_HAVE_HIP
StreamTimer::~StreamTimer() {
    if (ev0_) (void)hipEventDestroy(ev0_);
    if (ev1_) (void)hipEventDestroy(ev1_);
    if (stream_) (void)hipStreamDestroy(stream_);
}

bool StreamTimer::init(std::string* error) {
    hipError_t e = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (e != hipSuccess) { if (error) *error = std::string("stream: ") + hipGetErrorString(e); return false; }
    e = hipEventCreateWithFlags(&ev0_, hipEventDefault);
    if (e != hipSuccess) { if (error) *error = std::string("event0: ") + hipGetErrorString(e); return false; }
    e = hipEventCreateWithFlags(&ev1_, hipEventDefault);
    if (e != hipSuccess) { if (error) *error = std::string("event1: ") + hipGetErrorString(e); return false; }
    return true;
}
#endif  // SHELF_HAVE_HIP

}  // namespace shelf::bench
