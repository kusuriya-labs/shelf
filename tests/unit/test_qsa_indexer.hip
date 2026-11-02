#include <phaseshift/core/memory/types.h>
#include <phaseshift/kernels/optimized/attention/qsa_indexer.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef PHASESHIFT_SOURCE_ROOT
#error "PHASESHIFT_SOURCE_ROOT must be defined"
#endif

namespace {

using ps::bf16_t;
using ps::kernel::qsa_indexer_scratch_bytes;

constexpr uint32_t kRows = 2053;
constexpr uint32_t kHeads = 4;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kRatio = 4;
constexpr uint32_t kTopK = 512;
constexpr uint32_t kRotary = 64;
constexpr float kTheta = 10000000.0f;
constexpr float kEps = 1e-6f;
constexpr uint64_t kSeed = 0x5153415F49445831ull;
constexpr uint64_t kGoldenG = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kMixC1 = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t kMixC2 = 0x94D049BB133111EBull;
constexpr uint32_t kScoresStride = kRows / kRatio;

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mse = 0.0;
    uint64_t count = 0;
    uint32_t worst_index = 0;
};

Metrics score_metrics(const float* gpu, const float* ref, uint64_t n) {
    Metrics m;
    double scale = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double v = std::fabs(static_cast<double>(ref[i]));
        if (v > scale) scale = v;
    }
    const double floor = scale > 0.0 ? scale * 1e-2 : 1e-6;
    double sq = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(gpu[i]);
        const double y = static_cast<double>(ref[i]);
        const double d = std::fabs(x - y);
        const double denom = std::fabs(y) > floor ? std::fabs(y) : floor;
        if (d > m.max_abs) {
            m.max_abs = d;
            m.worst_index = static_cast<uint32_t>(i);
        }
        const double r = d / denom;
        if (r > m.max_rel) m.max_rel = r;
        sq += d * d;
    }
    m.count = n;
    m.mse = n > 0 ? sq / static_cast<double>(n) : 0.0;
    return m;
}

Metrics compare_floats(const float* a, const float* b, uint64_t n) {
    Metrics m;
    double sq = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        const double d = std::fabs(x - y);
        const double denom = std::fabs(y) > 1e-6 ? std::fabs(y) : 1e-6;
        if (d > m.max_abs) {
            m.max_abs = d;
            m.worst_index = static_cast<uint32_t>(i);
        }
        const double r = d / denom;
        if (r > m.max_rel) m.max_rel = r;
        sq += d * d;
    }
    m.count = n;
    m.mse = n > 0 ? sq / static_cast<double>(n) : 0.0;
    return m;
}

void report(const char* label, const Metrics& m) {
    std::printf("METRIC %s max_abs=%.9g max_rel=%.9g RMSE=%.9g n=%llu\n", label, m.max_abs,
                m.max_rel, std::sqrt(m.mse), static_cast<unsigned long long>(m.count));
}

bool sorted_blocks_equal(const uint32_t* a, const uint32_t* b, uint32_t n) {
    std::vector<uint32_t> va(a, a + n), vb(b, b + n);
    std::sort(va.begin(), va.end());
    std::sort(vb.begin(), vb.end());
    return va == vb;
}

void check_metrics(const char* label, const Metrics& m, double abs_tol, double rel_tol,
                  double rmse_tol) {
    const double rmse = std::sqrt(m.mse);
    const bool ok = m.max_abs <= abs_tol && m.max_rel <= rel_tol && rmse <= rmse_tol;
    if (!ok) {
        std::printf("METRIC-FAIL %s max_abs=%.9g tol=%.9g max_rel=%.9g tol=%.9g RMSE=%.9g "
                    "tol=%.9g\n",
                    label, m.max_abs, abs_tol, m.max_rel, rel_tol, rmse, rmse_tol);
    }
    check(ok, label);
}

struct SplitMix {
    uint64_t state = 0;

    uint64_t next_u64() {
        state += kGoldenG;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * kMixC1;
        z = (z ^ (z >> 27)) * kMixC2;
        z = z ^ (z >> 31);
        return z;
    }

    float next_f32() {
        const uint32_t u = static_cast<uint32_t>(next_u64() >> 32);
        return static_cast<float>(u) * (2.0f / 4294967296.0f) - 1.0f;
    }
};

uint32_t crc32(const void* data, size_t n) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256u; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        ready = true;
    }
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

float bf16_to_f32_host(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t f32_to_bf16_host(float f) {
    return ps::f32_to_bf16_rne(f);
}

void rope_host(const float* in, float* out, uint32_t rows, uint32_t features, uint32_t head_dim,
               uint32_t rotary, uint32_t position_base, uint32_t position_step, float theta) {
    const uint32_t half = rotary / 2u;
    for (uint32_t r = 0; r < rows; ++r) {
        const float* src = in + static_cast<size_t>(r) * features;
        float* dst = out + static_cast<size_t>(r) * features;
        const double pos = static_cast<double>(position_base + r * position_step);
        for (uint32_t h = 0; h < features / head_dim; ++h) {
            const float* x = src + static_cast<size_t>(h) * head_dim;
            float* y = dst + static_cast<size_t>(h) * head_dim;
            for (uint32_t c = 0; c < head_dim; ++c) y[c] = bf16_to_f32_host(f32_to_bf16_host(x[c]));
            for (uint32_t c = 0; c < rotary; ++c) {
                const uint32_t pi = c < half ? c : c - half;
                const double freq = 1.0 / std::pow(static_cast<double>(theta),
                                                   static_cast<double>(pi) /
                                                       static_cast<double>(half));
                const double angle = pos * freq;
                const float cs = static_cast<float>(std::cos(angle));
                const float sn = static_cast<float>(std::sin(angle));
                const uint32_t partner = c < half ? c + half : c - half;
                const float v = c < half ? x[c] * cs - x[partner] * sn
                                         : x[c] * cs + x[partner] * sn;
                y[c] = bf16_to_f32_host(f32_to_bf16_host(v));
            }
        }
    }
}

struct IndexerOracle {
    std::vector<float> scores;
    std::vector<uint32_t> selected;
    std::vector<uint32_t> counts;
    std::vector<uint8_t> mask;
};

void rmsnorm_group_host(const float* in, const uint16_t* weight, float* out, uint32_t rows,
                        uint32_t features, uint32_t group, float eps) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t g = 0; g < features / group; ++g) {
            const float* x = in + static_cast<size_t>(r) * features + g * group;
            float* y = out + static_cast<size_t>(r) * features + g * group;
            float sum = 0.0f;
            for (uint32_t d = 0; d < group; ++d) sum += x[d] * x[d];
            const float inv = 1.0f / std::sqrt(sum / static_cast<float>(group) + eps);
            for (uint32_t d = 0; d < group; ++d)
                y[d] = (1.0f + bf16_to_f32_host(weight[d])) * (x[d] * inv);
        }
    }
}

IndexerOracle run_oracle(const std::vector<uint16_t>& q_raw, const std::vector<uint16_t>& keys,
                         const std::vector<uint16_t>& q_weight,
                         const std::vector<uint16_t>& k_weight, uint32_t rows) {
    IndexerOracle o;
    const uint32_t q_features = kHeads * kHeadDim;
    const uint32_t n_blocks = rows / kRatio;
    std::vector<float> qn(static_cast<size_t>(rows) * q_features);
    std::vector<float> q_rope(static_cast<size_t>(rows) * q_features);
    std::vector<float> q_bf16(static_cast<size_t>(rows) * q_features);
    for (size_t i = 0; i < qn.size(); ++i) qn[i] = bf16_to_f32_host(q_raw[i]);
    rmsnorm_group_host(qn.data(), q_weight.data(), q_rope.data(), rows, q_features, kHeadDim,
                       kEps);
    rope_host(q_rope.data(), q_bf16.data(), rows, q_features, kHeadDim, kRotary, 0, 1, kTheta);

    std::vector<float> pooled(static_cast<size_t>(n_blocks) * kHeadDim);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < kRatio; ++t)
                acc += bf16_to_f32_host(
                    keys[static_cast<size_t>(b * kRatio + t) * kHeadDim + d]);
            pooled[static_cast<size_t>(b) * kHeadDim + d] =
                acc / static_cast<float>(kRatio);
        }
    }
    std::vector<float> kn(pooled.size());
    std::vector<float> block_bf16(pooled.size());
    rmsnorm_group_host(pooled.data(), k_weight.data(), kn.data(), n_blocks, kHeadDim, kHeadDim,
                       kEps);
    rope_host(kn.data(), block_bf16.data(), n_blocks, kHeadDim, kHeadDim, kRotary, 0,
                  kRatio, kTheta);

    o.scores.assign(static_cast<size_t>(rows) * kScoresStride, 0.0f);
    o.selected.assign(static_cast<size_t>(rows) * kTopK, 0xDEADBEEFu);
    o.counts.assign(rows, 0u);
    o.mask.assign(static_cast<size_t>(rows) * rows, 0u);

    struct Entry {
        float score;
        uint32_t id;
    };
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    std::vector<Entry> entries;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t ncb = (i + 1u) / kRatio;
        const uint32_t k = ncb < kTopK ? ncb : kTopK;
        float* row_scores = o.scores.data() + static_cast<size_t>(i) * kScoresStride;
        entries.clear();
        for (uint32_t b = 0; b < ncb; ++b) {
            float acc = 0.0f;
            for (uint32_t h = 0; h < kHeads; ++h) {
                const float* qh =
                    q_bf16.data() + static_cast<size_t>(i) * q_features + h * kHeadDim;
                const float* bk = block_bf16.data() + static_cast<size_t>(b) * kHeadDim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < kHeadDim; ++d) dot += qh[d] * bk[d];
                acc += dot > 0.0f ? dot : 0.0f;
            }
            row_scores[b] = acc * scale;
            entries.push_back(Entry{row_scores[b], b});
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.id < b.id;
        });
        uint32_t* row_sel = o.selected.data() + static_cast<size_t>(i) * kTopK;
        for (uint32_t s = 0; s < k; ++s) row_sel[s] = entries[s].id;
        o.counts[i] = k;
        for (uint32_t s = 0; s < k; ++s) {
            const uint32_t base = entries[s].id * kRatio;
            for (uint32_t t = 0; t < kRatio; ++t)
                o.mask[static_cast<size_t>(i) * rows + base + t] = 1u;
        }
        for (uint32_t tok = ncb * kRatio; tok <= i; ++tok)
            o.mask[static_cast<size_t>(i) * rows + tok] = 1u;
    }
    return o;
}

struct Reader {
    const std::vector<uint8_t>& data;
    size_t pos = 0;

    void bytes(void* out, size_t n) {
        std::memcpy(out, data.data() + pos, n);
        pos += n;
    }

    template <typename T>
    T pod() {
        T v{};
        bytes(&v, sizeof(T));
        return v;
    }

    std::string text(size_t n) {
        std::string s(reinterpret_cast<const char*>(data.data() + pos), n);
        pos += n;
        return s;
    }
};

struct GoldenRow {
    uint32_t row = 0;
    uint32_t ncb = 0;
    uint32_t k = 0;
    std::vector<float> scores;
    std::vector<uint32_t> blocks;
    std::vector<uint8_t> mask;
};

struct Golden {
    uint32_t rows = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t ratio = 0;
    uint32_t topk = 0;
    uint32_t crc = 0;
    std::vector<GoldenRow> entries;
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.size() < 64) return false;
    Reader r{buf};
    if (r.text(8) != "PSQSAIDX") return false;
    const uint32_t version = r.pod<uint32_t>();
    g.rows = r.pod<uint32_t>();
    const uint64_t seed = r.pod<uint64_t>();
    g.crc = static_cast<uint32_t>(r.pod<uint64_t>());
    g.heads = r.pod<uint32_t>();
    g.head_dim = r.pod<uint32_t>();
    g.ratio = r.pod<uint32_t>();
    g.topk = r.pod<uint32_t>();
    const uint32_t rotary = r.pod<uint32_t>();
    const uint32_t n_critical = r.pod<uint32_t>();
    (void)r.pod<uint32_t>();
    (void)r.pod<uint32_t>();
    const float theta = r.pod<float>();
    const float eps = r.pod<float>();
    const float scale = r.pod<float>();
    if (version != 1 || g.rows != kRows || g.heads != kHeads || g.head_dim != kHeadDim ||
        g.ratio != kRatio || g.topk != kTopK || rotary != kRotary || seed != kSeed)
        return false;
    if (theta != kTheta || eps != kEps) return false;
    if (scale <= 0.0f) return false;
    std::vector<uint32_t> critical(n_critical);
    r.bytes(critical.data(), n_critical * sizeof(uint32_t));
    for (uint32_t idx = 0; idx < n_critical; ++idx) {
        GoldenRow row;
        row.row = r.pod<uint32_t>();
        row.ncb = r.pod<uint32_t>();
        row.k = r.pod<uint32_t>();
        row.scores.resize(row.ncb);
        r.bytes(row.scores.data(), row.ncb * sizeof(float));
        row.blocks.resize(row.k);
        r.bytes(row.blocks.data(), row.k * sizeof(uint32_t));
        row.mask.resize(g.rows);
        r.bytes(row.mask.data(), g.rows);
        if (row.row != critical[idx]) return false;
        g.entries.push_back(std::move(row));
    }
    return r.pos == buf.size();
}

struct GpuResult {
    std::vector<float> scores;
    std::vector<uint32_t> selected;
    std::vector<uint32_t> counts;
    std::vector<uint8_t> mask;
};

bool run_gpu(const std::vector<uint16_t>& q_raw, const std::vector<uint16_t>& keys,
             const std::vector<uint16_t>& q_weight, const std::vector<uint16_t>& k_weight,
             uint32_t query_begin, uint32_t query_rows, uint32_t key_tokens, bool zero_q,
             GpuResult& out) {
    const uint32_t q_features = kHeads * kHeadDim;
    std::vector<uint16_t> q_src(
        q_raw.begin() + static_cast<size_t>(query_begin) * q_features,
        q_raw.begin() + static_cast<size_t>(query_begin + query_rows) * q_features);
    if (zero_q) std::fill(q_src.begin(), q_src.end(), 0u);

    bf16_t* d_q = nullptr;
    bf16_t* d_keys = nullptr;
    bf16_t* d_qw = nullptr;
    bf16_t* d_kw = nullptr;
    uint32_t* d_pos = nullptr;
    float* d_scores = nullptr;
    uint32_t* d_sel = nullptr;
    uint32_t* d_counts = nullptr;
    uint8_t* d_mask = nullptr;
    void* d_scratch = nullptr;
    bool ok = true;

    const size_t q_bytes = q_src.size() * sizeof(uint16_t);
    const size_t key_bytes = static_cast<size_t>(key_tokens) * kHeadDim * sizeof(uint16_t);
    const size_t score_count = static_cast<size_t>(query_rows) * kScoresStride;
    const size_t sel_count = static_cast<size_t>(query_rows) * kTopK;
    const size_t mask_count = static_cast<size_t>(query_rows) * key_tokens;
    const size_t scratch = qsa_indexer_scratch_bytes(query_rows, key_tokens, kHeads, kHeadDim,
                                                     kRatio);

    auto alloc = [&](void** p, size_t n) { return hipMalloc(p, n) == hipSuccess; };
    ok = ok && alloc(reinterpret_cast<void**>(&d_q), q_bytes);
    ok = ok && alloc(reinterpret_cast<void**>(&d_keys), key_bytes);
    ok = ok && alloc(reinterpret_cast<void**>(&d_qw), kHeadDim * sizeof(uint16_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d_kw), kHeadDim * sizeof(uint16_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d_pos), key_tokens * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d_scores), score_count * sizeof(float));
    ok = ok && alloc(reinterpret_cast<void**>(&d_sel), sel_count * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d_counts), query_rows * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d_mask), mask_count);
    ok = ok && alloc(&d_scratch, scratch);
    if (!ok) {
        std::printf("FAIL: device allocation\n");
        return false;
    }

    std::vector<uint32_t> positions(key_tokens);
    for (uint32_t i = 0; i < key_tokens; ++i) positions[i] = i;
    std::vector<uint32_t> sel_fill(sel_count, 0xDEADBEEFu);

    (void)hipMemcpy(d_q, q_src.data(), q_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_keys, keys.data(), key_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_qw, q_weight.data(), kHeadDim * sizeof(uint16_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_kw, k_weight.data(), kHeadDim * sizeof(uint16_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_pos, positions.data(), key_tokens * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(d_sel, sel_fill.data(), sel_count * sizeof(uint32_t),
                    hipMemcpyHostToDevice);

    ps::kernel::QsaIndexerArgs args;
    args.q_raw = reinterpret_cast<const bf16_t*>(d_q);
    args.key_cache = reinterpret_cast<const bf16_t*>(d_keys);
    args.key_positions = d_pos;
    args.q_weight = reinterpret_cast<const bf16_t*>(d_qw);
    args.k_weight = reinterpret_cast<const bf16_t*>(d_kw);
    args.query_begin = query_begin;
    args.query_rows = query_rows;
    args.key_tokens = key_tokens;
    args.scores = d_scores;
    args.scores_stride = kScoresStride;
    args.selected_blocks = d_sel;
    args.selected_block_counts = d_counts;
    args.mask = d_mask;
    args.scratch = d_scratch;
    args.scratch_bytes = scratch;

    const hipError_t err = ps::kernel::launch_qsa_indexer(args);
    if (err != hipSuccess) {
        std::printf("FAIL: launch_qsa_indexer: %s\n", hipGetErrorString(err));
        (void)hipDeviceSynchronize();
        hipFree(d_q); hipFree(d_keys); hipFree(d_qw); hipFree(d_kw); hipFree(d_pos);
        hipFree(d_scores); hipFree(d_sel); hipFree(d_counts); hipFree(d_mask);
        hipFree(d_scratch);
        return false;
    }
    const hipError_t sync = hipDeviceSynchronize();
    if (sync != hipSuccess) {
        std::printf("FAIL: synchronize: %s\n", hipGetErrorString(sync));
    }

    out.scores.resize(score_count);
    out.selected.resize(sel_count);
    out.counts.resize(query_rows);
    out.mask.resize(mask_count);
    (void)hipMemcpy(out.scores.data(), d_scores, score_count * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(out.selected.data(), d_sel, sel_count * sizeof(uint32_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(out.counts.data(), d_counts, query_rows * sizeof(uint32_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(out.mask.data(), d_mask, mask_count, hipMemcpyDeviceToHost);

    hipFree(d_q); hipFree(d_keys); hipFree(d_qw); hipFree(d_kw); hipFree(d_pos);
    hipFree(d_scores); hipFree(d_sel); hipFree(d_counts); hipFree(d_mask);
    hipFree(d_scratch);
    return sync == hipSuccess && err == hipSuccess;
}

void compare_rows(const GpuResult& gpu, const IndexerOracle& oracle, uint32_t begin, uint32_t end,
                  const char* tag) {
    std::vector<float> gs_all;
    std::vector<float> os_all;
    gs_all.reserve(static_cast<size_t>(end - begin) * kScoresStride);
    os_all.reserve(static_cast<size_t>(end - begin) * kScoresStride);
    bool ids_ok = true;
    bool counts_ok = true;
    bool mask_ok = true;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t ncb = (i + 1u) / kRatio;
        const float* gs = gpu.scores.data() + static_cast<size_t>(i) * kScoresStride;
        const float* os = oracle.scores.data() + static_cast<size_t>(i) * kScoresStride;
        gs_all.insert(gs_all.end(), gs, gs + ncb);
        os_all.insert(os_all.end(), os, os + ncb);
        for (uint32_t b = ncb; b < kScoresStride; ++b) {
            if (gs[b] != 0.0f) ids_ok = false;
        }
        if (gpu.counts[i] != oracle.counts[i]) counts_ok = false;
        const uint32_t* gsel = gpu.selected.data() + static_cast<size_t>(i) * kTopK;
        const uint32_t* osel = oracle.selected.data() + static_cast<size_t>(i) * kTopK;
        if (!sorted_blocks_equal(gsel, osel, kTopK)) ids_ok = false;
        const uint8_t* gmask = gpu.mask.data() + static_cast<size_t>(i) * kRows;
        const uint8_t* omask = oracle.mask.data() + static_cast<size_t>(i) * kRows;
        if (std::memcmp(gmask, omask, kRows) != 0) mask_ok = false;
    }
    const Metrics m = score_metrics(gs_all.data(), os_all.data(), gs_all.size());
    report(tag, m);
    check_metrics(tag, m, 6.0e-2, 2.5e-1, 1.0e-2);
    if (std::getenv("QSA_DEBUG") != nullptr) {
        for (uint32_t i : {100u, 2047u}) {
            if (i < begin || i >= end) continue;
            const float* gs = gpu.scores.data() + static_cast<size_t>(i) * kScoresStride;
            const float* os = oracle.scores.data() + static_cast<size_t>(i) * kScoresStride;
            const uint32_t ncb = (i + 1u) / kRatio;
            std::printf("DBG row=%u ncb=%u\n", i, ncb);
            for (uint32_t b = 0; b < ncb && b < 6u; ++b)
                std::printf("  b=%u gpu=%.9g cpu=%.9g\n", b, gs[b], os[b]);
        }
    }
    check(counts_ok, (std::string(tag) + " counts").c_str());
    check(ids_ok, (std::string(tag) + " selected block ids / invalid scores").c_str());
    check(mask_ok, (std::string(tag) + " mask bitwise").c_str());
}

std::string golden_path(const char* name) {
    return std::string(PHASESHIFT_SOURCE_ROOT) + "/tests/fixtures/qsa/" + name;
}

}

int main() {
    const uint32_t q_features = kHeads * kHeadDim;
    std::vector<uint16_t> q_raw(static_cast<size_t>(kRows) * q_features);
    std::vector<uint16_t> keys(static_cast<size_t>(kRows) * kHeadDim);
    std::vector<uint16_t> q_weight(kHeadDim);
    std::vector<uint16_t> k_weight(kHeadDim);

    auto fill = [](std::vector<uint16_t>& v, uint64_t seed) {
        SplitMix sm{seed};
        for (auto& x : v) x = f32_to_bf16_host(sm.next_f32());
    };
    fill(q_raw, kSeed);
    fill(keys, kSeed ^ 1ull);
    fill(q_weight, kSeed ^ 2ull);
    fill(k_weight, kSeed ^ 3ull);

    std::vector<uint8_t> payload;
    payload.reserve(q_raw.size() * 2 + keys.size() * 2 + (q_weight.size() + k_weight.size()) * 2);
    auto append = [&payload](const std::vector<uint16_t>& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
        payload.insert(payload.end(), p, p + v.size() * 2);
    };
    append(q_raw);
    append(keys);
    append(q_weight);
    append(k_weight);
    const uint32_t input_crc = crc32(payload.data(), payload.size());
    std::printf("INPUT crc32=%08x rows=%u seed=%016llx\n", input_crc, kRows,
                static_cast<unsigned long long>(kSeed));

    Golden golden;
    const bool has_golden = load_golden(golden_path("qsa_indexer_golden.bin"), golden);
    check(has_golden, "golden load");
    if (has_golden) {
        check(golden.crc == input_crc, "golden input checksum");
    }

    const IndexerOracle oracle = run_oracle(q_raw, keys, q_weight, k_weight, kRows);

    GpuResult gpu;
    if (!run_gpu(q_raw, keys, q_weight, k_weight, 0, kRows, kRows, false, gpu)) return 1;
    compare_rows(gpu, oracle, 0, kRows, "prefill gpu-vs-cpu-oracle");
    check(gpu.counts[0] == 0 && gpu.counts[3] == 1, "tail length 0..3 coverage");

    if (has_golden) {
        bool blocks_ok = true;
        bool mask_ok = true;
        std::vector<float> gs_all;
        std::vector<float> ref_all;
        for (const auto& e : golden.entries) {
            const float* gs = gpu.scores.data() + static_cast<size_t>(e.row) * kScoresStride;
            gs_all.insert(gs_all.end(), gs, gs + e.ncb);
            ref_all.insert(ref_all.end(), e.scores.data(), e.scores.data() + e.ncb);
            const uint32_t* gsel = gpu.selected.data() + static_cast<size_t>(e.row) * kTopK;
            if (!sorted_blocks_equal(gsel, e.blocks.data(), e.k)) blocks_ok = false;
            const uint8_t* gmask = gpu.mask.data() + static_cast<size_t>(e.row) * kRows;
            if (std::memcmp(gmask, e.mask.data(), kRows) != 0) mask_ok = false;
        }
        const Metrics m =
            score_metrics(gs_all.data(), ref_all.data(), gs_all.size());
        report("prefill gpu-vs-pytorch-golden", m);
        check_metrics("prefill gpu-vs-pytorch-golden tolerance", m, 5.0e-2, 2.5e-1, 1.0e-2);
        check(blocks_ok, "golden selected block ids bitwise");
        check(mask_ok, "golden mask bitwise");
    }

    bool causal_ok = true;
    bool self_ok = true;
    for (uint32_t i = 0; i < kRows; ++i) {
        const uint8_t* row = gpu.mask.data() + static_cast<size_t>(i) * kRows;
        for (uint32_t j = i + 1; j < kRows; ++j)
            if (row[j] != 0) causal_ok = false;
        const uint32_t ncb = (i + 1u) / kRatio;
        const bool self_required = ((i + 1u) % kRatio != 0u) || (ncb <= kTopK);
        if (self_required && row[i] == 0) self_ok = false;
    }
    check(causal_ok, "mask respects causality");
    check(self_ok, "mask contains the current token whenever the tail or a full "
                 "block selection guarantees it");

    GpuResult decode;
    if (!run_gpu(q_raw, keys, q_weight, k_weight, kRows - 8, 8, kRows, false, decode)) return 1;
    bool decode_ok = true;
    for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t i = kRows - 8 + r;
        const float* a = decode.scores.data() + static_cast<size_t>(r) * kScoresStride;
        const float* b = gpu.scores.data() + static_cast<size_t>(i) * kScoresStride;
        const uint32_t ncb = (i + 1u) / kRatio;
        for (uint32_t x = 0; x < ncb; ++x)
            if (a[x] != b[x]) {
                decode_ok = false;
                break;
            }
        const uint32_t* as = decode.selected.data() + static_cast<size_t>(r) * kTopK;
        const uint32_t* bs = gpu.selected.data() + static_cast<size_t>(i) * kTopK;
        if (!sorted_blocks_equal(as, bs, kTopK)) decode_ok = false;
        const uint8_t* am = decode.mask.data() + static_cast<size_t>(r) * kRows;
        const uint8_t* bm = gpu.mask.data() + static_cast<size_t>(i) * kRows;
        if (std::memcmp(am, bm, kRows) != 0) decode_ok = false;
        if (decode.counts[r] != gpu.counts[i]) decode_ok = false;
    }
    check(decode_ok, "decode rows match the corresponding prefill rows");

    GpuResult prefill_prefix;
    if (!run_gpu(q_raw, keys, q_weight, k_weight, 0, kRows - 1, kRows - 1, false,
                 prefill_prefix))
        return 1;
    GpuResult appended;
    if (!run_gpu(q_raw, keys, q_weight, k_weight, kRows - 1, 1, kRows, false, appended))
        return 1;
    {
        const uint32_t i = kRows - 1;
        const float* a = appended.scores.data();
        const float* b = gpu.scores.data() + static_cast<size_t>(i) * kScoresStride;
        const uint32_t ncb = (i + 1u) / kRatio;
        bool ok = true;
        for (uint32_t x = 0; x < ncb; ++x)
            if (a[x] != b[x]) ok = false;
        check(ok, "state append keeps the raw key cache consistent");
    }
    {
        bool prefix_ok = true;
        for (uint32_t i = 0; i + 1u < kRows; ++i) {
            const uint32_t ncb = (i + 1u) / kRatio;
            const float* a = prefill_prefix.scores.data() + static_cast<size_t>(i) * kScoresStride;
            const float* b = gpu.scores.data() + static_cast<size_t>(i) * kScoresStride;
            for (uint32_t x = 0; x < ncb; ++x)
                if (a[x] != b[x]) prefix_ok = false;
            if (!sorted_blocks_equal(prefill_prefix.selected.data() +
                                         static_cast<size_t>(i) * kTopK,
                                     gpu.selected.data() + static_cast<size_t>(i) * kTopK,
                                     kTopK))
                prefix_ok = false;
            if (std::memcmp(prefill_prefix.mask.data() + static_cast<size_t>(i) * (kRows - 1u),
                            gpu.mask.data() + static_cast<size_t>(i) * kRows,
                            kRows - 1u) != 0)
                prefix_ok = false;
        }
        check(prefix_ok, "indexer state prefix consistency");
    }

    GpuResult tie;
    if (!run_gpu(q_raw, keys, q_weight, k_weight, 0, 512, 512, true, tie)) return 1;
    bool tie_scores = true;
    bool tie_blocks = true;
    for (uint32_t i = 0; i < 512; ++i) {
        const uint32_t ncb = (i + 1u) / kRatio;
        const uint32_t k = ncb < kTopK ? ncb : kTopK;
        const float* s = tie.scores.data() + static_cast<size_t>(i) * kScoresStride;
        for (uint32_t b = 0; b < ncb; ++b)
            if (s[b] != 0.0f) tie_scores = false;
        const uint32_t* sel = tie.selected.data() + static_cast<size_t>(i) * kTopK;
        for (uint32_t x = 0; x < k; ++x)
            if (sel[x] != x) tie_blocks = false;
        if (tie.counts[i] != k) tie_blocks = false;
    }
    check(tie_scores, "top-k tie: all scores are exactly zero");
    check(tie_blocks, "top-k tie: lowest block ids win deterministically");

    std::printf("test_qsa_indexer: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
