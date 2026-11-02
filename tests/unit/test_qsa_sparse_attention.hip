#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/attention/qsa_sparse_attention.h>
#include <shelf/kernels/optimized/attention/rope.h>
#include <shelf/kernels/optimized/rmsnorm.h>

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

#ifndef SHELF_SOURCE_ROOT
#error "SHELF_SOURCE_ROOT must be defined"
#endif

namespace {

using ps::bf16_t;
using ps::kernel::KVCacheDType;
using ps::kernel::PagedAttentionCommonArgs;
using ps::kernel::PagedAttentionOutputDType;
using ps::kernel::QsaSparseAttentionArgs;
using ps::kernel::QsaSparseAttentionOutputDType;
using ps::runtime::ValueDType;

constexpr uint32_t kRows = 4096;
constexpr uint32_t kQHeads = 24;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kRotary = 64;
constexpr uint32_t kFeatures = kQHeads * kHeadDim;
constexpr uint32_t kKvFeatures = kKvHeads * kHeadDim;
constexpr uint32_t kPageTokens = 16;
constexpr uint32_t kNumPages = 512;
constexpr uint32_t kNumLayers = 1;
constexpr uint32_t kElemsPerToken = kKvFeatures;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr uint32_t kDenseRows = 64;
constexpr uint32_t kDecodeRows = 8;
constexpr uint32_t kMaskStride = 8;
constexpr uint32_t kMaskKeep = 2;
constexpr float kTheta = 10000000.0f;
constexpr float kEps = 1e-6f;
constexpr float kScale = 1.0f / 16.0f;
constexpr uint64_t kSeed = 0x5153415F53415454ull;
constexpr uint64_t kGoldenG = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kMixC1 = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t kMixC2 = 0x94D049BB133111EBull;

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
        const double d = std::fabs(static_cast<double>(gpu[i]) - static_cast<double>(ref[i]));
        const double denom = std::fabs(static_cast<double>(ref[i])) > floor
                                 ? std::fabs(static_cast<double>(ref[i]))
                                 : floor;
        if (d > m.max_abs) m.max_abs = d;
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

float bf16_quant_host(float f) {
    return bf16_to_f32_host(f32_to_bf16_host(f));
}

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

void rope_host(const float* in, float* out, uint32_t rows, uint32_t features, uint32_t head_dim,
               uint32_t rotary, float theta) {
    const uint32_t half = rotary / 2u;
    for (uint32_t r = 0; r < rows; ++r) {
        const float* src = in + static_cast<size_t>(r) * features;
        float* dst = out + static_cast<size_t>(r) * features;
        for (uint32_t h = 0; h < features / head_dim; ++h) {
            const float* x = src + static_cast<size_t>(h) * head_dim;
            float* y = dst + static_cast<size_t>(h) * head_dim;
            for (uint32_t c = 0; c < head_dim; ++c) y[c] = bf16_quant_host(x[c]);
            for (uint32_t c = 0; c < rotary; ++c) {
                const uint32_t pi = c < half ? c : c - half;
                const double freq = 1.0 / std::pow(static_cast<double>(theta),
                                                   static_cast<double>(pi) /
                                                       static_cast<double>(half));
                const double angle = static_cast<double>(r) * freq;
                const float cs = static_cast<float>(std::cos(angle));
                const float sn = static_cast<float>(std::sin(angle));
                const uint32_t partner = c < half ? c + half : c - half;
                const float v = c < half ? x[c] * cs - x[partner] * sn
                                         : x[c] * cs + x[partner] * sn;
                y[c] = bf16_quant_host(v);
            }
        }
    }
}

void prepare_host(const std::vector<uint16_t>& query_raw, const std::vector<uint16_t>& k_raw,
                  const std::vector<uint16_t>& v_raw, const std::vector<uint16_t>& q_weight,
                  const std::vector<uint16_t>& k_weight, std::vector<float>& q_host,
                  std::vector<float>& k_host, std::vector<float>& v_host,
                  std::vector<float>& q_normed_host) {
    std::vector<float> tmp(q_host.size());
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = bf16_to_f32_host(query_raw[i]);
    q_normed_host.resize(tmp.size());
    rmsnorm_group_host(tmp.data(), q_weight.data(), q_normed_host.data(), kRows, kFeatures,
                       kHeadDim, kEps);
    rope_host(q_normed_host.data(), q_host.data(), kRows, kFeatures, kHeadDim, kRotary, kTheta);

    std::vector<float> ktmp(k_host.size());
    std::vector<float> knorm(k_host.size());
    for (size_t i = 0; i < ktmp.size(); ++i) ktmp[i] = bf16_to_f32_host(k_raw[i]);
    rmsnorm_group_host(ktmp.data(), k_weight.data(), knorm.data(), kRows, kKvFeatures, kHeadDim,
                       kEps);
    rope_host(knorm.data(), k_host.data(), kRows, kKvFeatures, kHeadDim, kRotary, kTheta);

    v_host.resize(v_raw.size());
    for (size_t i = 0; i < v_host.size(); ++i)
        v_host[i] = bf16_quant_host(bf16_to_f32_host(v_raw[i]));
}

uint32_t page_perm(uint32_t logical) {
    return (logical * 137u + 41u) % kNumPages;
}

void build_dense_rule_mask(std::vector<uint8_t>& mask) {
    mask.assign(static_cast<size_t>(kRows) * kRows, 0u);
    for (uint32_t i = 0; i < kRows; ++i) {
        uint8_t* row = mask.data() + static_cast<size_t>(i) * kRows;
        if (i < kDenseRows) {
            for (uint32_t p = 0; p <= i; ++p) row[p] = 1u;
            continue;
        }
        const uint32_t ncb = (i + 1u) / 4u;
        for (uint32_t b = 0; b < ncb; ++b) {
            if ((b % kMaskStride) >= kMaskKeep) continue;
            for (uint32_t t = 0; t < 4u; ++t) row[b * 4u + t] = 1u;
        }
        for (uint32_t p = ncb * 4u; p <= i; ++p) row[p] = 1u;
    }
}

void run_oracle(const std::vector<float>& q_host, const std::vector<float>& k_host,
                const std::vector<float>& v_host, const std::vector<uint8_t>& mask,
                const std::vector<uint32_t>& rows, std::vector<float>& out) {
    out.assign(static_cast<size_t>(rows.size()) * kFeatures, 0.0f);
    const uint32_t group = kQHeads / kKvHeads;
    std::vector<float> scores;
    for (size_t ri = 0; ri < rows.size(); ++ri) {
        const uint32_t i = rows[ri];
        float* dst = out.data() + ri * kFeatures;
        const uint8_t* mrow = mask.data() + static_cast<size_t>(i) * kRows;
        scores.assign(static_cast<size_t>(i) + 1u, 0.0f);
        for (uint32_t h = 0; h < kQHeads; ++h) {
            const uint32_t kv = h / group;
            const float* qh = q_host.data() + static_cast<size_t>(i) * kFeatures + h * kHeadDim;
            float max_score = -3.402823466e+38f;
            size_t idx = 0;
            for (uint32_t p = 0; p <= i; ++p) {
                if (mrow[p] == 0u) continue;
                const float* kh = k_host.data() + static_cast<size_t>(p) * kKvFeatures +
                                  kv * kHeadDim;
                float sc = 0.0f;
                for (uint32_t d = 0; d < kHeadDim; ++d) sc += qh[d] * kh[d];
                sc *= kScale;
                scores[idx] = sc;
                if (sc > max_score) max_score = sc;
                ++idx;
            }
            const size_t count = idx;
            float sum = 0.0f;
            for (size_t s = 0; s < count; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum += scores[s];
            }
            float* oh = dst + h * kHeadDim;
            for (uint32_t d = 0; d < kHeadDim; ++d) oh[d] = 0.0f;
            idx = 0;
            for (uint32_t p = 0; p <= i; ++p) {
                if (mrow[p] == 0u) continue;
                const float w = scores[idx] / sum;
                ++idx;
                const float* vh = v_host.data() + static_cast<size_t>(p) * kKvFeatures +
                                  kv * kHeadDim;
                for (uint32_t d = 0; d < kHeadDim; ++d) oh[d] += w * vh[d];
            }
        }
    }
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

struct Golden {
    uint32_t rows = 0;
    uint32_t crc = 0;
    std::vector<uint32_t> row_ids;
    std::vector<std::vector<float>> outputs;
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.size() < 64) return false;
    Reader r{buf};
    if (r.text(8) != "PSQSASPT") return false;
    const uint32_t version = r.pod<uint32_t>();
    g.rows = r.pod<uint32_t>();
    const uint64_t seed = r.pod<uint64_t>();
    g.crc = static_cast<uint32_t>(r.pod<uint64_t>());
    const uint32_t q_heads = r.pod<uint32_t>();
    const uint32_t kv_heads = r.pod<uint32_t>();
    const uint32_t head_dim = r.pod<uint32_t>();
    const uint32_t rotary = r.pod<uint32_t>();
    const uint32_t page_tokens = r.pod<uint32_t>();
    const uint32_t num_pages = r.pod<uint32_t>();
    const uint32_t mask_stride = r.pod<uint32_t>();
    const uint32_t mask_keep = r.pod<uint32_t>();
    const uint32_t n_rows = r.pod<uint32_t>();
    const uint32_t features = r.pod<uint32_t>();
    const float theta = r.pod<float>();
    const float eps = r.pod<float>();
    const float scale = r.pod<float>();
    if (version != 1 || g.rows != kRows || q_heads != kQHeads || kv_heads != kKvHeads ||
        head_dim != kHeadDim || rotary != kRotary || page_tokens != kPageTokens ||
        num_pages != kNumPages || mask_stride != kMaskStride || mask_keep != kMaskKeep ||
        features != kFeatures || theta != kTheta || eps != kEps || seed != kSeed)
        return false;
    if (scale <= 0.0f) return false;
    g.row_ids.resize(n_rows);
    r.bytes(g.row_ids.data(), n_rows * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t row = r.pod<uint32_t>();
        if (row != g.row_ids[i]) return false;
        std::vector<float> out(features);
        r.bytes(out.data(), features * sizeof(float));
        g.outputs.push_back(std::move(out));
    }
    return r.pos == buf.size();
}

struct DeviceBuffers {
    bf16_t* q_raw = nullptr;
    bf16_t* k_raw = nullptr;
    bf16_t* q_weight = nullptr;
    bf16_t* k_weight = nullptr;
    bf16_t* q_rope = nullptr;
    bf16_t* k_rope = nullptr;
    float* normed = nullptr;
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
    uint32_t* block_table = nullptr;
    uint32_t* slots = nullptr;
    uint32_t* positions = nullptr;
    uint8_t* mask = nullptr;
    float* out = nullptr;
    float* dense_out = nullptr;
};

void free_buffers(DeviceBuffers& b) {
    hipFree(b.q_raw);
    hipFree(b.k_raw);
    hipFree(b.q_weight);
    hipFree(b.k_weight);
    hipFree(b.q_rope);
    hipFree(b.k_rope);
    hipFree(b.normed);
    hipFree(b.k_pool);
    hipFree(b.v_pool);
    hipFree(b.block_table);
    hipFree(b.slots);
    hipFree(b.positions);
    hipFree(b.mask);
    hipFree(b.out);
    hipFree(b.dense_out);
    b = DeviceBuffers{};
}

hipError_t launch_sparse(const QsaSparseAttentionArgs& base, const DeviceBuffers& d,
                         uint32_t row_begin, uint32_t rows) {
    QsaSparseAttentionArgs a = base;
    a.q = reinterpret_cast<const bf16_t*>(reinterpret_cast<const char*>(d.q_rope) +
                                          static_cast<size_t>(row_begin) * kFeatures * 2u);
    a.row_sequence_slots = d.slots + row_begin;
    a.row_positions = d.positions + row_begin;
    a.mask = d.mask + static_cast<size_t>(row_begin) * kRows;
    a.output = reinterpret_cast<char*>(d.out) +
               static_cast<size_t>(row_begin) * kFeatures * sizeof(float);
    a.rows = rows;
    return ps::kernel::launch_qsa_sparse_attention_bf16(a);
}

}

int main() {
    const size_t q_count = static_cast<size_t>(kRows) * kFeatures;
    const size_t kv_count = static_cast<size_t>(kRows) * kKvFeatures;
    std::vector<uint16_t> query_raw(q_count), k_raw(kv_count), v_raw(kv_count);
    std::vector<uint16_t> q_weight(kHeadDim), k_weight(kHeadDim);

    SplitMix stream{kSeed};
    for (auto& x : query_raw) x = f32_to_bf16_host(stream.next_f32());
    for (auto& x : k_raw) x = f32_to_bf16_host(stream.next_f32());
    for (auto& x : v_raw) x = f32_to_bf16_host(stream.next_f32());
    SplitMix wq{kSeed ^ 1ull};
    SplitMix wk{kSeed ^ 2ull};
    for (auto& x : q_weight) x = f32_to_bf16_host(wq.next_f32());
    for (auto& x : k_weight) x = f32_to_bf16_host(wk.next_f32());

    std::vector<uint8_t> payload;
    payload.reserve((q_count + 2 * kv_count + 2 * kHeadDim) * 2);
    auto append = [&payload](const std::vector<uint16_t>& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
        payload.insert(payload.end(), p, p + v.size() * 2);
    };
    append(query_raw);
    append(k_raw);
    append(v_raw);
    append(q_weight);
    append(k_weight);
    const uint32_t input_crc = crc32(payload.data(), payload.size());
    std::printf("INPUT crc32=%08x rows=%u seed=%016llx\n", input_crc, kRows,
                static_cast<unsigned long long>(kSeed));

    Golden golden;
    const bool has_golden = load_golden(
        std::string(SHELF_SOURCE_ROOT) + "/tests/fixtures/qsa/qsa_sparse_attention_golden.bin",
        golden);
    check(has_golden, "golden load");
    if (has_golden) check(golden.crc == input_crc, "golden input checksum");

    const ::ps::execution::PagedAttentionSelectorInput selector_input{
        KVCacheDType::BF16, ValueDType::F32, kDenseRows, kQHeads, kKvHeads, kHeadDim,
        kPageTokens, kRows};
    check(::ps::execution::select_paged_attention_implementation(selector_input) ==
              ::ps::execution::PagedAttentionImplementation::Correctness,
          "selector falls back to correctness for q=24 kv=2 head_dim=256");
    const ::ps::execution::PagedAttentionSelectorInput measured_input{
        KVCacheDType::BF16, ValueDType::F32, kDenseRows, 16, 4, kHeadDim, kPageTokens, kRows};
    check(::ps::execution::select_paged_attention_implementation(measured_input) ==
              ::ps::execution::PagedAttentionImplementation::Optimized,
          "selector still resolves the measured q=16 kv=4 geometry");

    std::vector<float> q_host(q_count), k_host(kv_count), v_host(kv_count), q_normed_host;
    prepare_host(query_raw, k_raw, v_raw, q_weight, k_weight, q_host, k_host, v_host,
                 q_normed_host);

    std::vector<uint8_t> mask;
    build_dense_rule_mask(mask);

    DeviceBuffers d;
    const size_t pool_bytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);
    auto alloc = [](void** p, size_t n) { return hipMalloc(p, n) == hipSuccess; };
    bool ok = true;
    ok = ok && alloc(reinterpret_cast<void**>(&d.q_raw), q_count * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.k_raw), kv_count * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.q_weight), kHeadDim * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.k_weight), kHeadDim * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.q_rope), q_count * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.k_rope), kv_count * 2u);
    ok = ok && alloc(reinterpret_cast<void**>(&d.normed), (q_count + kv_count) * sizeof(float));
    ok = ok && alloc(reinterpret_cast<void**>(&d.k_pool), pool_bytes);
    ok = ok && alloc(reinterpret_cast<void**>(&d.v_pool), pool_bytes);
    constexpr uint32_t kLogicalPages = kRows / kPageTokens;
    ok = ok && alloc(reinterpret_cast<void**>(&d.block_table), kLogicalPages * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d.slots), kRows * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d.positions), kRows * sizeof(uint32_t));
    ok = ok && alloc(reinterpret_cast<void**>(&d.mask), mask.size());
    ok = ok && alloc(reinterpret_cast<void**>(&d.out), q_count * sizeof(float));
    ok = ok && alloc(reinterpret_cast<void**>(&d.dense_out),
                     static_cast<size_t>(kDenseRows) * kFeatures * sizeof(float));
    if (!ok) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }

    std::vector<uint16_t> k_pool(pool_bytes / 2, 0u), v_pool(pool_bytes / 2, 0u);
    for (uint32_t t = 0; t < kRows; ++t) {
        const uint32_t phys = page_perm(t / kPageTokens);
        const size_t base = static_cast<size_t>(phys) * kElemsPerPage +
                            static_cast<size_t>(t % kPageTokens) * kElemsPerToken;
        for (uint32_t e = 0; e < kElemsPerToken; ++e) {
            k_pool[base + e] = f32_to_bf16_host(k_host[static_cast<size_t>(t) * kKvFeatures + e]);
            v_pool[base + e] = f32_to_bf16_host(v_host[static_cast<size_t>(t) * kKvFeatures + e]);
        }
    }
    bool pool_ok = true;
    for (uint32_t t = 0; t < kRows; t += 617u) {
        const uint32_t phys = page_perm(t / kPageTokens);
        const size_t base = static_cast<size_t>(phys) * kElemsPerPage +
                            static_cast<size_t>(t % kPageTokens) * kElemsPerToken;
        for (uint32_t e = 0; e < kElemsPerToken; e += 33u) {
            if (k_pool[base + e] !=
                f32_to_bf16_host(k_host[static_cast<size_t>(t) * kKvFeatures + e]))
                pool_ok = false;
        }
    }
    check(pool_ok, "paged KV pool pattern");

    (void)hipMemcpy(d.q_raw, query_raw.data(), q_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(d.k_raw, k_raw.data(), kv_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(d.q_weight, q_weight.data(), kHeadDim * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(d.k_weight, k_weight.data(), kHeadDim * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(d.k_pool, k_pool.data(), pool_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d.v_pool, v_pool.data(), pool_bytes, hipMemcpyHostToDevice);
    std::vector<uint32_t> block_table(kLogicalPages);
    for (uint32_t l = 0; l < kLogicalPages; ++l) block_table[l] = page_perm(l);
    (void)hipMemcpy(d.block_table, block_table.data(),
                    kLogicalPages * sizeof(uint32_t), hipMemcpyHostToDevice);
    std::vector<uint32_t> slots(kRows, 0u), positions(kRows);
    for (uint32_t i = 0; i < kRows; ++i) positions[i] = i;
    (void)hipMemcpy(d.slots, slots.data(), kRows * sizeof(uint32_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d.positions, positions.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(d.mask, mask.data(), mask.size(), hipMemcpyHostToDevice);

    float* q_normed = d.normed;
    float* k_normed = q_normed + q_count;

    hipError_t err = ps::kernel::launch_rmsnorm(
        d.q_raw, ps::kernel::RmsNormDataType::BF16, d.q_weight,
        ps::kernel::RmsNormDataType::BF16, ps::kernel::RmsNormWeightLayout::PER_GROUP,
        ps::kernel::RmsNormWeightMode::ONE_PLUS, q_normed, ps::kernel::RmsNormDataType::F32,
        kRows, kFeatures, kHeadDim, kFeatures, kFeatures, kEps, nullptr);
    check(err == hipSuccess, "q rmsnorm launch");
    err = ps::kernel::launch_rope_f32_bf16_generic(
        q_normed, kFeatures, d.q_rope, kFeatures, d.positions, kRows, kFeatures, kHeadDim,
        kRotary, kTheta, nullptr, 1.0f, nullptr);
    check(err == hipSuccess, "q partial rope launch");

    err = ps::kernel::launch_rmsnorm(
        d.k_raw, ps::kernel::RmsNormDataType::BF16, d.k_weight,
        ps::kernel::RmsNormDataType::BF16, ps::kernel::RmsNormWeightLayout::PER_GROUP,
        ps::kernel::RmsNormWeightMode::ONE_PLUS, k_normed, ps::kernel::RmsNormDataType::F32,
        kRows, kKvFeatures, kHeadDim, kKvFeatures, kKvFeatures, kEps, nullptr);
    check(err == hipSuccess, "k rmsnorm launch");
    err = ps::kernel::launch_rope_f32_bf16_generic(
        k_normed, kKvFeatures, d.k_rope, kKvFeatures, d.positions, kRows, kKvFeatures, kHeadDim,
        kRotary, kTheta, nullptr, 1.0f, nullptr);
    check(err == hipSuccess, "k partial rope launch");
    check(hipDeviceSynchronize() == hipSuccess, "prep synchronize");

    std::vector<uint16_t> gpu_q_rope(q_count);
    (void)hipMemcpy(gpu_q_rope.data(), d.q_rope, q_count * 2u, hipMemcpyDeviceToHost);
    std::vector<float> gpu_q_f32(q_count);
    for (size_t i = 0; i < q_count; ++i) gpu_q_f32[i] = bf16_to_f32_host(gpu_q_rope[i]);
    {
        const Metrics m = score_metrics(gpu_q_f32.data(), q_host.data(), q_count);
        report("q after norm+partial rope gpu-vs-cpu-oracle", m);
        check_metrics("q prep tolerance", m, 3.2e-2, 2.0e-2, 1.0e-4);
    }

    std::vector<float> gpu_normed(q_count);
    (void)hipMemcpy(gpu_normed.data(), d.normed, q_count * sizeof(float),
                    hipMemcpyDeviceToHost);
    bool rotary_tail_ok = true;
    for (uint32_t r : {0u, 1u, 7u, 1234u, kRows - 1u}) {
        for (uint32_t h = 0; h < kQHeads; ++h) {
            const size_t base = static_cast<size_t>(r) * kFeatures + h * kHeadDim;
            for (uint32_t c = kRotary; c < kHeadDim; ++c) {
                if (gpu_q_rope[base + c] != f32_to_bf16_host(gpu_normed[base + c]))
                    rotary_tail_ok = false;
            }
        }
    }
    check(rotary_tail_ok, "rotary tail is a pure bf16 passthrough");

    QsaSparseAttentionArgs args;
    args.q_row_stride = kFeatures;
    args.output_dtype = QsaSparseAttentionOutputDType::F32;
    args.output_row_stride = kFeatures;
    args.row_sequence_slots = d.slots;
    args.row_positions = d.positions;
    args.mask_row_stride = kRows;
    args.block_tables = d.block_table;
    args.block_table_stride = kLogicalPages;
    args.max_sequences = 1u;
    args.rows = kRows;
    args.q_heads = kQHeads;
    args.kv_heads = kKvHeads;
    args.head_dim = kHeadDim;
    args.layer = 0u;
    args.page_tokens = kPageTokens;
    args.num_pages = kNumPages;
    args.num_attention_layers = kNumLayers;
    args.elems_per_token = kElemsPerToken;
    args.elems_per_page = kElemsPerPage;
    args.elems_per_layer = kElemsPerLayer;
    args.k_pool = d.k_pool;
    args.v_pool = d.v_pool;
    args.scale = kScale;

    err = launch_sparse(args, d, 0, kRows);
    check(err == hipSuccess, "sparse attention launch (prefill)");
    check(hipDeviceSynchronize() == hipSuccess, "sparse attention synchronize");
    std::vector<float> gpu_out(q_count);
    (void)hipMemcpy(gpu_out.data(), d.out, q_count * sizeof(float), hipMemcpyDeviceToHost);

    std::vector<uint32_t> oracle_rows;
    for (uint32_t i = 0; i < kDenseRows; ++i) oracle_rows.push_back(i);
    if (has_golden)
        oracle_rows.insert(oracle_rows.end(), golden.row_ids.begin(), golden.row_ids.end());
    std::sort(oracle_rows.begin(), oracle_rows.end());
    oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()), oracle_rows.end());

    std::vector<float> oracle_out;
    run_oracle(q_host, k_host, v_host, mask, oracle_rows, oracle_out);
    std::vector<float> gpu_sel, ref_sel;
    gpu_sel.reserve(oracle_rows.size() * kFeatures);
    ref_sel.reserve(oracle_rows.size() * kFeatures);
    for (size_t ri = 0; ri < oracle_rows.size(); ++ri) {
        const float* g = gpu_out.data() + static_cast<size_t>(oracle_rows[ri]) * kFeatures;
        gpu_sel.insert(gpu_sel.end(), g, g + kFeatures);
        const float* r = oracle_out.data() + ri * kFeatures;
        ref_sel.insert(ref_sel.end(), r, r + kFeatures);
    }
    {
        const Metrics m = score_metrics(gpu_sel.data(), ref_sel.data(), gpu_sel.size());
        report("sparse output gpu-vs-cpu-oracle", m);
        check_metrics("sparse output tolerance vs cpu oracle", m, 1.0e-3, 5.0e-2, 2.0e-5);
    }

    bool causal_ok = true;
    for (uint32_t i = 0; i < kRows; i += 511u) {
        const uint8_t* row = mask.data() + static_cast<size_t>(i) * kRows;
        for (uint32_t p = i + 1u; p < kRows; ++p) {
            if (row[p] != 0u) {
                std::printf("DEBUG mask beyond causality row=%u p=%u\n", i, p);
                causal_ok = false;
                break;
            }
        }
        if ((i + 1u) % 4u != 0u && row[i] == 0u) {
            std::printf("DEBUG mask missing tail self row=%u\n", i);
            causal_ok = false;
        }
    }
    check(causal_ok, "mask respects causality and self");
    uint64_t selected_total = 0;
    uint64_t visible_total = 0;
    for (uint32_t i = kDenseRows; i < kRows; ++i) {
        const uint8_t* row = mask.data() + static_cast<size_t>(i) * kRows;
        for (uint32_t p = 0; p <= i; ++p) {
            selected_total += row[p];
            ++visible_total;
        }
    }
    const double sparse_fraction =
        static_cast<double>(selected_total) / static_cast<double>(visible_total);
    std::printf("SPARSE fraction of visible tokens selected=%.4f\n", sparse_fraction);
    check(sparse_fraction < 0.4, "fixture is truly sparse");

    if (has_golden) {
        std::vector<float> gpu_g, ref_g;
        for (size_t i = 0; i < golden.row_ids.size(); ++i) {
            const float* g = gpu_out.data() + static_cast<size_t>(golden.row_ids[i]) * kFeatures;
            gpu_g.insert(gpu_g.end(), g, g + kFeatures);
            ref_g.insert(ref_g.end(), golden.outputs[i].begin(), golden.outputs[i].end());
        }
        const Metrics m = score_metrics(gpu_g.data(), ref_g.data(), gpu_g.size());
        report("sparse output gpu-vs-pytorch-golden", m);
        check_metrics("sparse output tolerance vs pytorch golden", m, 1.0e-3, 5.0e-2, 2.0e-5);
    }

    PagedAttentionCommonArgs common;
    common.q = reinterpret_cast<const bf16_t*>(d.q_rope);
    common.q_row_stride = kFeatures;
    common.output = d.dense_out;
    common.output_dtype = PagedAttentionOutputDType::F32;
    common.output_row_stride = kFeatures;
    common.row_sequence_slots = d.slots;
    common.row_positions = d.positions;
    common.block_tables = d.block_table;
    common.block_table_stride = kLogicalPages;
    common.max_sequences = 1u;
    common.rows = kDenseRows;
    common.q_heads = kQHeads;
    common.kv_heads = kKvHeads;
    common.head_dim = kHeadDim;
    common.layer = 0u;
    common.page_tokens = kPageTokens;
    common.num_pages = kNumPages;
    common.num_attention_layers = kNumLayers;
    common.elems_per_token = kElemsPerToken;
    common.elems_per_page = kElemsPerPage;
    common.elems_per_layer = kElemsPerLayer;
    common.scale = kScale;
    err = ps::kernel::launch_attention_paged_bf16(common, d.k_pool, d.v_pool, nullptr);
    check(err == hipSuccess, "dense attention launch");
    check(hipDeviceSynchronize() == hipSuccess, "dense attention synchronize");
    std::vector<float> dense_out(static_cast<size_t>(kDenseRows) * kFeatures);
    (void)hipMemcpy(dense_out.data(), d.dense_out, dense_out.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    {
        const Metrics m = score_metrics(gpu_out.data(), dense_out.data(), dense_out.size());
        report("all-selected short text sparse-vs-dense", m);
        check_metrics("all-selected dense tolerance", m, 1.0e-5, 1.0e-3, 1.0e-6);
    }

    const uint32_t decode_begin = kRows - kDecodeRows;
    err = launch_sparse(args, d, decode_begin, kDecodeRows);
    check(err == hipSuccess, "sparse attention launch (decode)");
    check(hipDeviceSynchronize() == hipSuccess, "decode synchronize");
    std::vector<float> decode_out(static_cast<size_t>(kDecodeRows) * kFeatures);
    (void)hipMemcpy(decode_out.data(), d.out + static_cast<size_t>(decode_begin) * kFeatures,
                    decode_out.size() * sizeof(float), hipMemcpyDeviceToHost);
    bool decode_ok = true;
    for (uint32_t r = 0; r < kDecodeRows; ++r) {
        const float* a = decode_out.data() + static_cast<size_t>(r) * kFeatures;
        const float* b = gpu_out.data() + static_cast<size_t>(decode_begin + r) * kFeatures;
        if (std::memcmp(a, b, kFeatures * sizeof(float)) != 0) decode_ok = false;
    }
    check(decode_ok, "decode rows bitwise match the prefill rows");

    free_buffers(d);
    std::printf("test_qsa_sparse_attention: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
