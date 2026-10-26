// shelf unit test — vendored PhaseShift paged GQA decode attention + RoPE + kv_append.
// Parity vs fresh naive f64 attention honoring the kernel's physical layout
// (separate K/V pools, [block][slot][kvh][d] packing, uint32 block table).
// NOTE (ABI): the vendored kernels do NOT consume the shelf locked paged_kv
// ABI (single interleaved pool); this test builds the PhaseShift layout.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/attention/kv_append.h>
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/attention/rope.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::hipk::launch_attention_paged_bf16;
using shelf::hipk::launch_attention_paged_bf16_split;
using shelf::hipk::launch_attention_paged_bf16_split_reduce;
using shelf::hipk::launch_kv_append_bf16;
using shelf::hipk::launch_rope_f32_bf16;
using shelf::hipk::launch_rope_inv_freq_init;
using shelf::hipk::PagedAttentionCommonArgs;
using shelf::hipk::PagedAttentionOutputDType;
using shelf::hipk::KvAppendCommonArgs;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

#define HIP_OK(call)                                                        \
    do {                                                                    \
        const hipError_t e_ = (call);                                       \
        if (e_ != hipSuccess) {                                             \
            std::printf("FAIL hip: %s @%d: %s\n", #call, __LINE__,          \
                        hipGetErrorString(e_));                             \
            ++failures;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

float bf2f(shelf::bf16_t b) { return shelf::bf16_to_f32(b); }

constexpr uint32_t kFillPat = 0x3C3Cu;  // unused-slot fill pattern (bf16 bits)

// ------------------------------------------------------------------ layout
// PhaseShift pool geometry (what the kernel math actually addresses):
//   K/V are SEPARATE pools; element(p) within one pool =
//     layer*elems_per_layer + page*elems_per_page + off*elems_per_token
//       + kv_head*head_dim + d
//   page = block_table[slot*block_table_stride + p/page_tokens] (uint32)
struct PoolGeom {
    uint32_t page_tokens = 16;
    uint32_t table_width = 8;      // block_table_stride
    uint32_t num_pages = 16;       // pool capacity in blocks
    uint32_t kv_heads = 2;
    uint32_t head_dim = 256;
    std::size_t ept() const { return std::size_t(kv_heads) * head_dim; }
    std::size_t epp() const { return std::size_t(page_tokens) * ept(); }
    std::size_t epl() const { return std::size_t(num_pages) * epp(); }
    std::size_t idx(uint32_t page, uint32_t off, uint32_t h, uint32_t d) const {
        return std::size_t(page) * epp() + std::size_t(off) * ept()
             + std::size_t(h) * head_dim + d;
    }
};

// host fill helper: pattern everywhere, then place one sequence's tokens.
void fill_pool(std::vector<shelf::bf16_t>& pool, const PoolGeom& g, uint32_t pat) {
    for (auto& e : pool) e = shelf::bf16_t{static_cast<uint16_t>(pat)};
}

// ------------------------------------------------------------------ rope
// f64 reference for rope_f32_bf16: pair (i, i+half) per head, partial rotary.
void rope_ref_f64(const float* x, uint32_t head_dim, uint32_t rotary,
                  uint32_t pos, const double* inv_freq, double factor,
                  std::vector<double>& out) {
    const uint32_t half = rotary / 2u;
    out.assign(x, x + head_dim);
    for (uint32_t i = 0; i < half; ++i) {
        const double ang = static_cast<double>(pos) * inv_freq[i];
        const double c = std::cos(ang) * factor;
        const double s = std::sin(ang) * factor;
        const double a = x[i], b = x[i + half];
        out[i] = a * c - b * s;
        out[i + half] = b * c + a * s;
    }
}

void run_rope_case(uint32_t rows, uint32_t heads, uint32_t head_dim,
                   uint32_t rotary, double theta,
                   const std::vector<double>& inv_freq,
                   const std::vector<uint32_t>& positions) {
    const uint32_t features = heads * head_dim;
    std::mt19937 rng(99 + rows * 131 + head_dim);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> in(static_cast<size_t>(rows) * features);
    for (auto& v : in) v = dist(rng);

    float* d_in = nullptr; shelf::bf16_t* d_out = nullptr;
    uint32_t* d_pos = nullptr; double* d_if = nullptr;
    HIP_OK(hipMalloc(&d_in, in.size() * sizeof(float)));
    HIP_OK(hipMalloc(&d_out, in.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_pos, positions.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_if, inv_freq.size() * sizeof(double)));
    HIP_OK(hipMemcpy(d_in, in.data(), in.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_pos, positions.data(), positions.size() * sizeof(uint32_t),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_if, inv_freq.data(), inv_freq.size() * sizeof(double),
                     hipMemcpyHostToDevice));

    const hipError_t st = launch_rope_f32_bf16(
        d_in, features, d_out, features, d_pos, rows, features, head_dim,
        rotary, static_cast<float>(theta), d_if, 1.0f, nullptr);
    if (st != hipSuccess) {
        std::printf("FAIL: launch_rope_f32_bf16 D=%u rotary=%u: %s\n",
                    head_dim, rotary, hipGetErrorString(st));
        ++failures;
    }
    std::vector<shelf::bf16_t> out(static_cast<size_t>(rows) * features);
    HIP_OK(hipMemcpy(out.data(), d_out, out.size() * sizeof(shelf::bf16_t),
                     hipMemcpyDeviceToHost));
    hipFree(d_in); hipFree(d_out); hipFree(d_pos); hipFree(d_if);
    HIP_OK(hipGetLastError());

    // f64 reference + 5e-2 absolute tolerance (bf16 output rounding)
    std::vector<double> ref;
    double worst = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t h = 0; h < heads; ++h) {
            const float* xr = &in[static_cast<size_t>(r) * features
                                  + static_cast<size_t>(h) * head_dim];
            rope_ref_f64(xr, head_dim, rotary, positions[r], inv_freq.data(),
                         1.0, ref);
            for (uint32_t d = 0; d < head_dim; ++d) {
                const double got = bf2f(out[static_cast<size_t>(r) * features
                                           + static_cast<size_t>(h) * head_dim + d]);
                const double diff = std::fabs(got - ref[d]);
                if (diff > worst) worst = diff;
                if (diff > 5e-2) {
                    std::printf("FAIL: rope D=%u row=%u head=%u dim=%u diff=%g\n",
                                head_dim, r, h, d, diff);
                    ++failures;
                    d = head_dim; r = rows; break;
                }
            }
        }
    }
    std::printf("rope D=%u rotary=%u rows=%u heads=%u worst|diff|=%.4g\n",
                head_dim, rotary, rows, heads, worst);
}

// verify launch_rope_inv_freq_init against f64 table
void run_inv_freq_case(uint32_t half, float theta) {
    double* d_if = nullptr;
    HIP_OK(hipMalloc(&d_if, half * sizeof(double)));
    if (launch_rope_inv_freq_init(d_if, half, theta, nullptr) != hipSuccess) {
        std::printf("FAIL: launch_rope_inv_freq_init\n"); ++failures; return;
    }
    std::vector<double> got(half);
    HIP_OK(hipMemcpy(got.data(), d_if, half * sizeof(double), hipMemcpyDeviceToHost));
    hipFree(d_if);
    for (uint32_t i = 0; i < half; ++i) {
        const double want = 1.0 / std::pow(static_cast<double>(theta),
                                          static_cast<double>(i) / static_cast<double>(half));
        if (std::fabs(got[i] - want) > 1e-12 * want) {
            std::printf("FAIL: inv_freq[%u] got=%.17g want=%.17g\n", i, got[i], want);
            ++failures;
        }
    }
}

// ------------------------------------------------------------------ attention
struct AttnCase {
    uint32_t rows = 2;
    uint32_t q_heads = 4;
    uint32_t kv_heads = 2;
    uint32_t head_dim = 256;
    std::vector<uint32_t> seqused = {35, 20};   // spans >1 block (block 16)
    std::vector<uint32_t> table;                 // rows x table_width uint32
};

// fresh naive f64 attention honoring the PhaseShift physical layout
void reference_attention(const AttnCase& c, const PoolGeom& g,
                         const std::vector<shelf::bf16_t>& q,   // [rows][qh*D]
                         const std::vector<shelf::bf16_t>& kpool,
                         const std::vector<shelf::bf16_t>& vpool,
                         std::vector<double>& ref) {
    const uint32_t q_per_kv = c.q_heads / c.kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
    ref.assign(static_cast<size_t>(c.rows) * c.q_heads * c.head_dim, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        const uint32_t vis = c.seqused[r];
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const uint32_t kvh = h / q_per_kv;
            std::vector<double> score(vis);
            double m = -1e300;
            for (uint32_t p = 0; p < vis; ++p) {
                const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                              + p / g.page_tokens];
                const uint32_t off = p % g.page_tokens;
                double acc = 0.0;
                for (uint32_t d = 0; d < c.head_dim; ++d) {
                    const double qv = bf2f(q[(static_cast<size_t>(r) * c.q_heads + h)
                                            * c.head_dim + d]);
                    const double kv = bf2f(kpool[g.idx(page, off, kvh, d)]);
                    acc += qv * kv;
                }
                score[p] = acc * scale;
                if (score[p] > m) m = score[p];
            }
            double sum = 0.0;
            std::vector<double> w(vis);
            for (uint32_t p = 0; p < vis; ++p) {
                w[p] = std::exp(score[p] - m);
                sum += w[p];
            }
            for (uint32_t d = 0; d < c.head_dim; ++d) {
                double acc = 0.0;
                for (uint32_t p = 0; p < vis; ++p) {
                    const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                                  + p / g.page_tokens];
                    const uint32_t off = p % g.page_tokens;
                    acc += w[p] * bf2f(vpool[g.idx(page, off, kvh, d)]);
                }
                ref[(static_cast<size_t>(r) * c.q_heads + h) * c.head_dim + d]
                    = acc / sum;
            }
        }
    }
}

// original strict comparator (unchanged gate for pre-existing cases)
void compare_attention(const AttnCase& c, const char* tag,
                       const std::vector<shelf::bf16_t>& got,
                       const std::vector<double>& ref) {
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const size_t base = (static_cast<size_t>(r) * c.q_heads + h) * c.head_dim;
            double worst = 0.0;
            uint32_t amax_got = 0, amax_ref = 0;
            double bg = -1e300, br = -1e300;
            for (uint32_t d = 0; d < c.head_dim; ++d) {
                const double g = bf2f(got[base + d]);
                const double diff = std::fabs(g - ref[base + d]);
                if (diff > worst) worst = diff;
                if (g > bg) { bg = g; amax_got = d; }
                if (ref[base + d] > br) { br = ref[base + d]; amax_ref = d; }
            }
            char buf[128];
            std::snprintf(buf, sizeof buf, "%s row=%u head=%u worst|diff|=%.4g "
                          "argmax got=%u ref=%u", tag, r, h, worst, amax_got, amax_ref);
            if (worst > 5e-2) { ++failures; std::printf("FAIL: %s (tol)\n", buf); }
            else if (amax_got != amax_ref) { ++failures; std::printf("FAIL: %s (argmax)\n", buf); }
            else std::printf("ok: %s\n", buf);
        }
    }
}

// gap-aware variant: argmax equality only enforced when the f64 top-1/top-2
// gap is large compared to the tolerance (bf16 output rounding can otherwise
// flip near-ties, which is not a numerical defect).
void compare_attention_gap(const AttnCase& c, const char* tag,
                           const std::vector<shelf::bf16_t>& got,
                           const std::vector<double>& ref, double tol) {
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const size_t base = (static_cast<size_t>(r) * c.q_heads + h) * c.head_dim;
            double worst = 0.0;
            uint32_t amax_got = 0, amax_ref = 0;
            double bg = -1e300, br = -1e300, br2 = -1e300;
            for (uint32_t d = 0; d < c.head_dim; ++d) {
                const double g = bf2f(got[base + d]);
                const double diff = std::fabs(g - ref[base + d]);
                if (diff > worst) worst = diff;
                if (g > bg) { bg = g; amax_got = d; }
                if (ref[base + d] > br) { br2 = br; br = ref[base + d]; amax_ref = d; }
                else if (ref[base + d] > br2) br2 = ref[base + d];
            }
            char buf[128];
            std::snprintf(buf, sizeof buf, "%s row=%u head=%u worst|diff|=%.4g "
                          "argmax got=%u ref=%u gap=%.4g", tag, r, h, worst, amax_got, amax_ref, br - br2);
            if (worst > tol) { ++failures; std::printf("FAIL: %s (tol)\n", buf); }
            else if (amax_got != amax_ref && br - br2 > 4.0 * tol) { ++failures; std::printf("FAIL: %s (argmax)\n", buf); }
            else std::printf("ok: %s\n", buf);
        }
    }
}

// Build pool + table on host, run paged decode on device, f64 reference compare.
// If do_rope: apply the rope kernel to q (f32) and to generated K token rows
// (f32) before they enter the pool; reference consumes the kernel outputs.
void run_attention_case(uint32_t head_dim, bool do_rope, bool use_split,
                        uint32_t q_heads = 4, uint32_t splits = 4u) {
    AttnCase c;
    c.q_heads = q_heads;
    c.head_dim = head_dim;
    c.table = {3, 0, 5, ~0u, ~0u, ~0u, ~0u, ~0u,
               7, 1, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    PoolGeom g;
    g.head_dim = head_dim;
    const uint32_t total_tokens = c.seqused[0] + c.seqused[1];

    char tag[64];
    std::snprintf(tag, sizeof tag, "attn nq=%u kvh=%u D=%u %s", c.q_heads,
                  c.kv_heads, head_dim, use_split ? "split" : "bf16");

    std::mt19937 rng(7 + head_dim * 17u + q_heads * 101u + (use_split ? 5u : 0u));
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    // q rows (f32 source if rope, else straight bf16)
    const size_t q_elems = static_cast<size_t>(c.rows) * c.q_heads * head_dim;
    std::vector<float> q_f32(q_elems);
    for (auto& v : q_f32) v = dist(rng);
    std::vector<shelf::bf16_t> q(q_elems);
    if (!do_rope) for (size_t i = 0; i < q_elems; ++i) q[i] = shelf::f32_to_bf16(q_f32[i]);

    // K/V token stream (row-major per sequence, kvh*D per token)
    const size_t kv_elems = static_cast<size_t>(total_tokens) * c.kv_heads * head_dim;
    std::vector<float> k_f32(kv_elems);
    std::vector<shelf::bf16_t> v_bf16(kv_elems);
    for (auto& v : k_f32) v = dist(rng);
    for (auto& v : v_bf16) v = shelf::f32_to_bf16(dist(rng));

    // rope q and K on device if requested
    std::vector<double> inv_freq(head_dim / 2u < 32u ? head_dim / 2u : 32u);
    const double theta = 1e6;
    for (size_t i = 0; i < inv_freq.size(); ++i)
        inv_freq[i] = 1.0 / std::pow(theta, static_cast<double>(i)
                                    / static_cast<double>(inv_freq.size()));
    const uint32_t rotary = 64u;
    std::vector<shelf::bf16_t> k_bf16(kv_elems);
    if (do_rope) {
        // q: positions = last-token index per row
        std::vector<uint32_t> q_pos = {c.seqused[0] - 1u, c.seqused[1] - 1u};
        {
            float* d_in = nullptr; shelf::bf16_t* d_out = nullptr;
            uint32_t* d_pos = nullptr; double* d_if = nullptr;
            HIP_OK(hipMalloc(&d_in, q_f32.size() * sizeof(float)));
            HIP_OK(hipMalloc(&d_out, q_f32.size() * sizeof(shelf::bf16_t)));
            HIP_OK(hipMalloc(&d_pos, 2 * sizeof(uint32_t)));
            HIP_OK(hipMalloc(&d_if, inv_freq.size() * sizeof(double)));
            HIP_OK(hipMemcpy(d_in, q_f32.data(), q_f32.size() * sizeof(float), hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_pos, q_pos.data(), 2 * sizeof(uint32_t), hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_if, inv_freq.data(), inv_freq.size() * sizeof(double), hipMemcpyHostToDevice));
            if (launch_rope_f32_bf16(d_in, c.q_heads * head_dim, d_out,
                                     c.q_heads * head_dim, d_pos, c.rows,
                                     c.q_heads * head_dim, head_dim, rotary,
                                     1e6f, d_if, 1.0f, nullptr) != hipSuccess) {
                std::printf("FAIL: rope(q) %s\n", tag); ++failures;
            }
            HIP_OK(hipMemcpy(q.data(), d_out, q.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
            hipFree(d_in); hipFree(d_out); hipFree(d_pos); hipFree(d_if);
        }
        // K tokens: position within own sequence
        std::vector<uint32_t> k_pos(total_tokens);
        for (uint32_t t = 0; t < c.seqused[0]; ++t) k_pos[t] = t;
        for (uint32_t t = 0; t < c.seqused[1]; ++t) k_pos[c.seqused[0] + t] = t;
        {
            float* d_in = nullptr; shelf::bf16_t* d_out = nullptr;
            uint32_t* d_pos = nullptr; double* d_if = nullptr;
            HIP_OK(hipMalloc(&d_in, k_f32.size() * sizeof(float)));
            HIP_OK(hipMalloc(&d_out, k_f32.size() * sizeof(shelf::bf16_t)));
            HIP_OK(hipMalloc(&d_pos, k_pos.size() * sizeof(uint32_t)));
            HIP_OK(hipMalloc(&d_if, inv_freq.size() * sizeof(double)));
            HIP_OK(hipMemcpy(d_in, k_f32.data(), k_f32.size() * sizeof(float), hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_pos, k_pos.data(), k_pos.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_if, inv_freq.data(), inv_freq.size() * sizeof(double), hipMemcpyHostToDevice));
            if (launch_rope_f32_bf16(d_in, c.kv_heads * head_dim, d_out,
                                     c.kv_heads * head_dim, d_pos, total_tokens,
                                     c.kv_heads * head_dim, head_dim, rotary,
                                     1e6f, d_if, 1.0f, nullptr) != hipSuccess) {
                std::printf("FAIL: rope(K) %s\n", tag); ++failures;
            }
            HIP_OK(hipMemcpy(k_bf16.data(), d_out, k_bf16.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
            hipFree(d_in); hipFree(d_out); hipFree(d_pos); hipFree(d_if);
        }
    } else {
        for (size_t i = 0; i < kv_elems; ++i) k_bf16[i] = shelf::f32_to_bf16(k_f32[i]);
    }
    {
        // scatter into pool layout (host), tokens of seq0 then seq1
        std::vector<shelf::bf16_t> kpool(g.num_pages * g.epp()), vpool(g.num_pages * g.epp());
        fill_pool(kpool, g, kFillPat);
        fill_pool(vpool, g, kFillPat);
        for (uint32_t s = 0; s < c.rows; ++s) {
            for (uint32_t t = 0; t < c.seqused[s]; ++t) {
                const uint32_t page = c.table[static_cast<size_t>(s) * g.table_width
                                              + t / g.page_tokens];
                const uint32_t off = t % g.page_tokens;
                for (uint32_t h = 0; h < c.kv_heads; ++h)
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        const size_t src = ((static_cast<size_t>(s) ? static_cast<size_t>(c.seqused[0]) : 0u)
                                            + t) * c.kv_heads * head_dim
                                         + static_cast<size_t>(h) * head_dim + d;
                        kpool[g.idx(page, off, h, d)] = k_bf16[src];
                        vpool[g.idx(page, off, h, d)] = v_bf16[src];
                    }
            }
        }
        // device pools
        shelf::bf16_t *d_k = nullptr, *d_v = nullptr, *d_q = nullptr, *d_o = nullptr;
        uint32_t *d_tbl = nullptr, *d_slots = nullptr, *d_rows_pos = nullptr;
        HIP_OK(hipMalloc(&d_k, kpool.size() * sizeof(shelf::bf16_t)));
        HIP_OK(hipMalloc(&d_v, vpool.size() * sizeof(shelf::bf16_t)));
        HIP_OK(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)));
        HIP_OK(hipMalloc(&d_o, q.size() * sizeof(shelf::bf16_t)));
        HIP_OK(hipMalloc(&d_tbl, c.table.size() * sizeof(uint32_t)));
        HIP_OK(hipMalloc(&d_slots, c.rows * sizeof(uint32_t)));
        HIP_OK(hipMalloc(&d_rows_pos, c.rows * sizeof(uint32_t)));
        HIP_OK(hipMemcpy(d_k, kpool.data(), kpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_v, vpool.data(), vpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_tbl, c.table.data(), c.table.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
        {
            const uint32_t slots[2] = {0u, 1u};
            const uint32_t rows_pos[2] = {c.seqused[0] - 1u, c.seqused[1] - 1u};
            HIP_OK(hipMemcpy(d_slots, slots, sizeof slots, hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_rows_pos, rows_pos, sizeof rows_pos, hipMemcpyHostToDevice));
        }
        PagedAttentionCommonArgs a{};
        a.q = d_q;
        a.output = d_o;
        a.output_dtype = PagedAttentionOutputDType::BF16;
        a.q_row_stride = c.q_heads * head_dim;
        a.output_row_stride = c.q_heads * head_dim;
        a.row_sequence_slots = d_slots;
        a.row_positions = d_rows_pos;
        a.block_tables = d_tbl;
        a.block_table_stride = g.table_width;
        a.max_sequences = c.rows;
        a.rows = c.rows;
        a.layer = 0;
        a.page_tokens = g.page_tokens;
        a.num_pages = g.num_pages;
        a.num_attention_layers = 1;
        a.q_heads = c.q_heads;
        a.kv_heads = c.kv_heads;
        a.head_dim = head_dim;
        a.elems_per_token = static_cast<uint32_t>(g.ept());
        a.elems_per_page = static_cast<uint32_t>(g.epp());
        a.elems_per_layer = static_cast<uint32_t>(g.epl());
        a.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        hipError_t st = hipErrorInvalidValue;
        if (!use_split) {
            st = launch_attention_paged_bf16(a, d_k, d_v, nullptr);
        } else {
            const uint32_t qpk = c.q_heads / c.kv_heads;
            std::vector<float> partials(static_cast<size_t>(c.rows) * c.kv_heads
                                        * splits * qpk * (2u + 256u), 0.0f);
            float* d_part = nullptr;
            HIP_OK(hipMalloc(&d_part, partials.size() * sizeof(float)));
            st = launch_attention_paged_bf16_split(a, d_k, d_v, d_part, splits, nullptr);
            if (st == hipSuccess)
                st = launch_attention_paged_bf16_split_reduce(a, d_part, splits, nullptr);
            hipFree(d_part);
        }
        if (st != hipSuccess) {
            std::printf("FAIL: launch %s: %s\n", tag, hipGetErrorString(st));
            ++failures;
        }
        std::vector<shelf::bf16_t> got(q_elems);
        HIP_OK(hipMemcpy(got.data(), d_o, got.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
        hipFree(d_k); hipFree(d_v); hipFree(d_q); hipFree(d_o);
        hipFree(d_tbl); hipFree(d_slots); hipFree(d_rows_pos);
        HIP_OK(hipGetLastError());

        std::vector<double> ref;
        reference_attention(c, g, q, kpool, vpool, ref);
        compare_attention(c, tag, got, ref);
    }
}

// ------------------------------------------------------------------ kv_append
void run_kv_append_case(uint32_t head_dim) {
    AttnCase c;
    c.head_dim = head_dim;
    c.table = {3, 0, 5, ~0u, ~0u, ~0u, ~0u, ~0u,
               7, 1, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    PoolGeom g;
    g.head_dim = head_dim;

    std::mt19937 rng(55 + head_dim);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<shelf::bf16_t> kpool(g.num_pages * g.epp()), vpool(g.num_pages * g.epp());
    fill_pool(kpool, g, kFillPat);
    fill_pool(vpool, g, kFillPat);

    // append one new token per sequence: seq0 at pos 35 (block 5, off 3),
    // seq1 at pos 20 (block 1, off 4)
    const uint32_t rows = 2;
    const uint32_t rows_pos[2] = {35u, 20u};
    std::vector<shelf::bf16_t> kin(static_cast<size_t>(rows) * c.kv_heads * head_dim),
                           vin(static_cast<size_t>(rows) * c.kv_heads * head_dim);
    for (auto& v : kin) v = shelf::f32_to_bf16(dist(rng));
    for (auto& v : vin) v = shelf::f32_to_bf16(dist(rng));

    shelf::bf16_t *d_k = nullptr, *d_v = nullptr, *d_kin = nullptr, *d_vin = nullptr;
    uint32_t *d_tbl = nullptr, *d_slots = nullptr, *d_pos = nullptr;
    HIP_OK(hipMalloc(&d_k, kpool.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_v, vpool.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_kin, kin.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_vin, vin.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_tbl, c.table.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_slots, rows * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_pos, rows * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(d_k, kpool.data(), kpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_v, vpool.data(), vpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_kin, kin.data(), kin.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_vin, vin.data(), vin.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_tbl, c.table.data(), c.table.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    { const uint32_t slots[2] = {0u, 1u};
      HIP_OK(hipMemcpy(d_slots, slots, sizeof slots, hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(d_pos, rows_pos, sizeof rows_pos, hipMemcpyHostToDevice)); }

    KvAppendCommonArgs a{};
    a.k_input = d_kin;
    a.v_input = d_vin;
    a.k_row_stride = c.kv_heads * head_dim;
    a.v_row_stride = c.kv_heads * head_dim;
    a.row_sequence_slots = d_slots;
    a.row_positions = d_pos;
    a.block_tables = d_tbl;
    a.block_table_stride = g.table_width;
    a.max_sequences = rows;
    a.rows = rows;
    a.layer = 0;
    a.page_tokens = g.page_tokens;
    a.num_pages = g.num_pages;
    a.num_attention_layers = 1;
    a.kv_heads = c.kv_heads;
    a.head_dim = head_dim;
    a.elems_per_token = static_cast<uint32_t>(g.ept());
    a.elems_per_page = static_cast<uint32_t>(g.epp());
    a.elems_per_layer = static_cast<uint32_t>(g.epl());
    if (launch_kv_append_bf16(a, d_k, d_v, nullptr) != hipSuccess) {
        std::printf("FAIL: launch_kv_append_bf16 D=%u\n", head_dim); ++failures;
    }
    HIP_OK(hipMemcpy(kpool.data(), d_k, kpool.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(vpool.data(), d_v, vpool.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
    hipFree(d_k); hipFree(d_v); hipFree(d_kin); hipFree(d_vin);
    hipFree(d_tbl); hipFree(d_slots); hipFree(d_pos);
    HIP_OK(hipGetLastError());

    // exact copy semantics: every element of appended slot matches input bits
    bool ok = true;
    for (uint32_t s = 0; s < rows; ++s) {
        const uint32_t pos = rows_pos[s];
        const uint32_t page = c.table[static_cast<size_t>(s) * g.table_width
                                      + pos / g.page_tokens];
        const uint32_t off = pos % g.page_tokens;
        for (uint32_t h = 0; h < c.kv_heads && ok; ++h) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const shelf::bf16_t ek = kin[(static_cast<size_t>(s) * c.kv_heads + h)
                                             * head_dim + d];
                const shelf::bf16_t ev = vin[(static_cast<size_t>(s) * c.kv_heads + h)
                                             * head_dim + d];
                if (kpool[g.idx(page, off, h, d)].data != ek.data ||
                    vpool[g.idx(page, off, h, d)].data != ev.data) { ok = false; break; }
            }
        }
    }
    // untouched neighbor slot (block 3, off 4) still the fill pattern
    for (uint32_t h = 0; h < c.kv_heads && ok; ++h)
        for (uint32_t d = 0; d < head_dim; ++d)
            if (kpool[g.idx(3, 4, h, d)].data != kFillPat ||
                vpool[g.idx(3, 4, h, d)].data != kFillPat) { ok = false; break; }
    char tag[64];
    std::snprintf(tag, sizeof tag, "kv_append D=%u exact-write + untouched-neighbor", head_dim);
    check(ok, tag);
}

// ------------------------------------------------------------------ fp8 (PR-0 gap)
// host e4m3fn decode — must match device e4m3_decode_f32 bit-exactly for the
// values we generate (e4m3 values are exact in f32/bf16, so the kernel's
// bf16 LUT round-trip f2bf_p8 is lossless on them).
double e4m3_f64(uint8_t c) {
    const uint32_t sgn = (c >> 7) & 1u, e = (c >> 3) & 0xFu, m = c & 7u;
    if (e == 0xFu && m == 0x7u) return std::nan("");
    const double v = (e == 0u)
        ? static_cast<double>(m) / 8.0 * 0.015625   // m * 2^-9 (subnormal)
        : (1.0 + static_cast<double>(m) / 8.0) * std::pow(2.0, static_cast<int>(e) - 7);
    return sgn ? -v : v;
}

void run_fp8_attention_case(uint32_t head_dim, bool use_split, uint32_t q_heads = 4) {
    AttnCase c;
    c.q_heads = q_heads;
    c.head_dim = head_dim;
    c.table = {3, 0, 5, ~0u, ~0u, ~0u, ~0u, ~0u,
               7, 1, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    PoolGeom g;
    g.head_dim = head_dim;
    char tag[80];
    std::snprintf(tag, sizeof tag, "attn-fp8 nq=%u kvh=%u D=%u %s", c.q_heads,
                  c.kv_heads, head_dim, use_split ? "split" : "decode");

    std::mt19937 rng(2024 + head_dim * 13u + q_heads * 57u + (use_split ? 9u : 3u));
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    const size_t q_elems = static_cast<size_t>(c.rows) * c.q_heads * head_dim;
    std::vector<shelf::bf16_t> q(q_elems);
    for (auto& v : q) v = shelf::f32_to_bf16(dist(rng));

    // fp8 pools store one byte per element; fill pattern marks untouched slots
    const uint8_t fp8Fill = 0x00u;
    std::vector<uint8_t> kpool(g.num_pages * g.epp(), fp8Fill),
                       vpool(g.num_pages * g.epp(), fp8Fill);
    const uint32_t total_tokens = c.seqused[0] + c.seqused[1];
    // random non-NaN e4m3 codes, magnitude kept ≤ 3.75 so bf16 output rounding
    // stays well inside the 5e-2 compare tolerance
    auto next_code = [&]() {
        uint8_t x;
        do { x = static_cast<uint8_t>(rng() & 0xFFu); }
        while (((x >> 3) & 0xFu) > 8u || (x & 0x7Fu) == 0x7Fu);
        return x;
    };
    std::vector<float> kscale(static_cast<size_t>(g.num_pages) * g.page_tokens * c.kv_heads),
                     vscale(static_cast<size_t>(g.num_pages) * g.page_tokens * c.kv_heads);
    std::uniform_real_distribution<float> sdist(0.5f, 1.5f);
    for (auto& v : kscale) v = sdist(rng);
    for (auto& v : vscale) v = sdist(rng);

    size_t tsrc = 0;
    for (uint32_t s = 0; s < c.rows; ++s) {
        for (uint32_t t = 0; t < c.seqused[s]; ++t, ++tsrc) {
            const uint32_t page = c.table[static_cast<size_t>(s) * g.table_width
                                          + t / g.page_tokens];
            const uint32_t off = t % g.page_tokens;
            for (uint32_t h = 0; h < c.kv_heads; ++h)
                for (uint32_t d = 0; d < head_dim; ++d) {
                    kpool[g.idx(page, off, h, d)] = next_code();
                    vpool[g.idx(page, off, h, d)] = next_code();
                }
        }
    }

    uint8_t *d_k = nullptr, *d_v = nullptr;
    shelf::bf16_t *d_q = nullptr, *d_o = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    uint32_t *d_tbl = nullptr, *d_slots = nullptr, *d_rows_pos = nullptr;
    HIP_OK(hipMalloc(&d_k, kpool.size()));
    HIP_OK(hipMalloc(&d_v, vpool.size()));
    HIP_OK(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_o, q.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_ks, kscale.size() * sizeof(float)));
    HIP_OK(hipMalloc(&d_vs, vscale.size() * sizeof(float)));
    HIP_OK(hipMalloc(&d_tbl, c.table.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_slots, c.rows * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_rows_pos, c.rows * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(d_k, kpool.data(), kpool.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_v, vpool.data(), vpool.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_ks, kscale.data(), kscale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_vs, vscale.data(), vscale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_tbl, c.table.data(), c.table.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    {
        const uint32_t slots[2] = {0u, 1u};
        const uint32_t rows_pos[2] = {c.seqused[0] - 1u, c.seqused[1] - 1u};
        HIP_OK(hipMemcpy(d_slots, slots, sizeof slots, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_rows_pos, rows_pos, sizeof rows_pos, hipMemcpyHostToDevice));
    }
    (void)total_tokens;

    PagedAttentionCommonArgs a{};
    a.q = d_q;
    a.output = d_o;
    a.output_dtype = PagedAttentionOutputDType::BF16;
    a.q_row_stride = c.q_heads * head_dim;
    a.output_row_stride = c.q_heads * head_dim;
    a.row_sequence_slots = d_slots;
    a.row_positions = d_rows_pos;
    a.block_tables = d_tbl;
    a.block_table_stride = g.table_width;
    a.max_sequences = c.rows;
    a.rows = c.rows;
    a.layer = 0;
    a.page_tokens = g.page_tokens;
    a.num_pages = g.num_pages;
    a.num_attention_layers = 1;
    a.q_heads = c.q_heads;
    a.kv_heads = c.kv_heads;
    a.head_dim = head_dim;
    a.elems_per_token = static_cast<uint32_t>(g.ept());
    a.elems_per_page = static_cast<uint32_t>(g.epp());
    a.elems_per_layer = static_cast<uint32_t>(g.epl());
    a.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto fp8 = reinterpret_cast<shelf::hipk::fp8e4m3_storage_t*>(d_k);
    auto fp8v = reinterpret_cast<shelf::hipk::fp8e4m3_storage_t*>(d_v);

    hipError_t st = hipErrorInvalidValue;
    if (!use_split) {
        st = launch_attention_paged_fp8_e4m3(a, fp8, fp8v, d_ks, d_vs, nullptr);
    } else {
        const uint32_t splits = 4u;
        const uint32_t qpk = c.q_heads / c.kv_heads;
        std::vector<float> partials(static_cast<size_t>(c.rows) * c.kv_heads
                                    * splits * qpk * (2u + 256u), 0.0f);
        float* d_part = nullptr;
        HIP_OK(hipMalloc(&d_part, partials.size() * sizeof(float)));
        st = launch_attention_paged_fp8_split(a, fp8, fp8v, d_ks, d_vs, d_part, splits, nullptr);
        if (st == hipSuccess)
            st = launch_attention_paged_bf16_split_reduce(a, d_part, splits, nullptr);
        hipFree(d_part);
    }
    if (st != hipSuccess) {
        std::printf("FAIL: launch %s: %s\n", tag, hipGetErrorString(st));
        ++failures;
    }
    std::vector<shelf::bf16_t> got(q_elems);
    HIP_OK(hipMemcpy(got.data(), d_o, got.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
    hipFree(d_k); hipFree(d_v); hipFree(d_q); hipFree(d_o);
    hipFree(d_ks); hipFree(d_vs);
    hipFree(d_tbl); hipFree(d_slots); hipFree(d_rows_pos);
    HIP_OK(hipGetLastError());

    // f64 reference over decoded e4m3 values with per-token scales
    const uint32_t q_per_kv = c.q_heads / c.kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> ref(static_cast<size_t>(c.rows) * c.q_heads * head_dim, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        const uint32_t vis = c.seqused[r];
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const uint32_t kvh = h / q_per_kv;
            std::vector<double> score(vis);
            double m = -1e300;
            for (uint32_t p = 0; p < vis; ++p) {
                const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                              + p / g.page_tokens];
                const uint32_t off = p % g.page_tokens;
                const double ksc = kscale[(static_cast<size_t>(page) * g.page_tokens + off)
                                          * c.kv_heads + kvh];
                double acc = 0.0;
                for (uint32_t d = 0; d < head_dim; ++d)
                    acc += bf2f(q[(static_cast<size_t>(r) * c.q_heads + h) * head_dim + d])
                         * e4m3_f64(kpool[g.idx(page, off, kvh, d)]);
                score[p] = acc * ksc * scale;
                if (score[p] > m) m = score[p];
            }
            double sum = 0.0;
            std::vector<double> w(vis);
            for (uint32_t p = 0; p < vis; ++p) {
                w[p] = std::exp(score[p] - m);
                sum += w[p];
            }
            for (uint32_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (uint32_t p = 0; p < vis; ++p) {
                    const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                                  + p / g.page_tokens];
                    const uint32_t off = p % g.page_tokens;
                    const double vsc = vscale[(static_cast<size_t>(page) * g.page_tokens + off)
                                              * c.kv_heads + kvh];
                    acc += w[p] * vsc * e4m3_f64(vpool[g.idx(page, off, kvh, d)]);
                }
                ref[(static_cast<size_t>(r) * c.q_heads + h) * head_dim + d] = acc / sum;
            }
        }
    }
    compare_attention_gap(c, tag, got, ref, 5e-2);  // fp8 output magnitudes widen bf16 ulp
}

// ------------------------------------------------------------------ tree ancestors (PR-0 gap)
// Exercises the tree_anc_offsets path: a committed block-table window plus a
// non-contiguous ancestor list (row 0) and a contiguous ancestor merge (row 1).
void run_attention_tree_case(uint32_t head_dim) {
    AttnCase c;
    c.head_dim = head_dim;
    c.table = {3, 0, 5, ~0u, ~0u, ~0u, ~0u, ~0u,
               7, 1, 5, ~0u, ~0u, ~0u, ~0u, ~0u};
    PoolGeom g;
    g.head_dim = head_dim;

    // row 0: committed_end = 16 (page 3 only), ancestors at {16,17,20}
    //   (page 0: off 0,1,4) — NOT contiguous (gap at 18,19)
    // row 1: committed_end = 32 (pages 7,1), ancestors {32,33,34} on page 5
    //   — contiguous, must merge into visible=35 with anc_count=0
    const uint32_t committed[2] = {16u, 32u};
    const uint32_t anc_offsets[3] = {0u, 3u, 6u};
    const uint32_t anc_positions[6] = {16u, 17u, 20u, 32u, 33u, 34u};

    std::mt19937 rng(999 + head_dim);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    const size_t q_elems = static_cast<size_t>(c.rows) * c.q_heads * head_dim;
    std::vector<shelf::bf16_t> q(q_elems);
    for (auto& v : q) v = shelf::f32_to_bf16(dist(rng));

    std::vector<shelf::bf16_t> kpool(g.num_pages * g.epp()), vpool(g.num_pages * g.epp());
    fill_pool(kpool, g, kFillPat);
    fill_pool(vpool, g, kFillPat);

    // token streams per row, logical position order
    std::vector<shelf::bf16_t> k_tok(35 * c.kv_heads * head_dim),
                               v_tok(35 * c.kv_heads * head_dim);
    for (auto& v : k_tok) v = shelf::f32_to_bf16(dist(rng));
    for (auto& v : v_tok) v = shelf::f32_to_bf16(dist(rng));

    // visible positions per row: block-table part + ancestor part
    std::vector<std::vector<uint32_t>> pos(2);
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t p = 0; p < committed[r]; ++p) pos[r].push_back(p);
    }
    for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t base = anc_offsets[r], cnt = anc_offsets[r + 1] - base;
        // row 0 appends non-contiguous; row 1's anc merge extends visible
        if (r == 0) for (uint32_t i = 0; i < cnt; ++i) pos[r].push_back(anc_positions[base + i]);
        else        for (uint32_t i = 0; i < cnt; ++i) pos[r].push_back(committed[r] + i);
    }
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t t = 0; t < pos[r].size(); ++t) {
            const uint32_t p = pos[r][t];
            const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                          + p / g.page_tokens];
            const uint32_t off = p % g.page_tokens;
            for (uint32_t h = 0; h < c.kv_heads; ++h)
                for (uint32_t d = 0; d < head_dim; ++d) {
                    kpool[g.idx(page, off, h, d)] =
                        k_tok[(static_cast<size_t>(t) * c.kv_heads + h) * head_dim + d];
                    vpool[g.idx(page, off, h, d)] =
                        v_tok[(static_cast<size_t>(t) * c.kv_heads + h) * head_dim + d];
                }
        }
    }

    shelf::bf16_t *d_k = nullptr, *d_v = nullptr, *d_q = nullptr, *d_o = nullptr;
    uint32_t *d_tbl = nullptr, *d_slots = nullptr, *d_rows_pos = nullptr;
    uint32_t *d_anc_off = nullptr, *d_anc_pos = nullptr, *d_committed = nullptr;
    HIP_OK(hipMalloc(&d_k, kpool.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_v, vpool.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_o, q.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_tbl, c.table.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_slots, c.rows * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_rows_pos, c.rows * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_anc_off, sizeof anc_offsets));
    HIP_OK(hipMalloc(&d_anc_pos, sizeof anc_positions));
    HIP_OK(hipMalloc(&d_committed, sizeof committed));
    HIP_OK(hipMemcpy(d_k, kpool.data(), kpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_v, vpool.data(), vpool.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_tbl, c.table.data(), c.table.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    {
        const uint32_t slots[2] = {0u, 1u};
        const uint32_t rows_pos[2] = {committed[0], committed[1] + 2u};  // unused in tree path
        HIP_OK(hipMemcpy(d_slots, slots, sizeof slots, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_rows_pos, rows_pos, sizeof rows_pos, hipMemcpyHostToDevice));
    }
    HIP_OK(hipMemcpy(d_anc_off, anc_offsets, sizeof anc_offsets, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_anc_pos, anc_positions, sizeof anc_positions, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_committed, committed, sizeof committed, hipMemcpyHostToDevice));

    PagedAttentionCommonArgs a{};
    a.q = d_q;
    a.output = d_o;
    a.output_dtype = PagedAttentionOutputDType::BF16;
    a.q_row_stride = c.q_heads * head_dim;
    a.output_row_stride = c.q_heads * head_dim;
    a.row_sequence_slots = d_slots;
    a.row_positions = d_rows_pos;
    a.tree_anc_offsets = d_anc_off;
    a.tree_anc_positions = d_anc_pos;
    a.tree_committed_end = 0u;
    a.tree_committed_end_rows = d_committed;
    a.block_tables = d_tbl;
    a.block_table_stride = g.table_width;
    a.max_sequences = c.rows;
    a.rows = c.rows;
    a.layer = 0;
    a.page_tokens = g.page_tokens;
    a.num_pages = g.num_pages;
    a.num_attention_layers = 1;
    a.q_heads = c.q_heads;
    a.kv_heads = c.kv_heads;
    a.head_dim = head_dim;
    a.elems_per_token = static_cast<uint32_t>(g.ept());
    a.elems_per_page = static_cast<uint32_t>(g.epp());
    a.elems_per_layer = static_cast<uint32_t>(g.epl());
    a.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const hipError_t st = launch_attention_paged_bf16(a, d_k, d_v, nullptr);
    if (st != hipSuccess) {
        std::printf("FAIL: launch attn-tree D=%u: %s\n", head_dim, hipGetErrorString(st));
        ++failures;
    }
    std::vector<shelf::bf16_t> got(q_elems);
    HIP_OK(hipMemcpy(got.data(), d_o, got.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
    hipFree(d_k); hipFree(d_v); hipFree(d_q); hipFree(d_o);
    hipFree(d_tbl); hipFree(d_slots); hipFree(d_rows_pos);
    hipFree(d_anc_off); hipFree(d_anc_pos); hipFree(d_committed);
    HIP_OK(hipGetLastError());

    const uint32_t q_per_kv = c.q_heads / c.kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> ref(static_cast<size_t>(c.rows) * c.q_heads * head_dim, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        const uint32_t vis = static_cast<uint32_t>(pos[r].size());
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const uint32_t kvh = h / q_per_kv;
            std::vector<double> score(vis);
            double m = -1e300;
            for (uint32_t t = 0; t < vis; ++t) {
                const uint32_t p = pos[r][t];
                const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                              + p / g.page_tokens];
                const uint32_t off = p % g.page_tokens;
                double acc = 0.0;
                for (uint32_t d = 0; d < head_dim; ++d)
                    acc += bf2f(q[(static_cast<size_t>(r) * c.q_heads + h) * head_dim + d])
                         * bf2f(kpool[g.idx(page, off, kvh, d)]);
                score[t] = acc * scale;
                if (score[t] > m) m = score[t];
            }
            double sum = 0.0;
            std::vector<double> w(vis);
            for (uint32_t t = 0; t < vis; ++t) { w[t] = std::exp(score[t] - m); sum += w[t]; }
            for (uint32_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (uint32_t t = 0; t < vis; ++t) {
                    const uint32_t p = pos[r][t];
                    const uint32_t page = c.table[static_cast<size_t>(r) * g.table_width
                                                  + p / g.page_tokens];
                    const uint32_t off = p % g.page_tokens;
                    acc += w[t] * bf2f(vpool[g.idx(page, off, kvh, d)]);
                }
                ref[(static_cast<size_t>(r) * c.q_heads + h) * head_dim + d] = acc / sum;
            }
        }
    }
    char tag[64];
    std::snprintf(tag, sizeof tag, "attn-tree D=%u anc-mixed", head_dim);
    compare_attention_gap(c, tag, got, ref, 5e-2);  // tree: tie-gap rule (see comparator)
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        std::printf("test_hip_attention: failures=1\n");
        return 1;
    }

    run_inv_freq_case(32, 1e6f);

    // rope parity: partial rotary 64, pair (i, i+32)
    {
        std::vector<uint32_t> pos = {0, 7, 34, 999};
        std::vector<double> inv32(32);
        for (size_t i = 0; i < 32; ++i)
            inv32[i] = 1.0 / std::pow(1e6, static_cast<double>(i) / 32.0);
        run_rope_case(4, 4, 256, 64, 1e6, inv32, pos);  // pair path (D=256)
        run_rope_case(4, 2, 64, 64, 1e6, inv32, pos);   // generic path (D=64)
    }

    // paged GQA decode: rope first, then attention vs f64 reference
    run_attention_case(256, /*do_rope=*/true, /*use_split=*/false);
    run_attention_case(64, /*do_rope=*/true, /*use_split=*/false);
    // split-K path exists for q_per_kv in {4,6}; nq=8 kvh=2 exercises it
    run_attention_case(256, /*do_rope=*/false, /*use_split=*/true, /*q_heads=*/8);
    // REGRESSION (fix df37c07): q_per_kv=4 with head_dim != kHeadLimit(256)
    // once fell through a dead "if (full_head)" guard in the bf16
    // split/split_reduce launchers -> NO kernel launched, partials stayed
    // zero, output read back all-zero, and hipGetLastError still reported
    // success. The named AOT kernels only cover D=256, so this case pins
    // the template instantiations (impl<4u,false>) that the hole orphaned.
    run_attention_case(128, /*do_rope=*/false, /*use_split=*/true, /*q_heads=*/8);

    // PR-0 gap fill: split counts, q_per_kv=6, head-grouping reduce variants
    run_attention_case(256, false, true, 8, 2u);    // splits=2
    run_attention_case(256, false, true, 12);       // q_per_kv=6 split/reduce
    setenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH", "2", 1);
    run_attention_case(256, false, true, 8);        // reduce q4_full_qh2 entry
    setenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH", "4", 1);
    run_attention_case(256, false, true, 8);        // reduce q4_full_base entry
    unsetenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH");

    // PR-0 gap fill: tree ancestors (non-contiguous + contiguous merge)
    run_attention_tree_case(256);
    run_attention_tree_case(64);

    // PR-0 gap fill: FP8 all paths (decode full/partial; split requires
    // q_per_kv in {4,6}: nq=8→qpk4, nq=12→qpk6)
    run_fp8_attention_case(256, false);
    run_fp8_attention_case(64, false);
    run_fp8_attention_case(256, true, 8);
    run_fp8_attention_case(64, true, 8);
    run_fp8_attention_case(256, true, 12);          // fp8 split q_per_kv=6

    run_kv_append_case(256);
    run_kv_append_case(64);

    std::printf("test_hip_attention: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
