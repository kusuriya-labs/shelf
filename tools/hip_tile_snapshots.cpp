// shelf tool — bitwise before/after snapshots for the HIP Tile refactor
// (docs/architecture/hip-tile-refactor-procedure.md §2.3).
//
// Modes:
//   record <dir>     write deterministic kernel outputs (fixed inputs) to <dir>
//   verify <dir>     re-run and require byte-identical outputs vs <dir>
//   roundtrip        twice in-process, byte-compare (selfcheck, no dir needed)
//
// Same variant must compare bitwise; the numeric tolerance tests
// (unit.hip_gemm_bf16) remain the correctness gate.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::hipk::Bf16GemmConfig;
using shelf::hipk::Bf16GemmConfigId;
using shelf::hipk::Bf16GemmOutputDType;
using shelf::hipk::launch_gemm_bf16;

struct Case {
    uint32_t rows, n, k, is, os;
};

// fixed input matrix (same list as test_hip_gemm_bf16 PR-0b)
const Case kCases[] = {
    {1, 2560, 2560, 0, 0}, {1, 9216, 2560, 0, 0}, {4, 512, 256, 0, 0},
    {1, 271, 40, 0, 0},    {2, 64, 16, 0, 0},     {4, 17, 1, 0, 0},
    {15, 33, 63, 0, 0},    {16, 256, 256, 0, 0},  {17, 33, 63, 0, 0},
    {63, 16, 16, 0, 0},    {64, 32, 15, 0, 0},    {65, 64, 272, 0, 0},
    {4, 512, 256, 264, 520}, {16, 33, 17, 24, 40},
};
constexpr const char* kCfgNames[] = {"ExactRows", "ExactRowsDot2", "Wmma",
                                     "WmmaWide", "WmmaKPartition"};
constexpr const char* kOutNames[] = {"f32", "bf16"};

// deterministic host inputs for one case
void make_inputs(const Case& c, std::vector<shelf::bf16_t>& w,
                 std::vector<shelf::bf16_t>& x) {
    std::mt19937 rng(1234 + c.rows * 7919u + c.n * 31u + c.k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    w.assign(static_cast<size_t>(c.n) * c.k, shelf::bf16_t{});
    const uint32_t is = c.is ? c.is : c.k;
    x.assign(static_cast<size_t>(c.rows) * is, shelf::bf16_t{});
    for (auto& v : w) v = shelf::f32_to_bf16(dist(rng));
    for (auto& v : x) v = shelf::f32_to_bf16(dist(rng));
    for (uint32_t r = 0; r < c.rows; ++r)
        for (uint32_t kk = c.k; kk < is; ++kk)
            x[static_cast<size_t>(r) * is + kk] = shelf::f32_to_bf16(123.5f);
}

std::string case_tag(const Case& c) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "r%u_n%u_k%u_is%u_os%u", c.rows, c.n, c.k, c.is, c.os);
    return buf;
}

// one shot: run cfg/odt for case c, return raw output bytes (F32 word grid)
std::vector<uint8_t> snapshot_one(const Case& c, Bf16GemmConfigId id,
                                  Bf16GemmOutputDType odt) {
    std::vector<shelf::bf16_t> w, x;
    make_inputs(c, w, x);
    const uint32_t is = c.is ? c.is : c.k;
    const uint32_t os = c.os ? c.os : c.n;
    void *dw = nullptr, *dx = nullptr, *dy = nullptr;
    if (hipMalloc(&dw, w.size() * sizeof(shelf::bf16_t)) != hipSuccess) return {};
    if (hipMalloc(&dx, x.size() * sizeof(shelf::bf16_t)) != hipSuccess) return {};
    const size_t out_bytes = static_cast<size_t>(c.rows) * os * sizeof(float);
    if (hipMalloc(&dy, out_bytes) != hipSuccess) return {};
    hipMemcpy(dw, w.data(), w.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    hipMemset(dy, 0x5A, out_bytes);  // sentinel: untouched bytes are visible
    Bf16GemmConfig cfg{id, static_cast<uint8_t>(c.rows)};
    const hipError_t err = launch_gemm_bf16(cfg, reinterpret_cast<const shelf::bf16_t*>(dw),
                                            reinterpret_cast<const shelf::bf16_t*>(dx), dy,
                                            odt, c.rows, c.n, c.k, is, os, 0);
    std::vector<uint8_t> out;
    if (err == hipSuccess) {
        out.resize(out_bytes);
        hipMemcpy(out.data(), dy, out_bytes, hipMemcpyDeviceToHost);
    } else {
        out.push_back(0xFFu);  // unsupported marker (deterministic)
        out.push_back(0xF0u);
        hipMemcpy(out.data() + 2, dy, 2, hipMemcpyDeviceToHost);
    }
    hipFree(dw); hipFree(dx); hipFree(dy);
    return out;
}

// ---------------------------------------------------------------- attention
// fixed-geometry builders for paged decode/split snapshots (contract §4).
namespace attn {

constexpr uint32_t kPageTokens = 16u, kTableWidth = 8u, kNumPages = 16u;

struct Built {
    std::vector<uint8_t> out;       // output bytes (bf16 elements)
    std::vector<uint8_t> partials;  // split workspace bytes (empty if none)
    bool ok = false;
};

inline std::size_t ept(uint32_t kvh, uint32_t D) { return std::size_t(kvh) * D; }
inline std::size_t epp(uint32_t kvh, uint32_t D) { return std::size_t(kPageTokens) * ept(kvh, D); }
inline std::size_t idx(uint32_t page, uint32_t off, uint32_t h, uint32_t d, uint32_t kvh, uint32_t D) {
    return std::size_t(page) * epp(kvh, D) + std::size_t(off) * ept(kvh, D) + std::size_t(h) * D + d;
}

struct Cfg {
    uint32_t D, qh, kvh, rows;
    bool fp8, split, tree;
    uint32_t splits;
};

Built run(const Cfg& cf) {
    Built r;
    std::mt19937 rng(31337 + cf.D * 101u + cf.qh * 17u + cf.kvh + (cf.fp8 ? 7u : 0u) +
                     (cf.split ? 13u : 0u) + (cf.tree ? 29u : 0u));
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    const uint32_t seqused[2] = {35u, 20u};
    std::vector<uint32_t> table = {3, 0, 5, ~0u, ~0u, ~0u, ~0u, ~0u,
                                   7, 1, 5, ~0u, ~0u, ~0u, ~0u, ~0u};
    const size_t qn = size_t(cf.rows) * cf.qh * cf.D;
    std::vector<shelf::bf16_t> q(qn);
    for (auto& v : q) v = shelf::f32_to_bf16(dist(rng));

    const size_t pk = kNumPages * epp(cf.kvh, cf.D);
    std::vector<shelf::bf16_t> kb16, vb16;
    std::vector<uint8_t> k8, v8;
    std::vector<float> ks, vs;
    if (cf.fp8) {
        k8.assign(pk, 0x00u); v8.assign(pk, 0x00u);
        auto code = [&]() { uint8_t x; do { x = uint8_t(rng() & 0xFFu); } while (((x >> 3) & 0xFu) > 8u || (x & 0x7Fu) == 0x7Fu); return x; };
        for (auto& e : k8) e = code();
        for (auto& e : v8) e = code();
        ks.assign(size_t(kNumPages) * kPageTokens * cf.kvh, 1.0f);
        vs = ks;
        for (auto& e : ks) e = 0.75f + 0.5f * dist(rng) / 1.5f;
        for (auto& e : vs) e = 0.75f + 0.5f * dist(rng) / 1.5f;
    } else {
        kb16.assign(pk, shelf::bf16_t{uint16_t(0x3C3Cu)});
        vb16 = kb16;
        for (auto& e : kb16) e = shelf::f32_to_bf16(dist(rng));
        for (auto& e : vb16) e = shelf::f32_to_bf16(dist(rng));
    }
    // pools are fully pre-filled deterministically; unused slots never read

    void *d_k, *d_v, *d_q, *d_o, *d_ks = nullptr, *d_vs = nullptr, *d_part = nullptr;
    uint32_t *d_tbl, *d_slots, *d_pos, *d_ao = nullptr, *d_ap = nullptr, *d_ce = nullptr;
    if (cf.fp8) { hipMalloc(&d_k, pk); hipMalloc(&d_v, pk);
        hipMalloc(&d_ks, ks.size() * 4); hipMalloc(&d_vs, vs.size() * 4);
        hipMemcpy(d_k, k8.data(), pk, hipMemcpyHostToDevice);
        hipMemcpy(d_v, v8.data(), pk, hipMemcpyHostToDevice);
        hipMemcpy(d_ks, ks.data(), ks.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(d_vs, vs.data(), vs.size() * 4, hipMemcpyHostToDevice);
    } else {
        hipMalloc(&d_k, pk * 2); hipMalloc(&d_v, pk * 2);
        hipMemcpy(d_k, kb16.data(), pk * 2, hipMemcpyHostToDevice);
        hipMemcpy(d_v, vb16.data(), pk * 2, hipMemcpyHostToDevice);
    }
    hipMalloc(&d_q, q.size() * 2); hipMalloc(&d_o, q.size() * 2);
    hipMemset(d_o, 0x5A, q.size() * 2);
    hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&d_tbl, table.size() * 4);
    hipMemcpy(d_tbl, table.data(), table.size() * 4, hipMemcpyHostToDevice);
    hipMalloc(&d_slots, cf.rows * 4); hipMalloc(&d_pos, cf.rows * 4);
    {
        uint32_t slots[2] = {0, 1}, pos[2] = {34, 19};
        hipMemcpy(d_slots, slots, 8, hipMemcpyHostToDevice);
        hipMemcpy(d_pos, pos, 8, hipMemcpyHostToDevice);
    }
    using namespace shelf::hipk;
    PagedAttentionCommonArgs a{};
    a.q = (const shelf::bf16_t*)d_q; a.output = d_o;
    a.output_dtype = PagedAttentionOutputDType::BF16;
    a.q_row_stride = cf.qh * cf.D; a.output_row_stride = cf.qh * cf.D;
    a.row_sequence_slots = d_slots; a.row_positions = d_pos;
    a.block_tables = d_tbl; a.block_table_stride = kTableWidth;
    a.max_sequences = cf.rows; a.rows = cf.rows; a.layer = 0;
    a.page_tokens = kPageTokens; a.num_pages = kNumPages; a.num_attention_layers = 1;
    a.q_heads = cf.qh; a.kv_heads = cf.kvh; a.head_dim = cf.D;
    a.elems_per_token = (uint32_t)ept(cf.kvh, cf.D);
    a.elems_per_page = (uint32_t)epp(cf.kvh, cf.D);
    a.elems_per_layer = (uint32_t)(kNumPages * epp(cf.kvh, cf.D));
    a.scale = 1.0f / std::sqrt(float(cf.D));
    if (cf.tree) {
        uint32_t ao[3] = {0, 3, 6}, ap[6] = {16, 17, 20, 32, 33, 34}, ce[2] = {16, 32};
        hipMalloc(&d_ao, sizeof ao); hipMalloc(&d_ap, sizeof ap); hipMalloc(&d_ce, sizeof ce);
        hipMemcpy(d_ao, ao, sizeof ao, hipMemcpyHostToDevice);
        hipMemcpy(d_ap, ap, sizeof ap, hipMemcpyHostToDevice);
        hipMemcpy(d_ce, ce, sizeof ce, hipMemcpyHostToDevice);
        a.tree_anc_offsets = d_ao; a.tree_anc_positions = d_ap;
        a.tree_committed_end_rows = d_ce;
    }

    hipError_t st = hipErrorInvalidValue;
    const auto fp8k = reinterpret_cast<const shelf::hipk::fp8e4m3_storage_t*>(d_k);
    const auto fp8v = reinterpret_cast<const shelf::hipk::fp8e4m3_storage_t*>(d_v);
    if (!cf.split) {
        st = cf.fp8 ? launch_attention_paged_fp8_e4m3(a, fp8k, fp8v, (const float*)d_ks, (const float*)d_vs, nullptr)
                    : launch_attention_paged_bf16(a, (const shelf::bf16_t*)d_k, (const shelf::bf16_t*)d_v, nullptr);
    } else {
        const uint32_t qpk = cf.qh / cf.kvh;
        const size_t pn = size_t(cf.rows) * cf.kvh * cf.splits * qpk * (2u + 256u);
        hipMalloc(&d_part, pn * 4);
        hipMemset(d_part, 0x5A, pn * 4);
        st = cf.fp8 ? launch_attention_paged_fp8_split(a, fp8k, fp8v, (const float*)d_ks, (const float*)d_vs, (float*)d_part, cf.splits, nullptr)
                    : launch_attention_paged_bf16_split(a, (const shelf::bf16_t*)d_k, (const shelf::bf16_t*)d_v, (float*)d_part, cf.splits, nullptr);
        if (st == hipSuccess)
            st = launch_attention_paged_bf16_split_reduce(a, (const float*)d_part, cf.splits, nullptr);
        if (st == hipSuccess) {
            r.partials.resize(pn * 4);
            hipMemcpy(r.partials.data(), d_part, pn * 4, hipMemcpyDeviceToHost);
        }
        hipFree(d_part);
    }
    if (st == hipSuccess) {
        r.out.resize(q.size() * 2);
        hipMemcpy(r.out.data(), d_o, q.size() * 2, hipMemcpyDeviceToHost);
        r.ok = true;
    }
    hipFree(d_k); hipFree(d_v); hipFree(d_q); hipFree(d_o);
    if (d_ks) hipFree(d_ks);
    if (d_vs) hipFree(d_vs);
    hipFree(d_tbl); hipFree(d_slots); hipFree(d_pos);
    if (d_ao) hipFree(d_ao);
    if (d_ap) hipFree(d_ap);
    if (d_ce) hipFree(d_ce);
    return r;
}

const Cfg kCfgs[] = {
    {256, 4, 2, 2, false, false, false, 0},   // bf16 decode full head
    {64,  4, 2, 2, false, false, false, 0},   // bf16 decode partial head
    {256, 4, 2, 2, false, false, true,  0},   // bf16 tree ancestors
    {64,  4, 2, 2, false, false, true,  0},
    {256, 8, 2, 2, false, true,  false, 4u},  // bf16 split qpk4
    {64,  8, 2, 2, false, true,  false, 2u},  // bf16 split partial head splits=2
    {256, 12, 2, 2, false, true, false, 8u},  // bf16 split qpk6
    {256, 4, 2, 2, true,  false, false, 0},   // fp8 decode
    {64,  4, 2, 2, true,  false, false, 0},
    {256, 8, 2, 2, true,  true,  false, 4u},  // fp8 split
    {64,  8, 2, 2, true,  true,  false, 4u},
    {256, 12, 2, 2, true,  true,  false, 16u},// fp8 split qpk6 splits16
};
std::string name(const Cfg& c) {
    char b[96];
    std::snprintf(b, sizeof b, "attn_%s_D%u_qh%u_%s%s_sp%u", c.fp8 ? "fp8" : "bf16", c.D,
                  c.qh, c.split ? "split" : "decode", c.tree ? "_tree" : "", c.splits);
    return b;
}

}  // namespace attn

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: hip_tile_snapshots {record|verify} <dir> | roundtrip\n");
        return 2;
    }
    const std::string mode = argv[1];
    const std::filesystem::path dir = argc > 2 ? argv[2] : "";
    if (mode == "record" && argc != 3) { std::printf("record needs <dir>\n"); return 2; }
    if (mode == "verify" && argc != 3) { std::printf("verify needs <dir>\n"); return 2; }
    if (mode == "record") std::filesystem::create_directories(dir);
    if (hipSetDevice(0) != hipSuccess) { std::printf("FAIL hipSetDevice\n"); return 1; }

    auto collect = []() {
        std::vector<std::pair<std::string, std::vector<uint8_t>>> all;
        for (const auto& c : kCases)
            for (int ci = 0; ci < 5; ++ci)
                for (int oi = 0; oi < 2; ++oi) {
                    const std::string name =
                        std::string(kCfgNames[ci]) + "_" + kOutNames[oi] + "_" + case_tag(c);
                    all.emplace_back(name, snapshot_one(
                        c, Bf16GemmConfigId(ci),
                        oi ? Bf16GemmOutputDType::BF16 : Bf16GemmOutputDType::F32));
                }
        for (const auto& ac : attn::kCfgs) {
            auto built = attn::run(ac);
            all.emplace_back(attn::name(ac) + ".out", built.ok ? built.out : std::vector<uint8_t>{0xFE});
            if (built.ok && !built.partials.empty())
                all.emplace_back(attn::name(ac) + ".partials", built.partials);
        }
        return all;
    };

    const auto pass1 = collect();
    int mismatches = 0;
    if (mode == "record") {
        for (auto& [name, bytes] : pass1) {
            if (bytes.empty()) { std::printf("FAIL alloc %s\n", name.c_str()); ++mismatches; continue; }
            std::ofstream f(dir / (name + ".bin"), std::ios::binary);
            f.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        }
        std::ofstream mf(dir / "MANIFEST.txt");
        for (auto& [name, bytes] : pass1) mf << name << " " << bytes.size() << "\n";
    } else {
        auto pass2 = collect();
        if (pass2.size() != pass1.size()) { std::printf("FAIL case count\n"); return 1; }
        for (size_t i = 0; i < pass1.size(); ++i) {
            const auto& [name, b1] = pass1[i];
            const auto& [name2, b2] = pass2[i];
            if (mode == "roundtrip") {
                if (b1 != b2) { std::printf("NONDET %s\n", name.c_str()); ++mismatches; }
            } else {  // verify against recorded dir
                std::ifstream f(dir / (name + ".bin"), std::ios::binary);
                if (!f) { std::printf("MISSING %s\n", (dir / (name + ".bin")).string().c_str()); ++mismatches; continue; }
                std::vector<uint8_t> want(b1.size());
                f.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size()));
                const bool short_read = f.gcount() != static_cast<std::streamsize>(want.size());
                if (short_read || want != b2) {
                    size_t at = 0;
                    while (at < want.size() && at < b2.size() && want[at] == b2[at]) ++at;
                    std::printf("DIFF %s at byte %zu (baseline %zu vs now %zu)\n", name.c_str(), at,
                                want.size(), b2.size());
                    ++mismatches;
                }
            }
        }
    }
    std::printf("hip_tile_snapshots %s: cases=%d mismatches=%d -> %s\n", mode.c_str(),
                static_cast<int>(pass1.size()), mismatches,
                mismatches ? "FAIL" : (mode == "record" ? "OK (recorded)" : "OK (bitwise)"));
    return mismatches == 0 ? 0 : 1;
}
