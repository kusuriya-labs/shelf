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
