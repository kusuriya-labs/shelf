// Micro benchmark of the grouped mul24 EXL3 kernel: FP16 GEMM vs INT8 activations (iu8 WMMA), on the routed-expert
// shapes of a model config (default: Qwen3.8-Flash-Next mul24). One layer's gate_proj (k = hidden, n = moe
// intermediate, token-row gather) and down_proj (k = intermediate, n = hidden, group rows) are timed per row count.
//   bench_exl3_int8_grouped [--model-dir DIR] [--bits N] [--skew] [--iters N]
// The group plan is built on the host from a random top-k routing (uniform, or --skew: expert popularity ~ 1/(i+1)).
// SHELF_EXL3_GROUPED_FUSE / SHELF_EXL3_GROUPED_NTILES apply as in the runtime.
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", static_cast<int>(e_), hipGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

namespace {

namespace kernel = shelf::kernel;
constexpr uint32_t kRowTile = kernel::kExl3W16A16RowTile;

struct Plane {
    uint32_t k, n;
    void* trellis = nullptr;
    void* suh = nullptr;
    void* svh = nullptr;
    uint64_t trellis_bytes = 0;
};

Plane make_plane(uint32_t experts, uint32_t k, uint32_t n, uint32_t bits, std::mt19937& rng) {
    Plane p;
    p.k = k;
    p.n = n;
    p.trellis_bytes = static_cast<uint64_t>(k / 16u) * (n / 16u) * 16u * bits * 2u;
    HIP_CHECK(hipMalloc(&p.trellis, p.trellis_bytes * experts));
    std::vector<uint32_t> host(p.trellis_bytes / 4u);
    for (uint32_t e = 0; e < experts; ++e) {
        for (auto& w : host) w = static_cast<uint32_t>(rng());
        HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(p.trellis) + e * p.trellis_bytes, host.data(), p.trellis_bytes,
                            hipMemcpyHostToDevice));
    }
    std::vector<uint16_t> suh(static_cast<size_t>(experts) * k, 0x3C00u), svh(static_cast<size_t>(experts) * n, 0x3800u);
    for (auto& s : suh) s = (rng() & 1u) ? 0x3C00u : 0xBC00u;  // +-1
    HIP_CHECK(hipMalloc(&p.suh, suh.size() * 2));
    HIP_CHECK(hipMalloc(&p.svh, svh.size() * 2));
    HIP_CHECK(hipMemcpy(p.suh, suh.data(), suh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(p.svh, svh.data(), svh.size() * 2, hipMemcpyHostToDevice));
    return p;
}

struct Plan {
    uint32_t groups = 0, active_experts = 0, rows = 0;
    uint32_t* group_expert = nullptr;
    uint32_t* group_rows = nullptr;
    uint32_t* group_count = nullptr;
    uint32_t* expert_begin = nullptr;
    int32_t* row_token = nullptr;
};

Plan make_plan(uint32_t tokens, uint32_t experts, uint32_t topk, bool skew, std::mt19937& rng) {
    std::vector<double> w(experts, 1.0);
    if (skew)
        for (uint32_t i = 0; i < experts; ++i) w[i] = 1.0 / (i + 1.0);
    std::vector<std::vector<int32_t>> per_expert(experts);
    std::vector<uint32_t> perm(experts);
    for (uint32_t t = 0; t < tokens; ++t) {
        // topk distinct experts, sampled by weight (Efraimidis-Spirakis keys).
        std::vector<std::pair<double, uint32_t>> keys(experts);
        std::uniform_real_distribution<double> u(1e-12, 1.0);
        for (uint32_t e = 0; e < experts; ++e) keys[e] = {std::pow(u(rng), 1.0 / w[e]), e};
        std::partial_sort(keys.begin(), keys.begin() + topk, keys.end(), std::greater<>());
        for (uint32_t j = 0; j < topk; ++j) per_expert[keys[j].second].push_back(static_cast<int32_t>(t));
    }
    Plan p;
    std::vector<uint32_t> ge, gr, begin(experts + 1u, 0u);
    std::vector<int32_t> rt;
    for (uint32_t e = 0; e < experts; ++e) {
        const uint32_t c = static_cast<uint32_t>(per_expert[e].size());
        if (c != 0u) ++p.active_experts;
        p.rows += c;
        for (uint32_t off = 0; off < c; off += kRowTile) {
            const uint32_t rows = std::min(kRowTile, c - off);
            ge.push_back(e);
            gr.push_back(rows);
            for (uint32_t r = 0; r < kRowTile; ++r) rt.push_back(r < rows ? per_expert[e][off + r] : -1);
        }
        begin[e + 1u] = static_cast<uint32_t>(ge.size());
    }
    p.groups = static_cast<uint32_t>(ge.size());
    auto up = [](auto& v, auto*& d) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d), std::max<size_t>(v.size(), 1) * sizeof(v[0])));
        HIP_CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(v[0]), hipMemcpyHostToDevice));
    };
    up(ge, p.group_expert);
    up(gr, p.group_rows);
    up(begin, p.expert_begin);
    up(rt, p.row_token);
    std::vector<uint32_t> cnt = {p.groups};
    up(cnt, p.group_count);
    return p;
}

void free_plan(Plan& p) {
    (void)hipFree(p.group_expert);
    (void)hipFree(p.group_rows);
    (void)hipFree(p.group_count);
    (void)hipFree(p.expert_begin);
    (void)hipFree(p.row_token);
}

// Returns ms per launch.
double time_launch(const Plane& pl, const Plan& plan, const void* input, uint32_t input_stride, void* output,
                   uint32_t bits, bool token_rows, uint32_t int8_act, int iters) {
    kernel::Exl3GroupedArgs a;
    a.trellis = pl.trellis;
    a.suh = pl.suh;
    a.svh = pl.svh;
    a.input = input;
    a.output = output;
    a.out_features = pl.n;
    a.k = pl.k;
    a.input_row_stride = input_stride;
    a.output_row_stride = pl.n;
    a.output_dtype = 0;
    a.ka = bits;
    a.mul24 = 1;
    a.int8_act = int8_act;
    a.trellis_batch_stride = pl.trellis_bytes;
    a.suh_batch_stride = static_cast<uint64_t>(pl.k) * 2u;
    a.svh_batch_stride = static_cast<uint64_t>(pl.n) * 2u;
    a.group_expert = plan.group_expert;
    a.group_rows = plan.group_rows;
    a.group_count = plan.group_count;
    a.expert_group_begin = plan.expert_begin;
    a.row_token = token_rows ? plan.row_token : nullptr;
    a.group_capacity = std::max(plan.groups, 1u);
    for (int i = 0; i < 3; ++i) HIP_CHECK(kernel::launch_gemm_exl3_w16a16_grouped(a, nullptr));
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0, nullptr));
    for (int i = 0; i < iters; ++i) HIP_CHECK(kernel::launch_gemm_exl3_w16a16_grouped(a, nullptr));
    HIP_CHECK(hipEventRecord(t1, nullptr));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-flash-next-exl3-3.05bpw-mul24";
    uint32_t bits = 3;
    bool skew = false;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--bits") && i + 1 < argc) bits = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--skew")) skew = true;
    }
    nlohmann::json cfg;
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f.is_open()) { std::fprintf(stderr, "cannot read %s/config.json\n", model_dir.c_str()); return 1; }
        f >> cfg;
    }
    const nlohmann::json& t = cfg.contains("text_config") ? cfg["text_config"] : cfg;
    const uint32_t hidden = t["hidden_size"], inter = t["moe_intermediate_size"], experts = t["num_experts"],
                   topk = t["num_experts_per_tok"];
    std::printf("hidden=%u moe_intermediate=%u experts=%u topk=%u bits=%u routing=%s iters=%d fuse=%s\n", hidden, inter,
                experts, topk, bits, skew ? "skew" : "uniform", iters,
                std::getenv("SHELF_EXL3_GROUPED_FUSE") ? std::getenv("SHELF_EXL3_GROUPED_FUSE") : "default(2)");

    std::mt19937 rng(7);
    const Plane gate = make_plane(experts, hidden, inter, bits, rng);
    const Plane down = make_plane(experts, inter, hidden, bits, rng);

    const uint32_t max_tokens = 2048;
    void *d_tokens = nullptr, *d_gin = nullptr, *d_out = nullptr, *d_gout = nullptr;
    const size_t max_rows = static_cast<size_t>(max_tokens) * topk + static_cast<size_t>(experts) * kRowTile;
    HIP_CHECK(hipMalloc(&d_tokens, static_cast<size_t>(max_tokens) * hidden * 2));
    HIP_CHECK(hipMalloc(&d_gin, max_rows * inter * 2));
    HIP_CHECK(hipMalloc(&d_out, max_rows * std::max(inter, hidden) * 2));
    HIP_CHECK(hipMalloc(&d_gout, max_rows * 2));
    {
        std::vector<uint16_t> host(static_cast<size_t>(max_tokens) * hidden);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : host) v = shelf::f32_to_bf16_rne(nd(rng));
        HIP_CHECK(hipMemcpy(d_tokens, host.data(), host.size() * 2, hipMemcpyHostToDevice));
        std::vector<uint16_t> g(max_rows * inter);
        for (auto& v : g) v = shelf::f32_to_bf16_rne(nd(rng));
        HIP_CHECK(hipMemcpy(d_gin, g.data(), g.size() * 2, hipMemcpyHostToDevice));
    }

    std::printf("%6s %7s %7s %7s | %9s %9s %6s | %9s %9s %6s | %9s %9s %6s\n", "tokens", "experts", "groups", "rows",
                "gate fp16", "gate i8", "x", "down fp16", "down i8", "x", "sum fp16", "sum i8", "x");
    const uint32_t token_counts[] = {1, 2, 4, 8, 16, 64, 256, 512, 1024, 2048};
    for (const uint32_t tokens : token_counts) {
        Plan plan = make_plan(tokens, experts, topk, skew, rng);
        double gf = 0, g8 = 0, df = 0, d8 = 0;
        gf = time_launch(gate, plan, d_tokens, hidden, d_out, bits, true, 0, iters);
        g8 = time_launch(gate, plan, d_tokens, hidden, d_out, bits, true, 1, iters);
        df = time_launch(down, plan, d_gin, inter, d_out, bits, false, 0, iters);
        d8 = time_launch(down, plan, d_gin, inter, d_out, bits, false, 1, iters);
        std::printf("%6u %7u %7u %7u | %8.3fms %8.3fms %5.2fx | %8.3fms %8.3fms %5.2fx | %8.3fms %8.3fms %5.2fx\n",
                    tokens, plan.active_experts, plan.groups, plan.rows, gf, g8, gf / g8, df, d8, df / d8, gf + df,
                    g8 + d8, (gf + df) / (g8 + d8));
        free_plan(plan);
    }
    return 0;
}
