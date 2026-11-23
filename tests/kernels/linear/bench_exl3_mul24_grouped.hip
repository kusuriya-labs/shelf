#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/grouped.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>

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
    std::vector<uint32_t> host_group_rows;
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
    p.host_group_rows = gr;
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
                   uint32_t bits, bool token_rows, int iters,
                   const void* pre_codes = nullptr, const float* pre_scales = nullptr) {
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
    a.trellis_batch_stride = pl.trellis_bytes;
    a.suh_batch_stride = static_cast<uint64_t>(pl.k) * 2u;
    a.svh_batch_stride = static_cast<uint64_t>(pl.n) * 2u;
    a.group_expert = plan.group_expert;
    a.group_rows = plan.group_rows;
    a.group_count = plan.group_count;
    a.expert_group_begin = plan.expert_begin;
    a.row_token = token_rows ? plan.row_token : nullptr;
    a.group_capacity = std::max(plan.groups, 1u);
    a.pre_rotated_codes = pre_codes;
    a.pre_rotated_scales = pre_scales;
    for (int i = 0; i < 3; ++i) HIP_CHECK(kernel::launch_gemm_exl3_int8_grouped(a, nullptr));
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0, nullptr));
    for (int i = 0; i < iters; ++i) HIP_CHECK(kernel::launch_gemm_exl3_int8_grouped(a, nullptr));
    HIP_CHECK(hipEventRecord(t1, nullptr));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return ms / iters;
}


double time_producer(const Plane& pl, const Plan& plan, const void* input, uint32_t input_stride,
                     void* codes, float* scales, bool token_rows, int iters) {
    const uint32_t rows_cap = plan.groups * kRowTile;
    auto run = [&]() {
        return kernel::launch_exl3_grouped_activate(
            input, pl.suh, codes, scales, plan.group_expert, plan.group_rows,
            plan.group_count, nullptr, token_rows ? plan.row_token : nullptr, rows_cap,
            pl.k, input_stride, static_cast<uint64_t>(pl.k) * 2u, nullptr);
    };
    for (int i = 0; i < 3; ++i) HIP_CHECK(run());
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0, nullptr));
    for (int i = 0; i < iters; ++i) HIP_CHECK(run());
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
    std::string model_dir;
    if (const char* env = std::getenv("SHELF_QWEN4EXP_MODEL_DIR")) {
        model_dir = env;
    }
    uint32_t bits = 3;
    bool skew = false;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--bits") && i + 1 < argc) bits = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--skew")) skew = true;
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "set SHELF_QWEN4EXP_MODEL_DIR to the EXL3 model directory\n");
        return 1;
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
    void* d_codes = nullptr;
    float* d_scales = nullptr;
    void* d_out2 = nullptr;
    HIP_CHECK(hipMalloc(&d_codes, max_rows * hidden));
    HIP_CHECK(hipMalloc(&d_scales, max_rows * (hidden / 128u) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out2, max_rows * std::max(inter, hidden) * 2));
    {
        std::vector<uint16_t> host(static_cast<size_t>(max_tokens) * hidden);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : host) v = shelf::f32_to_bf16_rne(nd(rng));
        HIP_CHECK(hipMemcpy(d_tokens, host.data(), host.size() * 2, hipMemcpyHostToDevice));
        std::vector<uint16_t> g(max_rows * inter);
        for (auto& v : g) v = shelf::f32_to_bf16_rne(nd(rng));
        HIP_CHECK(hipMemcpy(d_gin, g.data(), g.size() * 2, hipMemcpyHostToDevice));
    }

    std::printf("%6s %7s %7s %7s | %9s %9s %9s\n", "tokens", "experts", "groups", "rows",
                "gate i8", "down i8", "sum i8");
    const uint32_t token_counts[] = {1, 2, 4, 8, 16, 64, 256, 512, 1024, 2048};
    for (const uint32_t tokens : token_counts) {
        Plan plan = make_plan(tokens, experts, topk, skew, rng);
        const double g8 = time_launch(gate, plan, d_tokens, hidden, d_out, bits, true, iters);
        const double d8 = time_launch(down, plan, d_gin, inter, d_out, bits, false, iters);
        const double g_prod = time_producer(gate, plan, d_tokens, hidden, d_codes, d_scales,
                                            true, iters);
        const double g_pre = time_launch(gate, plan, d_tokens, hidden, d_out2, bits, true,
                                         iters, d_codes, d_scales);
        const double d_prod = time_producer(down, plan, d_gin, inter, d_codes, d_scales,
                                            false, iters);
        const double d_pre = time_launch(down, plan, d_gin, inter, d_out2, bits, false,
                                         iters, d_codes, d_scales);
        std::printf("  gate prod %6.3f + pre %6.3f = %6.3f (fused %6.3f, %+5.1f%%) | "
                    "down prod %6.3f + pre %6.3f = %6.3f (fused %6.3f, %+5.1f%%) | "
                    "grouped %6.3f -> %6.3f (%+5.1f%%)\n",
                    g_prod, g_pre, g_prod + g_pre, g8, 100.0 * (g_prod + g_pre - g8) / g8,
                    d_prod, d_pre, d_prod + d_pre, d8, 100.0 * (d_prod + d_pre - d8) / d8,
                    g8 + d8, g_prod + g_pre + d_prod + d_pre,
                    100.0 * (g_prod + g_pre + d_prod + d_pre - g8 - d8) / (g8 + d8));
        HIP_CHECK(kernel::launch_gemm_exl3_int8_grouped(
            [&] {
                kernel::Exl3GroupedArgs a;
                a.trellis = gate.trellis; a.suh = gate.suh; a.svh = gate.svh;
                a.input = d_tokens; a.output = d_out; a.out_features = gate.n; a.k = gate.k;
                a.input_row_stride = hidden; a.output_row_stride = gate.n; a.output_dtype = 0u;
                a.ka = bits; a.trellis_batch_stride = gate.trellis_bytes;
                a.suh_batch_stride = static_cast<uint64_t>(gate.k) * 2u;
                a.svh_batch_stride = static_cast<uint64_t>(gate.n) * 2u;
                a.group_expert = plan.group_expert; a.group_rows = plan.group_rows;
                a.group_count = plan.group_count; a.expert_group_begin = plan.expert_begin;
                a.row_token = plan.row_token; a.group_capacity = std::max(plan.groups, 1u);
                return a;
            }(),
            nullptr));
        HIP_CHECK(kernel::launch_exl3_grouped_activate(
            d_tokens, gate.suh, d_codes, d_scales, plan.group_expert, plan.group_rows,
            plan.group_count, nullptr, plan.row_token, plan.groups * kRowTile, gate.k,
            hidden, static_cast<uint64_t>(gate.k) * 2u, nullptr));
        HIP_CHECK(kernel::launch_gemm_exl3_int8_grouped(
            [&] {
                kernel::Exl3GroupedArgs a;
                a.trellis = gate.trellis; a.suh = gate.suh; a.svh = gate.svh;
                a.input = d_tokens; a.output = d_out2; a.out_features = gate.n; a.k = gate.k;
                a.input_row_stride = hidden; a.output_row_stride = gate.n; a.output_dtype = 0u;
                a.ka = bits; a.trellis_batch_stride = gate.trellis_bytes;
                a.suh_batch_stride = static_cast<uint64_t>(gate.k) * 2u;
                a.svh_batch_stride = static_cast<uint64_t>(gate.n) * 2u;
                a.group_expert = plan.group_expert; a.group_rows = plan.group_rows;
                a.group_count = plan.group_count; a.expert_group_begin = plan.expert_begin;
                a.row_token = plan.row_token; a.group_capacity = std::max(plan.groups, 1u);
                a.pre_rotated_codes = d_codes; a.pre_rotated_scales = d_scales;
                return a;
            }(),
            nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        const uint32_t rows_cap = plan.groups * kRowTile;
        const size_t out_bytes = static_cast<size_t>(rows_cap) * gate.n * 2u;
        std::vector<uint8_t> ref(out_bytes), alt(out_bytes);
        HIP_CHECK(hipMemcpy(ref.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(alt.data(), d_out2, out_bytes, hipMemcpyDeviceToHost));
        bool exact = true;
        uint32_t first_diff = 0u, diff_col = 0u;
        for (uint32_t g = 0; g < plan.groups && exact; ++g) {
            const size_t byte_off = static_cast<size_t>(g) * kRowTile * gate.n * 2u;
            const size_t byte_len = static_cast<size_t>(plan.host_group_rows[g]) * gate.n * 2u;
            if (std::memcmp(ref.data() + byte_off, alt.data() + byte_off, byte_len) != 0) {
                exact = false;
                first_diff = g;
                for (uint32_t b = 0; b < byte_len; b += 2u) {
                    if (ref[byte_off + b] != alt[byte_off + b] ||
                        ref[byte_off + b + 1u] != alt[byte_off + b + 1u]) {
                        diff_col = b / 2u;
                        break;
                    }
                }
            }
        }
        if (!exact) {
            const uint32_t row0 = first_diff * kRowTile;
            const size_t b = (static_cast<size_t>(row0) * gate.n + diff_col) * 2u;
            uint16_t rf = 0u, al = 0u;
            std::memcpy(&rf, ref.data() + b, 2);
            std::memcpy(&al, alt.data() + b, 2);
            std::printf("  first diff group=%u col=%u fused=%04x pre=%04x\n", first_diff,
                        diff_col, rf, al);
        }
        std::printf("%6u %7u %7u %7u | %8.3fms %8.3fms %8.3fms | prod %7.3f pre %7.3f sum %7.3f gain %+6.1f%% exact %s\n",
                    tokens, plan.active_experts, plan.groups, plan.rows, g8, d8, g8 + d8,
                    g_prod, g_pre, g_prod + g_pre, 100.0 * ((g_prod + g_pre) - g8) / g8,
                    exact ? "yes" : "NO");
        free_plan(plan);
    }
    return 0;
}
