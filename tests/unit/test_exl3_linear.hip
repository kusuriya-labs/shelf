#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/detail/exl3_decode_device.h>
#include <shelf/quantization/exl3/exl3.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace shelf::quantization::exl3;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t g_rng = 0x12345678u;

static uint32_t next_u32() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

static float next_float(float scale) {
    const int32_t v = static_cast<int32_t>(next_u32() & 0xFFFFu) - 32768;
    return scale * static_cast<float>(v) / 32768.0f;
}

static float bf16_to_f32(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16u;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

__global__ void debug_codebook_kernel(uint16_t* out) {
    const uint32_t state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < 65536u) {
        out[state] = shelf::kernel::exl3_detail::codebook_f16_bits(state);
    }
}

__global__ void debug_fwht_kernel(const float* in, float* out) {
    const uint32_t lane = threadIdx.x;
    float v0 = in[lane * 4u + 0u];
    float v1 = in[lane * 4u + 1u];
    float v2 = in[lane * 4u + 2u];
    float v3 = in[lane * 4u + 3u];
    shelf::kernel::exl3_detail::fwht128(v0, v1, v2, v3, lane);
    out[lane * 4u + 0u] = v0;
    out[lane * 4u + 1u] = v1;
    out[lane * 4u + 2u] = v2;
    out[lane * 4u + 3u] = v3;
}

__global__ void debug_ring_kernel(const uint32_t* words, uint32_t word_count,
                                  uint32_t* states, uint32_t ka, uint32_t half) {
    const uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position < 256u) {
        states[position] = shelf::kernel::exl3_detail::state_from_ring(
            words, word_count, position, ka, half);
    }
}

static void test_device_helpers() {
    std::vector<uint16_t> expected(65536);
    for (uint32_t s = 0; s < 65536u; ++s) {
        expected[s] = codebook_f16_bits(static_cast<uint16_t>(s));
    }
    void* d_out = nullptr;
    check(hipMalloc(&d_out, 65536u * 2u) == hipSuccess, "malloc codebook");
    debug_codebook_kernel<<<256, 256>>>(static_cast<uint16_t*>(d_out));
    check(hipDeviceSynchronize() == hipSuccess, "codebook sync");
    std::vector<uint16_t> got(65536);
    check(hipMemcpy(got.data(), d_out, 65536u * 2u, hipMemcpyDeviceToHost) == hipSuccess,
          "codebook copy");
    uint32_t mismatches = 0;
    for (uint32_t s = 0; s < 65536u; ++s) {
        if (got[s] != expected[s]) ++mismatches;
    }
    check(mismatches == 0, "device codebook");
    check(hipFree(d_out) == hipSuccess, "free codebook");

    {
        std::vector<float> in(128);
        for (float& v : in) v = next_float(1.0f);
        std::vector<float> expected = in;
        hadamard128_inplace(expected.data());
        void* d_in = nullptr;
        void* d_out = nullptr;
        check(hipMalloc(&d_in, 128u * 4u) == hipSuccess, "malloc fwht in");
        check(hipMalloc(&d_out, 128u * 4u) == hipSuccess, "malloc fwht out");
        check(hipMemcpy(d_in, in.data(), 128u * 4u, hipMemcpyHostToDevice) == hipSuccess,
              "copy fwht in");
        debug_fwht_kernel<<<1, 32>>>(static_cast<const float*>(d_in),
                                     static_cast<float*>(d_out));
        check(hipDeviceSynchronize() == hipSuccess, "fwht sync");
        std::vector<float> got(128);
        check(hipMemcpy(got.data(), d_out, 128u * 4u, hipMemcpyDeviceToHost) == hipSuccess,
              "copy fwht out");
        double max_abs = 0.0;
        constexpr float kInvSqrt128 = 0.08838834764831844055f;
        for (uint32_t i = 0; i < 128u; ++i) {
            max_abs = std::max(
                max_abs, std::fabs(static_cast<double>(got[i] * kInvSqrt128) - expected[i]));
        }
        if (max_abs > 1e-4) std::printf("  fwht max_abs=%.6g\n", max_abs);
        check(max_abs <= 1e-4, "device fwht");
        check(hipFree(d_in) == hipSuccess, "free fwht in");
        check(hipFree(d_out) == hipSuccess, "free fwht out");
    }

    for (uint32_t ka : {1u, 2u, 3u, 4u, 6u, 8u}) {
        for (uint32_t half = 0u; half < 2u; ++half) {
            if (half != 0u && ka > 3u) continue;
            const uint32_t words = 16u * ka + (half ? 8u : 0u);
            const uint32_t word_count = words / 2u;
            std::vector<uint32_t> ring(std::max(word_count, 64u));
            for (uint32_t& w : ring) w = next_u32();
            std::vector<uint32_t> expected_states(256);
            for (uint32_t p = 0; p < 256u; ++p) {
                expected_states[p] = state_from_ring(ring.data(), word_count, p, ka, half != 0u);
            }
            void* d_ring = nullptr;
            void* d_states = nullptr;
            check(hipMalloc(&d_ring, ring.size() * 4u) == hipSuccess, "malloc ring");
            check(hipMalloc(&d_states, 256u * 4u) == hipSuccess, "malloc states");
            check(hipMemcpy(d_ring, ring.data(), ring.size() * 4u, hipMemcpyHostToDevice) ==
                      hipSuccess,
                  "copy ring");
            debug_ring_kernel<<<1, 256>>>(static_cast<const uint32_t*>(d_ring), word_count,
                                          static_cast<uint32_t*>(d_states), ka, half);
            check(hipDeviceSynchronize() == hipSuccess, "ring sync");
            std::vector<uint32_t> got(256);
            check(hipMemcpy(got.data(), d_states, 256u * 4u, hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  "ring copy");
            uint32_t mismatches = 0;
            for (uint32_t p = 0; p < 256u; ++p) {
                if (got[p] != expected_states[p]) ++mismatches;
            }
            if (mismatches != 0) {
                std::printf("  ring ka=%u half=%u mismatches=%u\n", ka, half, mismatches);
            }
            check(mismatches == 0, "device ring state");
            check(hipFree(d_ring) == hipSuccess, "free ring");
            check(hipFree(d_states) == hipSuccess, "free states");
        }
    }
}

struct Case {
    uint32_t rows;
    uint32_t n;
    uint32_t k;
    uint32_t ka;
    bool half;
    const char* tag;
    uint32_t splits = 0;
    bool pre = false;
};

static bool run_case(const Case& c) {
    const uint32_t words = words_per_tile(c.ka, c.half);
    const uint32_t kt = c.k / 16u;
    const uint32_t nt = c.n / 16u;
    std::vector<uint16_t> trellis(static_cast<std::size_t>(kt) * nt * words);
    for (uint16_t& v : trellis) v = static_cast<uint16_t>(next_u32() & 0xFFFFu);

    std::vector<uint16_t> suh(c.k);
    std::vector<uint16_t> svh(c.n);
    for (uint16_t& v : suh) v = f16_bits_from_f32(next_float(1.0f));
    for (uint16_t& v : svh) v = f16_bits_from_f32(next_float(1.0f));

    std::vector<shelf::bf16_t> input(static_cast<std::size_t>(c.rows) * c.k);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i].data = shelf::f32_to_bf16_rne(next_float(2.0f));
    }

    std::vector<float> weight(static_cast<std::size_t>(c.k) * c.n);
    if (!dequant(trellis.data(), suh.data(), svh.data(), c.k, c.n, c.ka, c.half,
                 weight.data(), true)) {
        check(false, std::string(c.tag) + " cpu dequant");
        return false;
    }

    std::vector<float> expected(static_cast<std::size_t>(c.rows) * c.n, 0.0f);
    for (uint32_t m = 0; m < c.rows; ++m) {
        for (uint32_t o = 0; o < c.n; ++o) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < c.k; ++kk) {
                acc += static_cast<double>(bf16_to_f32(input[m * c.k + kk].data)) *
                       static_cast<double>(weight[static_cast<std::size_t>(kk) * c.n + o]);
            }
            expected[static_cast<std::size_t>(m) * c.n + o] = static_cast<float>(acc);
        }
    }

    void* d_trellis = nullptr;
    void* d_suh = nullptr;
    void* d_svh = nullptr;
    void* d_input = nullptr;
    void* d_output = nullptr;
    const std::size_t trellis_bytes = trellis.size() * sizeof(uint16_t);
    check(hipMalloc(&d_trellis, trellis_bytes) == hipSuccess, "malloc trellis");
    check(hipMalloc(&d_suh, suh.size() * 2u) == hipSuccess, "malloc suh");
    check(hipMalloc(&d_svh, svh.size() * 2u) == hipSuccess, "malloc svh");
    check(hipMalloc(&d_input, input.size() * 2u) == hipSuccess, "malloc input");
    const std::size_t out_bytes = expected.size() * sizeof(float);
    check(hipMalloc(&d_output, out_bytes) == hipSuccess, "malloc output");
    constexpr std::size_t kScratchFloats = 3145728u;
    void* d_scratch = nullptr;
    check(hipMalloc(&d_scratch, kScratchFloats * sizeof(float)) == hipSuccess,
          "malloc splitk scratch");
    void* d_xrot = nullptr;
    if (c.pre) {
        check(hipMalloc(&d_xrot, static_cast<std::size_t>(c.rows) * c.k * 2u) == hipSuccess,
              "malloc xrot");
    }
    check(hipMemcpy(d_trellis, trellis.data(), trellis_bytes, hipMemcpyHostToDevice) ==
              hipSuccess,
          "copy trellis");
    check(hipMemcpy(d_suh, suh.data(), suh.size() * 2u, hipMemcpyHostToDevice) == hipSuccess,
          "copy suh");
    check(hipMemcpy(d_svh, svh.data(), svh.size() * 2u, hipMemcpyHostToDevice) == hipSuccess,
          "copy svh");
    check(hipMemcpy(d_input, input.data(), input.size() * 2u, hipMemcpyHostToDevice) ==
              hipSuccess,
          "copy input");

    shelf::kernel::Exl3GemmArgs args;
    args.trellis = d_trellis;
    args.suh = d_suh;
    args.svh = d_svh;
    args.input = d_input;
    args.output = d_output;
    args.rows = c.rows;
    args.out_features = c.n;
    args.k = c.k;
    args.input_row_stride = c.k;
    args.output_row_stride = c.n;
    args.output_dtype = 1u;
    args.ka = c.ka;
    args.half = c.half ? 1u : 0u;
    args.fast_decode = ((c.ka == 5u && !c.half) || c.ka == 3u) ? 1u : 0u;
    args.splitk_scratch = static_cast<float*>(d_scratch);
    args.splitk_floats = static_cast<uint32_t>(kScratchFloats);
    args.splitk_splits = c.splits;
    hipError_t herr = hipSuccess;
    if (c.pre) {
        herr = shelf::kernel::launch_rotate_exl3(args, d_xrot, nullptr);
        check(herr == hipSuccess, std::string(c.tag) + " rotate " + hipGetErrorString(herr));
        if (herr == hipSuccess) {
            herr = shelf::kernel::launch_gemm_exl3_w16a16_pre(args, d_xrot, nullptr);
        }
    } else {
        herr = shelf::kernel::launch_gemm_exl3_w16a16(args, nullptr);
    }
    check(herr == hipSuccess, std::string(c.tag) + " launch " + hipGetErrorString(herr));
    if (herr == hipSuccess) {
        check(hipDeviceSynchronize() == hipSuccess, std::string(c.tag) + " sync");
    }

    std::vector<float> got(expected.size(), 0.0f);
    check(hipMemcpy(got.data(), d_output, out_bytes, hipMemcpyDeviceToHost) == hipSuccess,
          std::string(c.tag) + " copy out");
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
        if (d > max_abs) max_abs = d;
        const double denom = 1e-6 + std::fabs(static_cast<double>(expected[i]));
        const double r = d / denom;
        if (r > max_rel) max_rel = r;
    }
    const bool ok = max_abs <= 2e-2 || max_rel <= 3e-3;
    if (!ok) {
        std::printf("  %s rows=%u n=%u k=%u max_abs=%.6g max_rel=%.6g\n", c.tag, c.rows, c.n,
                    c.k, max_abs, max_rel);
    }
    check(ok, std::string(c.tag) + " numeric");

    check(hipFree(d_trellis) == hipSuccess, "free trellis");
    check(hipFree(d_suh) == hipSuccess, "free suh");
    check(hipFree(d_svh) == hipSuccess, "free svh");
    check(hipFree(d_input) == hipSuccess, "free input");
    check(hipFree(d_output) == hipSuccess, "free output");
    check(hipFree(d_scratch) == hipSuccess, "free splitk scratch");
    if (c.pre) check(hipFree(d_xrot) == hipSuccess, "free xrot");
    return ok;
}

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL hipSetDevice\n");
        return 1;
    }

    test_device_helpers();

    run_case({1, 128, 128, 4, false, "mul1_4_rows1"});
    run_case({3, 128, 128, 4, false, "mul1_4_rows3"});
    run_case({16, 256, 256, 3, true, "mul1_3half_rows16"});
    run_case({17, 384, 128, 7, false, "mul1_7_rows17"});
    run_case({32, 256, 256, 4, false, "mul1_4_rows32"});
    run_case({32, 256, 256, 4, false, "nosplit_rows32", 1});
    run_case({32, 512, 512, 4, false, "nosplit_rows32_k512", 1});
    run_case({32, 512, 512, 4, false, "split2_rows32_k512", 2});
    run_case({64, 256, 256, 4, false, "mul1_4_rows64"});
    run_case({64, 8192, 128, 4, false, "groups2_rows64", 1});
    run_case({32, 1024, 512, 4, false, "split2_rows32", 2});
    run_case({64, 1024, 512, 4, false, "auto_rows64_n1024", 0});
    run_case({5, 256, 256, 6, false, "mul1_6_rows5"});
    run_case({1, 128, 128, 1, false, "mul1_1_rows1"});
    run_case({1, 128, 128, 5, false, "mul1_5_rows1"});
    run_case({16, 256, 256, 5, false, "mul1_5_rows16"});
    run_case({64, 1024, 512, 5, false, "auto_rows64_n1024_ka5", 0});
    run_case({2, 128, 128, 2, false, "mul1_2_rows2"});
    run_case({3, 256, 128, 5, false, "mul1_5_rows3"});
    run_case({1, 128, 128, 8, false, "mul1_8_rows1"});
    run_case({2, 128, 128, 1, true, "mul1_1half_rows2"});
    run_case({2, 256, 256, 2, true, "mul1_2half_rows2"});
    run_case({1, 8192, 128, 4, false, "nosplit_n8192", 1});
    run_case({5, 5120, 512, 4, false, "nosplit_rows5_n5120", 1});
    run_case({1, 1024, 512, 4, false, "split8_n1024", 8});
    run_case({1, 1024, 512, 4, false, "split16_n1024", 16});
    run_case({1, 1024, 512, 4, false, "split32_n1024", 32});
    run_case({3, 1024, 512, 4, false, "split4_rows3", 4});
    run_case({16, 1024, 512, 4, false, "split2_rows16", 2});
    run_case({1, 5120, 128, 4, false, "split4_n5120", 4});
    run_case({1, 17408, 128, 4, false, "split2_n17408", 2});
    run_case({5, 5120, 512, 4, false, "split2_rows5_n5120", 2});
    run_case({8, 5120, 512, 4, false, "auto_rows8_n5120", 0});
    run_case({1, 17408, 512, 4, false, "auto_n17408", 0});
    run_case({5, 5120, 512, 4, false, "auto_rows5_n5120", 0});
    run_case({4, 128, 128, 4, false, "pre_rows4", 1, true});
    run_case({16, 256, 256, 4, false, "pre_rows16", 1, true});
    run_case({64, 8192, 128, 4, false, "pre_groups2_rows64", 1, true});
    run_case({5, 5120, 512, 4, false, "pre_auto_rows5", 0, true});
    run_case({32, 1024, 512, 4, false, "pre_split2_rows32", 2, true});
    run_case({5, 2048, 512, 4, false, "nt2_rows5", 2});
    run_case({256, 2048, 512, 4, false, "auto_nt2_g4_rows256", 0});
    run_case({320, 2048, 512, 4, false, "pre_nt2_g4_rows320", 1, true});
    run_case({256, 2048, 512, 4, false, "pre_nt2_g4_rows256", 1, true});

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
