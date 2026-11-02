#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

namespace {

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t k = 2560;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
    int device = 0;
    uint32_t seed = 1234;
    bool check = false;
    std::string out;
};

void usage(FILE* f) {
    fprintf(f,
        "shelf-bench activation-quantize [options]\n"
        "  --variant correctness|optimized|e4m3-generic|e4m3-vec|e4m3-auto\n"
        "      correctness / optimized = a8 group-128 kernel\n"
        "      e4m3-generic / e4m3-vec / e4m3-auto = W4A8 group-128 e4m3 kernel\n"
        "                                          (production ACTIVATION_QUANTIZE_W4A8)\n"
        "  --rows 1,2,4,8,16,32,64,128       (default 1,2,4,8,16,32,64,128)\n"
        "  --k N                             (default 2560)\n"
        "  --warmup N                        (default 100)\n"
        "  --samples N                       (default 50)\n"
        "  --launches N                      (default 20)\n"
        "  --seed N                          (default 1234)\n"
        "  --device N                        (default 0)\n"
        "  --check                           verify codes+scales exact vs CPU reference\n"
        "  --out PATH                        (append CSV rows)\n");
}

bool parse_rows(const std::string& s, std::vector<uint32_t>& out) {
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (cur.empty()) return false;
            uint32_t v = (uint32_t)std::stoul(cur);
            if (v == 0) return false;
            out.push_back(v);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        uint32_t v = (uint32_t)std::stoul(cur);
        if (v == 0) return false;
        out.push_back(v);
    }
    return !out.empty();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

struct ActEnv {
    void* act_ws = nullptr;
    void* in_ptr = nullptr;
    void* values_dev = nullptr;
    void* ranges_dev = nullptr;
    void* ctx_dev = nullptr;
    float* scratch = nullptr;
    uint32_t* error_word = nullptr;
    ::ps::runtime::DeviceProgramView program{};
    ::ps::runtime::DispatchBinding binding{};
    std::vector<uint16_t> h_in;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t kp = 0;
    ::ps::runtime::Int8ActivationWorkspaceLayout layout{};
};

void teardown(ActEnv& e) {
    ps::gpu::discard_cleanup_result(hipFree(e.act_ws));
    ps::gpu::discard_cleanup_result(hipFree(e.in_ptr));
    ps::gpu::discard_cleanup_result(hipFree(e.values_dev));
    ps::gpu::discard_cleanup_result(hipFree(e.ranges_dev));
    ps::gpu::discard_cleanup_result(hipFree(e.ctx_dev));
    ps::gpu::discard_cleanup_result(hipFree(e.scratch));
    ps::gpu::discard_cleanup_result(hipFree(e.error_word));
}

bool setup(ActEnv& e, uint32_t rows, uint32_t k, uint32_t seed, hipStream_t stream) {
    e.rows = rows;
    e.k = k;
    e.kp = (k + 31u) & ~31u;
    e.layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    e.h_in.resize((size_t)rows * k);
    for (auto& v : e.h_in) {
        const float xv = xd(rng);
        uint32_t ui;
        std::memcpy(&ui, &xv, 4);
        v = (uint16_t)((ui + 0x8000u) >> 16);
    }
    if (hipMalloc(&e.in_ptr, (size_t)rows * k * 2) != hipSuccess) return false;
    if (hipMalloc(&e.act_ws, e.layout.total_bytes) != hipSuccess) { teardown(e); return false; }
    if (hipMemcpy(e.in_ptr, e.h_in.data(), e.h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess) {
        teardown(e); return false;
    }
    ::ps::runtime::DeviceValueBinding values[1];
    values[0] = {};
    values[0].logical_value = 0;
    values[0].storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = ::ps::runtime::ValueDType::BF16;
    values[0].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = k;
    values[0].row_stride = k;
    values[0].offset = reinterpret_cast<uint64_t>(e.in_ptr);
    values[0].bytes = (size_t)rows * k * 2;
    ::ps::runtime::DeviceWorkspaceRange range{};
    range.offset = 0;
    range.bytes = e.layout.total_bytes;
    if (hipMalloc(&e.values_dev, sizeof(values)) != hipSuccess) { teardown(e); return false; }
    if (hipMemcpy(e.values_dev, values, sizeof(values), hipMemcpyHostToDevice) != hipSuccess) {
        teardown(e); return false;
    }
    if (hipMalloc(&e.ranges_dev, sizeof(range)) != hipSuccess) { teardown(e); return false; }
    if (hipMemcpy(e.ranges_dev, &range, sizeof(range), hipMemcpyHostToDevice) != hipSuccess) {
        teardown(e); return false;
    }
    e.program.values = static_cast<const ::ps::runtime::DeviceValueBinding*>(e.values_dev);
    e.program.value_count = 1;
    e.program.workspace_ranges =
        static_cast<const ::ps::runtime::DeviceWorkspaceRange*>(e.ranges_dev);
    e.program.workspace_range_count = 1;
    e.program.workspace_base = reinterpret_cast<uint64_t>(e.act_ws);
    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_outputs = 0;
    if (hipMalloc(&e.ctx_dev, sizeof(ctx)) != hipSuccess) { teardown(e); return false; }
    if (hipMemcpy(e.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) {
        teardown(e); return false;
    }
    if (hipMalloc(&e.scratch, 2048 * sizeof(float)) != hipSuccess) { teardown(e); return false; }
    if (hipMalloc(&e.error_word, sizeof(uint32_t)) != hipSuccess) { teardown(e); return false; }
    if (hipMemset(e.error_word, 0, sizeof(uint32_t)) != hipSuccess) { teardown(e); return false; }
    e.binding = {};
    e.binding.kernel_id = ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8;
    e.binding.workspace_slot = 0;
    e.binding.group_size = 32;
    e.binding.activation_kp = e.kp;
    e.binding.activation_max_rows = rows;
    e.binding.activation_code_stride = e.layout.code_row_stride_bytes;
    e.binding.activation_scale_stride = e.layout.scale_row_stride_bytes;
    e.binding.input_slots[0] = 0;
    e.binding.input_count = 1;
    (void)stream;
    return true;
}

hipError_t launch_correctness(const ActEnv& e, hipStream_t stream) {
    if (hipMemsetAsync(e.error_word, 0, sizeof(uint32_t), stream) != hipSuccess)
        return hipGetLastError();
    auto st = ps::kernel::launch_model_dispatch_correctness(
        e.program, e.binding, static_cast<::ps::runtime::DeviceBatchContext*>(e.ctx_dev),
        ps::kernel::ModelDispatchStateView{}, e.scratch, e.error_word, stream);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const ActEnv& e, hipStream_t stream) {
    int8_t* codes = static_cast<int8_t*>(e.act_ws) + e.layout.codes_offset;
    float* scales = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.scales_offset);
    return ps::kernel::launch_activation_quantize_a8(
        static_cast<const ::ps::bf16_t*>(e.in_ptr),
        codes, scales,
        e.rows, e.k, e.kp,
        e.k, e.layout.code_row_stride_bytes, e.layout.scale_row_stride_bytes,
        stream);
}

hipError_t launch_e4m3_auto(const ActEnv& e, hipStream_t stream) {
    uint8_t* codes = reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.codes_offset;
    float* scales = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.scales_offset);
    return ps::kernel::launch_activation_quantize_e4m3(
        static_cast<const ::ps::bf16_t*>(e.in_ptr),
        codes, scales,
        e.rows, e.k, e.kp,
        e.k, e.layout.code_row_stride_bytes, e.layout.scale_row_stride_bytes,
        stream);
}

hipError_t launch_e4m3_generic(const ActEnv& e, hipStream_t stream) {
    uint8_t* codes = reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.codes_offset;
    float* scales = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.scales_offset);
    return ps::kernel::launch_activation_quantize_e4m3_generic(
        static_cast<const ::ps::bf16_t*>(e.in_ptr),
        codes, scales,
        e.rows, e.k, e.kp,
        e.k, e.layout.code_row_stride_bytes, e.layout.scale_row_stride_bytes,
        stream);
}

void cpu_reference(const std::vector<uint16_t>& h_in, uint32_t rows, uint32_t k,
                   uint32_t kp, uint32_t code_row_stride, uint32_t scale_row_stride,
                   std::vector<int8_t>& codes, std::vector<float>& scales) {
    const uint32_t groups = (kp + 127u) / 128u;
    const uint32_t scale_row_floats = scale_row_stride / 4u;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    codes.assign((size_t)rows_padded * code_row_stride, 0);
    scales.assign((size_t)rows * scale_row_floats, 0.f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            float amax = 0.f;
            float vals[128];
            for (uint32_t i = 0; i < 128u; ++i) {
                const uint32_t idx = i0 + i;
                const float v = idx < k ? bf16_bits_to_f32(h_in[(size_t)m * k + idx]) : 0.f;
                vals[i] = std::isfinite(v) ? v : 0.f;
                const float av = std::fabs(vals[i]);
                if (av > amax) amax = av;
            }
            const float scale = amax > 0.f ? amax / 127.0f : 0.f;
            scales[(size_t)m * scale_row_floats + g] = scale;
            const float inv = scale > 0.f ? 1.0f / scale : 0.f;
            for (uint32_t i = 0; i < 128u; ++i) {
                const uint32_t idx = i0 + i;
                if (idx >= kp) break;
                int c = 0;
                if (scale > 0.f) {
                    c = static_cast<int>(nearbyintf(vals[i] * inv));
                    if (c > 127) c = 127;
                    if (c < -127) c = -127;
                }
                codes[(size_t)(m >> 4) * (code_row_stride * 16u) +
                      (size_t)(idx >> 5u) * 512u + (size_t)((idx & 31u) >> 4u) * 256u +
                      (size_t)(m & 15u) * 16u + (size_t)(idx & 15u)] =
                    static_cast<int8_t>(c);
            }
        }
    }
}

}  // namespace

int run_activation_quantize(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_rows(next(), o.rows)) { fprintf(stderr, "bad --rows\n"); return 2; } }
        else if (a == "--k") o.k = (uint32_t)std::stoul(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--samples") o.samples = std::stoi(next());
        else if (a == "--launches") o.launches = std::stoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--out") o.out = next();
        else if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(stderr); return 2; }
    }
    const bool e4m3_variant = (o.variant == "e4m3-generic" || o.variant == "e4m3-vec" ||
                               o.variant == "e4m3-auto");
    if (o.variant != "correctness" && o.variant != "optimized" && !e4m3_variant) {
        fprintf(stderr, "bad --variant: %s\n", o.variant.c_str());
        return 2;
    }
    if (o.warmup < 0 || o.samples <= 0 || o.launches <= 0 || o.k == 0) {
        fprintf(stderr, "bad --warmup/--samples/--launches/--k\n");
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) { fprintf(stderr, "hipSetDevice failed\n"); return 1; }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "variant,rows,k,warmup,samples,launches_per_sample,min_us,p10_us,p50_us,p90_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        ActEnv e;
        if (!setup(e, rows, o.k, o.seed, stream)) {
            fprintf(stderr, "setup failed rows=%u\n", rows);
            return 1;
        }
        auto launch = (o.variant == "optimized")
            ? std::bind(launch_optimized, std::ref(e), std::placeholders::_1)
            : (o.variant == "e4m3-generic"
                   ? std::bind(launch_e4m3_generic, std::ref(e), std::placeholders::_1)
                   : (o.variant == "e4m3-vec"
                          ? std::bind(launch_e4m3_auto, std::ref(e), std::placeholders::_1)
                          : std::bind(launch_correctness, std::ref(e), std::placeholders::_1)));

        if (o.check && e4m3_variant) {
            const size_t total = e.layout.total_bytes;
            std::vector<uint8_t> got_vec(total);
            std::vector<uint8_t> got_gen(total);
            if (launch_e4m3_generic(e, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                fprintf(stderr, "check generic launch failed rows=%u\n", rows);
                rc = 1;
                teardown(e);
                continue;
            }
            if (hipMemcpy(got_gen.data(), e.act_ws, total, hipMemcpyDeviceToHost) != hipSuccess) {
                rc = 1; teardown(e); continue;
            }
            if (launch_e4m3_auto(e, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                fprintf(stderr, "check vec launch failed rows=%u\n", rows);
                rc = 1;
                teardown(e);
                continue;
            }
            if (hipMemcpy(got_vec.data(), e.act_ws, total, hipMemcpyDeviceToHost) != hipSuccess) {
                rc = 1; teardown(e); continue;
            }
            const bool same = (got_vec == got_gen);
            std::printf("CHECK e4m3 rows=%-4u k=%u vec_equals_generic=%d\n", rows, o.k, same ? 1 : 0);
            if (!same) rc = 1;
        }

        if (o.check && !e4m3_variant) {
            if (launch_optimized(e, stream) != hipSuccess) { fprintf(stderr, "check launch failed rows=%u\n", rows); rc = 1; teardown(e); continue; }
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; teardown(e); continue; }
            const uint32_t code_rs = e.layout.code_row_stride_bytes;
            const uint32_t scale_rs = e.layout.scale_row_stride_bytes;
            const uint32_t rows_padded = (rows + 15u) & ~15u;
            std::vector<int8_t> got_codes((size_t)rows_padded * code_rs, 0);
            std::vector<float> got_scales((size_t)rows * (scale_rs / 4u), 0.f);
            if (hipMemcpy(got_codes.data(), e.act_ws, (size_t)rows_padded * code_rs, hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "check read codes failed rows=%u\n", rows); rc = 1; teardown(e); continue;
            }
            const uint8_t* scales_ptr = static_cast<const uint8_t*>(e.act_ws) + e.layout.scales_offset;
            if (hipMemcpy(got_scales.data(), scales_ptr, got_scales.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "check read scales failed rows=%u\n", rows); rc = 1; teardown(e); continue;
            }
            std::vector<int8_t> ref_codes;
            std::vector<float> ref_scales;
            cpu_reference(e.h_in, rows, o.k, e.kp, code_rs, scale_rs, ref_codes, ref_scales);
            bool codes_ok = (got_codes == ref_codes);
            bool scales_ok = (got_scales == ref_scales);
            std::printf("CHECK %s rows=%-4u k=%u codes_exact=%d scales_exact=%d\n",
                        o.variant.c_str(), rows, o.k, codes_ok ? 1 : 0, scales_ok ? 1 : 0);
            if (!codes_ok || !scales_ok) rc = 1;
        }

        for (int i = 0; i < o.warmup; ++i)
            if (launch(stream) != hipSuccess) { fprintf(stderr, "launch failed rows=%u\n", rows); rc = 1; }
        if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; teardown(e); continue; }
        hipEvent_t ev0 = nullptr, ev1 = nullptr;
        if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; teardown(e); continue; }
        std::vector<double> per_launch_us;
        per_launch_us.reserve(o.samples);
        for (int s = 0; s < o.samples; ++s) {
            if (hipEventRecord(ev0, stream) != hipSuccess) break;
            for (int l = 0; l < o.launches; ++l)
                if (launch(stream) != hipSuccess) break;
            if (hipEventRecord(ev1, stream) != hipSuccess) break;
            if (hipEventSynchronize(ev1) != hipSuccess) break;
            float ms = 0.0f;
            if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
            per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
        }
        ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
        ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
        if (per_launch_us.empty()) { rc = 1; teardown(e); continue; }
        const double mn = percentile(per_launch_us, 0.0);
        const double p10 = percentile(per_launch_us, 0.1);
        const double p50 = percentile(per_launch_us, 0.5);
        const double p90 = percentile(per_launch_us, 0.9);
        const double p95 = percentile(per_launch_us, 0.95);
        double mean = 0.0;
        for (double v : per_launch_us) mean += v;
        mean /= (double)per_launch_us.size();
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                std::fprintf(f, "%s,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                             o.variant.c_str(), rows, o.k, o.warmup, o.samples, o.launches,
                             mn, p10, p50, p90, p95, mean);
                std::fclose(f);
            }
        }
        const char* path = "";
        if (o.variant == "e4m3-vec" || o.variant == "e4m3-auto") {
            uint8_t* codes = reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.codes_offset;
            path = ps::kernel::activation_quantize_e4m3_vec_supported(
                       static_cast<const ::ps::bf16_t*>(e.in_ptr), codes, e.k, e.kp, e.k)
                       ? " [vec]"
                       : " [generic-fallback]";
        }
        std::printf("%-13s rows=%-4u k=%-5u  p10=%.3f p50=%.3f p90=%.3f mean=%.3f us%s\n",
                    o.variant.c_str(), rows, o.k, p10, p50, p90, mean, path);
        teardown(e);
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
