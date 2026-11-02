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
    std::string variant = "e4m3-auto";
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
        "  --variant correctness|e4m3-generic|e4m3-vec|e4m3-auto\n"
        "      correctness = model dispatch の reference 実装\n"
        "      e4m3-generic / e4m3-vec / e4m3-auto = group-128 e4m3 kernel\n"
        "                                          (production ACTIVATION_QUANTIZE_FP8_FINEGRAINED)\n"
        "  --rows 1,2,4,8,16,32,64,128       (default 1,2,4,8,16,32,64,128)\n"
        "  --k N                             (default 2560)\n"
        "  --warmup N                        (default 100)\n"
        "  --samples N                       (default 50)\n"
        "  --launches N                      (default 20)\n"
        "  --seed N                          (default 1234)\n"
        "  --device N                        (default 0)\n"
        "  --check                           verify codes+scales exact vs CPU reference\n"
        "                                    and model-dispatch correctness path\n"
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
    e.binding.kernel_id = ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
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

hipError_t launch_fp8_finegrained_auto(const ActEnv& e, hipStream_t stream) {
    uint8_t* codes = reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.codes_offset;
    float* scales = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.scales_offset);
    return ps::kernel::launch_activation_quantize_fp8_finegrained(
        static_cast<const ::ps::bf16_t*>(e.in_ptr),
        codes, scales,
        e.rows, e.k, e.kp,
        e.k, e.layout.code_row_stride_bytes, e.layout.scale_row_stride_bytes,
        stream);
}

hipError_t launch_fp8_finegrained_generic(const ActEnv& e, hipStream_t stream) {
    uint8_t* codes = reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.codes_offset;
    float* scales = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(e.act_ws) + e.layout.scales_offset);
    return ps::kernel::launch_activation_quantize_fp8_finegrained_generic(
        static_cast<const ::ps::bf16_t*>(e.in_ptr),
        codes, scales,
        e.rows, e.k, e.kp,
        e.k, e.layout.code_row_stride_bytes, e.layout.scale_row_stride_bytes,
        stream);
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
    const bool fp8_finegrained_variant = (o.variant == "e4m3-generic" || o.variant == "e4m3-vec" ||
                               o.variant == "e4m3-auto");
    if (o.variant != "correctness" && !fp8_finegrained_variant) {
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
        auto launch = (o.variant == "e4m3-generic"
                           ? std::bind(launch_fp8_finegrained_generic, std::ref(e), std::placeholders::_1)
                           : (o.variant == "e4m3-vec"
                                  ? std::bind(launch_fp8_finegrained_auto, std::ref(e), std::placeholders::_1)
                                  : (o.variant == "e4m3-auto"
                                         ? std::bind(launch_fp8_finegrained_auto, std::ref(e), std::placeholders::_1)
                                         : std::bind(launch_correctness, std::ref(e), std::placeholders::_1))));

        if (o.check) {
            const size_t total = e.layout.total_bytes;
            std::vector<uint8_t> got_vec(total);
            std::vector<uint8_t> got_gen(total);
            std::vector<uint8_t> got_cor(total);
            if (launch_fp8_finegrained_generic(e, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                fprintf(stderr, "check generic launch failed rows=%u\n", rows);
                rc = 1;
                teardown(e);
                continue;
            }
            if (hipMemcpy(got_gen.data(), e.act_ws, total, hipMemcpyDeviceToHost) != hipSuccess) {
                rc = 1; teardown(e); continue;
            }
            if (launch_fp8_finegrained_auto(e, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                fprintf(stderr, "check vec launch failed rows=%u\n", rows);
                rc = 1;
                teardown(e);
                continue;
            }
            if (hipMemcpy(got_vec.data(), e.act_ws, total, hipMemcpyDeviceToHost) != hipSuccess) {
                rc = 1; teardown(e); continue;
            }
            if (launch_correctness(e, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                fprintf(stderr, "check correctness launch failed rows=%u\n", rows);
                rc = 1;
                teardown(e);
                continue;
            }
            if (hipMemcpy(got_cor.data(), e.act_ws, total, hipMemcpyDeviceToHost) != hipSuccess) {
                rc = 1; teardown(e); continue;
            }
            const bool same = (got_vec == got_gen);
            const bool cor_same = (got_cor == got_gen);
            std::printf("CHECK e4m3 rows=%-4u k=%u vec_equals_generic=%d "
                        "correctness_equals_generic=%d\n",
                        rows, o.k, same ? 1 : 0, cor_same ? 1 : 0);
            if (!cor_same) {
                std::size_t first = total;
                for (std::size_t i = 0; i < total; ++i) {
                    if (got_cor[i] != got_gen[i]) { first = i; break; }
                }
                std::printf("  first_diff=%zu codes_offset=%llu scales_offset=%llu "
                            "codes_bytes=%llu\n",
                            first,
                            static_cast<unsigned long long>(e.layout.codes_offset),
                            static_cast<unsigned long long>(e.layout.scales_offset),
                            static_cast<unsigned long long>(e.layout.codes_bytes()));
                {
                    const float* c = reinterpret_cast<const float*>(got_cor.data() + e.layout.scales_offset);
                    const float* g = reinterpret_cast<const float*>(got_gen.data() + e.layout.scales_offset);
                    std::printf("  scale0 cor=%g gen=%g\n",
                                static_cast<double>(c[0]), static_cast<double>(g[0]));
                }
                if (first < e.layout.scales_offset) {
                    std::printf("  code cor=%u gen=%u\n",
                                static_cast<unsigned>(got_cor[first]),
                                static_cast<unsigned>(got_gen[first]));
                } else {
                    const float* c = reinterpret_cast<const float*>(got_cor.data() + e.layout.scales_offset);
                    const float* g = reinterpret_cast<const float*>(got_gen.data() + e.layout.scales_offset);
                    const std::size_t idx = (first - e.layout.scales_offset) / sizeof(float);
                    std::printf("  scale idx=%zu cor=%g gen=%g\n", idx,
                                static_cast<double>(c[idx]),
                                static_cast<double>(g[idx]));
                }
            }
            if (!same || !cor_same) rc = 1;
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
            path = ps::kernel::activation_quantize_fp8_finegrained_vec_supported(
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
