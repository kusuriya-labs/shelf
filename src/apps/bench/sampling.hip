#include <shelf/kernels/optimized/sampling.h>
#include <shelf/kernels/optimized/stochastic_sampling.h>
#include <shelf/kernels/optimized/radix_topn.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;
using DT = ::shelf::runtime::ValueDType;

struct Options {
    std::string variant = "partitioned";
    uint32_t vocab = 248320;
    uint32_t outputs = 1;
    uint32_t partitions = 8;
    std::string flags = "all";
    std::string logit_pattern = "random";
    float temperature = 1.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0;
    uint32_t sample_index = 0;
    int warmup = 200;
    int samples = 100;
    int launches = 50;
    uint32_t seed = 1;
    int device = 0;
    std::string out;
    bool actual = false;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench sampling [options]\n"
                 "  --variant correctness|single|partitioned|stochastic|"
                 "stochastic-rejection|stochastic-radix\n"
                 "  --vocab N\n"
                 "  --outputs N\n"
                 "  --partitions N\n"
                 "  --flags all|none|mixed\n"
                 "  --logit-pattern random|unique|tie\n"
                 "  --temperature F\n"
                 "  --top-p F\n"
                 "  --top-k N\n"
                 "  --sample-index N\n"
                 "  --actual-qwen-profile\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --seed N\n"
                 "  --device N\n"
                 "  --out PATH\n");
}

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min((size_t)std::ceil(idx), (size_t)v.size() - 1);
    const double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Env {
    uint32_t vocab = 0;
    uint32_t outputs = 0;
    void* logits = nullptr;
    void* sampling = nullptr;
    void* out = nullptr;
    void* bc = nullptr;
    void* vals = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* partials = nullptr;
    void* attempts = nullptr;
    void* top_ids = nullptr;
    void* top_vals = nullptr;
    void* topk_scratch = nullptr;
    void* active = nullptr;
    P prog{};
    B binding{};
    ~Env() {
        for (auto* q : {logits, sampling, out, bc, vals, scratch, err, partials, attempts,
                        top_ids, top_vals, topk_scratch, active})
            if (q) shelf::gpu::discard_cleanup_result(hipFree(q));
    }
};

bool fill_logits(std::vector<float>& l, const std::string& pat, uint32_t vocab,
                 uint32_t outputs, uint32_t seed) {
    for (size_t i = 0; i < l.size(); ++i)
        l[i] = static_cast<float>(static_cast<int32_t>(h32((uint64_t)seed * 2654435761u + i) % 4001) -
                                  2000) /
               100.0f;
    if (pat == "unique") {
        for (uint32_t m = 0; m < outputs; ++m)
            l[(size_t)m * vocab + (h32((uint64_t)seed + m) % vocab)] = 1e6f;
    } else if (pat == "tie") {
        for (uint32_t m = 0; m < outputs; ++m) {
            l[(size_t)m * vocab + 10] = 1e6f;
            l[(size_t)m * vocab + 1000] = 1e6f;
        }
    }
    return true;
}

bool build_env(Env& e, const Options& o) {
    e.vocab = o.vocab;
    e.outputs = o.outputs;
    const size_t n = (size_t)o.outputs * o.vocab;
    if (hipMalloc(&e.logits, n * 4) != hipSuccess) return false;
    if (hipMalloc(&e.sampling, sizeof(::shelf::runtime::DeviceSamplingParams) * o.outputs) !=
        hipSuccess)
        return false;
    if (hipMalloc(&e.attempts, sizeof(uint32_t) * o.outputs) != hipSuccess) return false;
    if (hipMalloc(&e.out, (size_t)o.outputs * 4) != hipSuccess) return false;
    if (hipMalloc(&e.bc, sizeof(::shelf::runtime::DeviceBatchContext)) != hipSuccess)
        return false;
    if (hipMalloc(&e.vals, sizeof(V) * 2) != hipSuccess) return false;
    if (hipMalloc(&e.scratch, (size_t)2048 * 4) != hipSuccess) return false;
    if (hipMalloc(&e.err, sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMalloc(&e.partials, (size_t)o.outputs * o.partitions * 8) != hipSuccess)
        return false;
    const bool radix_variant = o.variant == "stochastic-radix";
    if (radix_variant) {
        const uint32_t top_k =
            o.top_k > 0u && o.top_k <= ::shelf::kernel::kSamplingMaxTopK
                ? o.top_k
                : ::shelf::kernel::kSamplingMaxTopK;
        if (hipMalloc(&e.top_ids, (size_t)o.outputs * top_k * 4) != hipSuccess) return false;
        if (hipMalloc(&e.top_vals, (size_t)o.outputs * top_k * 4) != hipSuccess)
            return false;
        const uint32_t partitions = ::shelf::kernel::dflash2_radix_default_partitions(o.vocab);
        const size_t radix_bytes =
            ::shelf::kernel::dflash2_radix_topn_scratch_bytes(o.outputs, partitions, o.vocab);
        if (hipMalloc(&e.topk_scratch, radix_bytes) != hipSuccess) return false;
        if (hipMalloc(&e.active, (size_t)o.outputs * 4) != hipSuccess) return false;
    }

    std::vector<float> l(n);
    fill_logits(l, o.logit_pattern, o.vocab, o.outputs, o.seed);
    if (hipMemcpy(e.logits, l.data(), n * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;

    std::vector<::shelf::runtime::DeviceSamplingParams> sp(o.outputs);
    const bool stochastic_variant =
        o.variant == "stochastic" || o.variant == "stochastic-rejection" ||
        o.variant == "stochastic-radix" || (o.variant == "correctness" && o.temperature > 0.0f);
    const uint8_t sample_mode = stochastic_variant ? 2u : 1u;
    for (size_t m = 0; m < sp.size(); ++m) {
        uint8_t mode = sample_mode;
        if (o.flags == "none") mode = 0;
        else if (o.flags == "mixed") mode = (m % 2) ? 0 : sample_mode;
        sp[m] = ::shelf::runtime::DeviceSamplingParams{};
        sp[m].seed = o.seed;
        sp[m].sample_index = o.sample_index + m;
        sp[m].temperature = o.temperature;
        sp[m].top_p = o.top_p;
        sp[m].top_k = o.top_k;
        sp[m].mode = mode;
    }
    if (hipMemcpy(e.sampling, sp.data(),
                  sizeof(::shelf::runtime::DeviceSamplingParams) * o.outputs,
                  hipMemcpyHostToDevice) != hipSuccess)
        return false;
    shelf::gpu::discard_cleanup_result(hipMemset(e.attempts, 0, sizeof(uint32_t) * o.outputs));

    ::shelf::runtime::DeviceBatchContext hbc{};
    hbc.actual_rows = o.outputs;
    hbc.num_requests = o.outputs;
    hbc.num_outputs = o.outputs;
    hbc.output_sampling_params = (::shelf::runtime::DeviceSamplingParams*)e.sampling;
    if (hipMemcpy(e.bc, &hbc, sizeof(hbc), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    auto* vals = (V*)e.vals;
    vals[0] = V{};
    vals[0].logical_value = 1;
    vals[0].storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    vals[0].dtype = DT::F32;
    vals[0].row_domain = ::shelf::runtime::ValueRowDomain::OUTPUT_ROWS;
    vals[0].feature_count = o.vocab;
    vals[0].row_stride = o.vocab;
    vals[0].offset = (uint64_t)(uintptr_t)e.logits;
    vals[1] = V{};
    vals[1].logical_value = 2;
    vals[1].storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    vals[1].dtype = DT::I32;
    vals[1].row_domain = ::shelf::runtime::ValueRowDomain::OUTPUT_ROWS;
    vals[1].feature_count = 1;
    vals[1].row_stride = 4;
    vals[1].offset = (uint64_t)(uintptr_t)e.out;

    e.prog = {};
    e.prog.values = vals;
    e.prog.value_count = 2;
    e.prog.workspace_base = 0;
    e.prog.weight_table = nullptr;
    e.prog.weight_slot_count = 0;
    e.binding = {};
    e.binding.kernel_id = ::shelf::runtime::KernelId::SAMPLING;
    e.binding.input_slots[0] = 1;
    e.binding.output_slots[0] = 2;
    e.binding.input_count = 1;
    e.binding.output_count = 1;
    e.binding.rotary_dim = o.vocab;
    return true;
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(e.err, 0, sizeof(uint32_t), s));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto st = shelf::kernel::launch_model_dispatch_correctness(
        e.prog, e.binding, (::shelf::runtime::DeviceBatchContext*)e.bc, sv,
        (float*)e.scratch, (uint32_t*)e.err, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_single(const Env& e, hipStream_t s) {
    ::shelf::kernel::ArgmaxF32Args a;
    a.logits = (const float*)e.logits;
    a.sampling = (const ::shelf::runtime::DeviceSamplingParams*)e.sampling;
    a.sampled_tokens = (int32_t*)e.out;
    a.outputs = e.outputs;
    a.vocab_size = e.vocab;
    a.logits_row_stride = e.vocab;
    return ::shelf::kernel::launch_argmax_f32_single(a, s);
}

hipError_t launch_stochastic(const Env& e, hipStream_t s) {
    ::shelf::kernel::StochasticSamplingF32Args a;
    a.logits = (const float*)e.logits;
    a.sampling = (const ::shelf::runtime::DeviceSamplingParams*)e.sampling;
    a.sampled_tokens = (int32_t*)e.out;
    a.outputs = e.outputs;
    a.vocab_size = e.vocab;
    a.logits_row_stride = e.vocab;
    a.error_word = (uint32_t*)e.err;
    a.attempts_out = (uint32_t*)e.attempts;
    return ::shelf::kernel::launch_stochastic_sampling_f32_singleblock(a, s);
}

hipError_t launch_partitioned(const Env& e, uint32_t partitions, hipStream_t s) {
    ::shelf::kernel::ArgmaxF32Args a;
    a.logits = (const float*)e.logits;
    a.sampling = (const ::shelf::runtime::DeviceSamplingParams*)e.sampling;
    a.sampled_tokens = (int32_t*)e.out;
    a.outputs = e.outputs;
    a.vocab_size = e.vocab;
    a.logits_row_stride = e.vocab;
    return ::shelf::kernel::launch_argmax_f32_partitioned(
        a, (::shelf::kernel::ArgMaxPair*)e.partials, e.outputs * partitions, partitions, s);
}

hipError_t launch_stochastic_radix(const Env& e, const Options& o, hipStream_t s) {
    if (e.top_ids == nullptr || e.top_vals == nullptr || e.topk_scratch == nullptr ||
        e.active == nullptr)
        return hipErrorInvalidValue;
    const uint32_t top_k =
        o.top_k > 0u && o.top_k <= ::shelf::kernel::kSamplingMaxTopK
            ? o.top_k
            : ::shelf::kernel::kSamplingMaxTopK;
    const uint32_t partitions = ::shelf::kernel::dflash2_radix_default_partitions(e.vocab);
    const size_t radix_bytes =
        ::shelf::kernel::dflash2_radix_topn_scratch_bytes(e.outputs, partitions, e.vocab);
    hipError_t err = ::shelf::kernel::launch_dflash2_radix_topn(
        (const float*)e.logits, e.outputs, e.vocab, e.vocab, top_k, (int32_t*)e.top_ids,
        (float*)e.top_vals, partitions, e.topk_scratch, radix_bytes, s);
    if (err != hipSuccess) return err;
    ::shelf::kernel::StochasticTopKSamplingArgs a;
    a.logits = (const float*)e.logits;
    a.sampling = (const ::shelf::runtime::DeviceSamplingParams*)e.sampling;
    a.sampled_tokens = (int32_t*)e.out;
    a.top_ids = (const int32_t*)e.top_ids;
    a.top_logits = (const float*)e.top_vals;
    a.outputs = e.outputs;
    a.vocab_size = e.vocab;
    a.logits_row_stride = e.vocab;
    a.top_k = top_k;
    a.error_word = (uint32_t*)e.err;
    a.active_count_out = (uint32_t*)e.active;
    return ::shelf::kernel::launch_stochastic_topk_sampling_f32(a, s);
}

}  // namespace

int run_sampling(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--vocab") o.vocab = (uint32_t)std::stoul(next());
        else if (a == "--outputs") o.outputs = (uint32_t)std::stoul(next());
        else if (a == "--partitions") o.partitions = (uint32_t)std::stoul(next());
        else if (a == "--flags") o.flags = next();
        else if (a == "--logit-pattern") o.logit_pattern = next();
        else if (a == "--temperature") o.temperature = std::stof(next());
        else if (a == "--top-p") o.top_p = std::stof(next());
        else if (a == "--top-k") o.top_k = (uint32_t)std::stoul(next());
        else if (a == "--sample-index") o.sample_index = (uint32_t)std::stoul(next());
        else if (a == "--actual-qwen-profile") o.actual = true;
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--out") o.out = next();
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "single" && o.variant != "partitioned" &&
        o.variant != "stochastic" && o.variant != "stochastic-rejection" &&
        o.variant != "stochastic-radix") {
        usage(stderr);
        return 2;
    }
    if (o.outputs == 0) return 2;
    if (o.actual) { o.vocab = 248320; o.outputs = 1; o.flags = "all"; }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    Env e;
    if (!build_env(e, o)) { std::printf("build fail\n"); return 1; }
    if (hipDeviceSynchronize() != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f,
                             "variant,vocab,outputs,partitions,flags,logit_pattern,temperature,"
                             "top_p,top_k,warmup,samples,launches_per_sample,min_us,p50_us,"
                             "p95_us,mean_us,avg_attempts,avg_active\n");
            std::fclose(f);
        }
    }

    auto launch = [&](hipStream_t s) -> hipError_t {
        if (o.variant == "correctness") return launch_correctness(e, s);
        if (o.variant == "single") return launch_single(e, s);
        if (o.variant == "stochastic" || o.variant == "stochastic-rejection")
            return launch_stochastic(e, s);
        if (o.variant == "stochastic-radix") return launch_stochastic_radix(e, o, s);
        return launch_partitioned(e, o.partitions, s);
    };
    for (int i = 0; i < o.warmup; ++i) {
        if (launch(stream) != hipSuccess) return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return 1;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess)
        return 1;
    std::vector<double> per;
    per.reserve(o.samples);
    for (int s = 0; s < o.samples; ++s) {
        if (hipEventRecord(ev0, stream) != hipSuccess) break;
        for (int l = 0; l < o.launches; ++l)
            if (launch(stream) != hipSuccess) break;
        if (hipEventRecord(ev1, stream) != hipSuccess) break;
        if (hipEventSynchronize(ev1) != hipSuccess) break;
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
        per.push_back(ms * 1000.0 / (double)o.launches);
    }
    shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
    shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
    if (per.empty()) return 1;
    const double mn = percentile(per, 0.0);
    const double p50 = percentile(per, 0.5);
    const double p95 = percentile(per, 0.95);
    double mean = 0.0;
    for (double x : per) mean += x;
    mean /= (double)per.size();
    double avg_attempts = 0.0;
    double avg_active = 0.0;
    const bool show_attempts =
        o.variant == "stochastic" || o.variant == "stochastic-rejection";
    const bool show_active = o.variant == "stochastic-radix";
    if (show_attempts) {
        std::vector<uint32_t> att(o.outputs);
        if (hipMemcpy(att.data(), e.attempts, sizeof(uint32_t) * o.outputs,
                      hipMemcpyDeviceToHost) == hipSuccess) {
            for (uint32_t x : att) avg_attempts += (double)x;
            avg_attempts /= (double)o.outputs;
        }
    }
    if (show_active && e.active != nullptr) {
        std::vector<uint32_t> act(o.outputs);
        if (hipMemcpy(act.data(), e.active, sizeof(uint32_t) * o.outputs,
                      hipMemcpyDeviceToHost) == hipSuccess) {
            for (uint32_t x : act) avg_active += (double)x;
            avg_active /= (double)o.outputs;
        }
    }
    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            std::fprintf(
                f, "%s,%u,%u,%u,%s,%s,%.4f,%.4f,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                o.variant.c_str(), o.vocab, o.outputs, o.partitions, o.flags.c_str(),
                o.logit_pattern.c_str(), o.temperature, o.top_p, o.top_k, o.warmup,
                o.samples, o.launches, mn, p50, p95, mean, avg_attempts, avg_active);
            std::fclose(f);
        }
    }
    if (show_active) {
        std::printf(
            "%-18s v=%-6u out=%-4u flags=%-6s pat=%-7s T=%.2f p=%.2f k=%-4u "
            "p50=%.2fus p95=%.2fus mean=%.2fus active=%.2f\n",
            o.variant.c_str(), o.vocab, o.outputs, o.flags.c_str(), o.logit_pattern.c_str(),
            o.temperature, o.top_p, o.top_k, p50, p95, mean, avg_active);
    } else {
        std::printf(
            "%-18s v=%-6u out=%-4u P=%-3u flags=%-6s pat=%-7s T=%.2f p=%.2f k=%-4u "
            "p50=%.2fus p95=%.2fus mean=%.2fus attempts=%.2f\n",
            o.variant.c_str(), o.vocab, o.outputs, o.partitions, o.flags.c_str(),
            o.logit_pattern.c_str(), o.temperature, o.top_p, o.top_k, p50, p95, mean,
            avg_attempts);
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return 0;
}
