#include <shelf/kernels/optimized/stochastic_sampling.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/device_program.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using ::ps::runtime::DeviceSamplingParams;

int g_fail = 0;

void check(bool cond, const char* msg) {
    if (!cond) ++g_fail;
    std::printf("  [%-42s] %s\n", msg, cond ? "PASS" : "FAIL");
}

DeviceSamplingParams mk(uint8_t mode, float temperature, float top_p, uint32_t top_k,
                        uint64_t seed, uint64_t index, float min_p = 0.0f) {
    DeviceSamplingParams p{};
    p.mode = mode;
    p.temperature = temperature;
    p.top_p = top_p;
    p.min_p = min_p;
    p.top_k = top_k;
    p.seed = seed;
    p.sample_index = index;
    return p;
}

struct BatchResult {
    std::vector<int32_t> tokens;
    std::vector<uint32_t> attempts;
    uint32_t error_word = 0;
};

bool run_batch(const std::vector<float>& logits, uint32_t vocab, uint32_t outputs,
               const std::vector<DeviceSamplingParams>& params, BatchResult& out,
               bool want_attempts) {
    if (params.size() != outputs) return false;
    if (logits.size() != static_cast<size_t>(outputs) * vocab) return false;

    float* d_logits = nullptr;
    DeviceSamplingParams* d_params = nullptr;
    int32_t* d_tokens = nullptr;
    uint32_t* d_err = nullptr;
    uint32_t* d_attempts = nullptr;
    bool ok = hipMalloc(&d_logits, sizeof(float) * logits.size()) == hipSuccess &&
              hipMalloc(&d_params, sizeof(DeviceSamplingParams) * outputs) == hipSuccess &&
              hipMalloc(&d_tokens, sizeof(int32_t) * outputs) == hipSuccess &&
              hipMalloc(&d_err, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&d_attempts, sizeof(uint32_t) * outputs) == hipSuccess;
    if (ok) {
        ok = hipMemcpy(d_logits, logits.data(), sizeof(float) * logits.size(),
                       hipMemcpyHostToDevice) == hipSuccess &&
             hipMemcpy(d_params, params.data(), sizeof(DeviceSamplingParams) * outputs,
                       hipMemcpyHostToDevice) == hipSuccess;
    }
    if (ok) {
        ps::gpu::discard_cleanup_result(hipMemset(d_err, 0, sizeof(uint32_t)));
        ::ps::kernel::StochasticSamplingF32Args args;
        args.logits = d_logits;
        args.sampling = d_params;
        args.sampled_tokens = d_tokens;
        args.outputs = outputs;
        args.vocab_size = vocab;
        args.logits_row_stride = vocab;
        args.error_word = d_err;
        args.attempts_out = want_attempts ? d_attempts : nullptr;
        ok = ::ps::kernel::launch_stochastic_sampling_f32_singleblock(args, nullptr) ==
                 hipSuccess &&
             hipDeviceSynchronize() == hipSuccess;
    }
    if (ok) {
        out.tokens.resize(outputs);
        out.attempts.assign(outputs, 0u);
        ok = hipMemcpy(out.tokens.data(), d_tokens, sizeof(int32_t) * outputs,
                       hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(&out.error_word, d_err, sizeof(uint32_t),
                       hipMemcpyDeviceToHost) == hipSuccess;
        if (ok && want_attempts) {
            ok = hipMemcpy(out.attempts.data(), d_attempts, sizeof(uint32_t) * outputs,
                           hipMemcpyDeviceToHost) == hipSuccess;
        }
    }
    if (d_logits) ps::gpu::discard_cleanup_result(hipFree(d_logits));
    if (d_params) ps::gpu::discard_cleanup_result(hipFree(d_params));
    if (d_tokens) ps::gpu::discard_cleanup_result(hipFree(d_tokens));
    if (d_err) ps::gpu::discard_cleanup_result(hipFree(d_err));
    if (d_attempts) ps::gpu::discard_cleanup_result(hipFree(d_attempts));
    return ok;
}

bool sample_many(uint32_t vocab, const std::vector<float>& logits, float temperature,
                 float top_p, uint32_t top_k, uint64_t seed, uint32_t count,
                 std::vector<int32_t>& tokens, std::vector<uint32_t>* attempts = nullptr,
                 float min_p = 0.0f) {
    std::vector<DeviceSamplingParams> params(count);
    for (uint32_t i = 0; i < count; ++i) {
        params[i] = mk(2u, temperature, top_p, top_k, seed, i, min_p);
    }
    std::vector<float> rows;
    rows.reserve(static_cast<size_t>(count) * vocab);
    for (uint32_t i = 0; i < count; ++i) {
        rows.insert(rows.end(), logits.begin(), logits.end());
    }
    BatchResult r;
    if (!run_batch(rows, vocab, count, params, r, attempts != nullptr)) return false;
    if (r.error_word != 0u) {
        std::printf("    error_word=%u\n", r.error_word);
        return false;
    }
    tokens = r.tokens;
    if (attempts != nullptr) *attempts = r.attempts;
    return true;
}

int32_t sample_one(uint32_t vocab, const std::vector<float>& logits, float temperature,
                   float top_p, uint32_t top_k, uint64_t seed, uint64_t index) {
    BatchResult r;
    if (!run_batch(logits, vocab, 1u, {mk(2u, temperature, top_p, top_k, seed, index)}, r,
                   false))
        return -2;
    if (r.error_word != 0u) return -3;
    return r.tokens[0];
}

int32_t greedy_one(uint32_t vocab, const std::vector<float>& logits) {
    BatchResult r;
    if (!run_batch(logits, vocab, 1u, {mk(1u, 0.0f, 1.0f, 0u, 0u, 0u)}, r, false))
        return -2;
    if (r.error_word != 0u) return -3;
    return r.tokens[0];
}

std::vector<float> prob_logits(const std::vector<float>& p) {
    std::vector<float> l(p.size());
    for (size_t i = 0; i < p.size(); ++i) l[i] = std::log(p[i]);
    return l;
}


// Runs the same batch through the correctness dispatch so the shared sampler can
// be compared against the optimized kernel for identical inputs.
bool run_correctness(const std::vector<float>& logits, uint32_t vocab, uint32_t outputs,
                     const std::vector<DeviceSamplingParams>& params,
                     std::vector<int32_t>& tokens, uint32_t& error_word) {
    using V = ::ps::runtime::DeviceValueBinding;
    using P = ::ps::runtime::DeviceProgramView;
    using B = ::ps::runtime::DispatchBinding;
    using DT = ::ps::runtime::ValueDType;

    float* d_logits = nullptr;
    DeviceSamplingParams* d_params = nullptr;
    int32_t* d_tokens = nullptr;
    uint32_t* d_err = nullptr;
    float* d_scratch = nullptr;
    V* d_vals = nullptr;
    ::ps::runtime::DeviceBatchContext* d_ctx = nullptr;

    constexpr uint32_t kBlocks = 64;
    bool ok = hipMalloc(&d_logits, sizeof(float) * logits.size()) == hipSuccess &&
              hipMalloc(&d_params, sizeof(DeviceSamplingParams) * outputs) == hipSuccess &&
              hipMalloc(&d_tokens, sizeof(int32_t) * outputs) == hipSuccess &&
              hipMalloc(&d_err, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&d_scratch, sizeof(float) * 2048u * kBlocks) == hipSuccess &&
              hipMalloc(&d_vals, sizeof(V) * 2u) == hipSuccess &&
              hipMalloc(&d_ctx, sizeof(::ps::runtime::DeviceBatchContext)) == hipSuccess;
    if (ok) {
        ok = hipMemcpy(d_logits, logits.data(), sizeof(float) * logits.size(),
                       hipMemcpyHostToDevice) == hipSuccess &&
             hipMemcpy(d_params, params.data(), sizeof(DeviceSamplingParams) * outputs,
                       hipMemcpyHostToDevice) == hipSuccess;
    }
    if (!ok) {
        if (d_logits) ps::gpu::discard_cleanup_result(hipFree(d_logits));
        if (d_params) ps::gpu::discard_cleanup_result(hipFree(d_params));
        if (d_tokens) ps::gpu::discard_cleanup_result(hipFree(d_tokens));
        if (d_err) ps::gpu::discard_cleanup_result(hipFree(d_err));
        if (d_scratch) ps::gpu::discard_cleanup_result(hipFree(d_scratch));
        if (d_vals) ps::gpu::discard_cleanup_result(hipFree(d_vals));
        if (d_ctx) ps::gpu::discard_cleanup_result(hipFree(d_ctx));
        return false;
    }

    V* vals = new V[2]();
    vals[0] = V{};
    vals[0].logical_value = 1;
    vals[0].storage = ::ps::runtime::ValueStorage::WORKSPACE;
    vals[0].dtype = DT::F32;
    vals[0].row_domain = ::ps::runtime::ValueRowDomain::OUTPUT_ROWS;
    vals[0].feature_count = vocab;
    vals[0].row_stride = vocab;
    vals[0].offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_logits));
    vals[1] = V{};
    vals[1].logical_value = 2;
    vals[1].storage = ::ps::runtime::ValueStorage::WORKSPACE;
    vals[1].dtype = DT::I32;
    vals[1].row_domain = ::ps::runtime::ValueRowDomain::OUTPUT_ROWS;
    vals[1].feature_count = 1;
    vals[1].row_stride = 4;
    vals[1].offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_tokens));
    ok = hipMemcpy(d_vals, vals, sizeof(V) * 2u, hipMemcpyHostToDevice) == hipSuccess;
    delete[] vals;

    P prog{};
    prog.values = d_vals;
    prog.value_count = 2;
    prog.workspace_base = 0;
    B binding{};
    binding.kernel_id = ::ps::runtime::KernelId::SAMPLING;
    binding.input_slots[0] = 1;
    binding.output_slots[0] = 2;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.rotary_dim = vocab;

    ::ps::runtime::DeviceBatchContext hctx{};
    hctx.actual_rows = outputs;
    hctx.num_requests = outputs;
    hctx.num_outputs = outputs;
    hctx.output_sampling_params = d_params;
    ok = ok && hipMemcpy(d_ctx, &hctx, sizeof(hctx), hipMemcpyHostToDevice) == hipSuccess;
    if (ok) {
        ps::gpu::discard_cleanup_result(hipMemset(d_err, 0, sizeof(uint32_t)));
        ::ps::kernel::ModelDispatchStateView sv{};
        const ps::Status st = ::ps::kernel::launch_model_dispatch_correctness(
            prog, binding, d_ctx, sv, d_scratch, d_err, nullptr);
        ok = st.ok() && hipDeviceSynchronize() == hipSuccess;
    }
    if (ok) {
        tokens.resize(outputs);
        ok = hipMemcpy(tokens.data(), d_tokens, sizeof(int32_t) * outputs,
                       hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(&error_word, d_err, sizeof(uint32_t), hipMemcpyDeviceToHost) ==
                 hipSuccess;
    }
    ps::gpu::discard_cleanup_result(hipFree(d_logits));
    ps::gpu::discard_cleanup_result(hipFree(d_params));
    ps::gpu::discard_cleanup_result(hipFree(d_tokens));
    ps::gpu::discard_cleanup_result(hipFree(d_err));
    ps::gpu::discard_cleanup_result(hipFree(d_scratch));
    ps::gpu::discard_cleanup_result(hipFree(d_vals));
    ps::gpu::discard_cleanup_result(hipFree(d_ctx));
    return ok;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    const std::vector<float> four = {10.0f, 9.0f, 8.0f, 7.0f};

    {
        std::printf("A. greedy mode\n");
        bool all_top = true;
        for (uint64_t seed = 0; seed < 32u; ++seed) {
            BatchResult r;
            if (!run_batch(four, 4u, 1u, {mk(1u, 0.0f, 1.0f, 0u, seed, 0u)}, r, false)) {
                all_top = false;
                break;
            }
            if (r.tokens[0] != 0) all_top = false;
        }
        check(all_top, "greedy returns the argmax for all seeds");
        check(greedy_one(4u, four) == 0, "greedy argmax token");
    }

    {
        std::printf("B. top_p collapses to the argmax\n");
        bool all_top = true;
        for (uint64_t seed = 0; seed < 256u; ++seed) {
            const int32_t t = sample_one(4u, four, 1.0f, 1e-6f, 0u, seed, 0u);
            if (t != 0) all_top = false;
        }
        check(all_top, "top_p=1e-6 -> argmax only");
    }

    {
        std::printf("C. top_p = 1 explores\n");
        std::vector<int32_t> tokens;
        bool ok = sample_many(4u, four, 1.0f, 1.0f, 0u, 7u, 256u, tokens);
        uint32_t non_top = 0;
        for (int32_t t : tokens)
            if (t != 0) ++non_top;
        check(ok, "full softmax launch");
        check(non_top > 0u, "tokens other than the argmax appear");
    }

    {
        std::printf("D. determinism\n");
        const int32_t first = sample_one(4u, four, 0.8f, 0.9f, 0u, 123u, 5u);
        bool same = true;
        for (int i = 0; i < 100; ++i) {
            if (sample_one(4u, four, 0.8f, 0.9f, 0u, 123u, 5u) != first) same = false;
        }
        check(same, "100 repeats bit exact");
    }

    {
        std::printf("E. sequence determinism\n");
        std::vector<int32_t> a, b;
        bool ok = sample_many(4u, four, 0.8f, 0.9f, 0u, 99u, 64u, a);
        ok = ok && sample_many(4u, four, 0.8f, 0.9f, 0u, 99u, 64u, b);
        check(ok && a == b, "index sequence bit exact");
        std::vector<int32_t> c;
        sample_many(4u, four, 0.8f, 0.9f, 0u, 100u, 64u, c);
        check(ok && c != a, "different seed changes the sequence");
    }

    {
        std::printf("F/G. top_k membership\n");
        std::vector<int32_t> tokens;
        bool ok = sample_many(4u, four, 1.0f, 1.0f, 1u, 3u, 256u, tokens);
        bool only_top = true;
        for (int32_t t : tokens)
            if (t != 0) only_top = false;
        check(ok && only_top, "top_k=1 -> argmax only");

        ok = sample_many(4u, four, 1.0f, 1.0f, 2u, 3u, 256u, tokens);
        bool within_two = true;
        for (int32_t t : tokens)
            if (t != 0 && t != 1) within_two = false;
        check(ok && within_two, "top_k=2 -> first two tokens only");
    }

    {
        std::printf("H. top-p distribution\n");
        const std::vector<float> l = prob_logits({0.60f, 0.25f, 0.10f, 0.05f});
        const uint32_t count = 10000u;
        std::vector<int32_t> tokens;
        std::vector<uint32_t> attempts;
        bool ok = sample_many(4u, l, 1.0f, 0.5f, 0u, 11u, count, tokens, &attempts);
        uint32_t c0 = 0;
        for (int32_t t : tokens)
            if (t == 0) ++c0;
        check(ok && c0 == count, "top_p=0.5 -> token0 always");

        ok = sample_many(4u, l, 1.0f, 0.7f, 0u, 11u, count, tokens, &attempts);
        double f0 = 0.0, f1 = 0.0;
        uint32_t other = 0;
        for (int32_t t : tokens) {
            if (t == 0) f0 += 1.0;
            else if (t == 1) f1 += 1.0;
            else ++other;
        }
        f0 /= count;
        f1 /= count;
        std::printf("    top_p=0.7: p0=%.4f p1=%.4f other=%u\n", f0, f1, other);
        check(ok && other == 0u, "top_p=0.7 -> nucleus is {0,1}");
        check(std::fabs(f0 - 0.70588) <= 0.03, "conditional p0 within 3%");
        check(std::fabs(f1 - 0.29412) <= 0.03, "conditional p1 within 3%");
        double avg = 0.0;
        for (uint32_t a : attempts) avg += static_cast<double>(a);
        avg /= attempts.empty() ? 1.0 : static_cast<double>(attempts.size());
        std::printf("    average attempts = %.3f\n", avg);
        check(avg > 0.9 && avg < 2.0, "average attempts bounded");
    }

    {
        std::printf("H2. min-p membership\n");
        const uint32_t count = 512u;
        std::vector<int32_t> tokens;
        bool ok = sample_many(4u, four, 1.0f, 1.0f, 0u, 13u, count, tokens, nullptr, 0.5f);
        bool only_max = true;
        for (int32_t t : tokens)
            if (t != 0) only_max = false;
        check(ok && only_max, "min_p=0.5 -> argmax only");

        ok = sample_many(4u, four, 1.0f, 1.0f, 0u, 13u, count, tokens, nullptr, 0.1f);
        bool within_three = true;
        for (int32_t t : tokens)
            if (t < 0 || t > 2) within_three = false;
        check(ok && within_three, "min_p=0.1 keeps weights >= 0.1");

        ok = sample_many(4u, four, 1.0f, 1.0f, 2u, 13u, count, tokens, nullptr, 0.1f);
        bool topk_and_minp = true;
        for (int32_t t : tokens)
            if (t != 0 && t != 1) topk_and_minp = false;
        check(ok && topk_and_minp, "min_p intersects top_k");
    }

    {
        std::printf("I. temperature\n");
        const std::vector<float> l = prob_logits({0.60f, 0.25f, 0.10f, 0.05f});
        const uint32_t count = 4000u;
        std::vector<int32_t> tokens;
        double f[3] = {0.0, 0.0, 0.0};
        const float temps[3] = {0.5f, 1.0f, 2.0f};
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            ok = ok && sample_many(4u, l, temps[i], 1.0f, 0u, 21u, count, tokens);
            for (int32_t t : tokens)
                if (t == 0) f[i] += 1.0;
            f[i] /= count;
        }
        std::printf("    p0: T=0.5 %.4f  T=1 %.4f  T=2 %.4f\n", f[0], f[1], f[2]);
        check(ok && f[0] > f[1] && f[1] > f[2], "entropy increases with temperature");
    }

    {
        std::printf("J. batch independence\n");
        const std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        BatchResult single;
        const bool ok1 =
            run_batch(four, 4u, 1u, {mk(2u, 0.8f, 0.9f, 0u, 55u, 7u)}, single, false);
        std::vector<float> batched;
        batched.insert(batched.end(), other.begin(), other.end());
        batched.insert(batched.end(), other.begin(), other.end());
        batched.insert(batched.end(), other.begin(), other.end());
        batched.insert(batched.end(), four.begin(), four.end());
        for (int i = 0; i < 4; ++i)
            batched.insert(batched.end(), other.begin(), other.end());
        std::vector<DeviceSamplingParams> params(8);
        for (uint32_t i = 0; i < 8u; ++i) {
            params[i] = mk(2u, 0.8f, 0.9f, 0u, 55u, 7u + i);
        }
        BatchResult multi;
        const bool ok2 = run_batch(batched, 4u, 8u, params, multi, false);
        check(ok1 && ok2 && single.tokens[0] == multi.tokens[3],
              "row3 of an 8-row batch matches the single row");
        check(ok2 && multi.tokens[3] != multi.tokens[0],
              "neighbouring rows differ");
    }

    {
        std::printf("K. mixed modes\n");
        std::vector<float> logits;
        const std::vector<float> a = {1.0f, 5.0f, 2.0f, 0.0f};
        const std::vector<float> b = {4.0f, 1.0f, 3.0f, 2.0f};
        logits.insert(logits.end(), four.begin(), four.end());
        logits.insert(logits.end(), a.begin(), a.end());
        logits.insert(logits.end(), four.begin(), four.end());
        logits.insert(logits.end(), b.begin(), b.end());
        logits.insert(logits.end(), four.begin(), four.end());
        std::vector<DeviceSamplingParams> params = {
            mk(0u, 0.0f, 1.0f, 0u, 1u, 0u),
            mk(1u, 0.0f, 1.0f, 0u, 1u, 1u),
            mk(2u, 0.8f, 0.9f, 0u, 1u, 2u),
            mk(1u, 0.0f, 1.0f, 0u, 1u, 3u),
            mk(2u, 0.8f, 0.9f, 0u, 1u, 4u),
        };
        BatchResult r;
        const bool ok = run_batch(logits, 4u, 5u, params, r, false);
        check(ok && r.error_word == 0u, "mixed launch");
        check(ok && r.tokens[0] == -1, "None -> -1");
        check(ok && r.tokens[1] == 1, "Greedy -> argmax");
        check(ok && r.tokens[3] == 0, "Greedy row 3 -> argmax");
        check(ok && r.tokens[2] >= 0 && r.tokens[4] >= 0, "Stochastic rows sampled");
        BatchResult r2;
        const bool ok2 = run_batch(logits, 4u, 5u, params, r2, false);
        check(ok2 && r2.tokens == r.tokens, "mixed repeat bit exact");
    }

    {
        std::printf("L. invalid logits\n");
        std::vector<float> bad = {1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
                                  3.0f};
        BatchResult r;
        const bool ok = run_batch(bad, 4u, 1u, {mk(2u, 1.0f, 1.0f, 0u, 1u, 0u)}, r, false);
        check(ok && r.error_word == 4u, "NaN logits -> DISPATCH_FAILED");
        check(ok && r.tokens[0] == -1, "NaN logits -> token -1");

        std::vector<float> inf = {1.0f, std::numeric_limits<float>::infinity(), 2.0f, 3.0f};
        BatchResult r2;
        const bool ok2 = run_batch(inf, 4u, 1u, {mk(2u, 1.0f, 1.0f, 0u, 1u, 0u)}, r2, false);
        check(ok2 && r2.error_word == 4u, "inf logits -> DISPATCH_FAILED");
        check(ok2 && r2.tokens[0] == -1, "inf logits -> token -1");
    }


    {
        std::printf("N. correctness vs optimized\n");
        std::vector<float> rows;
        std::vector<DeviceSamplingParams> params;
        const std::vector<float> a = {1.0f, 5.0f, 2.0f, 0.0f};
        const std::vector<float> b = {4.0f, 1.0f, 3.0f, 2.0f};
        const std::vector<float> rows4[4] = {four, a, b, four};
        for (uint32_t m = 0; m < 4u; ++m) {
            rows.insert(rows.end(), rows4[m].begin(), rows4[m].end());
            params.push_back(mk(m == 0u ? 0u : 2u, m == 0u ? 0.0f : 0.8f, 0.9f, 0u, 77u, m));
        }
        BatchResult opt;
        const bool ok1 = run_batch(rows, 4u, 4u, params, opt, false);
        std::vector<int32_t> corr;
        uint32_t corr_err = 0;
        const bool ok2 = run_correctness(rows, 4u, 4u, params, corr, corr_err);
        check(ok1 && ok2, "both backends launch");
        check(ok2 && corr_err == 0u, "correctness backend no error");
        check(ok1 && ok2 && corr == opt.tokens, "correctness and optimized agree");
    }

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
