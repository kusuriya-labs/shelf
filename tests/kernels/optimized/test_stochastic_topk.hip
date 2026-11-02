#include <shelf/kernels/optimized/radix_topn.h>
#include <shelf/kernels/optimized/stochastic_sampling.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
    std::printf("  [%-46s] %s\n", msg, cond ? "PASS" : "FAIL");
}

DeviceSamplingParams mk(uint8_t mode, float temperature, float top_p, uint32_t top_k,
                        uint64_t seed, uint64_t index) {
    DeviceSamplingParams p{};
    p.mode = mode;
    p.temperature = temperature;
    p.top_p = top_p;
    p.top_k = top_k;
    p.seed = seed;
    p.sample_index = index;
    p.reserved[0] = ::ps::kernel::kSamplingTopKPathFlag;
    return p;
}

struct TopKResult {
    std::vector<int32_t> tokens;
    std::vector<uint32_t> active;
    uint32_t error_word = 0;
};

__global__ void topk_reference_kernel(
    const float* __restrict__ logits,
    const DeviceSamplingParams* __restrict__ sampling,
    int32_t* __restrict__ tokens,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t top_k,
    uint32_t* __restrict__ error_word) {
    const uint32_t row = blockIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * row_stride;
    const ::ps::kernel::SamplingRowParams params =
        ::ps::kernel::sampling_row_params(sampling[row]);
    __shared__ int32_t ids[::ps::kernel::kSamplingMaxTopK];
    __shared__ float values[::ps::kernel::kSamplingMaxTopK];
    ::ps::kernel::sampling_topk_select_reference(row_logits, vocab, top_k, ids, values);
    ::ps::kernel::SamplingTopKView view;
    view.ids = ids;
    view.logits = values;
    view.count = top_k;
    ::ps::kernel::sampling_topk_active_row(row_logits, vocab, params, view,
                                           tokens + row, error_word);
}

bool run_radix(const std::vector<float>& logits, uint32_t vocab, uint32_t outputs,
               const std::vector<DeviceSamplingParams>& params, uint32_t top_k,
               TopKResult& out, bool want_active) {
    if (params.size() != outputs) return false;
    if (logits.size() != static_cast<std::size_t>(outputs) * vocab) return false;
    if (top_k == 0u || top_k > ::ps::kernel::kSamplingMaxTopK) return false;

    const uint32_t partitions = ::ps::kernel::dflash2_radix_default_partitions(vocab);
    const std::size_t scratch_bytes =
        ::ps::kernel::dflash2_radix_topn_scratch_bytes(outputs, partitions, vocab);

    float* d_logits = nullptr;
    DeviceSamplingParams* d_params = nullptr;
    int32_t* d_tokens = nullptr;
    uint32_t* d_err = nullptr;
    int32_t* d_ids = nullptr;
    float* d_vals = nullptr;
    void* d_scratch = nullptr;
    uint32_t* d_active = nullptr;
    bool ok = hipMalloc(&d_logits, sizeof(float) * logits.size()) == hipSuccess &&
              hipMalloc(&d_params, sizeof(DeviceSamplingParams) * outputs) == hipSuccess &&
              hipMalloc(&d_tokens, sizeof(int32_t) * outputs) == hipSuccess &&
              hipMalloc(&d_err, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&d_ids, sizeof(int32_t) * outputs * top_k) == hipSuccess &&
              hipMalloc(&d_vals, sizeof(float) * outputs * top_k) == hipSuccess &&
              hipMalloc(&d_scratch, scratch_bytes) == hipSuccess &&
              hipMalloc(&d_active, sizeof(uint32_t) * outputs) == hipSuccess;
    if (ok) {
        ok = hipMemcpy(d_logits, logits.data(), sizeof(float) * logits.size(),
                       hipMemcpyHostToDevice) == hipSuccess &&
             hipMemcpy(d_params, params.data(), sizeof(DeviceSamplingParams) * outputs,
                       hipMemcpyHostToDevice) == hipSuccess;
    }
    if (ok) {
        ps::gpu::discard_cleanup_result(hipMemset(d_err, 0, sizeof(uint32_t)));
        ok = ::ps::kernel::launch_dflash2_radix_topn(
                 d_logits, outputs, vocab, vocab, top_k, d_ids, d_vals, partitions,
                 d_scratch, scratch_bytes, nullptr) == hipSuccess;
    }
    if (ok) {
        ::ps::kernel::StochasticTopKSamplingArgs args;
        args.logits = d_logits;
        args.sampling = d_params;
        args.sampled_tokens = d_tokens;
        args.top_ids = d_ids;
        args.top_logits = d_vals;
        args.outputs = outputs;
        args.vocab_size = vocab;
        args.logits_row_stride = vocab;
        args.top_k = top_k;
        args.error_word = d_err;
        args.active_count_out = d_active;
        ok = ::ps::kernel::launch_stochastic_topk_sampling_f32(args, nullptr) == hipSuccess &&
             hipDeviceSynchronize() == hipSuccess;
    }
    if (ok) {
        out.tokens.resize(outputs);
        out.active.assign(outputs, 0u);
        ok = hipMemcpy(out.tokens.data(), d_tokens, sizeof(int32_t) * outputs,
                       hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(&out.error_word, d_err, sizeof(uint32_t),
                       hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(out.active.data(), d_active, sizeof(uint32_t) * outputs,
                       hipMemcpyDeviceToHost) == hipSuccess;
    }
    for (void* p : {reinterpret_cast<void*>(d_logits), reinterpret_cast<void*>(d_params),
                    reinterpret_cast<void*>(d_tokens), reinterpret_cast<void*>(d_err),
                    reinterpret_cast<void*>(d_ids), reinterpret_cast<void*>(d_vals),
                    d_scratch, reinterpret_cast<void*>(d_active)}) {
        if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
    (void)want_active;
    return ok;
}

bool run_reference(const std::vector<float>& logits, uint32_t vocab, uint32_t outputs,
                   const std::vector<DeviceSamplingParams>& params, uint32_t top_k,
                   TopKResult& out) {
    if (params.size() != outputs) return false;
    if (logits.size() != static_cast<std::size_t>(outputs) * vocab) return false;
    float* d_logits = nullptr;
    DeviceSamplingParams* d_params = nullptr;
    int32_t* d_tokens = nullptr;
    uint32_t* d_err = nullptr;
    bool ok = hipMalloc(&d_logits, sizeof(float) * logits.size()) == hipSuccess &&
              hipMalloc(&d_params, sizeof(DeviceSamplingParams) * outputs) == hipSuccess &&
              hipMalloc(&d_tokens, sizeof(int32_t) * outputs) == hipSuccess &&
              hipMalloc(&d_err, sizeof(uint32_t)) == hipSuccess;
    if (ok) {
        ok = hipMemcpy(d_logits, logits.data(), sizeof(float) * logits.size(),
                       hipMemcpyHostToDevice) == hipSuccess &&
             hipMemcpy(d_params, params.data(), sizeof(DeviceSamplingParams) * outputs,
                       hipMemcpyHostToDevice) == hipSuccess;
    }
    if (ok) {
        ps::gpu::discard_cleanup_result(hipMemset(d_err, 0, sizeof(uint32_t)));
        topk_reference_kernel<<<outputs, 256, 0, nullptr>>>(
            d_logits, d_params, d_tokens, vocab, vocab, top_k, d_err);
        ok = hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess;
    }
    if (ok) {
        out.tokens.resize(outputs);
        out.active.assign(outputs, 0u);
        ok = hipMemcpy(out.tokens.data(), d_tokens, sizeof(int32_t) * outputs,
                       hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(&out.error_word, d_err, sizeof(uint32_t),
                       hipMemcpyDeviceToHost) == hipSuccess;
    }
    for (void* p : {reinterpret_cast<void*>(d_logits), reinterpret_cast<void*>(d_params),
                    reinterpret_cast<void*>(d_tokens), reinterpret_cast<void*>(d_err)}) {
        if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
    return ok;
}

bool sample_topk(uint32_t vocab, const std::vector<float>& logits, float temperature,
                 float top_p, uint32_t top_k, uint64_t seed, uint32_t count,
                 TopKResult& out) {
    std::vector<DeviceSamplingParams> params(count);
    for (uint32_t i = 0; i < count; ++i)
        params[i] = mk(2u, temperature, top_p, top_k, seed, i);
    std::vector<float> rows;
    rows.reserve(static_cast<std::size_t>(count) * vocab);
    for (uint32_t i = 0; i < count; ++i)
        rows.insert(rows.end(), logits.begin(), logits.end());
    return run_radix(rows, vocab, count, params, top_k, out, false) && out.error_word == 0u;
}

bool sample_topk_reference(uint32_t vocab, const std::vector<float>& logits,
                           float temperature, float top_p, uint32_t top_k, uint64_t seed,
                           uint32_t count, TopKResult& out) {
    std::vector<DeviceSamplingParams> params(count);
    for (uint32_t i = 0; i < count; ++i)
        params[i] = mk(2u, temperature, top_p, top_k, seed, i);
    std::vector<float> rows;
    rows.reserve(static_cast<std::size_t>(count) * vocab);
    for (uint32_t i = 0; i < count; ++i)
        rows.insert(rows.end(), logits.begin(), logits.end());
    return run_reference(rows, vocab, count, params, top_k, out) &&
           out.error_word == 0u;
}

std::vector<float> prob_logits(const std::vector<float>& p) {
    std::vector<float> l(p.size());
    for (size_t i = 0; i < p.size(); ++i) l[i] = std::log(p[i]);
    return l;
}

std::vector<int32_t> cpu_topk_order(const std::vector<float>& row, uint32_t k) {
    std::vector<int32_t> order(row.size());
    for (size_t i = 0; i < row.size(); ++i) order[i] = static_cast<int32_t>(i);
    std::stable_sort(order.begin(), order.end(), [&row](int32_t a, int32_t b) {
        const float va = row[static_cast<size_t>(a)];
        const float vb = row[static_cast<size_t>(b)];
        if (va != vb) return va > vb;
        return a < b;
    });
    order.resize(k);
    return order;
}

std::vector<int32_t> cpu_active_set(const std::vector<float>& row, float temperature,
                                    float top_p, uint32_t top_k) {
    const std::vector<int32_t> order = cpu_topk_order(row, top_k);
    if (top_p >= 1.0f) return order;
    float lmax = -std::numeric_limits<float>::infinity();
    for (float v : row)
        if (std::isfinite(v) && v > lmax) lmax = v;
    double z = 0.0;
    for (float v : row) z += static_cast<double>(std::exp((v - lmax) / temperature));
    std::vector<int32_t> active;
    double prefix = 0.0;
    const double threshold = static_cast<double>(top_p) * z;
    for (int32_t t : order) {
        if (prefix < threshold) active.push_back(t);
        prefix += static_cast<double>(
            std::exp((row[static_cast<size_t>(t)] - lmax) / temperature));
    }
    return active;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    {
        std::printf("A. top_k=1 always argmax\n");
        const uint32_t vocab = 64u;
        std::vector<float> row(vocab);
        for (uint32_t i = 0; i < vocab; ++i)
            row[i] = static_cast<float>((i * 37u) % 101u) * 0.13f;
        const int32_t argmax =
            static_cast<int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
        TopKResult r;
        bool ok = sample_topk(vocab, row, 1.0f, 1.0f, 1u, 5u, 128u, r);
        bool all_top = ok;
        for (int32_t t : r.tokens)
            if (t != argmax) all_top = false;
        check(all_top, "top_k=1 -> argmax for all 128 samples");
        TopKResult ref;
        ok = sample_topk_reference(vocab, row, 1.0f, 1.0f, 1u, 5u, 16u, ref);
        bool ref_top = ok;
        for (int32_t t : ref.tokens)
            if (t != argmax) ref_top = false;
        check(ref_top, "reference top_k=1 -> argmax");
    }

    for (uint32_t top_k : {2u, 16u, 50u, 128u}) {
        const uint32_t vocab = 256u;
        std::vector<float> row(vocab);
        for (uint32_t i = 0; i < vocab; ++i)
            row[i] = std::sin(static_cast<float>(i) * 0.37f) * 4.0f +
                     std::cos(static_cast<float>(i) * 0.11f) * 2.0f;
        const std::vector<int32_t> order = cpu_topk_order(row, top_k);
        std::vector<int32_t> allowed = order;
        std::sort(allowed.begin(), allowed.end());
        TopKResult r;
        const bool ok = sample_topk(vocab, row, 1.0f, 1.0f, top_k, 9u, 512u, r);
        bool within = ok;
        for (int32_t t : r.tokens)
            if (!std::binary_search(allowed.begin(), allowed.end(), t)) within = false;
        char label[64];
        std::snprintf(label, sizeof(label), "top_k=%u tokens stay in top-%u set", top_k,
                      top_k);
        check(within, label);

        TopKResult ref;
        const bool ok_ref =
            sample_topk_reference(vocab, row, 1.0f, 1.0f, top_k, 9u, 64u, ref);
        bool ref_within = ok_ref;
        for (int32_t t : ref.tokens)
            if (!std::binary_search(allowed.begin(), allowed.end(), t)) ref_within = false;
        std::snprintf(label, sizeof(label), "reference top_k=%u membership", top_k);
        check(ref_within, label);
    }

    {
        std::printf("F. top_k >= vocab keeps every token\n");
        const uint32_t vocab = 8u;
        const std::vector<float> row = {1.0f, 5.0f, 2.0f, 0.0f, 4.0f, 3.0f, 7.0f, 6.0f};
        TopKResult r;
        const bool ok = sample_topk(vocab, row, 1.0f, 1.0f, 8u, 4u, 256u, r);
        check(ok && !r.active.empty() && r.active[0] == vocab,
              "top_k=vocab -> active set is the whole vocab");
        const int32_t argmax = 6;
        check(std::find(r.tokens.begin(), r.tokens.end(), argmax) != r.tokens.end(),
              "top_k=vocab -> argmax still sampled");

        const std::vector<float> flat(8, 1.0f);
        TopKResult all;
        const bool ok2 = sample_topk(8u, flat, 1.0f, 1.0f, 8u, 77u, 512u, all);
        bool seen[8] = {false, false, false, false, false, false, false, false};
        for (int32_t t : all.tokens)
            if (t >= 0 && t < 8) seen[t] = true;
        bool all_seen = ok2;
        for (uint32_t i = 0; i < 8u; ++i)
            if (!seen[i]) all_seen = false;
        check(all_seen, "top_k=vocab with flat logits reaches every token");
    }

    {
        std::printf("G. ties order by lower token id\n");
        const uint32_t vocab = 16u;
        std::vector<float> row(vocab, 3.0f);
        TopKResult r;
        const bool ok = sample_topk(vocab, row, 1.0f, 1.0f, 4u, 13u, 512u, r);
        bool only_low = ok;
        for (int32_t t : r.tokens)
            if (t < 0 || t >= 4) only_low = false;
        check(ok && only_low, "all-equal logits + top_k=4 -> only tokens 0..3");
    }

    {
        std::printf("H. top_k + top_p intersection\n");
        const std::vector<float> l = prob_logits({0.60f, 0.25f, 0.10f, 0.05f});
        TopKResult r;
        const bool ok = sample_topk(4u, l, 1.0f, 0.7f, 2u, 21u, 4000u, r);
        uint32_t c0 = 0, c1 = 0, other = 0;
        for (int32_t t : r.tokens) {
            if (t == 0) ++c0;
            else if (t == 1) ++c1;
            else ++other;
        }
        const double f0 = static_cast<double>(c0) / 4000.0;
        const double f1 = static_cast<double>(c1) / 4000.0;
        std::printf("    top_k=2 top_p=0.7: p0=%.4f p1=%.4f other=%u\n", f0, f1, other);
        check(ok && other == 0u, "top_k=2 + top_p=0.7 -> nucleus {0,1}");
        check(std::fabs(f0 - 0.70588) <= 0.03, "conditional p0 within 3%");
        check(std::fabs(f1 - 0.29412) <= 0.03, "conditional p1 within 3%");

        TopKResult r3;
        const bool ok3 = sample_topk(4u, l, 1.0f, 1.0f, 3u, 21u, 4000u, r3);
        uint32_t c3_0 = 0, c3_1 = 0, c3_2 = 0;
        for (int32_t t : r3.tokens) {
            if (t == 0) ++c3_0;
            else if (t == 1) ++c3_1;
            else if (t == 2) ++c3_2;
        }
        const double p0 = static_cast<double>(c3_0) / 4000.0;
        const double p1 = static_cast<double>(c3_1) / 4000.0;
        const double p2 = static_cast<double>(c3_2) / 4000.0;
        std::printf("    top_k=3: p0=%.4f p1=%.4f p2=%.4f\n", p0, p1, p2);
        check(ok3 && c3_0 + c3_1 + c3_2 == 4000u, "top_k=3 -> only first three tokens");
        check(std::fabs(p0 - 0.6 / 0.95) <= 0.03, "top_k=3 p0 within 3%");
        check(std::fabs(p1 - 0.25 / 0.95) <= 0.03, "top_k=3 p1 within 3%");
        check(std::fabs(p2 - 0.10 / 0.95) <= 0.03, "top_k=3 p2 within 3%");
    }

    {
        std::printf("I. determinism\n");
        const std::vector<float> row = {1.0f, 5.0f, 2.0f, 0.0f, 4.0f, 3.0f};
        TopKResult first;
        const bool ok0 = sample_topk(6u, row, 0.8f, 0.9f, 4u, 123u, 8u, first);
        bool same = ok0;
        for (int i = 0; i < 100; ++i) {
            TopKResult r;
            if (!sample_topk(6u, row, 0.8f, 0.9f, 4u, 123u, 8u, r)) same = false;
            else if (r.tokens != first.tokens) same = false;
        }
        check(same, "100 repeats bit exact");
        TopKResult other_seed;
        if (sample_topk(6u, row, 0.8f, 0.9f, 4u, 124u, 8u, other_seed))
            check(other_seed.tokens != first.tokens, "different seed changes tokens");
    }

    {
        std::printf("J. batch independence\n");
        const std::vector<float> a = {1.0f, 5.0f, 2.0f, 0.0f};
        const std::vector<float> b = {4.0f, 1.0f, 3.0f, 2.0f};
        TopKResult single;
        const bool ok1 = sample_topk(4u, a, 0.8f, 1.0f, 3u, 55u, 1u, single);
        std::vector<float> rows;
        rows.insert(rows.end(), b.begin(), b.end());
        rows.insert(rows.end(), a.begin(), a.end());
        rows.insert(rows.end(), b.begin(), b.end());
        TopKResult multi;
        std::vector<DeviceSamplingParams> params = {
            mk(2u, 0.8f, 1.0f, 3u, 55u, 0u), mk(2u, 0.8f, 1.0f, 3u, 55u, 1u),
            mk(2u, 0.8f, 1.0f, 3u, 55u, 2u)};
        const bool ok2 = run_radix(rows, 4u, 3u, params, 3u, multi, false);
        TopKResult single_at_one;
        const bool ok3 = sample_topk(4u, a, 0.8f, 1.0f, 3u, 55u, 2u, single_at_one);
        check(ok1 && ok2 && ok3, "batch launches");
        check(ok1 && ok2 && multi.tokens[1] == single_at_one.tokens[1],
              "row1 of a batch matches a single-row run");
        check(ok2 && multi.tokens[0] != multi.tokens[1], "neighbouring rows differ");
    }

    {
        std::printf("K/L. invalid logits\n");
        std::vector<float> bad = {1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
                                  3.0f, 4.0f, 5.0f};
        TopKResult r;
        const bool ok = run_radix(bad, 6u, 1u, {mk(2u, 1.0f, 1.0f, 3u, 1u, 0u)}, 3u, r, false);
        check(ok && r.error_word == 4u, "NaN logits -> DISPATCH_FAILED");
        check(ok && r.tokens[0] == -1, "NaN logits -> token -1");

        std::vector<float> inf = {1.0f, std::numeric_limits<float>::infinity(), 2.0f,
                                  3.0f, 4.0f, 5.0f};
        TopKResult r2;
        const bool ok2 = run_radix(inf, 6u, 1u, {mk(2u, 1.0f, 1.0f, 3u, 1u, 0u)}, 3u, r2, false);
        check(ok2 && r2.error_word == 4u, "inf logits -> DISPATCH_FAILED");
        check(ok2 && r2.tokens[0] == -1, "inf logits -> token -1");

        std::vector<float> ninf(6, -std::numeric_limits<float>::infinity());
        TopKResult r3;
        const bool ok3 =
            run_radix(ninf, 6u, 1u, {mk(2u, 1.0f, 1.0f, 3u, 1u, 0u)}, 3u, r3, false);
        check(ok3 && r3.error_word == 4u, "-inf logits -> DISPATCH_FAILED");
        check(ok3 && r3.tokens[0] == -1, "-inf logits -> token -1");
    }

    {
        std::printf("M. correctness reference vs optimized radix\n");
        const uint32_t vocab = 128u;
        bool exact = true;
        for (uint32_t seed = 0; seed < 8u; ++seed) {
            std::vector<float> rows;
            std::vector<DeviceSamplingParams> params;
            for (uint32_t r = 0; r < 4u; ++r) {
                std::vector<float> row(vocab);
                for (uint32_t i = 0; i < vocab; ++i) {
                    const uint32_t h = (i * 2654435761u + seed * 40503u + r * 7919u);
                    row[i] = static_cast<float>((h >> 8) % 1000u) * 0.01f;
                }
                rows.insert(rows.end(), row.begin(), row.end());
                params.push_back(mk(2u, 0.9f, 0.95f, 16u, 777u, r));
            }
            TopKResult opt, ref;
            const bool ok1 = run_radix(rows, vocab, 4u, params, 16u, opt, false);
            const bool ok2 = run_reference(rows, vocab, 4u, params, 16u, ref);
            if (!ok1 || !ok2 || opt.tokens != ref.tokens || opt.error_word != ref.error_word)
                exact = false;
        }
        check(exact, "reference and radix produce identical tokens");

        const std::vector<float> l = prob_logits({0.60f, 0.25f, 0.10f, 0.05f});
        TopKResult opt, ref;
        const bool ok1 = sample_topk(4u, l, 1.0f, 0.7f, 2u, 31u, 4096u, opt);
        const bool ok2 = sample_topk_reference(4u, l, 1.0f, 0.7f, 2u, 31u, 4096u, ref);
        check(ok1 && ok2 && opt.tokens == ref.tokens,
              "reference and radix agree on 4096 samples with top_p");
    }

    {
        std::printf("5.19. active set matches the full-softmax definition\n");
        const std::vector<float> l = prob_logits({0.60f, 0.25f, 0.10f, 0.05f});
        struct Case {
            float top_p;
            uint32_t top_k;
        };
        const Case cases[] = {{0.7f, 2u}, {0.7f, 3u}, {0.9f, 3u}, {1.0f, 2u},
                              {0.95f, 4u}, {0.5f, 4u}};
        for (const Case& c : cases) {
            const std::vector<int32_t> want =
                cpu_active_set(l, 1.0f, c.top_p, c.top_k);
            std::vector<int32_t> seen;
            TopKResult r;
            const bool ok = sample_topk(4u, l, 1.0f, c.top_p, c.top_k, 4242u, 8192u, r);
            for (uint32_t i = 0; i < r.tokens.size(); ++i)
                if (std::find(seen.begin(), seen.end(), r.tokens[i]) == seen.end())
                    seen.push_back(r.tokens[i]);
            std::sort(seen.begin(), seen.end());
            std::vector<int32_t> want_sorted = want;
            std::sort(want_sorted.begin(), want_sorted.end());
            char label[80];
            std::snprintf(label, sizeof(label),
                          "top_p=%.2f top_k=%u sampled set equals active set", c.top_p,
                          c.top_k);
            check(ok && seen == want_sorted, label);

            TopKResult one;
            const bool ok1 = run_radix(l, 4u, 1u, {mk(2u, 1.0f, c.top_p, c.top_k, 7u, 0u)},
                                       c.top_k, one, true);
            char label2[80];
            std::snprintf(label2, sizeof(label2), "top_p=%.2f top_k=%u active_count=%u",
                          c.top_p, c.top_k, one.active.empty() ? 0u : one.active[0]);
            std::printf("    %s (want %zu)\n", label2, want.size());
            check(ok1 && !one.active.empty() && one.active[0] == want.size(), label2);
        }
    }

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
