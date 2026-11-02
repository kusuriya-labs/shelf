#include <phaseshift/models/qwen4exp/ple/ple_conv.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ple = ps::qwen4exp::ple;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

struct Rng {
    std::uint64_t state;

    explicit Rng(std::uint64_t seed) : state(seed) {}

    std::uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    float uniform(float lo, float hi) {
        const float u = static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
        return lo + (hi - lo) * u;
    }
};

static void oracle_conv(const std::vector<float>& weight, const std::vector<float>& input,
                        std::uint32_t tokens, std::uint32_t conv_dim,
                        std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(tokens) * conv_dim, 0.0);
    std::vector<double> state(static_cast<std::size_t>(conv_dim) * ple::kPleConvStateLen, 0.0);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t c = 0; c < conv_dim; ++c) {
            const float* wf = weight.data() + static_cast<std::size_t>(c) * ple::kPleConvKernel;
            const double w[4] = {static_cast<double>(wf[0]), static_cast<double>(wf[1]),
                                 static_cast<double>(wf[2]), static_cast<double>(wf[3])};
            const double* hist = state.data() + c;
            const double x = static_cast<double>(input[static_cast<std::size_t>(t) * conv_dim + c]);
            out[static_cast<std::size_t>(t) * conv_dim + c] =
                w[0] * hist[0ull * conv_dim] + w[1] * hist[3ull * conv_dim] +
                w[2] * hist[6ull * conv_dim] + w[3] * x;
        }
        for (std::uint32_t c = 0; c < conv_dim; ++c) {
            double* hist = state.data() + c;
            for (std::uint32_t k = 0; k + 1u < ple::kPleConvStateLen; ++k) {
                hist[static_cast<std::size_t>(k) * conv_dim] =
                    hist[static_cast<std::size_t>(k + 1u) * conv_dim];
            }
            hist[static_cast<std::size_t>(ple::kPleConvStateLen - 1u) * conv_dim] =
                static_cast<double>(input[static_cast<std::size_t>(t) * conv_dim + c]);
        }
    }
}

static void test_hand_case() {
    const std::uint32_t dim = 1;
    const float weight[4] = {1.0f, 2.0f, 4.0f, 8.0f};
    const float input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float output[6] = {};
    ple::PleConvState state;
    state.reset(dim);
    check(ple::ple_dilated_conv1d(weight, input, 6, dim, state, output).ok(),
          "hand: conv runs");
    const float want[6] = {8.0f, 16.0f, 24.0f, 36.0f, 48.0f, 60.0f};
    bool ok = true;
    for (int t = 0; t < 6; ++t) {
        if (output[t] != want[t]) {
            ok = false;
            printf("  hand mismatch t=%d got=%f want=%f\n", t, output[t], want[t]);
        }
    }
    check(ok, "hand: dilated taps at t, t-3, t-6, t-9 exact");
}

static void test_oracle() {
    const std::uint32_t dim = 128;
    const std::uint32_t tokens = 37;
    Rng rng(1234);
    std::vector<float> weight(static_cast<std::size_t>(dim) * ple::kPleConvKernel);
    std::vector<float> input(static_cast<std::size_t>(tokens) * dim);
    for (float& v : weight) v = rng.uniform(-0.02f, 0.02f);
    for (float& v : input) v = rng.uniform(-1.0f, 1.0f);

    ple::PleConvState state;
    state.reset(dim);
    std::vector<float> output(static_cast<std::size_t>(tokens) * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input.data(), tokens, dim, state,
                                  output.data())
              .ok(),
          "oracle: conv runs");

    std::vector<double> want;
    oracle_conv(weight, input, tokens, dim, want);
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sq = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double diff = std::abs(static_cast<double>(output[i]) - want[i]);
        if (diff > max_abs) max_abs = diff;
        const double rel = diff / std::max(std::abs(want[i]), 1e-12);
        if (rel > max_rel) max_rel = rel;
        sq += diff * diff;
    }
    const double rmse = std::sqrt(sq / static_cast<double>(want.size()));
    printf("  conv raw: max_abs=%.3e max_rel=%.3e RMSE=%.3e\n", max_abs, max_rel, rmse);
    check(max_abs <= 1e-6 && rmse <= 1e-7, "oracle: float32 within float64 oracle threshold");

    bool state_ok = true;
    for (std::uint32_t k = 0; k < ple::kPleConvStateLen; ++k) {
        const std::uint32_t src_row = tokens - ple::kPleConvStateLen + k;
        for (std::uint32_t c = 0; c < dim; ++c) {
            if (state.buffer[static_cast<std::size_t>(k) * dim + c] !=
                input[static_cast<std::size_t>(src_row) * dim + c]) {
                state_ok = false;
            }
        }
    }
    check(state_ok, "state: holds the last 9 inputs oldest-first after run");
}

static void test_chunk_and_decode() {
    const std::uint32_t dim = 64;
    const std::uint32_t tokens = 24;
    Rng rng(555);
    std::vector<float> weight(static_cast<std::size_t>(dim) * ple::kPleConvKernel);
    std::vector<float> input(static_cast<std::size_t>(tokens + 1) * dim);
    for (float& v : weight) v = rng.uniform(-0.02f, 0.02f);
    for (float& v : input) v = rng.uniform(-1.0f, 1.0f);

    ple::PleConvState full_state;
    full_state.reset(dim);
    std::vector<float> full(static_cast<std::size_t>(tokens) * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input.data(), tokens, dim, full_state,
                                  full.data())
              .ok(),
          "chunk: full run");

    ple::PleConvState chunk_state;
    chunk_state.reset(dim);
    std::vector<float> chunked(static_cast<std::size_t>(tokens) * dim, 0.0f);
    const std::uint32_t cuts[3] = {9, 5, 10};
    std::uint32_t begin = 0;
    for (std::uint32_t cut : cuts) {
        check(ple::ple_dilated_conv1d(weight.data(), input.data() + static_cast<std::size_t>(begin) * dim,
                                      cut, dim, chunk_state,
                                      chunked.data() + static_cast<std::size_t>(begin) * dim)
                  .ok(),
              "chunk: sub-chunk runs");
        begin += cut;
    }
    check(full == chunked, "chunk: split prefill bitwise equals full prefill");

    std::vector<float> decoded(1ull * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input.data() + static_cast<std::size_t>(tokens) * dim,
                                  1, dim, full_state, decoded.data())
              .ok(),
          "decode: append runs");
    ple::PleConvState extra_state;
    extra_state.reset(dim);
    std::vector<float> longer(static_cast<std::size_t>(tokens + 1) * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input.data(), tokens + 1, dim, extra_state,
                                  longer.data())
              .ok(),
          "decode: longer run");
    const bool same = std::equal(decoded.begin(), decoded.end(),
                                 longer.begin() + static_cast<std::ptrdiff_t>(tokens) * dim);
    check(same, "decode: one-token append with state equals longer run");
}

static void test_reset_and_reuse() {
    const std::uint32_t dim = 32;
    Rng rng(999);
    std::vector<float> weight(static_cast<std::size_t>(dim) * ple::kPleConvKernel);
    for (float& v : weight) v = rng.uniform(-0.02f, 0.02f);

    std::vector<float> seq_a(17ull * dim);
    std::vector<float> seq_b(16ull * dim);
    for (std::uint32_t i = 0; i < 16ull * dim; ++i) seq_a[i] = rng.uniform(-1.0f, 1.0f);
    for (float& v : seq_b) v = rng.uniform(-1.0f, 1.0f);
    const float eos = 123456.0f;
    for (std::uint32_t c = 0; c < dim; ++c) seq_a[16ull * dim + c] = eos;

    ple::PleConvState state;
    state.reset(dim);
    std::vector<float> out_a(16ull * dim, 0.0f);
    std::vector<float> out_eos(dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), seq_a.data(), 16, dim, state, out_a.data())
              .ok(),
          "reuse: sequence A runs");
    check(ple::ple_dilated_conv1d(weight.data(), seq_a.data() + 16ull * dim, 1, dim, state,
                                  out_eos.data())
              .ok(),
          "reuse: EOS tail token runs");

    state.reset(dim);
    std::vector<float> out_b(16ull * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), seq_b.data(), 16, dim, state, out_b.data())
              .ok(),
          "reuse: sequence B runs after reset");

    ple::PleConvState fresh;
    fresh.reset(dim);
    std::vector<float> out_b_fresh(16ull * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), seq_b.data(), 16, dim, fresh,
                                  out_b_fresh.data())
              .ok(),
          "reuse: sequence B fresh state runs");
    check(out_b == out_b_fresh, "reuse: reset leaves no residue from previous sequence");

    state.reset(dim);
    std::vector<float> out_a2(16ull * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), seq_a.data(), 16, dim, state, out_a2.data())
              .ok(),
          "reuse: sequence A repeat runs");
    check(out_a2 == out_a, "reuse: repeated sequence is bitwise identical");
}

static void test_silu_gate() {
    const std::uint32_t dim = 4;
    Rng rng(4242);
    std::vector<float> weight(static_cast<std::size_t>(dim) * ple::kPleConvKernel);
    std::vector<float> input(8ull * dim);
    for (float& v : weight) v = rng.uniform(-0.02f, 0.02f);
    for (float& v : input) v = rng.uniform(-1.0f, 1.0f);

    ple::PleConvState state;
    state.reset(dim);
    std::vector<float> raw(8ull * dim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input.data(), 8, dim, state, raw.data())
              .ok(),
          "silu: raw conv runs");
    double max_abs = 0.0;
    double sq = 0.0;
    for (float v : raw) {
        const double s = static_cast<double>(v) / (1.0 + std::exp(-static_cast<double>(v)));
        const double f = static_cast<float>(s);
        const double diff = std::abs(f - s);
        if (diff > max_abs) max_abs = diff;
        sq += diff * diff;
    }
    printf("  silu: max_abs=%.3e RMSE=%.3e\n", max_abs,
           std::sqrt(sq / static_cast<double>(raw.size())));
    check(max_abs <= 1e-7, "silu: float32 silu within double reference");
}

static void test_error_paths() {
    ple::PleConvState state;
    const float w = 0.0f;
    const float x = 0.0f;
    float y = 0.0f;
    check(!ple::ple_dilated_conv1d(&w, &x, 1, 4, state, &y).ok(),
          "err: uninitialized state rejected");
    state.reset(4);
    check(!ple::ple_dilated_conv1d(&w, &x, 1, 8, state, &y).ok(),
          "err: conv_dim mismatch rejected");
    check(!ple::ple_dilated_conv1d(nullptr, &x, 1, 4, state, &y).ok(),
          "err: null weight rejected");
    check(!ple::ple_dilated_conv1d(&w, nullptr, 1, 4, state, &y).ok(),
          "err: null input rejected");
    check(ple::ple_dilated_conv1d(&w, &x, 0, 4, state, &y).ok(),
          "err: zero-length accepted");
}

int main() {
    printf("=== PLE dilated depthwise conv (kernel=4, dilation=3, state=9) ===\n");
    test_hand_case();
    test_oracle();
    test_chunk_and_decode();
    test_reset_and_reuse();
    test_silu_gate();
    test_error_paths();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
