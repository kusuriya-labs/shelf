#include <shelf/inference/execution/selectors/sampling_selector.h>

#include <cstdio>

using ::shelf::execution::SamplingChoice;
using ::shelf::execution::SamplingImplementation;
using ::shelf::execution::SamplingSelectorInput;
using ::shelf::execution::select_sampling_implementation;

static int g_fail = 0;

static SamplingSelectorInput mk(uint32_t vocab, uint32_t outputs, uint32_t sampled,
                                uint32_t scratch) {
    SamplingSelectorInput in;
    in.vocab_size = vocab;
    in.outputs = outputs;
    in.sampled_outputs = sampled;
    in.scratch_pair_capacity = scratch;
    return in;
}

static SamplingSelectorInput mk_stochastic(uint32_t vocab, uint32_t outputs,
                                           uint32_t sampled, uint32_t stochastic,
                                           uint32_t scratch) {
    SamplingSelectorInput in = mk(vocab, outputs, sampled, scratch);
    in.stochastic_outputs = stochastic;
    return in;
}

static void expect(const SamplingChoice& c, SamplingImplementation want,
                   uint32_t want_p, const char* msg) {
    const bool ok = c.implementation == want && c.partitions == want_p;
    if (!ok) ++g_fail;
    std::printf("  [%s] got=%d p=%u want=%d p=%u %s\n", msg, (int)c.implementation,
                c.partitions, (int)want, want_p, ok ? "PASS" : "FAIL");
}

int main() {
    const SamplingImplementation C = SamplingImplementation::Correctness;
    const SamplingImplementation S = SamplingImplementation::SingleBlock;
    const SamplingImplementation P = SamplingImplementation::Partitioned;
    const SamplingImplementation St = SamplingImplementation::StochasticSingleBlock;
    const SamplingImplementation R = SamplingImplementation::StochasticRadixTopK;

    auto mk_topk = [](uint32_t vocab, uint32_t outputs, uint32_t stochastic, uint32_t k,
                      bool eligible, bool workspace) {
        SamplingSelectorInput in = mk_stochastic(vocab, outputs, outputs, stochastic, 8192u);
        in.stochastic_topk_eligible = eligible;
        in.stochastic_top_k = k;
        in.stochastic_topk_workspace = workspace;
        return in;
    };

    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 50u, true, true)), R, 0, "radix topk eligible out=8");
    expect(select_sampling_implementation(
               mk_topk(248320u, 1, 1, 20u, true, true)), R, 0, "radix topk eligible out=1");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 128u, true, true)), R, 0, "radix topk k=128");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 50u, true, false)), St, 0, "radix topk no workspace");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 50u, false, true)), St, 0, "radix topk not eligible");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 0u, true, true)), St, 0, "radix topk k=0");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 129u, true, true)), St, 0, "radix topk k over max");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 8, 4096u, true, true)), St, 0, "radix topk k huge");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 3, 50u, true, true)), St, 0, "radix topk partial stochastic");
    expect(select_sampling_implementation(
               mk_topk(248320u, 8, 0, 50u, true, true)), P, 64, "radix topk no stochastic");

    expect(select_sampling_implementation(mk(123456789u, 1, 1, 8192u)), C, 0,
           "unknown vocab allgreedy");
    expect(select_sampling_implementation(mk(248320u, 0, 0, 8192u)), C, 0, "outputs=0");

    expect(select_sampling_implementation(mk(248320u, 1, 1, 8192u)), P, 64,
           "allgreedy out=1");
    expect(select_sampling_implementation(mk(248320u, 8, 8, 8192u)), P, 64,
           "allgreedy out=8");
    expect(select_sampling_implementation(mk(248320u, 64, 64, 8192u)), P, 64,
           "allgreedy out=64 max");
    expect(select_sampling_implementation(mk(248320u, 65, 65, 8192u)), C, 0,
           "allgreedy out=65 over");
    expect(select_sampling_implementation(mk(248320u, 16, 16, 1000u)), C, 0,
           "allgreedy scratch insufficient");

    expect(select_sampling_implementation(mk(248320u, 2, 1, 8192u)), P, 64,
           "mixed out=2 sampled=1");
    expect(select_sampling_implementation(mk(248320u, 8, 4, 8192u)), P, 64,
           "mixed out=8 sampled=4");
    expect(select_sampling_implementation(mk(248320u, 8, 5, 8192u)), P, 64,
           "mixed out=8 sampled=5");

    expect(select_sampling_implementation(mk(248320u, 1, 0, 8192u)), S, 0,
           "no-sampling out=1");
    expect(select_sampling_implementation(mk(248320u, 64, 0, 8192u)), S, 0,
           "no-sampling out=64");
    expect(select_sampling_implementation(mk(248320u, 256, 0, 8192u)), S, 0,
           "no-sampling out=256 max");
    expect(select_sampling_implementation(mk(248320u, 257, 0, 8192u)), C, 0,
           "no-sampling out=257 over");

    expect(select_sampling_implementation(mk_stochastic(248320u, 1, 1, 1, 8192u)), St, 0,
           "stochastic out=1");
    expect(select_sampling_implementation(mk_stochastic(248320u, 8, 8, 8, 8192u)), St, 0,
           "stochastic out=8 all");
    expect(select_sampling_implementation(mk_stochastic(248320u, 8, 4, 3, 8192u)), St, 0,
           "stochastic mixed");
    expect(select_sampling_implementation(mk_stochastic(248320u, 256, 256, 1, 8192u)), St,
           0, "stochastic out=256 max");
    expect(select_sampling_implementation(mk_stochastic(248320u, 257, 257, 1, 8192u)), C, 0,
           "stochastic out=257 over");
    expect(select_sampling_implementation(mk_stochastic(123456789u, 4, 4, 1, 8192u)), C, 0,
           "stochastic unknown vocab");
    expect(select_sampling_implementation(mk_stochastic(248320u, 8, 0, 0, 8192u)), S, 0,
           "stochastic=0 keeps no-sampling");

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
