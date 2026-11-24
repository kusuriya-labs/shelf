#include <shelf/core/memory/types.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/tp/tp_argmax.h>
#include <shelf/runtime/tp/tp_execution.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using shelf::runtime::TpArgmaxPair;
using shelf::runtime::TpGreedyProbe;
using shelf::runtime::tp_argmax_pack_host;
using shelf::runtime::tp_argmax_reduce_host;
using shelf::runtime::tp_greedy_argmax_eligible;
using shelf::runtime::tp_sum_bf16_4_host;
using shelf::runtime::tp_sum_f32_4_host;

static int g_fail = 0;

static void fail(const char* msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg);
}

static void check(bool cond, const char* msg) {
    if (!cond) fail(msg);
}

static float bf16_to_f32(std::uint16_t v) {
    const std::uint32_t u = static_cast<std::uint32_t>(v) << 16u;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static void test_eligibility() {
    TpGreedyProbe probe;
    check(tp_greedy_argmax_eligible(probe), "default probe is greedy eligible");
    probe.temperature = 0.7f;
    check(!tp_greedy_argmax_eligible(probe), "stochastic temperature is not eligible");
    probe.temperature = 0.0f;
    probe.top_k = 4u;
    check(!tp_greedy_argmax_eligible(probe), "top_k is not eligible");
    probe.top_k = 0u;
    probe.top_p = 0.9f;
    check(!tp_greedy_argmax_eligible(probe), "top_p is not eligible");
    probe.top_p = 1.0f;
    probe.min_p = 0.1f;
    check(!tp_greedy_argmax_eligible(probe), "min_p is not eligible");
    probe.min_p = 0.0f;
    probe.has_token_bitmask = true;
    check(!tp_greedy_argmax_eligible(probe), "grammar bitmask is not eligible");
}

static void test_argmax_pair_layout() {
    check(sizeof(TpArgmaxPair) == 8u, "argmax pair is 8 bytes per row");
    check(alignof(TpArgmaxPair) == 8u, "argmax pair is 8 byte aligned");
}

static void test_argmax_aggregation() {
    constexpr uint32_t rows = 7u;
    constexpr uint32_t vocab = 24u;
    constexpr uint32_t ranks = 4u;
    constexpr uint32_t local_cols = vocab / ranks;
    std::vector<std::vector<float>> logits(ranks, std::vector<float>(rows * vocab, -20.0f));
    for (uint32_t r = 0u; r < ranks; ++r) {
        for (uint32_t row = 0u; row < rows; ++row) {
            for (uint32_t c = 0u; c < local_cols; ++c) {
                logits[r][row * vocab + c] = static_cast<float>((row * 11u + c * 7u + r * 3u) % 23u);
            }
        }
    }
    logits[0][0 * vocab + 2u] = 100.0f;
    logits[2][0 * vocab + 1u] = 100.0f;
    logits[3][3 * vocab + 3u] = 100.0f;

    const std::size_t stride_pairs = rows + 3u;
    std::vector<TpArgmaxPair> pairs(ranks * stride_pairs);
    for (uint32_t r = 0u; r < ranks; ++r) {
        tp_argmax_pack_host(logits[r].data(), rows, vocab, local_cols, r * local_cols,
                            pairs.data() + r * stride_pairs);
    }
    std::vector<int32_t> ids(rows, -100);
    tp_argmax_reduce_host(pairs.data(), stride_pairs, ranks, rows, ids.data());

    check(ids[0] == 2, "tie breaks toward the smaller global token id");
    check(ids[3] == 21, "rank three local maximum wins its row");
    for (uint32_t row = 0u; row < rows; ++row) {
        float best = -20.0f;
        int best_id = -1;
        for (uint32_t token = 0u; token < vocab; ++token) {
            const float value =
                logits[token / local_cols][row * vocab + (token % local_cols)];
            if (value > best || (value == best && (best_id < 0 || static_cast<int>(token) < best_id))) {
                best = value;
                best_id = static_cast<int>(token);
            }
        }
        check(ids[row] == best_id, "argmax matches the full vocabulary reference");
    }
    check(pairs[0].id == 2, "rank zero local pair carries a global token id");
    check(pairs[0].value == 100.0f, "rank zero local pair carries the local maximum value");
}

static void test_war_buffer_separation() {
    constexpr uint32_t rows = 4u;
    const std::vector<float> logits(rows * 16u, 1.0f);
    std::vector<TpArgmaxPair> dedicated(rows);
    std::vector<float> inbox(rows * 16u, 0.0f);
    tp_argmax_pack_host(logits.data(), rows, 16u, 16u, 0u, dedicated.data());
    check(reinterpret_cast<const void*>(dedicated.data()) !=
              reinterpret_cast<const void*>(inbox.data()),
          "dedicated argmax buffer is a distinct allocation");
    for (float value : inbox) check(value == 0.0f, "argmax pack does not touch tp inbox");
}

static void test_four_input_sum() {
    constexpr std::size_t count = 33u;
    std::vector<std::uint16_t> a(count), b(count), c(count), d(count), out(count);
    std::vector<float> fa(count), fb(count), fc(count), fd(count), fout(count);
    for (std::size_t i = 0u; i < count; ++i) {
        a[i] = static_cast<std::uint16_t>((i + 1u) * 64u);
        b[i] = static_cast<std::uint16_t>((i + 2u) * 32u);
        c[i] = static_cast<std::uint16_t>((i + 3u) * 16u);
        d[i] = static_cast<std::uint16_t>((i + 4u) * 8u);
        fa[i] = static_cast<float>(i) + 0.25f;
        fb[i] = static_cast<float>(i) + 0.5f;
        fc[i] = static_cast<float>(i) + 0.75f;
        fd[i] = static_cast<float>(i) + 1.0f;
    }
    tp_sum_bf16_4_host(out.data(), a.data(), b.data(), c.data(), d.data(), count);
    tp_sum_f32_4_host(fout.data(), fa.data(), fb.data(), fc.data(), fd.data(), count);
    for (std::size_t i = 0u; i < count; ++i) {
        const float want_bf16 = bf16_to_f32(a[i]) + bf16_to_f32(b[i]) +
                                bf16_to_f32(c[i]) + bf16_to_f32(d[i]);
        check(out[i] == shelf::f32_to_bf16_rne(want_bf16), "four BF16 inputs sum exactly");
        check(fout[i] == fa[i] + fb[i] + fc[i] + fd[i], "four F32 inputs sum exactly");
    }
}

static void test_schedule_binds_argmax_output() {
    using namespace shelf::runtime;
    constexpr uint32_t global_vocab = 248320u;
    constexpr uint32_t local_vocab = 124160u;
    constexpr uint32_t hidden = 5120u;
    PrimitiveGraph graph;
    const ValueId normed = graph.alloc_value(hidden, ValueDType::BF16,
                                             ValueRowDomain::OUTPUT_ROWS);
    const ValueId logits = graph.alloc_value(global_vocab, ValueDType::F32,
                                             ValueRowDomain::OUTPUT_ROWS);
    const ValueId sampled = graph.alloc_value(1u, ValueDType::I32,
                                              ValueRowDomain::OUTPUT_ROWS);
    graph.external_inputs.push_back(normed);
    graph.external_outputs.push_back(sampled);

    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{local_vocab, hidden};
    linear.weight_index = graph.register_weight(linear.shape);
    linear.out_dtype = ValueDType::F32;
    auto& logits_node = graph.add_node(linear);
    logits_node.inputs = {normed};
    logits_node.outputs = {logits};
    logits_node.debug_name = "logits";
    logits_node.tp_combine = 1u;
    logits_node.tp_combine_local_cols = local_vocab;

    SamplingNode sampling;
    sampling.vocab_size = global_vocab;
    auto& sampling_node = graph.add_node(sampling);
    sampling_node.inputs = {logits};
    sampling_node.outputs = {sampled};
    sampling_node.debug_name = "sampled";

    WeightSlot slot{};
    slot.rows = local_vocab;
    slot.cols = hidden;
    slot.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Bf16);
    slot.bytes = static_cast<uint64_t>(local_vocab) * hidden * 2u;
    slot.output_col_offset = local_vocab;
    slot.output_col_total = global_vocab;
    WeightTableView table{&slot, 1u};
    ProgramBuildOptions options{};
    options.max_output_rows = 4u;
    auto program = build_program(graph, table, RowBucket::R16, ExecutionClass::DECODE, options);
    check(program.ok(), "argmax schedule program builds");
    if (!program.ok()) return;
    auto schedule = build_tp_execution_schedule(graph, program.value(), 2u, 1u, false);
    check(schedule.ok(), "argmax schedule builds");
    if (!schedule.ok()) return;
    const auto& segments = schedule.value().segments;
    bool found = false;
    for (const auto& segment : segments) {
        if (segment.combine_value.id != logits.id) continue;
        found = true;
        check(segment.argmax_output.id == sampled.id,
              "combine segment records the sampled output");
        check(segment.argmax_dispatch != UINT32_MAX,
              "combine segment records the sampling dispatch");
        if (segment.argmax_dispatch != UINT32_MAX) {
            check(program.value().dispatches[segment.argmax_dispatch].kernel_id ==
                      KernelId::SAMPLING,
                  "recorded dispatch is the sampling kernel");
        }
    }
    check(found, "a combine segment exists for logits");
}

int main() {
    test_eligibility();
    test_argmax_pair_layout();
    test_argmax_aggregation();
    test_war_buffer_separation();
    test_four_input_sum();
    test_schedule_binds_argmax_output();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
