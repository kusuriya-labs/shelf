#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/runtime/tp/tp_execution.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using shelf::Result;
using shelf::Status;
using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::Qwen35TensorParallelContext;
using shelf::qwen35::make_qwen35_tensor_parallel_context;
using shelf::runtime::ExecutionClass;
using shelf::runtime::LinearNode;
using shelf::runtime::MatrixwiseShapeKey;
using shelf::runtime::PrimitiveGraph;
using shelf::runtime::PrimitiveGraphNode;
using shelf::runtime::Program;
using shelf::runtime::ProgramBuildOptions;
using shelf::runtime::RowBucket;
using shelf::runtime::TpBarrierKind;
using shelf::runtime::ValueBinding;
using shelf::runtime::ValueDType;
using shelf::runtime::ValueId;
using shelf::runtime::ValueRowDomain;
using shelf::runtime::ValueStorage;
using shelf::runtime::WeightSlot;
using shelf::runtime::WeightTableView;
using shelf::runtime::build_program;
using shelf::runtime::build_tp_execution_schedule;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static Qwen35TextConfig base_config() {
    Qwen35TextConfig c;
    c.vocab_size = 128;
    c.hidden_size = 64;
    c.ffn.intermediate_size = 128;
    c.num_hidden_layers = 4;
    c.full_attention_interval = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.attention_head_dim = 16;
    c.linear_num_key_heads = 4;
    c.linear_num_value_heads = 8;
    c.linear_key_head_dim = 8;
    c.linear_value_head_dim = 8;
    c.linear_conv_kernel_dim = 4;
    c.layer_types = {0, 1, 0, 1};
    return c;
}

// The GDN state pool and the KV pool size themselves from this counter, so it has to
// follow the same rule the loader uses, including for a checkpoint that declares only
// full_attention_interval and no layer_types.
static void check_topology_counter() {
    using shelf::qwen_hybrid::count_layer_topology;
    using shelf::qwen_hybrid::layer_is_gdn;

    Qwen35TextConfig interval_only = base_config();
    interval_only.layer_types.clear();
    interval_only.num_hidden_layers = 8;
    interval_only.full_attention_interval = 4;
    const auto by_interval = count_layer_topology(interval_only);
    check(by_interval.gdn_layers == 6 && by_interval.attention_layers == 2,
          "interval-only config counts every layer");
    std::size_t gdn = 0;
    for (std::size_t l = 0; l < interval_only.num_hidden_layers; ++l) {
        if (layer_is_gdn(interval_only, l)) ++gdn;
    }
    check(gdn == by_interval.gdn_layers, "counter follows layer_is_gdn");
    check(by_interval.gdn_layers + by_interval.attention_layers ==
              interval_only.num_hidden_layers,
          "every layer is counted exactly once");

    Qwen35TextConfig typed = base_config();
    typed.num_hidden_layers = 4;
    typed.full_attention_interval = 2;
    typed.layer_types = {1, 1, 0, 1};
    const auto by_types = count_layer_topology(typed);
    check(by_types.gdn_layers == 1 && by_types.attention_layers == 3,
          "declared layer_types win over the interval rule");

    check(shelf::qwen35::make_qwen35_gdn_state_layout(typed).num_gdn_states == 1u,
          "gdn state pool follows the same rule (declared layer_types)");
    check(shelf::qwen35::make_qwen35_gdn_state_layout(interval_only).num_gdn_states == 6u,
          "gdn state pool counts interval-only configs over every layer");
}

static void test_context_valid() {
    Qwen35TextConfig c = base_config();

    auto single = make_qwen35_tensor_parallel_context(c, 1, 0);
    check(single.ok(), "tp_size=1 context builds");
    if (single.ok()) {
        const Qwen35TensorParallelContext& ctx = single.value();
        check(ctx.tp_size == 1 && ctx.tp_rank == 0, "tp=1 geometry");
        check(ctx.local_intermediate_size == 128, "tp=1 local intermediate");
        check(ctx.local_attention_heads == 4, "tp=1 local q heads");
        check(ctx.local_key_value_heads == 2, "tp=1 local kv heads");
        check(ctx.local_gdn_key_heads == 4, "tp=1 local gdn key heads");
        check(ctx.local_gdn_value_heads == 8, "tp=1 local gdn value heads");
        check(ctx.local_vocab_size == 128, "tp=1 local vocab");
        check(!ctx.vocab_parallel_lm_head, "tp=1 lm_head is replicated");
    }

    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        auto ctx = make_qwen35_tensor_parallel_context(c, 2, rank);
        check(ctx.ok(), "tp=2 context builds");
        if (!ctx.ok()) continue;
        check(ctx.value().tp_size == 2 && ctx.value().tp_rank == rank, "tp=2 geometry");
        check(ctx.value().local_intermediate_size == 64, "tp=2 local intermediate");
        check(ctx.value().local_attention_heads == 2, "tp=2 local q heads");
        check(ctx.value().local_key_value_heads == 1, "tp=2 local kv heads");
        check(ctx.value().local_gdn_key_heads == 2, "tp=2 local gdn key heads");
        check(ctx.value().local_gdn_value_heads == 4, "tp=2 local gdn value heads");
        check(ctx.value().local_vocab_size == 64, "tp=2 local vocab");
        check(ctx.value().vocab_parallel_lm_head, "tp=2 lm_head is vocab parallel");
    }

    Qwen35TextConfig tied = base_config();
    tied.tie_word_embeddings = true;
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        auto ctx = make_qwen35_tensor_parallel_context(tied, 2, rank);
        check(ctx.ok(), "tied tp=2 context builds");
        if (!ctx.ok()) continue;
        check(ctx.value().local_vocab_size == 128, "tied tp=2 local vocab");
        check(!ctx.value().vocab_parallel_lm_head, "tied tp=2 lm_head is replicated");
    }
}

static void test_context_invalid() {
    Qwen35TextConfig c = base_config();
    check(!make_qwen35_tensor_parallel_context(c, 0, 0).ok(), "tp_size=0 rejected");
    check(!make_qwen35_tensor_parallel_context(c, 2, 2).ok(), "tp_rank>=tp_size rejected");

    Qwen35TextConfig odd_i = base_config();
    odd_i.ffn.intermediate_size = 131;
    check(!make_qwen35_tensor_parallel_context(odd_i, 2, 0).ok(),
          "indivisible intermediate rejected");

    Qwen35TextConfig odd_q = base_config();
    odd_q.num_attention_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_q, 2, 0).ok(),
          "indivisible query heads rejected");

    Qwen35TextConfig odd_kv = base_config();
    odd_kv.num_key_value_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_kv, 2, 0).ok(),
          "indivisible kv heads rejected");

    Qwen35TextConfig odd_gdn_k = base_config();
    odd_gdn_k.linear_num_key_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_gdn_k, 2, 0).ok(),
          "indivisible gdn key heads rejected");

    Qwen35TextConfig odd_gdn_v = base_config();
    odd_gdn_v.linear_num_value_heads = 5;
    check(!make_qwen35_tensor_parallel_context(odd_gdn_v, 2, 0).ok(),
          "indivisible gdn value heads rejected");

    Qwen35TextConfig odd_vocab = base_config();
    odd_vocab.vocab_size = 129;
    check(!make_qwen35_tensor_parallel_context(odd_vocab, 2, 0).ok(),
          "indivisible vocab_size rejected");

    Qwen35TextConfig zero_vocab = base_config();
    zero_vocab.vocab_size = 0;
    check(!make_qwen35_tensor_parallel_context(zero_vocab, 2, 0).ok(),
          "zero vocab_size rejected");
}

namespace {

std::uint32_t add_node(PrimitiveGraph& graph, std::string name, bool combine,
                       std::uint32_t output_value, std::uint32_t local_cols = 0) {
    PrimitiveGraphNode node;
    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{64, 64};
    linear.weight_index = 0;
    node.node = linear;
    if (output_value != 0xFFFFFFFFu) {
        node.outputs.push_back(ValueId{output_value});
    }
    node.debug_name = std::move(name);
    node.tp_combine = combine ? 1u : 0u;
    node.tp_combine_local_cols = local_cols;
    graph.nodes.push_back(std::move(node));
    return static_cast<std::uint32_t>(graph.nodes.size() - 1);
}

void bind_value(Program& program, std::uint32_t value) {
    ValueBinding binding;
    binding.logical_value = ValueId{value};
    binding.storage = ValueStorage::WORKSPACE;
    binding.dtype = ValueDType::BF16;
    binding.row_domain = ValueRowDomain::TOKEN_ROWS;
    binding.slot = 0;
    binding.feature_count = 64;
    binding.row_stride = 64;
    binding.offset = 0;
    binding.bytes = 64u * 64u * 2u;
    program.values.push_back(binding);
}

}

static void test_schedule_build() {
    PrimitiveGraph graph;
    Program program;

    const std::uint32_t n0 = add_node(graph, "L0.input_norm", false, 10);
    const std::uint32_t n1 = add_node(graph, "L0.o_proj", true, 11);
    const std::uint32_t n2 = add_node(graph, "L0.mlp_gate", false, 12);
    const std::uint32_t n3 = add_node(graph, "L0.mlp_down", true, 13, 32);
    (void)n0;
    (void)n2;

    program.dispatches.resize(5);
    program.dispatch_source_node = {n0, n0, n1, n2, n3};
    bind_value(program, 11);
    bind_value(program, 13);

    auto schedule = build_tp_execution_schedule(graph, program);
    check(schedule.ok(), "schedule builds");
    if (!schedule.ok()) return;
    const auto& segments = schedule.value().segments;
    check(schedule.value().active(), "schedule is active");
    check(segments.size() == 2, "two barrier segments");
    if (segments.size() != 2) return;

    check(segments[0].dispatch_begin == 0 && segments[0].dispatch_end == 3,
          "first segment covers dispatches [0,3)");
    check(segments[0].barrier_after == TpBarrierKind::SumHidden,
          "first segment barrier after o_proj");
    check(segments[0].combine_value.id == 11, "first segment combine value");

    check(segments[1].dispatch_begin == 3 && segments[1].dispatch_end == 5,
          "second segment covers dispatches [3,5)");
    check(segments[1].barrier_after == TpBarrierKind::SumHidden,
          "second segment barrier after mlp_down");
    check(segments[1].combine_value.id == 13, "second segment combine value");
    check(segments[0].combine_local_cols == 0, "row parallel segment has no local width");
    check(segments[1].combine_local_cols == 32, "vocab parallel segment carries local width");

    Program contiguous = program;
    contiguous.dispatch_source_node = {n0, n0, n1, n2};
    auto short_map = build_tp_execution_schedule(graph, contiguous);
    check(!short_map.ok(), "source map size mismatch rejected");

    PrimitiveGraph no_combine_graph;
    Program plain;
    const std::uint32_t plain_node = add_node(no_combine_graph, "L0.input_norm", false, 20);
    plain.dispatches.resize(2);
    plain.dispatch_source_node = {plain_node, plain_node};
    auto plain_schedule = build_tp_execution_schedule(no_combine_graph, plain);
    check(plain_schedule.ok(), "schedule without combine nodes builds");
    if (plain_schedule.ok()) {
        check(plain_schedule.value().segments.size() == 1,
              "no combine nodes yields a single segment");
        check(plain_schedule.value().segments[0].barrier_after == TpBarrierKind::None,
              "no barrier without combine nodes");
        check(plain_schedule.value().segments[0].dispatch_end == 2,
              "single segment covers all dispatches");
    }

    PrimitiveGraph bad_index;
    Program index_program;
    index_program.dispatches.resize(1);
    index_program.dispatch_source_node = {7};
    auto bad = build_tp_execution_schedule(bad_index, index_program);
    check(!bad.ok(), "out-of-range source node rejected");

    PrimitiveGraph empty_out;
    PrimitiveGraphNode node;
    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{64, 64};
    node.node = linear;
    node.tp_combine = 1;
    empty_out.nodes.push_back(node);
    Program empty_program;
    empty_program.dispatches.resize(1);
    empty_program.dispatch_source_node = {0};
    auto no_output = build_tp_execution_schedule(empty_out, empty_program);
    check(!no_output.ok(), "combine node without outputs rejected");

    PrimitiveGraph unbound;
    const std::uint32_t unbound_node = add_node(unbound, "L0.o_proj", true, 99);
    Program unbound_program;
    unbound_program.dispatches.resize(1);
    unbound_program.dispatch_source_node = {unbound_node};
    auto unbound_result = build_tp_execution_schedule(unbound, unbound_program);
    check(!unbound_result.ok(), "combine value without binding rejected");
}

static void test_vocab_parallel_program() {
    constexpr std::uint32_t kGlobalVocab = 248320;
    constexpr std::uint32_t kLocalVocab = 124160;
    constexpr std::uint32_t kHidden = 5120;

    PrimitiveGraph graph;
    const ValueId normed = graph.alloc_value(kHidden, ValueDType::BF16,
                                             ValueRowDomain::OUTPUT_ROWS);
    const ValueId logits = graph.alloc_value(kGlobalVocab, ValueDType::F32,
                                             ValueRowDomain::OUTPUT_ROWS);
    graph.external_inputs.push_back(normed);
    graph.external_outputs.push_back(logits);

    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{kLocalVocab, kHidden};
    linear.weight_index = graph.register_weight(linear.shape);
    linear.out_dtype = ValueDType::F32;
    auto& node = graph.add_node(linear);
    node.inputs = {normed};
    node.outputs = {logits};
    node.debug_name = "logits";
    node.tp_combine = 1;
    node.tp_combine_local_cols = kLocalVocab;

    WeightSlot slot{};
    slot.rows = kLocalVocab;
    slot.cols = kHidden;
    slot.encoding = static_cast<std::uint8_t>(shelf::weights::MatrixEncoding::Bf16);
    slot.bytes = static_cast<std::uint64_t>(kLocalVocab) * kHidden * 2u;
    slot.output_col_offset = kLocalVocab;
    slot.output_col_total = kGlobalVocab;

    WeightTableView table{&slot, 1};
    ProgramBuildOptions opts{};
    opts.max_output_rows = 4;

    auto program = build_program(graph, table, RowBucket::R16, ExecutionClass::DECODE, opts);
    check(program.ok(), "vocab parallel program builds");
    if (program.ok()) {
        auto schedule = build_tp_execution_schedule(graph, program.value());
        check(schedule.ok(), "vocab parallel schedule builds");
        if (schedule.ok()) {
            const auto& segments = schedule.value().segments;
            check(segments.size() == 1, "vocab parallel schedule has one segment");
            if (segments.size() == 1) {
                check(segments[0].barrier_after == TpBarrierKind::SumHidden,
                      "vocab parallel segment sums the logits");
                check(segments[0].combine_value.id == logits.id,
                      "vocab parallel combine value is logits");
                check(segments[0].combine_local_cols == kLocalVocab,
                      "vocab parallel segment carries the local vocab width");
            }
        }
    }

    WeightSlot rank0_slot = slot;
    rank0_slot.output_col_offset = 0;
    WeightTableView rank0_table{&rank0_slot, 1};
    auto rank0_program = build_program(graph, rank0_table, RowBucket::R16,
                                       ExecutionClass::DECODE, opts);
    check(rank0_program.ok(), "rank0 column offset accepted");

    WeightSlot bad_slot = slot;
    bad_slot.output_col_total = 0;
    WeightTableView bad_table{&bad_slot, 1};
    auto bad_program = build_program(graph, bad_table, RowBucket::R16,
                                     ExecutionClass::DECODE, opts);
    check(!bad_program.ok(), "unsharded width with declared logits width rejected");

    WeightSlot short_slot = slot;
    short_slot.output_col_offset = kLocalVocab + 1u;
    WeightTableView short_table{&short_slot, 1};
    auto short_program = build_program(graph, short_table, RowBucket::R16,
                                       ExecutionClass::DECODE, opts);
    check(!short_program.ok(), "oversized column offset rejected");
}

int main() {
    check_topology_counter();
    test_context_valid();
    test_context_invalid();
    test_schedule_build();
    test_vocab_parallel_program();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
