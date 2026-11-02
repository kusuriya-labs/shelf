#include <shelf/models/qwen35/weights/tensor_parallel_plan.h>
#include <shelf/weights/tensor_partition.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using shelf::Result;
using shelf::Status;
using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;
using shelf::weights::validate_tensor_partition;
using shelf::weights::validate_tensor_partition_plan;
using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::build_qwen35_tensor_partition_plan;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static Qwen35TextConfig mini_config() {
    Qwen35TextConfig c;
    c.vocab_size = 128;
    c.hidden_size = 64;
    c.intermediate_size = 128;
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
    c.tie_word_embeddings = false;
    c.layer_types = {0, 1, 0, 1};
    return c;
}

static Qwen35TextConfig qwen38_27b_config() {
    Qwen35TextConfig c;
    c.vocab_size = 248320;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.full_attention_interval = 4;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.attention_head_dim = 256;
    c.linear_num_key_heads = 16;
    c.linear_num_value_heads = 48;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.tie_word_embeddings = false;
    for (std::size_t l = 0; l < c.num_hidden_layers; ++l) {
        c.layer_types.push_back((l % c.full_attention_interval ==
                                 c.full_attention_interval - 1) ? 1 : 0);
    }
    return c;
}

static const TensorPartitionDesc* find(const TensorPartitionPlan& plan,
                                       const std::string& name) {
    auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

static void check_ranges(const std::string& tag, const TensorPartitionDesc* d,
                         std::vector<TensorPartitionRange> want) {
    if (d == nullptr) {
        fail(tag + " missing entry");
        return;
    }
    if (d->ranges != want) {
        fail(tag + " ranges differ");
        return;
    }
}

static void check_axis_shape(const std::string& tag, const TensorPartitionDesc* d,
                             int32_t axis, std::vector<int64_t> shape) {
    if (d == nullptr) {
        fail(tag + " missing entry");
        return;
    }
    if (d->axis != axis) fail(tag + " axis mismatch");
    if (d->global_shape != shape) fail(tag + " global_shape mismatch");
}

static void test_basic_args() {
    Qwen35TextConfig c = mini_config();
    auto single = build_qwen35_tensor_partition_plan(c, 1, 0);
    check(single.ok(), "tp_size=1 plan builds");
    if (single.ok()) {
        check(single.value().tensors.empty(), "tp_size=1 plan is fully replicated");
    }
    check(!build_qwen35_tensor_partition_plan(c, 0, 0).ok(), "tp_size=0 rejected");
    check(!build_qwen35_tensor_partition_plan(c, 2, 2).ok(), "tp_rank>=tp_size rejected");
    check(!build_qwen35_tensor_partition_plan(c, 3, 0).ok(),
          "indivisible geometry rejected");
    Qwen35TextConfig empty = mini_config();
    empty.num_hidden_layers = 0;
    check(!build_qwen35_tensor_partition_plan(empty, 2, 0).ok(),
          "zero layers rejected");

    Qwen35TextConfig odd_vocab = mini_config();
    odd_vocab.vocab_size = 129;
    check(!build_qwen35_tensor_partition_plan(odd_vocab, 2, 0).ok(),
          "indivisible vocab_size rejected");
}

static void test_lm_head_fail_open() {
    Qwen35TextConfig tied = mini_config();
    tied.tie_word_embeddings = true;
    auto tied_plan = build_qwen35_tensor_partition_plan(tied, 2, 0);
    check(tied_plan.ok(), "tied plan builds");
    if (tied_plan.ok()) {
        check(tied_plan.value().tensors.size() == 36, "tied plan entry count");
        check(find(tied_plan.value(), "model.language_model.lm_head.weight") == nullptr,
              "tied lm_head is not partitioned");
        check(find(tied_plan.value(), "lm_head.weight") == nullptr,
              "tied lm_head alias is not partitioned");
    }

    Qwen35TextConfig unsharded = mini_config();
    auto unsharded_plan = build_qwen35_tensor_partition_plan(unsharded, 2, 0, false);
    check(unsharded_plan.ok(), "unsharded plan builds");
    if (unsharded_plan.ok()) {
        check(unsharded_plan.value().tensors.size() == 36,
              "unsharded lm_head plan entry count");
        check(find(unsharded_plan.value(), "lm_head.weight") == nullptr,
              "unsharded lm_head is not partitioned");
    }
}

static void test_mini_plan_rank(uint32_t rank) {
    Qwen35TextConfig c = mini_config();
    auto plan_result = build_qwen35_tensor_partition_plan(c, 2, rank);
    check(plan_result.ok(), "mini tp=2 plan builds");
    if (!plan_result.ok()) return;
    const TensorPartitionPlan& plan = plan_result.value();
    check(plan.tp_size == 2 && plan.tp_rank == rank, "mini plan geometry");
    check(plan.tensors.size() == 38, "mini plan entry count");
    check(validate_tensor_partition_plan(plan).ok(), "mini plan validates");

    const std::string p0 = "model.language_model.layers.0.";
    const std::string p1 = "model.language_model.layers.1.";
    const uint64_t mlp_off = rank * 64;
    const TensorPartitionRange mlp_range{mlp_off, 64};
    const TensorPartitionRange vocab_range{rank * 64, 64};

    check_axis_shape("lm_head", find(plan, "model.language_model.lm_head.weight"), 0,
                     {128, 64});
    check_ranges("lm_head", find(plan, "model.language_model.lm_head.weight"),
                 {vocab_range});
    check_axis_shape("lm_head alias", find(plan, "lm_head.weight"), 0, {128, 64});
    check_ranges("lm_head alias", find(plan, "lm_head.weight"), {vocab_range});

    check_axis_shape("gate", find(plan, p0 + "mlp.gate_proj.weight"), 0, {128, 64});
    check_ranges("gate", find(plan, p0 + "mlp.gate_proj.weight"), {mlp_range});
    check_axis_shape("up", find(plan, p0 + "mlp.up_proj.weight"), 0, {128, 64});
    check_ranges("up", find(plan, p0 + "mlp.up_proj.weight"), {mlp_range});
    check_axis_shape("down", find(plan, p0 + "mlp.down_proj.weight"), 1, {64, 128});
    check_ranges("down", find(plan, p0 + "mlp.down_proj.weight"), {mlp_range});

    const TensorPartitionDesc* gate = find(plan, p0 + "mlp.gate_proj.weight");
    const TensorPartitionDesc* up = find(plan, p0 + "mlp.up_proj.weight");
    const TensorPartitionDesc* down = find(plan, p0 + "mlp.down_proj.weight");
    if (gate && up && down) {
        check(gate->ranges == up->ranges && gate->ranges == down->ranges,
              "gate/up/down share the intermediate range");
    }

    const TensorPartitionDesc* q = find(plan, p1 + "self_attn.q_proj.weight");
    check_axis_shape("q_proj", q, 0, {128, 64});
    check_ranges("q_proj", q, {TensorPartitionRange{rank * 64, 64}});
    if (q != nullptr) {
        constexpr uint64_t head_pair_rows = 2 * 16;
        check((q->ranges[0].global_offset % head_pair_rows) == 0,
              "q_proj offset keeps q/gate head pairs");
        check((q->ranges[0].extent % head_pair_rows) == 0,
              "q_proj extent keeps q/gate head pairs");
    }
    check_axis_shape("k_proj", find(plan, p1 + "self_attn.k_proj.weight"), 0, {32, 64});
    check_ranges("k_proj", find(plan, p1 + "self_attn.k_proj.weight"),
                 {TensorPartitionRange{rank * 16, 16}});
    check_ranges("v_proj", find(plan, p1 + "self_attn.v_proj.weight"),
                 {TensorPartitionRange{rank * 16, 16}});
    check_axis_shape("o_proj", find(plan, p1 + "self_attn.o_proj.weight"), 1, {64, 64});
    check_ranges("o_proj", find(plan, p1 + "self_attn.o_proj.weight"),
                 {TensorPartitionRange{rank * 32, 32}});

    const TensorPartitionDesc* qkv = find(plan, p0 + "linear_attn.in_proj_qkv.weight");
    check_axis_shape("gdn qkv", qkv, 0, {128, 64});
    check_ranges("gdn qkv", qkv,
                 {TensorPartitionRange{rank * 16, 16},
                  TensorPartitionRange{32 + rank * 16, 16},
                  TensorPartitionRange{64 + rank * 32, 32}});
    const TensorPartitionDesc* conv = find(plan, p0 + "linear_attn.conv1d.weight");
    check_axis_shape("gdn conv", conv, 0, {128, 1, 4});
    if (qkv != nullptr && conv != nullptr) {
        check(qkv->ranges == conv->ranges, "gdn conv shares the QKV segmented ranges");
    }
    check_axis_shape("gdn z", find(plan, p0 + "linear_attn.in_proj_z.weight"), 0, {64, 64});
    check_ranges("gdn z", find(plan, p0 + "linear_attn.in_proj_z.weight"),
                 {TensorPartitionRange{rank * 32, 32}});
    check_ranges("gdn a", find(plan, p0 + "linear_attn.in_proj_a.weight"),
                 {TensorPartitionRange{rank * 4, 4}});
    check_ranges("gdn b", find(plan, p0 + "linear_attn.in_proj_b.weight"),
                 {TensorPartitionRange{rank * 4, 4}});
    check_axis_shape("gdn out", find(plan, p0 + "linear_attn.out_proj.weight"), 1, {64, 64});
    check_ranges("gdn out", find(plan, p0 + "linear_attn.out_proj.weight"),
                 {TensorPartitionRange{rank * 32, 32}});
    check_ranges("dt_bias", find(plan, p0 + "linear_attn.dt_bias"),
                 {TensorPartitionRange{rank * 4, 4}});
    check_ranges("A_log", find(plan, p0 + "linear_attn.A_log"),
                 {TensorPartitionRange{rank * 4, 4}});

    for (const auto& kv : plan.tensors) {
        check(validate_tensor_partition(kv.second).ok(), "entry validates: " + kv.first);
        const bool vocab_shard =
            kv.first == "model.language_model.lm_head.weight" ||
            kv.first == "lm_head.weight";
        const bool replicated_shape =
            !vocab_shard &&
            (kv.first.find("norm") != std::string::npos ||
             kv.first.find("embed_tokens") != std::string::npos ||
             kv.first.find("lm_head") != std::string::npos ||
             kv.first.rfind("mtp.", 0) == 0);
        if (replicated_shape) fail("replicated tensor has a partition entry: " + kv.first);
    }
}

static void test_gdn_qkv_coverage() {
    Qwen35TextConfig c = mini_config();
    auto p0 = build_qwen35_tensor_partition_plan(c, 2, 0);
    auto p1 = build_qwen35_tensor_partition_plan(c, 2, 1);
    if (!p0.ok() || !p1.ok()) {
        fail("coverage plans build");
        return;
    }
    const std::string name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    const TensorPartitionDesc* d0 = find(p0.value(), name);
    const TensorPartitionDesc* d1 = find(p1.value(), name);
    if (d0 == nullptr || d1 == nullptr) {
        fail("coverage entries present");
        return;
    }
    std::vector<TensorPartitionRange> all = d0->ranges;
    all.insert(all.end(), d1->ranges.begin(), d1->ranges.end());
    std::sort(all.begin(), all.end(),
              [](const TensorPartitionRange& a, const TensorPartitionRange& b) {
                  return a.global_offset < b.global_offset;
              });
    check(all.size() == 6, "coverage has six ranges");
    uint64_t cursor = 0;
    for (const TensorPartitionRange& r : all) {
        if (r.global_offset != cursor) {
            fail("coverage gap or overlap at " + std::to_string(cursor));
            break;
        }
        cursor += r.extent;
    }
    check(cursor == 128, "coverage spans the full conv_features axis");
}

static void test_qwen38_27b_plan() {
    Qwen35TextConfig c = qwen38_27b_config();
    auto p0 = build_qwen35_tensor_partition_plan(c, 2, 0);
    auto p1 = build_qwen35_tensor_partition_plan(c, 2, 1);
    check(p0.ok() && p1.ok(), "qwen3.8-27b tp=2 plan builds");
    if (!p0.ok() || !p1.ok()) return;
    const TensorPartitionPlan& rank0 = p0.value();
    const TensorPartitionPlan& rank1 = p1.value();
    check(rank0.tensors.size() == 642, "qwen3.8-27b entry count");
    check(validate_tensor_partition_plan(rank0).ok(), "qwen3.8-27b rank0 plan validates");
    check(validate_tensor_partition_plan(rank1).ok(), "qwen3.8-27b rank1 plan validates");

    check_axis_shape("27b lm_head rank0", find(rank0, "lm_head.weight"), 0, {248320, 5120});
    check_ranges("27b lm_head rank0", find(rank0, "lm_head.weight"),
                 {TensorPartitionRange{0, 124160}});
    check_ranges("27b lm_head rank1", find(rank1, "lm_head.weight"),
                 {TensorPartitionRange{124160, 124160}});
    check_ranges("27b lm_head canonical rank0",
                 find(rank0, "model.language_model.lm_head.weight"),
                 {TensorPartitionRange{0, 124160}});

    const std::string p = "model.language_model.layers.0.";
    check_ranges("27b gate rank0", find(rank0, p + "mlp.gate_proj.weight"),
                 {TensorPartitionRange{0, 8704}});
    check_ranges("27b gate rank1", find(rank1, p + "mlp.gate_proj.weight"),
                 {TensorPartitionRange{8704, 8704}});
    check_axis_shape("27b gate", find(rank0, p + "mlp.gate_proj.weight"), 0,
                     {17408, 5120});
    check_ranges("27b down rank0", find(rank0, p + "mlp.down_proj.weight"),
                 {TensorPartitionRange{0, 8704}});
    check_axis_shape("27b down", find(rank0, p + "mlp.down_proj.weight"), 1,
                     {5120, 17408});

    const std::string f = "model.language_model.layers.3.";
    const TensorPartitionDesc* q = find(rank0, f + "self_attn.q_proj.weight");
    check_axis_shape("27b q_proj", q, 0, {12288, 5120});
    check_ranges("27b q_proj rank0", q, {TensorPartitionRange{0, 6144}});
    check_ranges("27b q_proj rank1", find(rank1, f + "self_attn.q_proj.weight"),
                 {TensorPartitionRange{6144, 6144}});
    if (q != nullptr) {
        check((q->ranges[0].extent % (2u * 256u)) == 0,
              "27b q_proj extent keeps q/gate head pairs");
    }
    check_ranges("27b k_proj rank0", find(rank0, f + "self_attn.k_proj.weight"),
                 {TensorPartitionRange{0, 512}});
    check_ranges("27b k_proj rank1", find(rank1, f + "self_attn.k_proj.weight"),
                 {TensorPartitionRange{512, 512}});
    check_ranges("27b v_proj rank0", find(rank0, f + "self_attn.v_proj.weight"),
                 {TensorPartitionRange{0, 512}});
    check_axis_shape("27b o_proj", find(rank0, f + "self_attn.o_proj.weight"), 1,
                     {5120, 6144});
    check_ranges("27b o_proj rank0", find(rank0, f + "self_attn.o_proj.weight"),
                 {TensorPartitionRange{0, 3072}});
    check_ranges("27b o_proj rank1", find(rank1, f + "self_attn.o_proj.weight"),
                 {TensorPartitionRange{3072, 3072}});

    const TensorPartitionDesc* qkv = find(rank0, p + "linear_attn.in_proj_qkv.weight");
    check_axis_shape("27b qkv", qkv, 0, {10240, 5120});
    check_ranges("27b qkv rank0", qkv,
                 {TensorPartitionRange{0, 1024},
                  TensorPartitionRange{2048, 1024},
                  TensorPartitionRange{4096, 3072}});
    check_ranges("27b qkv rank1", find(rank1, p + "linear_attn.in_proj_qkv.weight"),
                 {TensorPartitionRange{1024, 1024},
                  TensorPartitionRange{3072, 1024},
                  TensorPartitionRange{7168, 3072}});
    check_axis_shape("27b conv", find(rank0, p + "linear_attn.conv1d.weight"), 0,
                     {10240, 1, 4});
    check_ranges("27b conv rank0", find(rank0, p + "linear_attn.conv1d.weight"),
                 {TensorPartitionRange{0, 1024},
                  TensorPartitionRange{2048, 1024},
                  TensorPartitionRange{4096, 3072}});
    check_ranges("27b dt_bias rank0", find(rank0, p + "linear_attn.dt_bias"),
                 {TensorPartitionRange{0, 24}});
    check_ranges("27b dt_bias rank1", find(rank1, p + "linear_attn.dt_bias"),
                 {TensorPartitionRange{24, 24}});
    check_ranges("27b A_log rank0", find(rank0, p + "linear_attn.A_log"),
                 {TensorPartitionRange{0, 24}});

    for (const auto& kv : rank0.tensors) {
        check(validate_tensor_partition(kv.second).ok(),
              "27b entry validates: " + kv.first);
        const bool vocab_shard =
            kv.first == "model.language_model.lm_head.weight" ||
            kv.first == "lm_head.weight";
        const bool replicated_shape =
            !vocab_shard &&
            (kv.first.find("norm") != std::string::npos ||
             kv.first.find("embed_tokens") != std::string::npos ||
             kv.first.find("lm_head") != std::string::npos ||
             kv.first.rfind("mtp.", 0) == 0);
        if (replicated_shape) fail("27b replicated tensor partitioned: " + kv.first);
    }
}

static void test_layer_type_fallback() {
    Qwen35TextConfig c = mini_config();
    c.layer_types.clear();
    auto plan = build_qwen35_tensor_partition_plan(c, 2, 0);
    check(plan.ok(), "layer type fallback plan builds");
    if (plan.ok()) {
        check(plan.value().tensors.size() == 38,
              "layer type fallback matches explicit layer types");
    }
}

int main() {
    test_basic_args();
    test_lm_head_fail_open();
    test_mini_plan_rank(0);
    test_mini_plan_rank(1);
    test_gdn_qkv_coverage();
    test_qwen38_27b_plan();
    test_layer_type_fallback();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
