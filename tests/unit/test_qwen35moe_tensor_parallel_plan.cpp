#include <shelf/models/qwen35/weights/tensor_parallel_plan.h>
#include <shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/weights/tensor_partition.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;
using shelf::weights::validate_tensor_partition_plan;
using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::build_qwen35_tensor_partition_plan;
using shelf::qwen35moe::Qwen35MoeTextConfig;
using shelf::qwen35moe::build_qwen35moe_tensor_partition_plan;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

const char* kLayer = "model.language_model.layers.0.";

Qwen35MoeTextConfig moe_config() {
    Qwen35MoeTextConfig c;
    c.vocab_size = 128;
    c.hidden_size = 64;
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
    c.ffn.num_experts = 8;
    c.ffn.num_experts_per_tok = 2;
    c.ffn.moe_intermediate_size = 32;
    c.ffn.shared_expert_intermediate_size = 48;
    return c;
}

const TensorPartitionDesc* find(const TensorPartitionPlan& plan, const std::string& name) {
    auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

void check_entry(const TensorPartitionPlan& plan, const std::string& name, int32_t axis,
                 std::vector<int64_t> global_shape, std::vector<TensorPartitionRange> ranges,
                 const std::string& tag) {
    const TensorPartitionDesc* d = find(plan, name);
    if (d == nullptr) {
        ++g_fail;
        std::printf("FAIL %s: no entry for %s\n", tag.c_str(), name.c_str());
        return;
    }
    if (d->axis != axis || d->global_shape != global_shape || d->ranges != ranges) {
        ++g_fail;
        std::printf("FAIL %s: %s differs\n", tag.c_str(), name.c_str());
    }
    if (d->index != plan.tp_rank || d->count != plan.tp_size) {
        ++g_fail;
        std::printf("FAIL %s: %s carries the wrong rank\n", tag.c_str(), name.c_str());
    }
}

}  // namespace

int main() {
    const Qwen35MoeTextConfig cfg = moe_config();

    {
        auto solo = build_qwen35moe_tensor_partition_plan(cfg, 1, 0);
        check(solo.ok() && solo.value().tensors.empty(),
              "tp_size 1 builds a MoE plan with no partition entries");
    }

    {
        auto p0 = build_qwen35moe_tensor_partition_plan(cfg, 2, 0);
        auto p1 = build_qwen35moe_tensor_partition_plan(cfg, 2, 1);
        check(p0.ok() && p1.ok(), "a MoE 2-way plan builds for both ranks");
        if (!p0.ok() || !p1.ok()) {
            std::printf("%s\n%s\n", p0.status().message().c_str(), p1.status().message().c_str());
            return 1;
        }
        const TensorPartitionPlan& r0 = p0.value();
        const TensorPartitionPlan& r1 = p1.value();
        check(validate_tensor_partition_plan(r0).ok() && validate_tensor_partition_plan(r1).ok(),
              "both rank plans validate");

        // The shared expert divides along its intermediate axis, the way a dense FFN does.
        check_entry(r0, std::string(kLayer) + "mlp.shared_expert.gate_proj.weight", 0, {48, 64},
                    {{0, 24}}, "rank 0 shared gate_proj");
        check_entry(r1, std::string(kLayer) + "mlp.shared_expert.gate_proj.weight", 0, {48, 64},
                    {{24, 24}}, "rank 1 shared gate_proj");
        check_entry(r0, std::string(kLayer) + "mlp.shared_expert.up_proj.weight", 0, {48, 64},
                    {{0, 24}}, "rank 0 shared up_proj");
        check_entry(r1, std::string(kLayer) + "mlp.shared_expert.down_proj.weight", 1, {64, 48},
                    {{24, 24}}, "rank 1 shared down_proj splits the K axis");

        // The router and the shared-expert gate stay whole: cutting the router logits would
        // make the ranks select different experts.
        check(find(r0, std::string(kLayer) + "mlp.gate.weight") == nullptr,
              "the router gate is not partitioned");
        check(find(r0, std::string(kLayer) + "mlp.shared_expert_gate.weight") == nullptr,
              "the shared-expert gate is not partitioned");

        // No plan entry describes the routed expert planes: they split along the expert
        // axis, which MoeExpertGroupPlan carries.
        check(find(r0, std::string(kLayer) + "mlp.experts.0.gate_proj.weight") == nullptr,
              "the routed experts are not part of the matrix partition plan");

        // The backbone entries are the same ones a dense plan emits for the same geometry.
        Qwen35TextConfig dense;
        static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(dense) = cfg;
        dense.ffn.intermediate_size = 128;
        auto dense_plan = build_qwen35_tensor_partition_plan(dense, 2, 0);
        check(dense_plan.ok(), "the dense plan for the same backbone geometry builds");
        if (dense_plan.ok()) {
            const char* body[] = {
                "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                "linear_attn.out_proj.weight", "linear_attn.conv1d.weight",
                "linear_attn.dt_bias", "linear_attn.A_log",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            };
            bool same = true;
            for (std::size_t l = 0; l < cfg.num_hidden_layers; ++l) {
                const std::string prefix = "model.language_model.layers." + std::to_string(l) + ".";
                for (const char* suffix : body) {
                    const TensorPartitionDesc* a = find(r0, prefix + suffix);
                    const TensorPartitionDesc* b = find(dense_plan.value(), prefix + suffix);
                    if ((a == nullptr) != (b == nullptr)) same = false;
                    else if (a != nullptr && *a != *b) same = false;
                }
            }
            check(same, "the MoE and dense plans partition the hybrid backbone identically");
        }

        // The vocab-parallel lm_head is shared with the dense path.
        check_entry(r0, "model.language_model.lm_head.weight", 0, {128, 64}, {{0, 64}},
                    "rank 0 lm_head");
        check_entry(r1, "lm_head.weight", 0, {128, 64}, {{64, 64}}, "rank 1 lm_head");

        Qwen35MoeTextConfig tied = moe_config();
        tied.tie_word_embeddings = true;
        auto no_head = build_qwen35moe_tensor_partition_plan(tied, 2, 0);
        check(no_head.ok() && find(no_head.value(), "lm_head.weight") == nullptr,
              "a tied checkpoint gets no lm_head partition");

        Qwen35MoeTextConfig odd = moe_config();
        odd.ffn.num_experts = 7;
        check(!build_qwen35moe_tensor_partition_plan(odd, 2, 0).ok(),
              "an expert count that does not divide by tp_size is rejected");
        odd = moe_config();
        odd.ffn.shared_expert_intermediate_size = 49;
        check(!build_qwen35moe_tensor_partition_plan(odd, 2, 0).ok(),
              "a shared expert width that does not divide by tp_size is rejected");
        odd = moe_config();
        odd.num_key_value_heads = 3;
        check(!build_qwen35moe_tensor_partition_plan(odd, 2, 0).ok(),
              "a kv head count that does not divide by tp_size is rejected");
        check(!build_qwen35moe_tensor_partition_plan(cfg, 2, 2).ok(),
              "tp_rank out of range is rejected");
        check(!build_qwen35moe_tensor_partition_plan(cfg, 0, 0).ok(), "tp_size 0 is rejected");
    }

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
