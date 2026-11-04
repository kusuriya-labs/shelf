#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <cstdio>
#include <string>

using shelf::qwen35moe::build_moe_expert_group_plan;
using shelf::qwen35moe::MoeExpertProjection;
using shelf::qwen35moe::moe_expert_tensor_count;
using shelf::qwen35moe::moe_expert_tensor_name;
using shelf::qwen35moe::Qwen35MoeTextConfig;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

// Measured text_config of models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4.
Qwen35MoeTextConfig moe_config() {
    Qwen35MoeTextConfig c;
    c.vocab_size = 248320;
    c.hidden_size = 2048;
    c.num_hidden_layers = 40;
    c.ffn.num_experts = 256;
    c.ffn.num_experts_per_tok = 8;
    c.ffn.moe_intermediate_size = 512;
    c.ffn.shared_expert_intermediate_size = 512;
    c.num_attention_heads = 16;
    c.num_key_value_heads = 2;
    c.attention_head_dim = 256;
    return c;
}

}  // namespace

int main() {
    const Qwen35MoeTextConfig cfg = moe_config();

    check(moe_expert_tensor_count(cfg) == 40u * 256u * 3u * 2u,
          "routed expert tensor count is 61440");
    check(moe_expert_tensor_count(cfg) == 61440u, "routed expert tensor count value");

    check(moe_expert_tensor_name(12, 7, "gate_proj", true) ==
              "model.language_model.layers.12.mlp.experts.7.gate_proj.weight_scale",
          "expert scale tensor name");
    check(moe_expert_tensor_name(0, 0, "down_proj", false) ==
              "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
          "expert codes tensor name");

    auto plan = build_moe_expert_group_plan(cfg, 0);
    check(plan.ok(), "build_moe_expert_group_plan(0)");
    if (!plan.ok()) {
        std::printf("FAIL %s\n", plan.status().message().c_str());
        return 1;
    }
    const auto& p = plan.value();

    check(p.layer == 0u && p.experts == 256u && p.scale_block_k == 32u, "plan header");
    check(p.w13_rows == 1024u && p.w13_cols == 2048u && p.w13_k_padded == 2048u, "w13 shape");
    check(p.w13_gate_rows == 512u && p.w13_up_rows == 512u, "w13 gate rows come first");
    check(p.w2_rows == 2048u && p.w2_cols == 512u && p.w2_k_padded == 512u, "w2 shape");

    check(p.w13_codes_row_stride_bytes == 64u * 256u, "w13 native codes row stride");
    check(p.w13_scales_row_stride_bytes == 64u, "w13 native scale row stride");
    check(p.w2_codes_row_stride_bytes == 16u * 256u, "w2 native codes row stride");
    check(p.w2_scales_row_stride_bytes == 16u, "w2 native scale row stride");

    check(p.w13_codes_batch_stride_bytes == 1048576ULL, "w13 codes batch stride");
    check(p.w13_scales_batch_stride_bytes == 65536u, "w13 scales batch stride");
    check(p.w2_codes_batch_stride_bytes == 524288u, "w2 codes batch stride");
    check(p.w2_scales_batch_stride_bytes == 32768u, "w2 scales batch stride");

    check(p.w13_codes_bytes == 256ULL * 1048576ULL, "w13 codes plane size");
    check(p.w13_scales_bytes == 256ULL * 65536ULL, "w13 scales plane size");
    check(p.w2_codes_bytes == 256ULL * 524288ULL, "w2 codes plane size");
    check(p.w2_scales_bytes == 256ULL * 32768ULL, "w2 scales plane size");
    check(p.w13_row_ref_bytes_per_expert == 1024u && p.w2_row_ref_bytes_per_expert == 2048u,
          "row reference plane size per expert");

    // Resident size anchor: the checkpoint's routed-expert payload measured in
    // docs/research/20261103/qwen36_35b_a3b_moe_structure.md (427,819,008 B per layer,
    // fully contiguous, 17,112,760,320 B total).
    const uint64_t layer_span = p.w13_codes_bytes + p.w13_scales_bytes + p.w2_codes_bytes +
                                p.w2_scales_bytes;
    check(layer_span == 427819008ULL, "one layer of experts is 427819008 bytes");
    check(layer_span * cfg.num_hidden_layers == 17112760320ULL,
          "40 layers of experts match the checkpoint expert payload");

    // Fusing gate and up is stride-equivalent to keeping them apart.
    const uint64_t unfused_projection_stride = (512ULL / 16ULL) * p.w13_codes_row_stride_bytes;
    check(p.w13_codes_batch_stride_bytes == 2u * unfused_projection_stride,
          "fused w13 batch stride equals two unfused projections");

    check(p.sources.size() == 256u * 3u, "sources enumerate every expert projection");
    bool order_ok = true;
    bool range_ok = true;
    const MoeExpertProjection expected[3] = {
        MoeExpertProjection::Down, MoeExpertProjection::Gate, MoeExpertProjection::Up,
    };
    for (std::size_t i = 0; i < p.sources.size(); ++i) {
        const auto& src = p.sources[i];
        const uint32_t e = static_cast<uint32_t>(i / 3u);
        if (src.expert != e || src.projection != expected[i % 3u])
            order_ok = false;
        if (src.projection == MoeExpertProjection::Gate) {
            if (src.row_offset != 0u || src.rows != 512u)
                range_ok = false;
        } else if (src.projection == MoeExpertProjection::Up) {
            if (src.row_offset != 512u || src.rows != 512u)
                range_ok = false;
        } else if (src.row_offset != 0u || src.rows != 2048u) {
            range_ok = false;
        }
    }
    check(order_ok, "sources are expert-major in checkpoint file order");
    check(range_ok, "sources carry the destination row ranges (gate first, then up)");
    check(p.sources.back().name == "model.language_model.layers.0.mlp.experts.255.up_proj.weight",
          "last source name");

    {
        Qwen35MoeTextConfig dense = moe_config();
        dense.ffn.num_experts = 0;
        dense.ffn.num_experts_per_tok = 0;
        check(!build_moe_expert_group_plan(dense, 0).ok(), "a dense config has no expert group");
        check(moe_expert_tensor_count(dense) == 0u, "a dense config has no expert tensors");
    }
    {
        check(!build_moe_expert_group_plan(cfg, cfg.num_hidden_layers).ok(),
              "layer out of range is rejected");
        Qwen35MoeTextConfig odd = moe_config();
        odd.hidden_size = 2064;  // a multiple of 16, not of 64
        auto res = build_moe_expert_group_plan(odd, 0);
        check(res.ok() && res.value().w13_k_padded == 2112u,
              "k_padded rounds up to the native 64 K alignment");
        odd.hidden_size = 2049;  // expert output rows are not a multiple of 16
        check(!build_moe_expert_group_plan(odd, 0).ok(),
              "expert output rows that are not a multiple of 16 are rejected");
    }

    std::printf("mxfp4 moe plan: layer span %llu bytes, w13 %ux%u, w2 %ux%u, sources %zu\n",
                static_cast<unsigned long long>(layer_span), p.w13_rows, p.w13_cols, p.w2_rows,
                p.w2_cols, p.sources.size());
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
