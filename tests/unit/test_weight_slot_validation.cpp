#include <shelf/inference/execution/weight_slot_validation.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

namespace RT = ::shelf::runtime;
namespace Q = ::shelf::quantization::mxfp4;
namespace MW = ::shelf::weights;

alignas(64) unsigned char g_backing[64];
const void* g_ptr = g_backing;

// Measured text_config of models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4.
shelf::qwen35moe::Qwen35MoeTextConfig moe_config() {
    shelf::qwen35moe::Qwen35MoeTextConfig c;
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

RT::WeightSlot mxfp4_slot(uint32_t rows, uint32_t cols, uint32_t k_padded,
                          uint32_t experts, uint64_t codes_batch,
                          uint64_t scales_batch) {
    RT::WeightSlot s{};
    s.rows = rows;
    s.cols = cols;
    s.k_padded = k_padded;
    s.encoding = static_cast<uint8_t>(MW::MatrixEncoding::Mxfp4);
    s.compute_spec = static_cast<uint8_t>(MW::encoding_to_compute_spec(MW::MatrixEncoding::Mxfp4));
    s.codes = g_ptr;
    s.scales = g_ptr;
    s.row_ref = g_ptr;
    s.codes_row_stride_bytes = (k_padded / 32u) * Q::mxfp4_native_block_bytes;
    s.scale_row_stride_bytes = k_padded / 32u;
    s.weight_scale_group = 32u;
    s.preshuffled = true;
    s.experts = experts;
    s.codes_batch_stride_bytes = codes_batch;
    s.scales_batch_stride_bytes = scales_batch;
    return s;
}

RT::WeightSlot exl3_slot() {
    RT::WeightSlot s{};
    s.rows = 128u;
    s.cols = 128u;
    s.k_padded = 128u;
    s.encoding = static_cast<uint8_t>(MW::MatrixEncoding::Exl3);
    s.compute_spec = static_cast<uint8_t>(MW::encoding_to_compute_spec(MW::MatrixEncoding::Exl3));
    s.codes = g_ptr;
    s.exl3_suh = g_ptr;
    s.exl3_svh = g_ptr;
    s.exl3_marker = g_ptr;
    s.exl3_ka = 4u;
    s.exl3_half = 0u;
    s.exl3_codebook = 2u;
    s.codes_row_stride_bytes = 8u * 64u * 2u;
    return s;
}

MW::MatrixWeight mxfp4_matrix(uint32_t rows, uint32_t cols, uint32_t k_padded,
                              uint32_t experts) {
    MW::MatrixWeight w{};
    w.encoding = MW::MatrixEncoding::Mxfp4;
    shelf::weights::apply_default_compute_spec(w);
    w.rows = rows;
    w.cols = cols;
    w.k_padded = k_padded;
    w.codes_row_stride_bytes = (k_padded / 32u) * Q::mxfp4_native_block_bytes;
    w.scale_row_stride_bytes = k_padded / 32u;
    w.weight_scale_group = 32u;
    w.preshuffled = true;
    w.experts = experts;
    if (experts > 1u) {
        w.codes_batch_stride_bytes = Q::mxfp4_native_codes_batch_stride_bytes(rows, k_padded);
        w.scales_batch_stride_bytes = Q::mxfp4_native_scale_batch_stride_bytes(rows, k_padded);
    }
    return w;
}

}  // namespace

int main() {
    const shelf::qwen35moe::Qwen35MoeTextConfig cfg = moe_config();
    auto plan = shelf::qwen35moe::build_moe_expert_group_plan(cfg, 0);
    if (!plan.ok()) {
        std::printf("FAIL build_moe_expert_group_plan: %s\n", plan.status().message().c_str());
        return 1;
    }

    const auto& pl = plan.value();
    const uint32_t experts = pl.experts;
    check(pl.w13_codes_batch_stride_bytes ==
              Q::mxfp4_native_codes_batch_stride_bytes(pl.w13_rows, pl.w13_k_padded),
          "w13 codes batch stride comes from the grouped layout helper");
    check(pl.w13_scales_batch_stride_bytes ==
              Q::mxfp4_native_scale_batch_stride_bytes(pl.w13_rows, pl.w13_k_padded),
          "w13 scales batch stride comes from the grouped layout helper");
    check(pl.w2_codes_batch_stride_bytes ==
              Q::mxfp4_native_codes_batch_stride_bytes(pl.w2_rows, pl.w2_k_padded),
          "w2 codes batch stride comes from the grouped layout helper");
    check(pl.w2_scales_batch_stride_bytes ==
              Q::mxfp4_native_scale_batch_stride_bytes(pl.w2_rows, pl.w2_k_padded),
          "w2 scales batch stride comes from the grouped layout helper");
    check(pl.w13_codes_bytes == pl.w13_codes_batch_stride_bytes * experts &&
              pl.w13_scales_bytes == pl.w13_scales_batch_stride_bytes * experts,
          "w13 plane is experts copies of one expert's strides");
    check(pl.w2_codes_bytes == pl.w2_codes_batch_stride_bytes * experts &&
              pl.w2_scales_bytes == pl.w2_scales_batch_stride_bytes * experts,
          "w2 plane is experts copies of one expert's strides");

    auto ok_slot = [&](const RT::WeightSlot& s, const char* what) {
        auto st = shelf::execution::validate_host_weight_slot(s, 0);
        check(st.ok(), what);
        if (!st.ok()) std::printf("     %s\n", st.message().c_str());
    };
    auto rejected = [&](const RT::WeightSlot& s, const char* what) {
        check(!shelf::execution::validate_host_weight_slot(s, 0).ok(), what);
    };

    RT::WeightSlot dense_bf16{};
    dense_bf16.rows = 512;
    dense_bf16.cols = 2048;
    dense_bf16.encoding = static_cast<uint8_t>(MW::MatrixEncoding::Bf16);
    dense_bf16.device_ptr = g_ptr;
    dense_bf16.bytes = static_cast<uint64_t>(dense_bf16.rows) * dense_bf16.cols * 2u;
    ok_slot(dense_bf16, "dense bf16 slot accepted");

    ok_slot(mxfp4_slot(1024, 2048, 2048, 1u, 0, 0), "dense mxfp4 slot accepted");
    ok_slot(exl3_slot(), "exl3 slot accepted");
    {
        RT::WeightSlot s = exl3_slot();
        s.exl3_suh = nullptr;
        rejected(s, "exl3 slot without suh rejected");
        s = exl3_slot();
        s.exl3_ka = 9u;
        rejected(s, "exl3 slot with a bad bit rate rejected");
        s = exl3_slot();
        s.codes_row_stride_bytes = 7u;
        rejected(s, "exl3 slot with a bad trellis stride rejected");
        s = exl3_slot();
        s.cols = 64u;
        s.k_padded = 64u;
        rejected(s, "exl3 slot with a non-128 shape rejected");
        s = exl3_slot();
        s.preshuffled = true;
        rejected(s, "exl3 slot with a preshuffle flag rejected");
    }

    ok_slot(mxfp4_slot(pl.w13_rows, pl.w13_cols, pl.w13_k_padded, experts,
                       pl.w13_codes_batch_stride_bytes, pl.w13_scales_batch_stride_bytes),
          "rank-3 w13 expert plane slot accepted");
    ok_slot(mxfp4_slot(pl.w2_rows, pl.w2_cols, pl.w2_k_padded, experts,
                       pl.w2_codes_batch_stride_bytes, pl.w2_scales_batch_stride_bytes),
          "rank-3 w2 expert plane slot accepted");

    const uint32_t rows = pl.w13_rows;
    const uint32_t cols = pl.w13_cols;
    const uint32_t kp = pl.w13_k_padded;

    rejected(mxfp4_slot(rows, cols, kp, 0u, pl.w13_codes_batch_stride_bytes,
                        pl.w13_scales_batch_stride_bytes),
             "experts 0 rejected");
    rejected(mxfp4_slot(rows, cols, kp, experts, 0, pl.w13_scales_batch_stride_bytes),
             "rank-3 with zero codes batch stride rejected");
    rejected(mxfp4_slot(rows, cols, kp, experts, pl.w13_codes_batch_stride_bytes, 0),
             "rank-3 with zero scales batch stride rejected");
    rejected(mxfp4_slot(rows, cols, kp, experts, pl.w13_codes_batch_stride_bytes + 1,
                        pl.w13_scales_batch_stride_bytes),
             "rank-3 with tampered codes batch stride rejected");
    rejected(mxfp4_slot(rows, cols, kp, experts, pl.w13_codes_batch_stride_bytes,
                        pl.w13_scales_batch_stride_bytes + 1),
             "rank-3 with tampered scales batch stride rejected");
    rejected(mxfp4_slot(rows - 1u, cols, kp, experts,
                        Q::mxfp4_native_codes_batch_stride_bytes(rows - 1u, kp),
                        Q::mxfp4_native_scale_batch_stride_bytes(rows - 1u, kp)),
             "rank-3 with output rows not a multiple of 16 rejected");
    rejected(mxfp4_slot(rows, cols, kp + 64u, experts,
                        Q::mxfp4_native_codes_batch_stride_bytes(rows, kp + 64u),
                        Q::mxfp4_native_scale_batch_stride_bytes(rows, kp + 64u)),
             "rank-3 with k_padded not a multiple of 128 rejected");

    {
        RT::WeightSlot s = mxfp4_slot(rows, cols, kp, experts,
                                      Q::mxfp4_native_codes_batch_stride_bytes(rows, kp),
                                      Q::mxfp4_native_scale_batch_stride_bytes(rows, kp));
        s.encoding = static_cast<uint8_t>(MW::MatrixEncoding::Bf16);
        s.device_ptr = g_ptr;
        s.codes = nullptr;
        s.scales = nullptr;
        rejected(s, "rank-3 bf16 encoding rejected");
    }

    {
        const MW::MatrixWeight w = mxfp4_matrix(rows, cols, kp, experts);
        const RT::WeightSlot s = shelf::qwen_hybrid::make_weight_slot(w);
        check(s.experts == experts, "make_weight_slot carries the expert count");
        check(s.codes_batch_stride_bytes == w.codes_batch_stride_bytes,
              "make_weight_slot carries the codes batch stride");
        check(s.scales_batch_stride_bytes == w.scales_batch_stride_bytes,
              "make_weight_slot carries the scales batch stride");
        check(s.codes_row_stride_bytes == w.codes_row_stride_bytes &&
                  s.scale_row_stride_bytes == w.scale_row_stride_bytes,
              "make_weight_slot keeps one expert's row strides");
    }
    {
        const MW::MatrixWeight w = mxfp4_matrix(5120, 2048, 2048, 1u);
        const RT::WeightSlot s = shelf::qwen_hybrid::make_weight_slot(w);
        check(s.experts == 1u, "a dense weight slot has one expert");
        check(s.codes_batch_stride_bytes == 0u && s.scales_batch_stride_bytes == 0u,
              "a dense weight slot has no batch stride");
    }

    {
        shelf::qwen35moe::Qwen35MoeTextConfig bad = cfg;
        bad.hidden_size = 1984;
        check(!shelf::qwen35moe::build_moe_expert_group_plan(bad, 0).ok(),
              "a k_padded that the grouped layout cannot read has no expert group plan");
    }

    if (g_fail != 0) {
        std::printf("FAILED %d checks\n", g_fail);
        return 1;
    }
    std::printf("all weight slot validation checks passed\n");
    return 0;
}
