#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>

#include <cstdio>
#include <string>

using shelf::qwen35::runtime::mtp_batch_rank_state_reserve_bytes;
using shelf::qwen35::runtime::mtp_spec_rows;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static shelf::qwen_hybrid::QwenHybridConfig reference_config() {
    shelf::qwen_hybrid::QwenHybridConfig config;
    config.vocab_size = 248320;
    config.hidden_size = 2560;
    config.num_hidden_layers = 48;
    config.full_attention_interval = 4;
    config.linear_num_key_heads = 16;
    config.linear_num_value_heads = 48;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;
    config.num_key_value_heads = 2;
    config.attention_head_dim = 256;
    return config;
}

int main() {
    check(mtp_spec_rows(1u, 4u) == 5u, "one slot reserves one verify round");
    check(mtp_spec_rows(3u, 7u) == 24u, "every slot reserves its own verify round");
    check(mtp_spec_rows(0u, 4u) == 0u, "no slots reserve nothing");

    const auto full = reference_config();
    const auto gdn_full = shelf::qwen35::make_qwen35_gdn_state_layout(full);
    const auto gdn_local = shelf::qwen35::make_qwen35_gdn_state_layout_local(
        full, 36u, 8u, 24u);

    const std::size_t rows = 5u;
    const std::size_t history = rows * gdn_full.num_gdn_states *
        gdn_full.conv_history_stride * gdn_full.conv_history * sizeof(shelf::bf16_t);
    const std::size_t vdim =
        static_cast<std::size_t>(gdn_full.num_v_heads) * gdn_full.head_v;
    const std::size_t qkdim =
        static_cast<std::size_t>(gdn_full.num_v_heads) * gdn_full.head_k;
    const std::size_t compact = gdn_full.num_gdn_states * rows *
        (vdim + qkdim + gdn_full.num_v_heads) * sizeof(float);
    check(mtp_batch_rank_state_reserve_bytes(gdn_full, static_cast<std::uint32_t>(rows)) ==
              history + compact,
          "batch rank state covers the history and compact log");

    const std::size_t small = mtp_batch_rank_state_reserve_bytes(gdn_full, 5u);
    const std::size_t prefill_rows = mtp_batch_rank_state_reserve_bytes(gdn_full, 512u);
    check(small * 100u < prefill_rows,
          "batch rank state follows the verify rows, not the prefill row limit");

    check(mtp_batch_rank_state_reserve_bytes(gdn_local, 5u) < small,
          "a tp rank reserves its local gdn geometry");

    check(mtp_batch_rank_state_reserve_bytes(gdn_full, 0u) == 0u,
          "no verify rows reserve nothing");

    shelf::qwen35::runtime::MtpDrafterConfig options;
    options.num_drafts = 4u;
    options.max_seq_len = 524288u;
    options.page_tokens = 16u;
    options.max_batch_rows = 512u;
    options.kv_dtype = shelf::kernel::KVCacheDType::FP8_E4M3;

    const std::size_t drafter_full = shelf::qwen35::runtime::mtp_drafter_reserve_bytes(
        full, options, 640u, 0u, true);
    const std::size_t drafter_local = shelf::qwen35::runtime::mtp_drafter_reserve_bytes(
        full, options, 640u, 0u, true, {}, &gdn_local);
    check(drafter_local < drafter_full,
          "a tp rank reserves its local target gdn snapshot");

    shelf::weights::Exl3LinearShape mtp_shape;
    mtp_shape.max_k = 2560u;
    mtp_shape.max_out_features = 6144u;
    mtp_shape.max_dense_bytes = 2560u * 6144u * 2u;
    const std::size_t drafter_shaped = shelf::qwen35::runtime::mtp_drafter_reserve_bytes(
        full, options, 640u, 0u, true, mtp_shape, &gdn_local);
    check(drafter_shaped < drafter_local,
          "the drafter reserves its own exl3 staging planes, not the generic bound");

    if (g_fail != 0) {
        std::printf("%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("mtp reserve ok\n");
    return 0;
}
