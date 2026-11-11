#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>

#include <cstdio>
#include <string>

using shelf::qwen35::dflash2::DFlash2Config;
using shelf::qwen35::runtime::DFlash2Footprint;
using shelf::qwen35::runtime::dflash2_context_state_bytes;
using shelf::qwen35::runtime::dflash2_draft_scratch_bytes;
using shelf::qwen35::runtime::dflash2_prefill_rows;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static DFlash2Config dflash2_reference_config() {
    DFlash2Config config;
    config.vocab_size = 248320;
    config.hidden_size = 5120;
    config.intermediate_size = 17408;
    config.num_hidden_layers = 5;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.sliding_window = 2048;
    config.is_causal = true;
    config.block_size = 8;
    config.conv_group_size = 16;
    config.conv_kernel_size = 2;
    config.selector_rank = 256;
    config.selector_top_k = 16;
    config.num_target_layers = 64;
    config.target_layer_ids = {5, 19, 33, 47, 61};
    config.num_target_layer_ids = 5;
    return config;
}

int main() {
    const DFlash2Config config = dflash2_reference_config();
    using shelf::qwen35::runtime::dflash2_session_rows;
    check(dflash2_session_rows(8u, 7u) == 8u, "default chain capacity");
    check(dflash2_session_rows(8u, 3u, 0u) == 4u, "zero budget retains chain capacity");
    check(dflash2_session_rows(8u, 7u, 10u) == 10u, "tree budget increases verify rows");
    check(dflash2_session_rows(8u, 7u, 6u) == 8u, "tree capacity never shrinks chain");

    uint32_t rows = 0u;
    check(dflash2_prefill_rows(config, 1u, nullptr, rows) && rows == 256u,
          "serving defaults to 256 feature rows");
    check(dflash2_prefill_rows(config, 1u, "8", rows) && rows == 0u,
          "8 restores legacy feature rows");
    check(dflash2_prefill_rows(config, 1u, "256", rows) && rows == 256u,
          "256 explicitly enables feature rows");
    check(dflash2_prefill_rows(config, 2u, nullptr, rows) && rows == 0u,
          "tensor parallel defaults to the legacy capacity");
    check(dflash2_prefill_rows(config, 2u, "256", rows) && rows == 0u,
          "tensor parallel falls back");
    DFlash2Config unsupported = config;
    unsupported.sliding_window = 255u;
    check(dflash2_prefill_rows(unsupported, 1u, nullptr, rows) && rows == 0u,
          "small sliding window falls back");
    unsupported = config;
    unsupported.block_size = 16u;
    check(dflash2_prefill_rows(unsupported, 1u, nullptr, rows) && rows == 0u,
          "other block sizes fall back");
    for (const char* invalid : {"", "0", "9", "257", "-1", " 8", "8x"}) {
        check(!dflash2_prefill_rows(config, 1u, invalid, rows), "invalid value is rejected");
    }

    const std::size_t context = dflash2_context_state_bytes(config);
    check(context == 2u * 5u * 2048u * (8u * 128u) * sizeof(shelf::bf16_t),
          "context state bytes match the ring geometry");
    check(config.context_ring_bytes() == context,
          "context state bytes stay aligned to the config accessor");

    const std::size_t scratch = dflash2_draft_scratch_bytes(config, 8u, 0u);
    check(scratch > 0u, "draft scratch is positive");
    check((scratch % 256u) == 0u, "draft scratch is alignment padded");
    const std::size_t prefill_scratch = dflash2_draft_scratch_bytes(config, 8u, 0u, 256u);
    const std::size_t feature_delta = (256u - 8u) *
        (config.tap_feature_size() + 2u * config.hidden_size) * sizeof(shelf::bf16_t);
    check(prefill_scratch - scratch == feature_delta,
          "prefill scratch accounts for exactly the additional feature rows");
    const std::size_t batch_scratch = dflash2_draft_scratch_bytes(config, 32u, 0u);
    const std::size_t batch_prefill = dflash2_draft_scratch_bytes(config, 32u, 0u, 256u);
    check(batch_prefill - batch_scratch == (256u - 32u) *
              (config.tap_feature_size() + 2u * config.hidden_size) * sizeof(shelf::bf16_t),
          "prefill scratch accounts for the existing batched row capacity");
    check(dflash2_draft_scratch_bytes(config, 8u, 0u, 4096u) ==
              dflash2_draft_scratch_bytes(config, 8u, 0u, 2048u),
          "prefill scratch is capped to context capacity");

    DFlash2Config small_vocab = config;
    small_vocab.vocab_size = config.vocab_size / 2u;
    const std::size_t small_scratch = dflash2_draft_scratch_bytes(small_vocab, 8u, 0u);
    const std::size_t logits_delta = 8u * (config.vocab_size - small_vocab.vocab_size) *
                                     sizeof(float);
    check(scratch > small_scratch, "draft scratch grows with the vocabulary");
    check(scratch - small_scratch >= 2u * logits_delta,
          "draft scratch covers the proposal logits and EXL3 accumulator");

    const std::size_t wide_rows = dflash2_draft_scratch_bytes(config, 64u, 0u);
    check(wide_rows > scratch, "draft scratch grows with the row count");
    check(wide_rows - scratch >= 2u * (64u - 8u) * config.vocab_size * sizeof(float),
          "batched logits and EXL3 accumulator cover all rows");

    using shelf::qwen35::runtime::dflash2_batch_tree_metadata_bytes;
    check(dflash2_batch_tree_metadata_bytes(16u, 0u) == 0u,
          "zero tree budget adds no batch metadata");
    check(dflash2_batch_tree_metadata_bytes(16u, 10u) == 6u * 256u + 768u,
          "tree metadata footprint accounts for aligned arrays");

    DFlash2Footprint footprint;
    footprint.draft_weights = 1024u;
    footprint.draft_scratch = 2048u;
    footprint.session_slot = 512u;
    footprint.sessions = 3u;
    check(footprint.total() == 1024u + 2048u + 512u * 3u,
          "footprint total sums weights, scratch and slots");

    if (g_fail != 0) {
        std::printf("dflash2 footprint: failed=%d\n", g_fail);
        return 1;
    }
    std::printf("dflash2 footprint: ok\n");
    return 0;
}
