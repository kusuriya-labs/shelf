#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>

#include <cstdio>
#include <string>

using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::runtime::DFlash2Footprint;
using ps::qwen35::runtime::dflash2_context_state_bytes;
using ps::qwen35::runtime::dflash2_draft_scratch_bytes;

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

    const std::size_t context = dflash2_context_state_bytes(config);
    check(context == 2u * 5u * 2048u * (8u * 128u) * sizeof(ps::bf16_t),
          "context state bytes match the ring geometry");
    check(config.context_ring_bytes() == context,
          "context state bytes stay aligned to the config accessor");

    const std::size_t scratch = dflash2_draft_scratch_bytes(config, 8u, 0u);
    check(scratch > 0u, "draft scratch is positive");
    check((scratch % 256u) == 0u, "draft scratch is alignment padded");

    DFlash2Config small_vocab = config;
    small_vocab.vocab_size = config.vocab_size / 2u;
    const std::size_t small_scratch = dflash2_draft_scratch_bytes(small_vocab, 8u, 0u);
    const std::size_t logits_delta = 8u * (config.vocab_size - small_vocab.vocab_size) *
                                     sizeof(float);
    check(scratch > small_scratch, "draft scratch grows with the vocabulary");
    check(scratch - small_scratch >= logits_delta,
          "draft scratch covers the proposal logits");

    const std::size_t wide_rows = dflash2_draft_scratch_bytes(config, 64u, 0u);
    check(wide_rows > scratch, "draft scratch grows with the row count");

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
