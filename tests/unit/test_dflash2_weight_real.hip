#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2Weights;
using ps::qwen35::dflash2::dflash2_expected_tensors;
using ps::qwen35::dflash2::load_dflash2_weights;
using ps::qwen35::dflash2::read_dflash2_config;
using ps::qwen35::dflash2::validate_dflash2_tensor_contract;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

int main() {
    const char* dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    if (dir == nullptr || dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = read_dflash2_config(dir);
    check(config_result.ok(), "config parse: " +
                                  (config_result.ok() ? "" : config_result.status().message()));
    if (!config_result.ok()) {
        std::printf("test_dflash2_weight_real: failed=%d\n", g_fail);
        return 1;
    }
    const DFlash2Config config = config_result.value();

    check(config.architecture == "DFlash2DraftModel", "architecture");
    check(config.vocab_size == 248320u, "vocab_size");
    check(config.hidden_size == 5120u, "hidden_size");
    check(config.intermediate_size == 17408u, "intermediate_size");
    check(config.num_hidden_layers == 5u, "num_hidden_layers");
    check(config.num_attention_heads == 32u, "num_attention_heads");
    check(config.num_key_value_heads == 8u, "num_key_value_heads");
    check(config.head_dim == 128u, "head_dim");
    check(config.sliding_window == 2048u, "sliding_window");
    check(!config.is_causal, "is_causal");
    check(config.block_size == 8u, "block_size");
    check(config.max_draft_tokens() == 7u, "max_draft_tokens");
    check(config.conv_group_size == 16u, "conv_group_size");
    check(config.conv_kernel_size == 2u, "conv_kernel_size");
    check(config.mask_token_id == 248070u, "mask_token_id");
    check(config.selector_rank == 256u, "selector_rank");
    check(config.selector_top_k == 16u, "selector_top_k");
    check(config.num_target_layers == 64u, "num_target_layers");
    check(config.num_target_layer_ids == 5u, "num_target_layer_ids");
    check(config.target_layer_ids[0] == 5u && config.target_layer_ids[1] == 19u &&
              config.target_layer_ids[2] == 33u && config.target_layer_ids[3] == 47u &&
              config.target_layer_ids[4] == 61u,
          "target_layer_ids");
    check(!config.tie_word_embeddings, "tie_word_embeddings");
    check(config.conv_groups() == 320u, "conv_groups");
    check(config.conv_projection_rows() == 1280u, "conv_projection_rows");
    check(config.attention_q_rows() == 4096u, "attention_q_rows");
    check(config.attention_kv_rows() == 1024u, "attention_kv_rows");
    check(config.tap_feature_size() == 25600u, "tap_feature_size");

    {
        const auto expected = dflash2_expected_tensors(config);
        check(expected.size() == 81u, "expected tensor count 81");
        std::size_t total_bytes = 0;
        for (const auto& e : expected) {
            std::size_t n = 1;
            for (std::size_t d : e.shape) n *= d;
            total_bytes += n * 2u;
        }
        check(total_bytes == 3848808960ull, "expected total payload bytes");
    }

    {
        auto st = validate_dflash2_tensor_contract(dir, config);
        check(st.ok(), "tensor contract: " + (st.ok() ? "" : st.message()));
    }

    auto arena_result = ps::gpu::GpuArena::create(0, 8ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    {
        auto loaded = load_dflash2_weights(dir, config, arena, stream);
        check(loaded.ok(), "load: " + (loaded.ok() ? "" : loaded.status().message()));
        if (loaded.ok()) {
            DFlash2Weights w = loaded.release();
            check(w.present, "present flag");
            check(w.layers.size() == 5u, "layer count");
            check(w.fc.rows == 5120u && w.fc.cols == 25600u, "fc rows/cols");
            check(w.hidden_norm_weight.dim(0) == 5120u, "hidden_norm size");
            check(w.final_norm_weight.dim(0) == 5120u, "final_norm size");
            check(w.selector.hidden_projection.rows == 256u &&
                      w.selector.hidden_projection.cols == 5120u,
                  "selector hidden_projection rows/cols");
            check(w.selector.predecessor_codebook.dim(0) == 248320u &&
                      w.selector.predecessor_codebook.dim(1) == 256u,
                  "predecessor_codebook shape");
            check(w.selector.successor_codebook.dim(0) == 248320u &&
                      w.selector.successor_codebook.dim(1) == 256u,
                  "successor_codebook shape");
            for (std::size_t i = 0; i < w.layers.size(); ++i) {
                const auto& lw = w.layers[i];
                check(lw.attn_q_proj.rows == 4096u && lw.attn_q_proj.cols == 5120u,
                      "layer q_proj rows/cols " + std::to_string(i));
                check(lw.attn_k_proj.rows == 1024u && lw.attn_k_proj.cols == 5120u,
                      "layer k_proj rows/cols " + std::to_string(i));
                check(lw.attn_o_proj.rows == 5120u && lw.attn_o_proj.cols == 4096u,
                      "layer o_proj rows/cols " + std::to_string(i));
                check(lw.mlp_gate_proj.rows == 17408u && lw.mlp_gate_proj.cols == 5120u,
                      "layer gate_proj rows/cols " + std::to_string(i));
                check(lw.mlp_down_proj.rows == 5120u && lw.mlp_down_proj.cols == 17408u,
                      "layer down_proj rows/cols " + std::to_string(i));
                check(lw.attn_q_norm_weight.dim(0) == 128u,
                      "layer q_norm size " + std::to_string(i));
                check(lw.attn_k_norm_weight.dim(0) == 128u,
                      "layer k_norm size " + std::to_string(i));
                check(lw.attention_conv_base_kernel.ndim() == 3u &&
                          lw.attention_conv_base_kernel.dim(0) == 2u &&
                          lw.attention_conv_base_kernel.dim(1) == 2u &&
                          lw.attention_conv_base_kernel.dim(2) == 5120u,
                      "layer attention_conv base_kernel shape " + std::to_string(i));
                check(lw.attention_conv_kernel_projection.rows == 1280u &&
                          lw.attention_conv_kernel_projection.cols == 5120u,
                      "layer attention_conv kernel_projection rows/cols " + std::to_string(i));
                check(lw.mlp_conv_base_kernel.ndim() == 3u &&
                          lw.mlp_conv_base_kernel.dim(2) == 5120u,
                      "layer mlp_conv base_kernel shape " + std::to_string(i));
                check(lw.mlp_conv_kernel_projection.rows == 1280u &&
                          lw.mlp_conv_kernel_projection.cols == 5120u,
                      "layer mlp_conv kernel_projection rows/cols " + std::to_string(i));
            }
        }
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    {
        const ps::Status shutdown = arena.shutdown();
        check(shutdown.ok(), "arena shutdown: " + shutdown.message());
        check(arena.is_shutdown(), "arena reports shutdown");
    }

    std::printf("test_dflash2_weight_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
