#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

using shelf::qwen35::dflash2::DFlash2Config;
using shelf::qwen35::dflash2::DFlash2Executor;
using shelf::qwen35::dflash2::DFlash2ExecutorConfig;
using shelf::qwen35::dflash2::DFlash2Weights;
using shelf::qwen35::runtime::dflash2_draft_scratch_bytes;
using shelf::quantization::QuantSpecId;
using shelf::weights::MatrixWeight;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static DFlash2Config test_config() {
    DFlash2Config config;
    config.vocab_size = 4096;
    config.hidden_size = 512;
    config.intermediate_size = 1024;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 128;
    config.sliding_window = 64;
    config.is_causal = false;
    config.block_size = 8;
    config.conv_group_size = 16;
    config.conv_kernel_size = 2;
    config.mask_token_id = 0;
    config.selector_rank = 64;
    config.selector_top_k = 16;
    config.num_target_layers = 8;
    config.target_layer_ids = {0, 1, 2, 3, 4};
    config.num_target_layer_ids = 5;
    return config;
}

static MatrixWeight make_head(const DFlash2Config& config, QuantSpecId spec) {
    MatrixWeight head;
    head.rows = static_cast<uint32_t>(config.vocab_size);
    head.cols = static_cast<uint32_t>(config.hidden_size);
    head.k_padded = static_cast<uint32_t>(config.hidden_size);
    head.quant_spec = spec;
    head.encoding = shelf::weights::quant_spec_to_encoding(spec);
    return head;
}

struct AllocResult {
    bool ok = false;
    std::size_t actual_bytes = 0;
    bool accum_present = false;
    std::size_t accum_bytes = 0;
};

static AllocResult create_and_measure(const DFlash2Config& config, QuantSpecId spec) {
    AllocResult result;
    auto arena_result = shelf::gpu::GpuArena::create(0, 64u * 1024u * 1024u);
    if (!arena_result.ok()) return result;
    shelf::gpu::GpuArena arena = arena_result.release();

    DFlash2Weights weights;
    weights.present = true;
    weights.layers.resize(config.num_hidden_layers);
    weights.fc.encoding = shelf::weights::MatrixEncoding::Exl3;
    MatrixWeight head = make_head(config, spec);

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = static_cast<uint32_t>(config.block_size);
    exec_config.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    exec_config.target_lm_head = &head;

    auto exec_result = shelf::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("create failed: %s\n", exec_result.status().message().c_str());
        (void)arena.shutdown();
        return result;
    }
    DFlash2Executor executor = exec_result.release();
    result.actual_bytes = arena.used();
    result.accum_present = executor.exl3_lm_head_accum.ndim() != 0;
    if (result.accum_present) {
        std::size_t bytes = 0;
        if (executor.exl3_lm_head_accum.required_span_bytes(bytes).ok()) {
            result.accum_bytes = bytes;
        }
    }
    result.ok = true;
    (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(executor);
    (void)arena.shutdown();
    return result;
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    const DFlash2Config config = test_config();
    const std::size_t rows = config.block_size;
    const std::size_t logits_bytes = rows * config.vocab_size * sizeof(float);
    const std::size_t logits_align256 = ((logits_bytes + 255u) / 256u) * 256u;

    const AllocResult mul24 =
        create_and_measure(config, QuantSpecId::EXL3_MUL24_INT8_V1);
    const AllocResult bf16 = create_and_measure(config, QuantSpecId::BF16_V1);
    const AllocResult fp8 =
        create_and_measure(config, QuantSpecId::FP8_FINEGRAINED_F32_V1);

    check(mul24.ok && bf16.ok && fp8.ok, "all executors created");
    if (!mul24.ok || !bf16.ok || !fp8.ok) return 1;

    check(mul24.accum_present, "mul24 head allocates exl3_lm_head_accum");
    check(!bf16.accum_present, "bf16 head does not allocate exl3_lm_head_accum");
    check(!fp8.accum_present, "fp8 head does not allocate exl3_lm_head_accum");
    check(mul24.accum_bytes == logits_bytes,
          "exl3_lm_head_accum is one draft x vocab F32 buffer");

    check(mul24.actual_bytes > bf16.actual_bytes, "mul24 allocates more than bf16");
    check(mul24.actual_bytes - bf16.actual_bytes >= logits_bytes &&
              mul24.actual_bytes - bf16.actual_bytes <= logits_bytes + 127u,
          "mul24 minus bf16 is exactly the exl3_lm_head_accum allocation");
    check(bf16.actual_bytes == fp8.actual_bytes, "bf16 and fp8 allocate the same bytes");

    const std::size_t fp_mul24 =
        dflash2_draft_scratch_bytes(config, rows, config.hidden_size, 0u, true);
    const std::size_t fp_bf16 =
        dflash2_draft_scratch_bytes(config, rows, config.hidden_size, 0u, false);
    check(fp_mul24 - fp_bf16 == logits_align256,
          "footprint mul24 minus bf16 is one aligned logits buffer");
    check(fp_mul24 >= mul24.actual_bytes, "footprint covers the mul24 allocation");
    check(fp_bf16 >= bf16.actual_bytes, "footprint covers the bf16 allocation");
    check(fp_mul24 - mul24.actual_bytes < logits_align256,
          "mul24 footprint over-allocation is below one logits buffer");
    check(fp_bf16 - bf16.actual_bytes < logits_align256,
          "bf16 footprint over-allocation is below one logits buffer");

    if (g_fail != 0) {
        std::printf("dflash2 executor alloc: failed=%d\n", g_fail);
        return 1;
    }
    std::printf("dflash2 executor alloc: ok\n");
    return 0;
}