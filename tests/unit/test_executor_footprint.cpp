#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35/runtime/exl3_dense_workspace.h>

#include <cstdio>
#include <string>

using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::ExecutorConfig;
using shelf::qwen35::runtime::qwen35_executor_reserve_bytes;
using shelf::qwen35::runtime::qwen35_padded_row_count;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static Qwen35TextConfig reference_text_config() {
    Qwen35TextConfig config;
    config.hidden_size = 5120;
    config.vocab_size = 248320;
    config.ffn.intermediate_size = 17408;
    config.num_key_value_heads = 4;
    return config;
}

static ExecutorConfig exec_config(std::uint32_t tokens, std::uint32_t requests,
                                  std::uint32_t output_rows, std::uint32_t taps = 0u) {
    ExecutorConfig config;
    config.max_batch_rows = tokens;
    config.max_scheduled_requests = requests;
    config.max_scheduled_output_rows = output_rows;
    config.target_hidden_tap_count = taps;
    return config;
}

constexpr std::size_t kMiB = 1024ull * 1024ull;

int main() {
    const Qwen35TextConfig text = reference_text_config();

    check(qwen35_padded_row_count(512u) >= 512u, "row padding keeps the requested rows");
    check(qwen35_padded_row_count(8192u) >= 8192u, "row padding covers the largest bucket");

    const std::size_t rows512 = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(512u, 4u, 0u), 1u, false);
    const std::size_t rows2048 = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(2048u, 4u, 0u), 1u, false);
    const std::size_t rows4096 = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(4096u, 4u, 0u), 1u, false);

    check((rows512 % 256u) == 0u, "reserve is alignment padded");
    check(rows4096 > rows2048 && rows2048 > rows512, "reserve grows with the row bucket");

    // Lower bounds are the executor footprints measured on gfx1201 with the 27B FP8
    // target (0.25 / 0.91 / 1.78 GiB). The reserve must never fall below them.
    check(rows512 >= 256ull * kMiB, "reserve covers the measured 512 bucket");
    check(rows2048 >= 932ull * kMiB, "reserve covers the measured 2048 bucket");
    check(rows4096 >= 1823ull * kMiB, "reserve covers the measured 4096 bucket");
    check(rows4096 < 2048ull * kMiB, "reserve stays within 15 percent of the measurement");

    const std::size_t taps = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(4096u, 4u, 0u, 5u), 1u, false);
    const std::size_t tap_bytes = 5ull * 4096ull * text.hidden_size * sizeof(shelf::bf16_t);
    check(taps >= rows4096 + tap_bytes, "reserve covers the target hidden tap buffers");

    const std::size_t wide = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(4096u, 4u, 1024u), 1u, false);
    const std::size_t logits = 1020ull * text.vocab_size * sizeof(float);
    check(wide >= rows4096 + logits, "reserve covers the extra logits for large output rows");

    const std::size_t clamped = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(64u, 4u, 4096u), 1u, false);
    const std::size_t base = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(64u, 4u, 0u), 1u, false);
    const std::size_t clamped_logits = 60ull * text.vocab_size * sizeof(float);
    check(clamped >= base + clamped_logits,
          "output rows are clamped to the scheduled token count");

    const std::size_t tp2 = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(8192u, 4u, 0u), 2u, false);
    const std::size_t tp1 = qwen35_executor_reserve_bytes(text, text.ffn.intermediate_size, exec_config(8192u, 4u, 0u), 1u, false);
    check(tp2 < tp1, "reserve shrinks when intermediate_size is sharded");
    check(tp2 >= 1885ull * kMiB, "reserve covers the measured tp2 executor footprint");

    // The rotated-domain trellis term is large and only a model that carries its linears in
    // that form pays for it: the KV pool is sized from what the reserve leaves behind.
    const std::size_t exl3_4096 = qwen35_executor_reserve_bytes(
        text, text.ffn.intermediate_size, exec_config(4096u, 4u, 0u), 1u, true);
    check(exl3_4096 - rows4096 ==
              shelf::qwen35::runtime::exl3_workspace_bytes(qwen35_padded_row_count(4096u)),
          "reserve adds the trellis staging term only when the model uses it");

    if (g_fail != 0) {
        std::printf("executor footprint: failed=%d\n", g_fail);
        return 1;
    }
    std::printf("executor footprint: ok\n");
    return 0;
}
