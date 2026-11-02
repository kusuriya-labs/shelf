#include <shelf/models/qwen35/runtime/executor_footprint.h>

#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/kernels/optimized/radix_topn.h>
#include <shelf/kernels/optimized/sampling_limits.h>

#include <algorithm>

namespace ps::qwen35 {
namespace runtime {

namespace {

constexpr std::size_t kAlignBytes = 256u;

constexpr std::size_t kProgramValuesPerFullRow = 44u;

constexpr std::size_t kScratchFloats = 8192u;

std::size_t align_up(std::size_t bytes) noexcept {
    return ((bytes + kAlignBytes - 1u) / kAlignBytes) * kAlignBytes;
}

}  // namespace

std::size_t qwen35_padded_row_count(std::uint32_t max_batch_rows) noexcept {
    std::size_t rows = max_batch_rows;
    auto bucket = ::ps::runtime::resolve_row_bucket(max_batch_rows);
    if (bucket.ok()) {
        const std::uint32_t limit = ::ps::runtime::row_bucket_limit(bucket.value());
        if (limit > rows) {
            rows = limit;
        }
    }
    return rows;
}

std::size_t qwen35_executor_reserve_bytes(const Qwen35TextConfig& text_config,
                                          const ExecutorConfig& config,
                                          std::size_t tp_size) noexcept {
    if (tp_size == 0u) {
        tp_size = 1u;
    }
    const std::size_t hidden = text_config.hidden_size != 0u ? text_config.hidden_size : 1u;
    const std::size_t vocab = text_config.vocab_size != 0u ? text_config.vocab_size : 1u;
    const std::size_t rows = qwen35_padded_row_count(config.max_batch_rows);
    const std::size_t requests =
        config.max_scheduled_requests != 0u ? config.max_scheduled_requests : 1u;

    std::size_t output_rows = config.max_scheduled_output_rows > requests
        ? config.max_scheduled_output_rows
        : requests;
    if (output_rows > config.max_batch_rows) {
        output_rows = config.max_batch_rows;
    }

    std::size_t total = 0u;

    const std::size_t intermediate =
        text_config.intermediate_size != 0u ? text_config.intermediate_size : hidden;
    const std::size_t full_span = hidden + intermediate;
    const std::size_t rank_span = hidden + intermediate / tp_size;
    const std::size_t values_per_row =
        std::max<std::size_t>(1u, kProgramValuesPerFullRow * rank_span / full_span);
    total += align_up(values_per_row * rows * hidden * sizeof(bf16_t));
    total += align_up(kScratchFloats * sizeof(float));
    total += align_up(::ps::execution::kDecodeAttnPartialBytes);

    total += align_up(rows * sizeof(int32_t));
    total += align_up(output_rows * hidden * sizeof(bf16_t));
    total += align_up(output_rows * vocab * sizeof(float));
    total += align_up(output_rows * sizeof(int32_t));

    total += align_up(output_rows * ::ps::kernel::kSamplingMaxTopK * sizeof(int32_t));
    total += align_up(output_rows * ::ps::kernel::kSamplingMaxTopK * sizeof(float));
    total += align_up(::ps::kernel::dflash2_radix_topn_scratch_bytes(
        static_cast<std::uint32_t>(output_rows),
        ::ps::kernel::dflash2_radix_default_partitions(static_cast<std::uint32_t>(vocab)),
        static_cast<std::uint32_t>(vocab)));
    total += align_up(output_rows * sizeof(std::uint32_t));

    const std::size_t taps = config.target_hidden_tap_count;
    total += align_up((1u + taps) * rows * hidden * sizeof(bf16_t));

    const std::size_t out_cap = std::max(requests, rows);
    total += align_up(3u * rows * sizeof(std::uint32_t));
    total += align_up(requests * sizeof(::ps::runtime::DeviceRequestDescriptor));
    total += align_up(out_cap * sizeof(std::uint32_t));
    total += align_up(out_cap * sizeof(::ps::runtime::DeviceSamplingParams));
    total += align_up(sizeof(::ps::runtime::DeviceBatchContext));

    total += align_up(sizeof(std::uint32_t));
    total += align_up(32u * sizeof(double));
    return total;
}

}  // namespace runtime
}  // namespace ps::qwen35
