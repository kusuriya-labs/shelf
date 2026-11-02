#include <shelf/inference/execution/embedding_dispatch.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/embedding.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/inference/execution/selectors/embedding_selector.h>
#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/gdn_recurrence_selector.h>
#include <shelf/inference/execution/selectors/kv_append_selector.h>
#include <shelf/inference/execution/selectors/l2_normalize_selector.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/inference/execution/selectors/rmsnorm_selector.h>
#include <shelf/inference/execution/selectors/rope_selector.h>
#include <shelf/inference/execution/selectors/sampling_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::execution {
namespace {

using ::shelf::runtime::WeightSlot;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueRowDomain;

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable};

}  // namespace

Result<PhysicalResolveStatus> resolve_embedding_physical(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalEmbeddingLaunch& out) {
    out = PhysicalEmbeddingLaunch{};
    if (b.input_count != 1u || b.output_count != 1u)
        return PhysicalResolveStatus::NotApplicable;
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return PhysicalResolveStatus::NotApplicable;
    const WeightSlot we = ctx.weights[b.weight_index];

    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue outv = out_res.value();
    if (outv.dtype != ValueDType::BF16)
        return PhysicalResolveStatus::NotApplicable;
    if (outv.row_domain != ValueRowDomain::TOKEN_ROWS)
        return PhysicalResolveStatus::NotApplicable;
    if (outv.ptr == nullptr || outv.feature_count == 0u ||
        outv.row_stride < outv.feature_count)
        return PhysicalResolveStatus::NotApplicable;

    auto tok_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!tok_res.ok()) return tok_res.status();
    const HostResolvedValue tok = tok_res.value();
    if (tok.dtype != ValueDType::I32)
        return Status::invalid_state("embedding token input dtype != I32",
                                     __FILE__, __LINE__);
    if (tok.ptr == nullptr) return PhysicalResolveStatus::NotApplicable;

    const uint32_t hidden = outv.feature_count;
    if (we.cols != hidden) return PhysicalResolveStatus::NotApplicable;
    const uint32_t vocab = we.rows;
    if (vocab == 0u) return PhysicalResolveStatus::NotApplicable;
    const uint32_t rows = ctx.actual_rows;
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;

    EmbeddingStorage storage;
    const ::shelf::bf16_t* bf16_table = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    uint32_t k_padded = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;

    if (we.encoding == 0u) {
        if (we.device_ptr == nullptr || we.codes != nullptr || we.scales != nullptr)
            return PhysicalResolveStatus::NotApplicable;
        storage = EmbeddingStorage::Bf16;
        bf16_table = static_cast<const ::shelf::bf16_t*>(we.device_ptr);
    } else {        return PhysicalResolveStatus::NotApplicable;
    }

    EmbeddingSelectorInput si;
    si.storage = storage;
    si.rows = rows;
    si.hidden_size = hidden;
    if (select_embedding_implementation(si) != EmbeddingImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    out.storage = storage;
    out.table = reinterpret_cast<uint64_t>(bf16_table);
    out.codes = reinterpret_cast<uint64_t>(codes);
    out.scales = reinterpret_cast<uint64_t>(scales);
    out.codes_row_stride_bytes = code_stride;
    out.scale_row_stride_bytes = scale_stride;
    out.k_padded = k_padded;
    out.token_ids = reinterpret_cast<uint64_t>(tok.ptr);
    out.output = reinterpret_cast<uint64_t>(outv.ptr);
    out.rows = rows;
    out.vocab_size = vocab;
    out.hidden_size = hidden;
    out.output_row_stride = outv.row_stride;
    return PhysicalResolveStatus::Applicable;
}

Result<OptimizedLaunchResult> try_launch_embedding(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != ::shelf::runtime::KernelId::EMBEDDING_LOOKUP) return kNone;

    PhysicalEmbeddingLaunch phys{};
    const auto resolved = resolve_embedding_physical(program, b, ctx, phys);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable) return kNone;

    if (phys.storage == EmbeddingStorage::Bf16) {
        const hipError_t herr = ::shelf::kernel::launch_embedding_bf16(
            reinterpret_cast<const ::shelf::bf16_t*>(phys.table),
            reinterpret_cast<const int32_t*>(phys.token_ids),
            reinterpret_cast<::shelf::bf16_t*>(phys.output), phys.rows,
            phys.vocab_size, phys.hidden_size, phys.output_row_stride,
            ctx.error_word, stream);
        if (herr != hipSuccess)
            return Status::hip_error("embedding optimized launch",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
    }
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
