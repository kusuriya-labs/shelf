#include <shelf/inference/execution/kv_append_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/runtime/graph/primitive_graph.h>
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
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace shelf::execution {
namespace {

using ::shelf::state::PagedKVPoolBF16View;
using ::shelf::state::PagedKVPoolFP8View;
using ::shelf::state::PagedKVPoolOctaveView;
using ::shelf::runtime::KernelId;

const ::shelf::runtime::StateBinding* find_state_binding_for_descriptor(
    const ::shelf::runtime::Program& program, uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& s : program.states) {
        if (s.slot + 1u == descriptor_slot) return &s;
    }
    return nullptr;
}

}  // namespace

Result<KvAppendResolvedArgs> resolve_kv_append_args(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx) {
    if (dispatch_index >= program.dispatches.size())
        return Status::invalid_state("kv-append: dispatch_index out of range", __FILE__, __LINE__);
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != KernelId::KV_APPEND)
        return Status::invalid_state("kv-append: kernel_id != KV_APPEND", __FILE__, __LINE__);
    if (b.input_count != 2 || b.output_count != 0)
        return Status::invalid_state("kv-append: io count != 2/0", __FILE__, __LINE__);

    const auto* sb = find_state_binding_for_descriptor(program, b.descriptor_slot);
    if (sb == nullptr || sb->kind != ::shelf::runtime::PrimitiveStateKind::KV_CACHE)
        return Status::invalid_state("kv-append: KV_CACHE state not found", __FILE__, __LINE__);
    const uint32_t layer = sb->state_index;

    auto k_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!k_res.ok()) return k_res.status();
    const HostResolvedValue k_in = k_res.value();
    auto v_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[1]}, ctx);
    if (!v_res.ok()) return v_res.status();
    const HostResolvedValue v_in = v_res.value();

    if (k_in.dtype != ::shelf::runtime::ValueDType::BF16 || v_in.dtype != ::shelf::runtime::ValueDType::BF16)
        return Status::invalid_state("kv-append: input dtype != BF16", __FILE__, __LINE__);
    if (k_in.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS ||
        v_in.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS)
        return Status::invalid_state("kv-append: input row_domain != TOKEN_ROWS", __FILE__, __LINE__);
    if (k_in.feature_count == 0 || v_in.feature_count == 0)
        return Status::invalid_state("kv-append: input feature_count == 0", __FILE__, __LINE__);
    if (k_in.row_stride < k_in.feature_count || v_in.row_stride < v_in.feature_count)
        return Status::invalid_state("kv-append: row_stride < feature_count", __FILE__, __LINE__);
    if (k_in.feature_count != v_in.feature_count)
        return Status::invalid_state("kv-append: K/V feature_count mismatch", __FILE__, __LINE__);

    const uint32_t rows = ctx.actual_rows;
    if (rows > 0u) {
        if (ctx.row_sequence_slots == nullptr || ctx.row_positions == nullptr)
            return Status::invalid_state("kv-append: row metadata null", __FILE__, __LINE__);
    }

    const auto& bt = ctx.model_state.block_table;
    if (bt.block_tables == nullptr || bt.max_sequences == 0u || bt.block_table_stride == 0u)
        return Status::invalid_state("kv-append: block table invalid", __FILE__, __LINE__);

    const uint32_t dtype_raw = ctx.model_state.kv_dtype;
    if (dtype_raw > static_cast<uint32_t>(::shelf::kernel::KVCacheDType::OCTAVE))
        return Status::invalid_state("kv-append: unknown KV dtype", __FILE__, __LINE__);
    const ::shelf::kernel::KVCacheDType kv_dtype =
        static_cast<::shelf::kernel::KVCacheDType>(dtype_raw);
    const bool fp8 = kv_dtype == ::shelf::kernel::KVCacheDType::FP8_E4M3;
    const bool octave = kv_dtype == ::shelf::kernel::KVCacheDType::OCTAVE;

    KvAppendResolvedArgs out;
    out.kv_dtype = kv_dtype;

    uint32_t page_tokens = 0, num_pages = 0, num_layers = 0, kv_heads = 0, head_dim = 0;
    uint32_t ept = 0, epp = 0, epl = 0;
    if (octave) {
        const PagedKVPoolOctaveView& kv = ctx.model_state.kv_octave;
        if (kv.k_pool == nullptr || kv.v_pool == nullptr || kv.levels == nullptr)
            return Status::invalid_state("kv-append: Octave pool null", __FILE__, __LINE__);
        page_tokens = kv.page_tokens; num_pages = kv.num_pages;
        num_layers = kv.num_attention_layers; kv_heads = kv.kv_heads; head_dim = kv.head_dim;
        ept = kv.kv_heads * kv.head_dim;
        epp = kv.page_tokens * ept;
        epl = kv.num_pages * epp;
        out.k_octave_pool = kv.k_pool; out.v_octave_pool = kv.v_pool;
        out.octave_levels = kv.levels; out.octave_format = kv.format;
    } else if (fp8) {
        const PagedKVPoolFP8View& kv = ctx.model_state.kv_fp8;
        if (kv.k_pool == nullptr || kv.v_pool == nullptr ||
            kv.k_scale_pool == nullptr || kv.v_scale_pool == nullptr)
            return Status::invalid_state("kv-append: fp8 pool null", __FILE__, __LINE__);
        page_tokens = kv.page_tokens; num_pages = kv.num_pages;
        num_layers = kv.num_attention_layers; kv_heads = kv.kv_heads; head_dim = kv.head_dim;
        ept = kv.elems_per_token; epp = kv.elems_per_page; epl = kv.elems_per_layer;
        out.k_fp8_pool = kv.k_pool; out.v_fp8_pool = kv.v_pool;
        out.k_scale_pool = kv.k_scale_pool; out.v_scale_pool = kv.v_scale_pool;
    } else {
        const PagedKVPoolBF16View& kv = ctx.model_state.kv_bf16;
        if (kv.k_pool == nullptr || kv.v_pool == nullptr)
            return Status::invalid_state("kv-append: bf16 pool null", __FILE__, __LINE__);
        page_tokens = kv.page_tokens; num_pages = kv.num_pages;
        num_layers = kv.num_attention_layers; kv_heads = kv.kv_heads; head_dim = kv.head_dim;
        ept = kv.elems_per_token; epp = kv.elems_per_page; epl = kv.elems_per_layer;
        out.k_bf16_pool = kv.k_pool; out.v_bf16_pool = kv.v_pool;
    }

    if (page_tokens == 0u || num_pages == 0u || num_layers == 0u || kv_heads == 0u || head_dim == 0u)
        return Status::invalid_state("kv-append: pool geometry invalid", __FILE__, __LINE__);
    if (layer >= num_layers)
        return Status::invalid_state("kv-append: layer out of range", __FILE__, __LINE__);
    if (b.group_size != head_dim)
        return Status::invalid_state("kv-append: group_size != head_dim", __FILE__, __LINE__);

    const uint64_t exp_ept = static_cast<uint64_t>(kv_heads) * head_dim;
    const uint64_t exp_epp = static_cast<uint64_t>(page_tokens) * exp_ept;
    const uint64_t exp_epl = static_cast<uint64_t>(num_pages) * exp_epp;
    if (exp_ept != static_cast<uint64_t>(k_in.feature_count))
        return Status::invalid_state("kv-append: features != kv_heads*head_dim", __FILE__, __LINE__);
    if (static_cast<uint64_t>(ept) != exp_ept || static_cast<uint64_t>(epp) != exp_epp ||
        static_cast<uint64_t>(epl) != exp_epl)
        return Status::invalid_state("kv-append: pool stride inconsistent", __FILE__, __LINE__);
    if (exp_ept > UINT32_MAX || exp_epp > UINT32_MAX || exp_epl > UINT32_MAX)
        return Status::invalid_state("kv-append: pool exceeds uint32", __FILE__, __LINE__);

    auto& c = out.common;
    c.k_input = static_cast<const bf16_t*>(k_in.ptr);
    c.v_input = static_cast<const bf16_t*>(v_in.ptr);
    c.k_row_stride = k_in.row_stride;
    c.v_row_stride = v_in.row_stride;
    c.row_sequence_slots = ctx.row_sequence_slots;
    c.row_positions = ctx.row_positions;
    c.block_tables = bt.block_tables;
    c.block_table_stride = bt.block_table_stride;
    c.max_sequences = bt.max_sequences;
    c.rows = rows;
    c.layer = layer;
    c.page_tokens = page_tokens;
    c.num_pages = num_pages;
    c.num_attention_layers = num_layers;
    c.kv_heads = kv_heads;
    c.head_dim = head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;
    return out;
}

Result<PhysicalResolveStatus> resolve_kv_append_physical(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx, PhysicalKvAppendLaunch& out) {
    out = PhysicalKvAppendLaunch{};
    const auto resolved = resolve_kv_append_args(program, dispatch_index, ctx);
    if (!resolved.ok()) return resolved.status();
    const KvAppendResolvedArgs a = resolved.value();
    if (a.common.rows == 0u) return PhysicalResolveStatus::Skipped;
    KvAppendSelectorInput si;
    si.kv_dtype = a.kv_dtype;
    si.rows = a.common.rows;
    si.kv_heads = a.common.kv_heads;
    si.head_dim = a.common.head_dim;
    si.page_tokens = a.common.page_tokens;
    if (select_kv_append_implementation(si) != KvAppendImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    out.common = a.common;
    switch (a.kv_dtype) {
        case ::shelf::kernel::KVCacheDType::BF16:
            out.variant = PhysicalKvAppendVariant::Bf16;
            out.k_pool = reinterpret_cast<uint64_t>(a.k_bf16_pool);
            out.v_pool = reinterpret_cast<uint64_t>(a.v_bf16_pool);
            break;
        case ::shelf::kernel::KVCacheDType::FP8_E4M3:
            out.variant = PhysicalKvAppendVariant::Fp8E4M3;
            out.k_pool = reinterpret_cast<uint64_t>(a.k_fp8_pool);
            out.v_pool = reinterpret_cast<uint64_t>(a.v_fp8_pool);
            out.k_scale_pool = reinterpret_cast<uint64_t>(a.k_scale_pool);
            out.v_scale_pool = reinterpret_cast<uint64_t>(a.v_scale_pool);
            break;
        case ::shelf::kernel::KVCacheDType::OCTAVE:
            out.variant = PhysicalKvAppendVariant::Octave;
            out.k_pool = reinterpret_cast<uint64_t>(a.k_octave_pool);
            out.v_pool = reinterpret_cast<uint64_t>(a.v_octave_pool);
            out.k_extra_pool = reinterpret_cast<uint64_t>(a.octave_levels);
            out.octave_format = a.octave_format;
            break;
        default:
            return PhysicalResolveStatus::NotApplicable;
    }
    out.grid = static_cast<uint32_t>(
        static_cast<uint64_t>(a.common.rows) * a.common.kv_heads);
    const uint32_t chunks = (a.common.head_dim + 3u) / 4u;
    out.workgroup = chunks <= 32u ? 32u
                   : chunks <= 64u ? 64u
                   : chunks <= 128u ? 128u
                                    : 256u;
    return PhysicalResolveStatus::Applicable;
}

hipError_t launch_kv_append_physical(const PhysicalKvAppendLaunch& d,
                                     hipStream_t stream) {
    switch (d.variant) {
        case PhysicalKvAppendVariant::Bf16:
            return ::shelf::kernel::launch_kv_append_bf16(
                d.common, reinterpret_cast<::shelf::bf16_t*>(d.k_pool),
                reinterpret_cast<::shelf::bf16_t*>(d.v_pool), stream);
        case PhysicalKvAppendVariant::Fp8E4M3:
            return ::shelf::kernel::launch_kv_append_fp8_e4m3(
                d.common,
                reinterpret_cast<::shelf::kernel::fp8e4m3_storage_t*>(d.k_pool),
                reinterpret_cast<::shelf::kernel::fp8e4m3_storage_t*>(d.v_pool),
                reinterpret_cast<float*>(d.k_scale_pool),
                reinterpret_cast<float*>(d.v_scale_pool), stream);
        case PhysicalKvAppendVariant::Octave:
            return ::shelf::kernel::launch_kv_append_octave(
                d.common, reinterpret_cast<uint8_t*>(d.k_pool),
                reinterpret_cast<uint8_t*>(d.v_pool),
                reinterpret_cast<const float*>(d.k_extra_pool), d.octave_format, stream);
        default:
            return hipErrorNotSupported;
    }
}

Result<OptimizedLaunchResult> try_launch_kv_append(
    const ::shelf::runtime::Program& program, size_t dispatch_index, HostExecutionContext& ctx,
    hipStream_t stream) {
    PhysicalKvAppendLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_kv_append_physical(program, dispatch_index, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() == PhysicalResolveStatus::NotApplicable)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (resolved.value() == PhysicalResolveStatus::Skipped)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const hipError_t herr = launch_kv_append_physical(launch, stream);
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("kv-append optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
