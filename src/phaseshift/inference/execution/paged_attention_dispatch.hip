#include <phaseshift/inference/execution/paged_attention_dispatch.h>
#include <phaseshift/inference/execution/program_executor.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/inference/execution/physical_launch.h>
#include <phaseshift/inference/execution/decode_perf_stats.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/inference/execution/selectors/elementwise_selector.h>
#include <phaseshift/inference/execution/selectors/embedding_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_conv_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_recurrence_selector.h>
#include <phaseshift/inference/execution/selectors/kv_append_selector.h>
#include <phaseshift/inference/execution/selectors/l2_normalize_selector.h>
#include <phaseshift/inference/execution/selectors/linear_selector.h>
#include <phaseshift/inference/execution/selectors/paged_attention_selector.h>
#include <phaseshift/inference/execution/selectors/rmsnorm_selector.h>
#include <phaseshift/inference/execution/selectors/rope_selector.h>
#include <phaseshift/inference/execution/selectors/sampling_selector.h>
#include <phaseshift/inference/execution/selectors/selector_limits.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdio>

namespace ps::execution {
namespace {

using ::ps::state::PagedKVPoolBF16View;
using ::ps::state::PagedKVPoolFP8View;
using ::ps::runtime::KernelId;

const ::ps::runtime::StateBinding* find_state_binding_for_descriptor(
    const ::ps::runtime::Program& program, uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& s : program.states) {
        if (s.slot + 1u == descriptor_slot) return &s;
    }
    return nullptr;
}

constexpr uint32_t kPrefillKernelMinRows = kPagedAttentionPrefillMinRows;

bool should_use_paged_attention_prefill(uint32_t rows, const HostExecutionContext& ctx) {
    if (rows < kPrefillKernelMinRows) return false;
    const auto* bc = ctx.batch_context;
    if (bc == nullptr || bc->requests == nullptr) return false;
    if (bc->num_requests != 1u) return false;
    const auto& req = bc->requests[0];
    return req.row_begin == 0u && req.row_count >= rows;
}

}  // namespace

Result<PagedAttentionResolvedArgs> resolve_paged_attention_args(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx) {
    if (dispatch_index >= program.dispatches.size())
        return Status::invalid_state("paged-attention: dispatch_index out of range", __FILE__, __LINE__);
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != KernelId::PAGED_ATTENTION)
        return Status::invalid_state("paged-attention: kernel_id != PAGED_ATTENTION", __FILE__, __LINE__);
    if (b.input_count != 1 || b.output_count != 1)
        return Status::invalid_state("paged-attention: io count != 1/1", __FILE__, __LINE__);

    const auto* sb = find_state_binding_for_descriptor(program, b.descriptor_slot);
    if (sb == nullptr || sb->kind != ::ps::runtime::PrimitiveStateKind::KV_CACHE)
        return Status::invalid_state("paged-attention: KV_CACHE state not found", __FILE__, __LINE__);
    const uint32_t layer = sb->state_index;

    auto q_res = resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!q_res.ok()) return q_res.status();
    const HostResolvedValue q_in = q_res.value();
    auto out_res = resolve_host_value(program, ::ps::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();

    if (q_in.dtype != ::ps::runtime::ValueDType::BF16)
        return Status::invalid_state("paged-attention: Q dtype != BF16", __FILE__, __LINE__);
    if (out.dtype != ::ps::runtime::ValueDType::BF16 && out.dtype != ::ps::runtime::ValueDType::F32)
        return Status::invalid_state("paged-attention: output dtype != BF16/F32", __FILE__, __LINE__);
    if (q_in.row_domain != ::ps::runtime::ValueRowDomain::TOKEN_ROWS ||
        out.row_domain != ::ps::runtime::ValueRowDomain::TOKEN_ROWS)
        return Status::invalid_state("paged-attention: row_domain != TOKEN_ROWS", __FILE__, __LINE__);
    if (q_in.feature_count == 0 || out.feature_count == 0)
        return Status::invalid_state("paged-attention: feature_count == 0", __FILE__, __LINE__);
    if (q_in.row_stride < q_in.feature_count || out.row_stride < out.feature_count)
        return Status::invalid_state("paged-attention: row_stride < feature_count", __FILE__, __LINE__);
    if (q_in.feature_count != out.feature_count)
        return Status::invalid_state("paged-attention: Q/output feature_count mismatch", __FILE__, __LINE__);

    const uint32_t head_dim = b.group_size;
    if (head_dim == 0u || head_dim > 256u || q_in.feature_count % head_dim != 0u)
        return Status::invalid_state("paged-attention: head_dim invalid", __FILE__, __LINE__);
    const uint32_t q_heads = q_in.feature_count / head_dim;

    const uint32_t dtype_raw = ctx.model_state.kv_dtype;
    const ::ps::kernel::KVCacheDType kv_dtype =
        dtype_raw == 1u ? ::ps::kernel::KVCacheDType::FP8_E4M3
                        : ::ps::kernel::KVCacheDType::BF16;
    const bool fp8 = kv_dtype == ::ps::kernel::KVCacheDType::FP8_E4M3;

    uint32_t page_tokens = 0, num_pages = 0, num_layers = 0, kv_heads = 0, pool_head_dim = 0;
    uint32_t ept = 0, epp = 0, epl = 0;
    PagedAttentionResolvedArgs out_args;
    out_args.kv_dtype = kv_dtype;
    out_args.output_dtype = out.dtype;

    if (fp8) {
        const PagedKVPoolFP8View& kv = ctx.model_state.kv_fp8;
        if (kv.k_pool == nullptr || kv.v_pool == nullptr ||
            kv.k_scale_pool == nullptr || kv.v_scale_pool == nullptr)
            return Status::invalid_state("paged-attention: fp8 pool null", __FILE__, __LINE__);
        page_tokens = kv.page_tokens; num_pages = kv.num_pages;
        num_layers = kv.num_attention_layers; kv_heads = kv.kv_heads; pool_head_dim = kv.head_dim;
        ept = kv.elems_per_token; epp = kv.elems_per_page; epl = kv.elems_per_layer;
        out_args.k_fp8_pool = kv.k_pool; out_args.v_fp8_pool = kv.v_pool;
        out_args.k_scale_pool = kv.k_scale_pool; out_args.v_scale_pool = kv.v_scale_pool;
    } else {
        const PagedKVPoolBF16View& kv = ctx.model_state.kv_bf16;
        if (kv.k_pool == nullptr || kv.v_pool == nullptr)
            return Status::invalid_state("paged-attention: bf16 pool null", __FILE__, __LINE__);
        page_tokens = kv.page_tokens; num_pages = kv.num_pages;
        num_layers = kv.num_attention_layers; kv_heads = kv.kv_heads; pool_head_dim = kv.head_dim;
        ept = kv.elems_per_token; epp = kv.elems_per_page; epl = kv.elems_per_layer;
        out_args.k_bf16_pool = kv.k_pool; out_args.v_bf16_pool = kv.v_pool;
    }

    if (page_tokens == 0u || num_pages == 0u || num_layers == 0u || kv_heads == 0u || pool_head_dim == 0u)
        return Status::invalid_state("paged-attention: pool geometry invalid", __FILE__, __LINE__);
    if (layer >= num_layers)
        return Status::invalid_state("paged-attention: layer out of range", __FILE__, __LINE__);
    if (pool_head_dim != head_dim)
        return Status::invalid_state("paged-attention: pool head_dim != group_size", __FILE__, __LINE__);
    if (q_heads % kv_heads != 0u)
        return Status::invalid_state("paged-attention: q_heads % kv_heads != 0", __FILE__, __LINE__);

    const uint64_t exp_ept = static_cast<uint64_t>(kv_heads) * head_dim;
    const uint64_t exp_epp = static_cast<uint64_t>(page_tokens) * exp_ept;
    const uint64_t exp_epl = static_cast<uint64_t>(num_pages) * exp_epp;
    if (static_cast<uint64_t>(ept) != exp_ept || static_cast<uint64_t>(epp) != exp_epp ||
        static_cast<uint64_t>(epl) != exp_epl)
        return Status::invalid_state("paged-attention: pool stride inconsistent", __FILE__, __LINE__);
    if (exp_ept > UINT32_MAX || exp_epp > UINT32_MAX || exp_epl > UINT32_MAX)
        return Status::invalid_state("paged-attention: pool exceeds uint32", __FILE__, __LINE__);

    const uint32_t rows = ctx.actual_rows;
    if (rows > 0u) {
        if (ctx.row_sequence_slots == nullptr || ctx.row_positions == nullptr)
            return Status::invalid_state("paged-attention: row metadata null", __FILE__, __LINE__);
    }
    const auto& bt = ctx.model_state.block_table;
    if (bt.block_tables == nullptr || bt.max_sequences == 0u || bt.block_table_stride == 0u)
        return Status::invalid_state("paged-attention: block table invalid", __FILE__, __LINE__);

    const uint64_t capacity = static_cast<uint64_t>(bt.block_table_stride) * page_tokens;
    if (static_cast<uint64_t>(ctx.spec.max_visible_tokens) > capacity)
        return Status::invalid_state("paged-attention: max_visible > capacity", __FILE__, __LINE__);

    auto& c = out_args.common;
    c.q = static_cast<const bf16_t*>(q_in.ptr);
    c.output = out.ptr;
    c.output_dtype = (out.dtype == ::ps::runtime::ValueDType::F32)
        ? ::ps::kernel::PagedAttentionOutputDType::F32
        : ::ps::kernel::PagedAttentionOutputDType::BF16;
    c.q_row_stride = q_in.row_stride;
    c.output_row_stride = out.row_stride;
    c.row_sequence_slots = ctx.row_sequence_slots;
    c.row_positions = ctx.row_positions;
    c.tree_anc_offsets = ctx.spec.tree_anc_offsets;
    c.tree_anc_positions = ctx.spec.tree_anc_positions;
    c.tree_committed_end = ctx.spec.tree_committed_end;
    c.tree_committed_end_rows = ctx.spec.tree_committed_end_rows;
    if ((c.tree_anc_offsets == nullptr) != (c.tree_anc_positions == nullptr))
        return Status::invalid_state("paged-attention: tree visibility pair mismatch",
                                     __FILE__, __LINE__);
    if (c.tree_anc_offsets != nullptr &&
        static_cast<uint64_t>(c.tree_committed_end) > capacity)
        return Status::invalid_state("paged-attention: tree committed > capacity",
                                     __FILE__, __LINE__);
    c.block_tables = bt.block_tables;
    c.block_table_stride = bt.block_table_stride;
    c.max_sequences = bt.max_sequences;
    c.rows = rows;
    c.layer = layer;
    c.page_tokens = page_tokens;
    c.num_pages = num_pages;
    c.num_attention_layers = num_layers;
    c.q_heads = q_heads;
    c.kv_heads = kv_heads;
    c.head_dim = head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;
    c.scale = b.scalar_a;
    return out_args;
}

PagedAttentionPlan plan_paged_attention(
    const PagedAttentionResolvedArgs& args,
    const HostExecutionContext& ctx,
    bool use_prefill) {
    PagedAttentionPlan plan;
    if (args.common.rows == 0u)
        return plan;
    PagedAttentionSelectorInput si;
    si.kv_dtype = args.kv_dtype;
    si.output_dtype = args.output_dtype;
    si.rows = args.common.rows;
    si.q_heads = args.common.q_heads;
    si.kv_heads = args.common.kv_heads;
    si.head_dim = args.common.head_dim;
    si.page_tokens = args.common.page_tokens;
    si.max_visible_tokens = ctx.spec.max_visible_tokens;
    if (select_paged_attention_implementation(si) != PagedAttentionImplementation::Optimized)
        return plan;
    plan.optimized = true;
    plan.use_prefill = use_prefill;
    if (!plan.use_prefill
        && args.common.tree_anc_offsets == nullptr
        && ctx.decode_attn_partials != nullptr
        && ctx.spec.max_visible_tokens >= kPagedAttentionSplitMinVisible
        && args.common.q_heads % args.common.kv_heads == 0u) {
        const uint32_t q_per_kv = args.common.q_heads / args.common.kv_heads;
        const uint32_t s = 16u;
        const uint64_t need =
            static_cast<uint64_t>(args.common.rows) * args.common.kv_heads * s
            * q_per_kv * (2u + ::ps::kernel::kPagedAttentionHeadLimit) * 4u;
        if (need <= ctx.decode_attn_partial_bytes)
            plan.splits = s;
    }
    return plan;
}

namespace {

uint32_t paged_attention_output_element_bytes(
    ::ps::kernel::PagedAttentionOutputDType dtype) {
    return dtype == ::ps::kernel::PagedAttentionOutputDType::F32 ? 4u : 2u;
}

::ps::kernel::PagedAttentionCommonArgs offset_paged_attention_rows(
    const ::ps::kernel::PagedAttentionCommonArgs& base, uint32_t row_begin,
    uint32_t rows) {
    ::ps::kernel::PagedAttentionCommonArgs region = base;
    region.q = base.q + static_cast<uint64_t>(row_begin) * base.q_row_stride;
    if (base.output != nullptr) {
        region.output = static_cast<unsigned char*>(base.output) +
                        static_cast<uint64_t>(row_begin) *
                            base.output_row_stride *
                            paged_attention_output_element_bytes(base.output_dtype);
    }
    if (base.row_sequence_slots != nullptr)
        region.row_sequence_slots = base.row_sequence_slots + row_begin;
    if (base.row_positions != nullptr)
        region.row_positions = base.row_positions + row_begin;
    if (base.tree_anc_offsets != nullptr)
        region.tree_anc_offsets = base.tree_anc_offsets + row_begin;
    region.rows = rows;
    return region;
}

void fill_decode_pools(const PagedAttentionResolvedArgs& a,
                       PhysicalPagedAttentionKernel& k) {
    switch (a.kv_dtype) {
        case ::ps::kernel::KVCacheDType::BF16:
            k.k_pool = reinterpret_cast<uint64_t>(a.k_bf16_pool);
            k.v_pool = reinterpret_cast<uint64_t>(a.v_bf16_pool);
            break;
        case ::ps::kernel::KVCacheDType::FP8_E4M3:
            k.k_pool = reinterpret_cast<uint64_t>(a.k_fp8_pool);
            k.v_pool = reinterpret_cast<uint64_t>(a.v_fp8_pool);
            k.k_scale_pool = reinterpret_cast<uint64_t>(a.k_scale_pool);
            k.v_scale_pool = reinterpret_cast<uint64_t>(a.v_scale_pool);
            break;
        default:
            break;
    }
    switch (a.kv_dtype) {
        case ::ps::kernel::KVCacheDType::BF16:
            k.variant = a.common.head_dim == ::ps::kernel::kPagedAttentionHeadLimit
                            ? PhysicalPagedAttentionVariant::Bf16FullHead
                            : PhysicalPagedAttentionVariant::Bf16PartialHead;
            break;
        case ::ps::kernel::KVCacheDType::FP8_E4M3:
            k.variant = PhysicalPagedAttentionVariant::Fp8E4M3;
            break;
        default:
            k.variant = PhysicalPagedAttentionVariant::Unsupported;
            break;
    }
}

void fill_prefill_pools(const PagedAttentionResolvedArgs& a,
                        PhysicalPagedAttentionKernel& k) {
    switch (a.kv_dtype) {
        case ::ps::kernel::KVCacheDType::BF16:
            k.variant = PhysicalPagedAttentionVariant::Bf16Prefill;
            k.k_pool = reinterpret_cast<uint64_t>(a.k_bf16_pool);
            k.v_pool = reinterpret_cast<uint64_t>(a.v_bf16_pool);
            break;
        case ::ps::kernel::KVCacheDType::FP8_E4M3:
            k.variant = PhysicalPagedAttentionVariant::Fp8E4M3Prefill;
            k.k_pool = reinterpret_cast<uint64_t>(a.k_fp8_pool);
            k.v_pool = reinterpret_cast<uint64_t>(a.v_fp8_pool);
            k.k_scale_pool = reinterpret_cast<uint64_t>(a.k_scale_pool);
            k.v_scale_pool = reinterpret_cast<uint64_t>(a.v_scale_pool);
            break;
        default:
            k.variant = PhysicalPagedAttentionVariant::Unsupported;
            break;
    }
}

PhysicalResolveStatus append_decode_region(
    const PagedAttentionResolvedArgs& a,
    const ::ps::kernel::PagedAttentionCommonArgs& region,
    const HostExecutionContext& ctx, PhysicalPagedAttentionPlan& out) {
    uint32_t splits = 1u;
    if (region.tree_anc_offsets == nullptr &&
        ctx.decode_attn_partials != nullptr &&
        ctx.spec.max_visible_tokens >= kPagedAttentionSplitMinVisible &&
        a.common.kv_heads != 0u && a.common.q_heads % a.common.kv_heads == 0u) {
        const uint32_t q_per_kv = a.common.q_heads / a.common.kv_heads;
        const uint64_t need =
            static_cast<uint64_t>(region.rows) * a.common.kv_heads * 16u
            * q_per_kv * (2u + ::ps::kernel::kPagedAttentionHeadLimit) * 4u;
        if (need <= ctx.decode_attn_partial_bytes) splits = 16u;
    }

    if (splits <= 1u) {
        if (out.physical_count >= kPhysicalPagedAttentionMaxKernels)
            return PhysicalResolveStatus::NotApplicable;
        PhysicalPagedAttentionKernel& k = out.kernels[out.physical_count++];
        k.common = region;
        k.grid = region.rows * region.q_heads;
        k.workgroup = ::ps::kernel::kPagedAttentionBlock;
        fill_decode_pools(a, k);
        if (k.variant == PhysicalPagedAttentionVariant::Unsupported)
            return PhysicalResolveStatus::NotApplicable;
        return PhysicalResolveStatus::Applicable;
    }

    if (out.physical_count + 2u > kPhysicalPagedAttentionMaxKernels)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t q_per_kv = a.common.q_heads / a.common.kv_heads;
    const bool full_head =
        a.common.head_dim == ::ps::kernel::kPagedAttentionHeadLimit;
    const uint32_t qh_per_wg =
        ::ps::kernel::paged_attention_split_reduce_qh_per_wg();

    PhysicalPagedAttentionKernel& split = out.kernels[out.physical_count++];
    split.common = region;
    split.partials = reinterpret_cast<uint64_t>(ctx.decode_attn_partials);
    split.splits = splits;
    split.q_per_kv = q_per_kv;
    split.grid = region.rows * region.kv_heads * splits;
    split.workgroup = ::ps::kernel::kPagedAttentionSplitBlock;
    {
        split.variant = (q_per_kv == 4u && full_head)
                            ? (a.kv_dtype == ::ps::kernel::KVCacheDType::FP8_E4M3
                                   ? PhysicalPagedAttentionVariant::Fp8SplitQ4Full
                                   : PhysicalPagedAttentionVariant::Bf16SplitQ4Full)
                            : (a.kv_dtype == ::ps::kernel::KVCacheDType::FP8_E4M3
                                   ? PhysicalPagedAttentionVariant::Fp8SplitOther
                                   : PhysicalPagedAttentionVariant::Bf16SplitOther);
        if (a.kv_dtype == ::ps::kernel::KVCacheDType::FP8_E4M3) {
            split.k_pool = reinterpret_cast<uint64_t>(a.k_fp8_pool);
            split.v_pool = reinterpret_cast<uint64_t>(a.v_fp8_pool);
            split.k_scale_pool = reinterpret_cast<uint64_t>(a.k_scale_pool);
            split.v_scale_pool = reinterpret_cast<uint64_t>(a.v_scale_pool);
        } else {
            split.k_pool = reinterpret_cast<uint64_t>(a.k_bf16_pool);
            split.v_pool = reinterpret_cast<uint64_t>(a.v_bf16_pool);
        }
    }

    PhysicalPagedAttentionKernel& reduce = out.kernels[out.physical_count++];
    reduce.common = region;
    reduce.partials = reinterpret_cast<uint64_t>(ctx.decode_attn_partials);
    reduce.splits = splits;
    reduce.q_per_kv = q_per_kv;
    reduce.qh_per_wg = qh_per_wg;
    reduce.workgroup = ::ps::kernel::kPagedAttentionSplitBlock;
    if (q_per_kv != 4u || !full_head) {
        reduce.variant = PhysicalPagedAttentionVariant::Bf16ReduceOther;
        reduce.grid = region.rows * region.kv_heads;
    } else if (qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0u) {
        if (qh_per_wg == 1u) {
            reduce.variant = PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh1;
        } else if (qh_per_wg == 2u) {
            reduce.variant = PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh2;
        } else {
            reduce.variant = PhysicalPagedAttentionVariant::Bf16ReduceOther;
        }
        reduce.grid = region.rows * region.kv_heads * (q_per_kv / qh_per_wg);
    } else {
        reduce.variant = PhysicalPagedAttentionVariant::Bf16ReduceQ4FullBase;
        reduce.grid = region.rows * region.kv_heads;
    }
    return PhysicalResolveStatus::Applicable;
}

PhysicalResolveStatus append_prefill_region(
    const PagedAttentionResolvedArgs& a,
    const ::ps::kernel::PagedAttentionCommonArgs& region,
    PhysicalPagedAttentionPlan& out) {
    if (out.physical_count >= kPhysicalPagedAttentionMaxKernels)
        return PhysicalResolveStatus::NotApplicable;
    PhysicalPagedAttentionKernel& k = out.kernels[out.physical_count++];
    k.common = region;
    fill_prefill_pools(a, k);
    if (k.variant == PhysicalPagedAttentionVariant::Unsupported)
        return PhysicalResolveStatus::NotApplicable;
    return PhysicalResolveStatus::Applicable;
}

}  // namespace

namespace {

enum class PagedAttentionResolveMode : uint8_t {
    HostImmediate = 0,
    StaticPlan = 1,
    StaticPrefill = 2,
};

Result<PhysicalResolveStatus> resolve_paged_attention_physical_impl(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx, PagedAttentionResolveMode mode,
    PhysicalPagedAttentionPlan& out) {
    out = PhysicalPagedAttentionPlan{};
    auto res = resolve_paged_attention_args(program, dispatch_index, ctx);
    if (!res.ok()) return res.status();
    const PagedAttentionResolvedArgs a = res.value();
    if (a.common.rows == 0u) return PhysicalResolveStatus::Skipped;

    const auto* bc = ctx.batch_context;
    const bool static_plan = mode != PagedAttentionResolveMode::HostImmediate;
    const bool prefill_intent =
        mode == PagedAttentionResolveMode::StaticPrefill;
    const bool mixed = !static_plan && bc != nullptr &&
                       bc->requests != nullptr && bc->num_requests > 1u;

    if (!mixed) {
        const bool use_prefill =
            static_plan
                ? prefill_intent
                : should_use_paged_attention_prefill(a.common.rows, ctx);
        const PagedAttentionPlan plan = plan_paged_attention(a, ctx, use_prefill);
        if (!plan.optimized) return PhysicalResolveStatus::NotApplicable;
        out.optimized = true;
        out.use_prefill = plan.use_prefill;
        out.kv_dtype = a.kv_dtype;
        out.splits = plan.splits;
        if (plan.use_prefill) {
            if (append_prefill_region(a, a.common, out) !=
                PhysicalResolveStatus::Applicable)
                return PhysicalResolveStatus::NotApplicable;
            return PhysicalResolveStatus::Applicable;
        }
        if (append_decode_region(a, a.common, ctx, out) !=
            PhysicalResolveStatus::Applicable)
            return PhysicalResolveStatus::NotApplicable;
        return PhysicalResolveStatus::Applicable;
    }

    const PagedAttentionPlan plan = plan_paged_attention(a, ctx, false);
    if (!plan.optimized) return PhysicalResolveStatus::NotApplicable;
    out.optimized = true;
    out.kv_dtype = a.kv_dtype;

    auto is_region_request = [](const ::ps::runtime::DeviceRequestDescriptor& d) {
        return d.execution_class == ::ps::runtime::ExecutionClass::PREFILL ||
               d.execution_class == ::ps::runtime::ExecutionClass::SPEC_VERIFY;
    };

    uint32_t decode_rows = a.common.rows;
    for (uint32_t i = 0u; i < bc->num_requests; ++i) {
        const auto& d = bc->requests[i];
        if (!is_region_request(d)) continue;
        if (d.row_begin < decode_rows) decode_rows = d.row_begin;
    }
    auto stamp_row_begin = [&](uint32_t before, uint32_t row_begin) {
        for (uint32_t i = before; i < out.physical_count; ++i) {
            out.kernels[i].row_begin = row_begin;
        }
    };
    if (decode_rows > 0u) {
        const auto region = offset_paged_attention_rows(a.common, 0u, decode_rows);
        const uint32_t before = out.physical_count;
        if (append_decode_region(a, region, ctx, out) !=
            PhysicalResolveStatus::Applicable)
            return PhysicalResolveStatus::NotApplicable;
        stamp_row_begin(before, 0u);
    }

    uint32_t prefill_regions = 0u;
    for (uint32_t i = 0u; i < bc->num_requests; ++i) {
        const auto& d = bc->requests[i];
        if (!is_region_request(d)) continue;
        if (d.row_count == 0u) continue;
        if (prefill_regions >= kPhysicalPagedAttentionMaxPrefillRegions)
            return PhysicalResolveStatus::NotApplicable;
        ++prefill_regions;
        const auto region =
            offset_paged_attention_rows(a.common, d.row_begin, d.row_count);
        const uint32_t before = out.physical_count;
        if (d.row_count >= kPagedAttentionPrefillMinRows) {
            if (append_prefill_region(a, region, out) !=
                PhysicalResolveStatus::Applicable)
                return PhysicalResolveStatus::NotApplicable;
            out.use_prefill = true;
        } else if (append_decode_region(a, region, ctx, out) !=
                   PhysicalResolveStatus::Applicable) {
            return PhysicalResolveStatus::NotApplicable;
        }
        stamp_row_begin(before, d.row_begin);
    }
    if (out.physical_count == 0u) return PhysicalResolveStatus::NotApplicable;
    return PhysicalResolveStatus::Applicable;
}

}  // namespace

Result<PhysicalResolveStatus> resolve_paged_attention_physical(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx, PhysicalPagedAttentionPlan& out) {
    return resolve_paged_attention_physical_impl(
        program, dispatch_index, ctx, PagedAttentionResolveMode::HostImmediate,
        out);
}

Result<PhysicalResolveStatus> resolve_paged_attention_physical_static_plan(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx, PhysicalPagedAttentionPlan& out) {
    return resolve_paged_attention_physical_impl(
        program, dispatch_index, ctx, PagedAttentionResolveMode::StaticPlan,
        out);
}

Result<PhysicalResolveStatus> resolve_paged_attention_physical_static_prefill(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const HostExecutionContext& ctx, PhysicalPagedAttentionPlan& out) {
    return resolve_paged_attention_physical_impl(
        program, dispatch_index, ctx, PagedAttentionResolveMode::StaticPrefill,
        out);
}

hipError_t launch_paged_attention_physical(const PhysicalPagedAttentionPlan& plan,
                                           hipStream_t stream) {
    if (plan.physical_count == 0u) return hipErrorNotSupported;
    hipError_t herr = hipSuccess;
    for (uint32_t i = 0u; i < plan.physical_count; ++i) {
        const PhysicalPagedAttentionKernel& k = plan.kernels[i];
        switch (k.variant) {
            case PhysicalPagedAttentionVariant::Bf16FullHead:
            case PhysicalPagedAttentionVariant::Bf16PartialHead:
                herr = ::ps::kernel::launch_attention_paged_bf16(
                    k.common, reinterpret_cast<const ::ps::bf16_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::bf16_t*>(k.v_pool), stream);
                break;
            case PhysicalPagedAttentionVariant::Bf16SplitQ4Full:
            case PhysicalPagedAttentionVariant::Bf16SplitOther:
                herr = ::ps::kernel::launch_attention_paged_bf16_split(
                    k.common, reinterpret_cast<const ::ps::bf16_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::bf16_t*>(k.v_pool),
                    reinterpret_cast<float*>(k.partials), k.splits, stream);
                break;
            case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh1:
            case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh2:
            case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullBase:
            case PhysicalPagedAttentionVariant::Bf16ReduceOther:
                herr = ::ps::kernel::launch_attention_paged_bf16_split_reduce(
                    k.common, reinterpret_cast<const float*>(k.partials), k.splits,
                    stream);
                break;
            case PhysicalPagedAttentionVariant::Fp8E4M3:
                herr = ::ps::kernel::launch_attention_paged_fp8_e4m3(
                    k.common, reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.v_pool),
                    reinterpret_cast<const float*>(k.k_scale_pool),
                    reinterpret_cast<const float*>(k.v_scale_pool), stream);
                break;
            case PhysicalPagedAttentionVariant::Fp8SplitQ4Full:
            case PhysicalPagedAttentionVariant::Fp8SplitOther:
                herr = ::ps::kernel::launch_attention_paged_fp8_split(
                    k.common, reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.v_pool),
                    reinterpret_cast<const float*>(k.k_scale_pool),
                    reinterpret_cast<const float*>(k.v_scale_pool),
                    reinterpret_cast<float*>(k.partials), k.splits, stream);
                break;
            case PhysicalPagedAttentionVariant::Bf16Prefill:
                herr = ::ps::kernel::launch_attention_paged_prefill_bf16(
                    k.common, reinterpret_cast<const ::ps::bf16_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::bf16_t*>(k.v_pool), stream);
                break;
            case PhysicalPagedAttentionVariant::Fp8E4M3Prefill:
                herr = ::ps::kernel::launch_attention_paged_prefill_fp8_e4m3(
                    k.common,
                    reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.k_pool),
                    reinterpret_cast<const ::ps::kernel::fp8e4m3_storage_t*>(k.v_pool),
                    reinterpret_cast<const float*>(k.k_scale_pool),
                    reinterpret_cast<const float*>(k.v_scale_pool), stream);
                break;
            default:
                return hipErrorNotSupported;
        }
        if (herr != hipSuccess) return herr;
    }
    return herr;
}

uint32_t make_paged_attention_plan_key(const PagedAttentionPlan& plan) {
    uint32_t key = plan.optimized ? 1u : 0u;
    key |= plan.use_prefill ? 2u : 0u;
    key |= (plan.splits & 0xFFFFu) << 2;
    return key;
}

Result<uint32_t> paged_attention_plan_signature(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    bool use_prefill) {
    auto res = resolve_paged_attention_args(program, dispatch_index, ctx);
    if (!res.ok()) return res.status();
    const PagedAttentionPlan plan = plan_paged_attention(res.value(), ctx, use_prefill);
    return make_paged_attention_plan_key(plan);
}

Result<OptimizedLaunchResult> try_launch_paged_attention(
    const ::ps::runtime::Program& program, size_t dispatch_index, HostExecutionContext& ctx,
    hipStream_t stream) {
    PhysicalPagedAttentionPlan plan{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_paged_attention_physical(program, dispatch_index, ctx, plan);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() == PhysicalResolveStatus::NotApplicable) {
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                "paged-attention: outside measured envelope (rows=%u max_visible=%u) "
                "falling back to correctness path\n",
                ctx.actual_rows, ctx.spec.max_visible_tokens);
        }
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable, 0};
    }
    if (resolved.value() == PhysicalResolveStatus::Skipped)
        return OptimizedLaunchResult{OptimizedLaunchKind::Skipped, 1};
    const hipError_t herr = launch_paged_attention_physical(plan, stream);
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable, 0};
    if (herr != hipSuccess)
        return Status::hip_error("paged-attention optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(plan.physical_count != 0u ? plan.physical_count : 1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 1};
}

}  // namespace ps::execution
