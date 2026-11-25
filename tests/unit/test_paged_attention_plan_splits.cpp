#include <shelf/inference/execution/paged_attention_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/attention/paged_verify.h>

#include <cstdio>
#include <initializer_list>
#include <cstdint>

namespace {

int failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++failed;
    }
}

using shelf::execution::HostExecutionContext;
using shelf::execution::PagedAttentionPlan;
using shelf::execution::PagedAttentionResolvedArgs;
using shelf::kernel::KVCacheDType;
using shelf::runtime::ValueDType;

// The pointer members only feed the workspace-size comparison; they are never read here.
alignas(64) uint8_t kPartials[64] = {};
alignas(64) uint8_t kBlockMask[64] = {};
uint32_t kTreeAnc[2] = {0u, 1u};

PagedAttentionPlan plan_for(uint32_t q_heads, uint32_t kv_heads, uint32_t visible,
                            uint32_t partial_bytes, bool tree,
                            KVCacheDType kv_dtype = KVCacheDType::BF16) {
    PagedAttentionResolvedArgs a;
    a.kv_dtype = kv_dtype;
    a.output_dtype = ValueDType::F32;
    a.common.rows = 1u;
    a.common.q_heads = q_heads;
    a.common.kv_heads = kv_heads;
    a.common.head_dim = 256u;
    a.common.page_tokens = 16u;
    a.common.tree_anc_offsets = tree ? kTreeAnc : nullptr;

    HostExecutionContext ctx;
    ctx.spec.max_visible_tokens = visible;
    ctx.decode_attn_partials = partial_bytes == 0u ? nullptr : kPartials;
    ctx.decode_attn_partial_bytes = partial_bytes;
    return shelf::execution::plan_paged_attention(a, ctx, false);
}

constexpr uint32_t kPlentyBytes = 1u << 20;

PagedAttentionPlan masked_plan(bool prefill_intent, uint32_t rows = 1u,
                               KVCacheDType dtype = KVCacheDType::BF16) {
    PagedAttentionResolvedArgs a;
    a.kv_dtype = dtype;
    a.output_dtype = ValueDType::F32;
    a.common.rows = rows;
    a.common.q_heads = dtype == KVCacheDType::OCTAVE ? 24u : 16u;
    a.common.kv_heads = 4u;
    a.common.head_dim = 256u;
    a.common.page_tokens = 16u;
    a.common.block_mask = kBlockMask;
    a.common.mask_stride = 8u;
    a.common.mask_blocks = 64u;
    a.common.mask_block_shift = 2u;
    HostExecutionContext ctx;
    ctx.spec.max_visible_tokens = 4096u;
    ctx.decode_attn_partials = kPartials;
    ctx.decode_attn_partial_bytes = kPlentyBytes;
    return shelf::execution::plan_paged_attention(a, ctx, prefill_intent);
}

void test_previous_verify_descriptors() {
    namespace rt = shelf::runtime;
    rt::Program program;
    auto& attention = program.dispatches.emplace_back();
    attention.kernel_id = rt::KernelId::PAGED_ATTENTION;
    attention.descriptor_slot = 1u;
    attention.group_size = 256u;
    attention.input_count = attention.output_count = 1u;
    attention.output_slots[0] = 1u;
    program.states.emplace_back();
    constexpr uint32_t width = 16u * 256u;
    program.values = {
        {rt::ValueId{0}, rt::ValueStorage::WORKSPACE, ValueDType::BF16,
         rt::ValueRowDomain::TOKEN_ROWS, 0u, width, width},
        {rt::ValueId{1}, rt::ValueStorage::WORKSPACE, ValueDType::F32,
         rt::ValueRowDomain::TOKEN_ROWS, 0u, width, width},
    };
    rt::DeviceRequestDescriptor previous[3]{};
    for (uint32_t i = 0; i < 3u; ++i) {
        previous[i].execution_class = rt::ExecutionClass::SPEC_VERIFY;
        previous[i].row_begin = 8u * i;
        previous[i].row_count = 8u;
    }
    rt::DeviceBatchContext device{};
    device.requests = previous;
    device.num_requests = 3u;
    rt::DeviceRequestDescriptor current{};
    current.execution_class = rt::ExecutionClass::DECODE;
    current.row_count = 1u;
    HostExecutionContext ctx{};
    ctx.batch_context = &device;
    ctx.staged_requests = &current;
    ctx.staged_request_count = ctx.num_requests = ctx.num_decode_requests = ctx.actual_rows = 1u;
    ctx.spec.max_visible_tokens = 16u;
    static uint8_t workspace[24u * width * sizeof(float)]{};
    shelf::bf16_t kv{};
    uint32_t positions[24]{};
    ctx.workspace = workspace;
    ctx.row_positions = ctx.row_sequence_slots = positions;
    ctx.model_state.block_table = {positions, 1u, 1u};
    ctx.model_state.kv_bf16 = {&kv, &kv, 16u, 1u, 1u, 4u, 256u, 1024u, 16384u, 16384u};
    shelf::execution::PhysicalPagedAttentionPlan plan;
    auto status = shelf::execution::resolve_paged_attention_physical(program, 0u, ctx, plan);
    check(status.ok() && status.value() == shelf::execution::PhysicalResolveStatus::Applicable &&
          plan.physical_count == 1u && plan.kernels[0].common.rows == 1u,
          "直前の verify の行数を次の decode に持ち越さない");
}

}  // namespace

int main() {
    test_previous_verify_descriptors();
    for (auto dtype : {KVCacheDType::BF16, KVCacheDType::FP8_E4M3, KVCacheDType::OCTAVE}) {
        PagedAttentionResolvedArgs args{};
        args.kv_dtype = dtype;
        args.output_dtype = ValueDType::F32;
        auto& c = args.common;
        c.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
        c.rows = 16; c.q_heads = 24; c.kv_heads = 4; c.head_dim = 256; c.page_tokens = 16;
        HostExecutionContext ctx{};
        ctx.num_requests = 1;
        ctx.num_verify_requests = 1;
        ctx.spec.max_visible_tokens = 2048;
        ctx.decode_attn_partials = kPartials;
        ctx.decode_attn_partial_bytes = shelf::kernel::paged_verify_workspace_bytes(
            c.rows, c.q_heads, shelf::kernel::kPagedVerifySplits, dtype);
        auto plan = [&] { return shelf::execution::plan_paged_attention(args, ctx, false); };
        const auto tiled = plan();
        check(tiled.verify_tile && tiled.splits == shelf::kernel::kPagedVerifySplits,
              "単一要求の長文 verify");
        --ctx.decode_attn_partial_bytes;
        check(!plan().verify_tile, "一時領域の容量境界");
        ++ctx.decode_attn_partial_bytes;
        ctx.num_requests = 2;
        check(!plan().verify_tile, "複数要求のページ表を分離する");
        ctx.num_requests = 1;
        ctx.spec.max_visible_tokens = 2047;
        check(!plan().verify_tile && shelf::execution::make_paged_attention_plan_key(plan()) !=
              shelf::execution::make_paged_attention_plan_key(tiled), "長文境界の Graph 再選択");
        ctx.spec.max_visible_tokens = 2048;
        c.tree_anc_offsets = kTreeAnc;
        check(!plan().verify_tile, "木の可視性を維持する");
        c.tree_anc_offsets = nullptr;
        c.block_mask = kBlockMask;
        check(!plan().verify_tile, "ブロックマスクを維持する");
    }
    for (uint32_t rows : {1u, 23u}) {
        PagedAttentionResolvedArgs a;
        a.kv_dtype = KVCacheDType::BF16;
        a.output_dtype = ValueDType::BF16;
        a.common.rows = rows; a.common.q_heads = 16; a.common.kv_heads = 1;
        a.common.head_dim = 128; a.common.page_tokens = 16;
        HostExecutionContext ctx;
        ctx.spec.max_visible_tokens = 2048;
        ctx.decode_attn_partials = kPartials;
        ctx.decode_attn_partial_bytes = kPlentyBytes;
        auto plan = shelf::execution::plan_paged_attention(a, ctx, true);
        check(plan.optimized && !plan.use_prefill && plan.splits == 1,
              "PLaMo uses partial-head decode kernel for all rows");
    }
    check(plan_for(16, 4, 4096, kPlentyBytes, false).splits == 16u,
          "q_per_kv 4 with long visible takes splits");
    check(plan_for(12, 2, 4096, kPlentyBytes, false).splits == 16u,
          "q_per_kv 6 with long visible takes splits");
    check(plan_for(16, 2, 4096, kPlentyBytes, false).splits == 16u,
          "q_per_kv 8 takes splits in bf16");
    check(plan_for(16, 2, 262144, kPlentyBytes, false).splits == 16u,
          "q_per_kv 8 takes splits at the measured max visible");
    {
        PagedAttentionResolvedArgs a;
        a.kv_dtype = KVCacheDType::BF16;
        a.output_dtype = ValueDType::BF16;
        a.common.rows = 1u;
        a.common.q_heads = 16u;
        a.common.kv_heads = 1u;
        a.common.head_dim = 128u;
        a.common.page_tokens = 16u;
        a.common.block_mask = kBlockMask;
        a.qsa_top_k = 512u;
        HostExecutionContext ctx;
        ctx.spec.max_visible_tokens = 1024u;
        check(shelf::execution::plan_paged_attention(a, ctx, false).mask_all_visible,
              "visible blocks within the QSA budget drop the dense mask");
        ctx.spec.max_visible_tokens = 4096u;
        check(!shelf::execution::plan_paged_attention(a, ctx, false).mask_all_visible,
              "over-budget visible blocks keep the dense mask");
        a.common.block_mask = nullptr;
        ctx.spec.max_visible_tokens = 1024u;
        check(!shelf::execution::plan_paged_attention(a, ctx, false).mask_all_visible,
              "a batch without a mask has nothing to drop");
    }
    check(plan_for(16, 2, 4096, kPlentyBytes, false).optimized,
          "q_per_kv 8 is still the optimized path");
    check(plan_for(16, 2, 4096, kPlentyBytes, false, KVCacheDType::FP8_E4M3).splits == 16u,
          "fp8 q_per_kv 8 takes splits");
    check(plan_for(16, 2, 4096, kPlentyBytes, false, KVCacheDType::FP8_E4M3).optimized,
          "fp8 q_per_kv 8 is the optimized path");
    check(plan_for(16, 2, 4096, 0u, false, KVCacheDType::FP8_E4M3).splits == 1u,
          "fp8 q_per_kv 8 without a partial workspace stays non-split");

    check(plan_for(16, 4, 1024, kPlentyBytes, false).splits == 1u,
          "short visible stays non-split");
    check(plan_for(16, 4, 4096, 0u, false).splits == 1u,
          "no partial workspace stays non-split");
    check(plan_for(16, 4, 4096, 1024u, false).splits == 1u,
          "partial workspace too small stays non-split");
    check(plan_for(16, 4, 4096, kPlentyBytes, true).splits == 1u,
          "tree rows stay non-split");
    check(masked_plan(false).splits == 1u,
          "a block mask disables the split decode kernel");
    check(!masked_plan(false).use_prefill && !masked_plan(true).use_prefill,
          "a block mask keeps the decode kernel below the prefill row threshold");
    check(masked_plan(true, 256u).use_prefill,
          "bf16 block mask uses the tiled prefill at 256 rows");
    check(masked_plan(true, 256u, KVCacheDType::FP8_E4M3).use_prefill,
          "fp8 block mask uses the tiled prefill at 256 rows");
    check(masked_plan(true, 256u, KVCacheDType::OCTAVE).use_prefill,
          "octave block mask uses the tiled prefill at 256 rows");

    check(::shelf::kernel::paged_attention_bf16_split_supports(4u) &&
              ::shelf::kernel::paged_attention_bf16_split_supports(6u) &&
              ::shelf::kernel::paged_attention_bf16_split_supports(8u) &&
              !::shelf::kernel::paged_attention_bf16_split_supports(1u),
          "bf16 split capability is the single source and covers 4 / 6 / 8");
    check(::shelf::kernel::paged_attention_fp8_split_supports(4u) &&
              ::shelf::kernel::paged_attention_fp8_split_supports(6u) &&
              ::shelf::kernel::paged_attention_fp8_split_supports(8u) &&
              !::shelf::kernel::paged_attention_fp8_split_supports(1u),
          "fp8 split capability is the single source and covers 4 / 6 / 8");

    std::printf("\nResults: %d failed\n", failed);
    return failed > 0 ? 1 : 0;
}
