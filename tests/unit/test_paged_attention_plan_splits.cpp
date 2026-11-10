#include <shelf/inference/execution/paged_attention_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>

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

PagedAttentionPlan masked_plan(bool prefill_intent) {
    PagedAttentionResolvedArgs a;
    a.kv_dtype = KVCacheDType::BF16;
    a.output_dtype = ValueDType::F32;
    a.common.rows = 1u;
    a.common.q_heads = 16u;
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

}  // namespace

int main() {
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
          "a block mask disables the tiled prefill kernel");

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
