#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>

#include <cstdio>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

using shelf::kernel::KVCacheDType;
using shelf::execution::PagedAttentionImplementation;
using shelf::execution::PagedAttentionSelectorInput;
using shelf::runtime::ValueDType;

PagedAttentionImplementation
sel(KVCacheDType dt, ValueDType out, uint32_t qh, uint32_t kvh, uint32_t hd,
    uint32_t pt, uint32_t rows, uint32_t vis) {
    PagedAttentionSelectorInput si;
    si.kv_dtype = dt;
    si.output_dtype = out;
    si.q_heads = qh;
    si.kv_heads = kvh;
    si.head_dim = hd;
    si.page_tokens = pt;
    si.rows = rows;
    si.max_visible_tokens = vis;
    return shelf::execution::select_paged_attention_implementation(si);
}

}  // namespace

int main() {
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 1, 1) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual rows=1 vis=1 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 64, 1024) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual rows=64 vis=1024 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 8, 1, 256, 16, 2048, 4096) ==
              PagedAttentionImplementation::Optimized,
          "bf16 tp2 rank geometry (q 8, kv 1) rows=2048 vis=4096 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 8, 1, 256, 16, 1, 262144) ==
              PagedAttentionImplementation::Optimized,
          "fp8 tp2 rank geometry (q 8, kv 1) rows=1 vis=262144 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 8, 1, 256, 16, 1, 262145) ==
              PagedAttentionImplementation::Correctness,
          "bf16 tp2 rank geometry above the measured visible range -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 2048, 4096) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual rows=2048 vis=4096 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 0, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 actual rows=0 -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 8193, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 actual rows=8193 (above measured) -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 1, 4097) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual vis=4097 (below measured) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 1, 32768) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual vis=32768 (measured max) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 1, 32769) ==
              PagedAttentionImplementation::Correctness,
          "bf16 actual vis=32769 (above measured) -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 4, 256, 16, 1, 1) ==
              PagedAttentionImplementation::Optimized,
          "fp8 actual rows=1 vis=1 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 4, 256, 16, 2048, 4096) ==
              PagedAttentionImplementation::Optimized,
          "fp8 actual rows=2048 vis=4096 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 4, 256, 16, 0, 16) ==
              PagedAttentionImplementation::Correctness,
          "fp8 actual rows=0 -> Correctness");

    check(sel(KVCacheDType::BF16, ValueDType::BF16, 16, 4, 256, 16, 1, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 output BF16 (unmeasured) -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 8, 8, 256, 16, 1, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 unknown q_heads/kv_heads -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 128, 16, 1, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 unknown head_dim -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 32, 1, 16) ==
              PagedAttentionImplementation::Correctness,
          "bf16 unknown page_tokens -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 8, 8, 256, 16, 1, 16) ==
              PagedAttentionImplementation::Correctness,
          "fp8 unknown q_heads/kv_heads -> Correctness");

    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 1, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 decode rows=1 vis=64 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 64, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 prefill rows=64 vis=64 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 1, 2048) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 decode rows=1 vis=2048 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 8193, 64) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 rows=8193 above measured -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 1, 524288) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 vis=524288 (measured max) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 1, 524289) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 vis=524289 above measured -> Correctness");

    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 decode rows=1 vis=64 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 8, 1697) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 spec verify rows=8 vis=1697 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 64, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 prefill rows=64 vis=64 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 524288) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 vis=524288 (measured max) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 8193, 64) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 fp8 rows=8193 above measured -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 524289) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 fp8 vis=524289 above measured -> Correctness");

    check(sel(KVCacheDType::BF16, ValueDType::F32, 24, 4, 256, 16, 1, 524288) ==
              PagedAttentionImplementation::Optimized,
          "27b vis=524288 (measured max) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 24, 4, 256, 16, 1, 524289) ==
              PagedAttentionImplementation::Correctness,
          "27b vis=524289 above measured -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 24, 4, 256, 16, 1, 524288) ==
              PagedAttentionImplementation::Optimized,
          "27b fp8 vis=524288 (measured max) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 24, 4, 256, 16, 1, 524289) ==
              PagedAttentionImplementation::Correctness,
          "27b fp8 vis=524289 above measured -> Correctness");

    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 2, 256, 16, 1, 64) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 rows=1 vis=64 (measured) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 2, 256, 16, 8192, 8192) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 rows=8192 vis=8192 (measured) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 2, 256, 16, 1, 262144) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 vis=262144 (measured max) -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 2, 256, 16, 1, 262145) ==
              PagedAttentionImplementation::Correctness,
          "moe 16/2/256 vis=262145 above measured -> Correctness");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 2, 256, 16, 8193, 64) ==
              PagedAttentionImplementation::Correctness,
          "moe 16/2/256 rows=8193 above measured -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 2, 256, 16, 1, 64) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 fp8 kv rows=1 vis=64 (measured) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 2, 256, 16, 2048, 4096) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 fp8 kv rows=2048 vis=4096 (measured) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 2, 256, 16, 1, 262144) ==
              PagedAttentionImplementation::Optimized,
          "moe 16/2/256 fp8 kv vis=262144 (measured max) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 2, 256, 16, 1, 262145) ==
              PagedAttentionImplementation::Correctness,
          "moe 16/2/256 fp8 kv vis=262145 above measured -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 16, 2, 256, 16, 8193, 64) ==
              PagedAttentionImplementation::Correctness,
          "moe 16/2/256 fp8 kv rows=8193 above measured -> Correctness");

    check(::shelf::kernel::paged_attention_bf16_split_supports(4u), "bf16 split q_per_kv 4");
    check(::shelf::kernel::paged_attention_bf16_split_supports(6u), "bf16 split q_per_kv 6");
    check(::shelf::kernel::paged_attention_bf16_split_supports(8u), "bf16 split q_per_kv 8");
    check(!::shelf::kernel::paged_attention_bf16_split_supports(1u), "bf16 split q_per_kv 1");
    check(::shelf::kernel::paged_attention_fp8_split_supports(4u), "fp8 split q_per_kv 4");
    check(::shelf::kernel::paged_attention_fp8_split_supports(6u), "fp8 split q_per_kv 6");
    check(::shelf::kernel::paged_attention_fp8_split_supports(8u),
          "fp8 split q_per_kv 8 has an instantiation");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
