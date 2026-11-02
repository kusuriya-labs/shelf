#include <shelf/inference/execution/selectors/paged_attention_selector.h>

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

using ps::kernel::KVCacheDType;
using ps::execution::PagedAttentionImplementation;
using ps::execution::PagedAttentionSelectorInput;
using ps::runtime::ValueDType;

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
    return ps::execution::select_paged_attention_implementation(si);
}

}  // namespace

int main() {
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 1, 1) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual rows=1 vis=1 -> Optimized");
    check(sel(KVCacheDType::BF16, ValueDType::F32, 16, 4, 256, 16, 64, 1024) ==
              PagedAttentionImplementation::Optimized,
          "bf16 actual rows=64 vis=1024 -> Optimized");
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
    check(sel(KVCacheDType::BF16, ValueDType::F32, 12, 2, 256, 16, 1, 262145) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 vis=262145 above measured -> Correctness");

    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 decode rows=1 vis=64 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 8, 1697) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 spec verify rows=8 vis=1697 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 64, 64) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 prefill rows=64 vis=64 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 262144) ==
              PagedAttentionImplementation::Optimized,
          "27b tp2 fp8 vis=262144 (measured max) -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 8193, 64) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 fp8 rows=8193 above measured -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, ValueDType::F32, 12, 2, 256, 16, 1, 262145) ==
              PagedAttentionImplementation::Correctness,
          "27b tp2 fp8 vis=262145 above measured -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
