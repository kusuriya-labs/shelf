#include <shelf/inference/execution/selectors/kv_append_selector.h>

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
using ps::execution::KvAppendImplementation;
using ps::execution::KvAppendSelectorInput;

ps::execution::KvAppendImplementation
sel(KVCacheDType dt, uint32_t kvh, uint32_t hd, uint32_t pt, uint32_t rows) {
    KvAppendSelectorInput si;
    si.kv_dtype = dt;
    si.rows = rows;
    si.kv_heads = kvh;
    si.head_dim = hd;
    si.page_tokens = pt;
    return ps::execution::select_kv_append_implementation(si);
}

}  // namespace

int main() {
    check(sel(KVCacheDType::BF16, 4, 256, 16, 1) == KvAppendImplementation::Optimized,
          "bf16 actual rows=1 -> Optimized");
    check(sel(KVCacheDType::BF16, 4, 256, 16, 64) == KvAppendImplementation::Optimized,
          "bf16 actual rows=64 -> Optimized");
    check(sel(KVCacheDType::BF16, 4, 256, 16, 2048) == KvAppendImplementation::Optimized,
          "bf16 actual rows=2048 -> Optimized");
    check(sel(KVCacheDType::BF16, 4, 256, 16, 0) == KvAppendImplementation::Correctness,
          "bf16 actual rows=0 -> Correctness");
    check(sel(KVCacheDType::BF16, 4, 256, 16, 8193) == KvAppendImplementation::Correctness,
          "bf16 actual rows=8193 (above measured) -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, 4, 256, 16, 1) == KvAppendImplementation::Optimized,
          "fp8 actual rows=1 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, 4, 256, 16, 2048) == KvAppendImplementation::Optimized,
          "fp8 actual rows=2048 -> Optimized");
    check(sel(KVCacheDType::FP8_E4M3, 4, 256, 16, 0) == KvAppendImplementation::Correctness,
          "fp8 actual rows=0 -> Correctness");

    check(sel(KVCacheDType::BF16, 8, 256, 16, 1) == KvAppendImplementation::Correctness,
          "bf16 unknown kv_heads -> Correctness");
    check(sel(KVCacheDType::BF16, 4, 128, 16, 1) == KvAppendImplementation::Correctness,
          "bf16 unknown head_dim -> Correctness");
    check(sel(KVCacheDType::BF16, 4, 256, 32, 1) == KvAppendImplementation::Correctness,
          "bf16 unknown page_tokens -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, 8, 256, 16, 1) == KvAppendImplementation::Correctness,
          "fp8 unknown kv_heads -> Correctness");
    check(sel(KVCacheDType::BF16, 2, 256, 16, 1) == KvAppendImplementation::Optimized,
          "27b tp2 bf16 rows=1 -> Optimized");
    check(sel(KVCacheDType::BF16, 2, 256, 16, 2048) == KvAppendImplementation::Optimized,
          "27b tp2 bf16 rows=2048 -> Optimized");
    check(sel(KVCacheDType::BF16, 2, 256, 16, 8193) == KvAppendImplementation::Correctness,
          "27b tp2 bf16 rows=8193 -> Correctness");
    check(sel(KVCacheDType::FP8_E4M3, 2, 256, 16, 1) == KvAppendImplementation::Correctness,
          "27b tp2 fp8 kv not enabled -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
