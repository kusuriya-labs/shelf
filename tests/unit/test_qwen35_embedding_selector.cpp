#include <phaseshift/inference/execution/selectors/embedding_selector.h>

#include <cstdio>

using ::ps::execution::EmbeddingImplementation;
using ::ps::execution::EmbeddingSelectorInput;
using ::ps::execution::EmbeddingStorage;
using ::ps::execution::select_embedding_implementation;

static int g_fail = 0;

static EmbeddingSelectorInput mk(EmbeddingStorage s, uint32_t rows, uint32_t hidden) {
    EmbeddingSelectorInput in;
    in.storage = s;
    in.rows = rows;
    in.hidden_size = hidden;
    return in;
}

static void expect(EmbeddingImplementation got, EmbeddingImplementation want, const char* msg) {
    const bool ok = got == want;
    if (!ok) ++g_fail;
    std::printf("  [%s] got=%d want=%d %s\n", msg, (int)got, (int)want, ok ? "PASS" : "FAIL");
}

int main() {
    const EmbeddingImplementation C = EmbeddingImplementation::Correctness;
    const EmbeddingImplementation O = EmbeddingImplementation::Optimized;
    const EmbeddingStorage BF = EmbeddingStorage::Bf16;
    expect(select_embedding_implementation(mk(BF, 4, 2560u)), O, "bf16 2560 rows=4 in range");
    expect(select_embedding_implementation(mk(BF, 512, 5120u)), O, "bf16 5120 rows=512 in range");
    expect(select_embedding_implementation(mk(BF, 0, 5120u)), C, "bf16 rows=0");
    expect(select_embedding_implementation(mk(BF, 4, 123456789u)), C, "bf16 unknown hidden");
    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
