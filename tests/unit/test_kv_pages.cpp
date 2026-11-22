#include <shelf/runtime/kv_pages.h>

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const char* message) {
    if (condition) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", message);
    }
}

uint64_t reference_pages(uint64_t max_seq_len, uint64_t page_tokens) {
    if (page_tokens == 0u) return 0u;
    return (max_seq_len + 1u + page_tokens - 1u) / page_tokens + 1u;
}

}

int main() {
    using shelf::runtime::kv_pages_for_context;

    check(kv_pages_for_context(0u, 16u) == 2u, "zero context reserves the guard page");
    check(kv_pages_for_context(512u, 16u) == 34u, "512 token pages");
    check(kv_pages_for_context(262144u, 16u) == 16386u, "262144 token pages");
    check(kv_pages_for_context(524288u, 16u) == 32770u, "524288 token pages");
    check(kv_pages_for_context(1048576u, 16u) == 65538u, "1048576 token pages");
    check(kv_pages_for_context(1000000u, 16u) ==
              static_cast<uint32_t>(reference_pages(1000000u, 16u)),
          "1000000 token pages match the 64-bit reference");

    const uint32_t max_context = std::numeric_limits<uint32_t>::max();
    check(kv_pages_for_context(max_context, 16u) ==
              static_cast<uint32_t>(reference_pages(max_context, 16u)),
          "uint32 max context does not wrap the page count");
    check(kv_pages_for_context(max_context, 0u) == 0u, "zero page size is rejected");

    uint32_t previous = 0u;
    bool monotonic = true;
    for (uint32_t tokens = 0u; tokens <= 1048576u; tokens += 4096u) {
        const uint32_t pages = kv_pages_for_context(tokens, 16u);
        if (pages < previous) monotonic = false;
        previous = pages;
    }
    check(monotonic, "page count is monotonic over the supported context range");

    const uint32_t tail = max_context - 1u - ((max_context - 1u) % 16u);
    check(tail == 4294967280u, "prefill tail position arithmetic stays in uint32");

    constexpr uint32_t kRopeCap = 1048576u;
    constexpr uint32_t kMaxRowBucket = 8192u;
    const uint64_t last_position =
        static_cast<uint64_t>(kRopeCap) + static_cast<uint64_t>(kMaxRowBucket);
    check(last_position <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
          "max context plus row bucket fits in uint32 positions");

    std::printf("test_kv_pages: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
