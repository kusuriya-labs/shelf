#include <shelf/models/qwen35/runtime/execution_types.h>
#include <shelf/core/status.h>
#include <cstdio>
#include <cstdlib>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

}  // namespace

int main() {
    using ps::qwen35::runtime::KernelMode;
    using ps::qwen35::runtime::read_kernel_mode;

    unsetenv("SHELF_QWEN35_KERNEL_MODE");
    auto r0 = read_kernel_mode();
    check(r0.ok() && r0.value() == KernelMode::Auto, "unset -> Auto");

    setenv("SHELF_QWEN35_KERNEL_MODE", "auto", 1);
    auto r1 = read_kernel_mode();
    check(r1.ok() && r1.value() == KernelMode::Auto, "auto -> Auto");

    setenv("SHELF_QWEN35_KERNEL_MODE", "correctness", 1);
    auto r2 = read_kernel_mode();
    check(r2.ok() && r2.value() == KernelMode::Correctness, "correctness -> Correctness");

    setenv("SHELF_QWEN35_KERNEL_MODE", "bogus", 1);
    auto r3 = read_kernel_mode();
    check(!r3.ok() && r3.status().code() == ps::Status::Code::invalid_argument,
          "invalid string -> invalid_argument");

    unsetenv("SHELF_QWEN35_KERNEL_MODE");
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
