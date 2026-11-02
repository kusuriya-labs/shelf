#include <phaseshift/runtime/staging_layout.h>
#include <cstdio>
#include <cstdint>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        printf("PASS: %s\n", name);
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

::ps::runtime::DispatchBinding make_binding() {
    ::ps::runtime::DispatchBinding b{};
    b.kernel_id = ::ps::runtime::KernelId::LINEAR_BF16;
    b.input_slots[0] = 0;
    b.input_slots[1] = 1;
    b.input_count = 2;
    b.output_slots[0] = 2;
    b.output_count = 1;
    return b;
}

::ps::runtime::Program make_program(uint32_t n_dispatches, uint32_t n_ranges,
                                    uint32_t n_states) {
    ::ps::runtime::Program p{};
    p.dispatches.reserve(n_dispatches);
    for (uint32_t i = 0; i < n_dispatches; ++i) p.dispatches.push_back(make_binding());
    p.workspace.ranges.resize(n_ranges);
    for (uint32_t i = 0; i < n_ranges; ++i) {
        p.workspace.ranges[i].offset = static_cast<std::size_t>(i) * 64;
        p.workspace.ranges[i].bytes = 64;
    }
    p.states.resize(n_states);
    return p;
}

}

int main() {
    printf("=== Dispatch Staging Size Contract ===\n");

    const ::ps::runtime::Program a = make_program(10, 10, 2);
    const ::ps::runtime::Program b = make_program(1000, 1000, 200);
    const ::ps::runtime::DispatchBinding binding = make_binding();

    const uint64_t image = ::ps::runtime::staging_bytes_for(binding);
    check(image == ::ps::runtime::staging_section_offsets(3).total,
          "image size equals section layout");
    check(image < 512, "image is small (no embedded range list)");

    const uint64_t old_full_image =
        16 + 3 * sizeof(::ps::runtime::DeviceValueBinding) +
        1000ull * sizeof(::ps::runtime::DeviceWorkspaceRange);
    check(image < old_full_image / 4,
          "image does not scale with workspace range count");

    check(::ps::runtime::max_host_staging_bytes(a) == image, "scratch A equals one image");
    check(::ps::runtime::max_host_staging_bytes(a) == ::ps::runtime::max_host_staging_bytes(b),
          "scratch independent of dispatch count");

    const uint64_t meta_a = ::ps::runtime::program_staging_meta_bytes(10, 2);
    const uint64_t meta_b = ::ps::runtime::program_staging_meta_bytes(1000, 200);
    const uint64_t expect_a =
        ((10ull * sizeof(::ps::runtime::DeviceWorkspaceRange) + 15) & ~15ull) +
        ((2ull * sizeof(::ps::runtime::DeviceStateBinding) + 15) & ~15ull);
    const uint64_t expect_b =
        ((1000ull * sizeof(::ps::runtime::DeviceWorkspaceRange) + 15) & ~15ull) +
        ((200ull * sizeof(::ps::runtime::DeviceStateBinding) + 15) & ~15ull);
    check(meta_a == expect_a, "meta A exact bytes");
    check(meta_b == expect_b, "meta B exact bytes");
    check(meta_b > meta_a, "meta scales only with static range/state count");

    const uint64_t pool_a = 10ull * image + meta_a;
    const uint64_t pool_b = 1000ull * image + meta_b;
    const uint64_t old_pool_b =
        1000ull * old_full_image;
    check(pool_b < old_pool_b / 10, "pool B far below dispatch count x full image");
    const uint64_t delta = pool_b - pool_a;
    const uint64_t explainable = 990ull * image + (meta_b - meta_a);
    check(delta == explainable, "pool delta is per-dispatch small image plus meta delta");

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
