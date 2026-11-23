#include "apps/compute/arena_budget.h"

#include <cstddef>
#include <cstdio>

int main() {
    using namespace shelf::app;
    std::size_t bytes = 0;
    std::size_t capacity = 0;
    const std::size_t total = 16ull * 1024 * 1024 * 1024;
    const auto check = [](bool ok, const char* name) {
        if (!ok) std::fprintf(stderr, "%s failed\n", name);
        return ok;
    };
    bool ok = true;
    ok &= check(arena_budget_bytes(2.0, bytes) && bytes == static_cast<std::size_t>(1.7 * kGiBBytes), "positive");
    ok &= check(!arena_budget_bytes(0.3, bytes), "boundary");
    ok &= check(!arena_budget_bytes(0.1, bytes), "below boundary");
    ok &= check(!arena_budget_bytes(0.0, bytes), "zero");
    ok &= check(arena_budget_bytes(-2.0, bytes) && bytes == static_cast<std::size_t>(2.3 * kGiBBytes), "negative");
    ok &= check(arena_total_minus(total, bytes, capacity) && capacity == total - bytes, "total minus");
    ok &= check(arena_budget_bytes(-15.7, bytes) && !arena_total_minus(total, bytes, capacity), "equal total");
    ok &= check(arena_budget_bytes(-16.0, bytes) && !arena_total_minus(total, bytes, capacity), "adjusted exceeds total");
    ok &= check(!arena_total_minus(total, total + 1, capacity), "exceeds total");
    return ok ? 0 : 1;
}
