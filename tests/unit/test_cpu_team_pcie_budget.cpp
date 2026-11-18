#include <shelf/runtime/cpu_team.h>
#include <shelf/core/pcie_budget.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

void check_team(std::uint32_t threads, const char* message) {
    auto team = shelf::runtime::CpuTeam::create(threads);
    check(team->size() == threads, "team size matches the request");
    const std::uint32_t count = 1000u;
    std::vector<std::uint32_t> hits(count, 0u);
    for (int round = 0; round < 3; ++round) {
        team->parallel_for(count, [&hits](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t i = begin; i < end; ++i) ++hits[i];
        });
    }
    bool covered = true;
    for (const std::uint32_t value : hits) covered = covered && value == 3u;
    check(covered, message);
    team->parallel_for(0u, [](std::uint32_t, std::uint32_t) {});
}

}  // namespace

int main() {
    check_team(1u, "serial team covers every index exactly once");
    check_team(4u, "parallel team covers every index exactly once");

    shelf::PcieBudget budget(0.25, 20.0);
    check(budget.bytes_for(1000u) == 5000000000ull, "bytes_for scales with the share");
    check(budget.fraction() == 0.25 && budget.gbps() == 20.0, "fraction and rate are kept");
    check(budget.try_acquire(1000000u), "a small copy fits the initial burst");
    check(!budget.try_acquire(1000000000000ull), "a copy larger than the burst is deferred");
    check(budget.deferred() == 1u, "deferrals are counted");
    check(budget.available() <= 10000000ull, "the burst is bounded by the share");

    shelf::PcieBudget tight(0.05, 20.0);
    shelf::PcieBudget open(1.0, 20.0);
    check(open.bytes_for(1000u) > tight.bytes_for(1000u), "a larger share admits more bytes");
    check(tight.fraction() == 0.05 && open.fraction() == 1.0, "shares are kept");

    shelf::PcieBudget clamped(0.0, -1.0);
    check(clamped.fraction() >= 0.05 && clamped.gbps() > 0.0, "invalid settings are clamped");

    if (failures == 0) {
        std::printf("PASS: cpu team and pcie budget\n");
        return 0;
    }
    std::printf("FAIL: %d checks\n", failures);
    return 1;
}
