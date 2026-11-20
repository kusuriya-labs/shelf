#include <shelf/runtime/parallel/parallel_topology.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using shelf::runtime::ParallelAxis;
using shelf::runtime::ParallelCoord;
using shelf::runtime::ParallelShape;
using shelf::runtime::ParallelTopology;
using shelf::runtime::RankGroup;

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

bool same(const RankGroup& group, const std::vector<std::uint32_t>& want) {
    return group.ranks == want;
}

bool same(const std::vector<std::uint32_t>& got,
          const std::vector<std::uint32_t>& want) {
    return got == want;
}

ParallelTopology must_create(ParallelShape shape, std::vector<int> devices) {
    auto result = ParallelTopology::create(shape, std::move(devices));
    if (!result.ok()) {
        std::printf("create failed: %s\n", result.status().message().c_str());
        std::exit(1);
    }
    return result.release();
}

}  // namespace

int main() {
    std::printf("=== ParallelTopology Tests ===\n");

    {
        auto bad = ParallelTopology::create(ParallelShape{1, 1, 1}, {0, 1});
        check(!bad.ok(), "device count mismatch rejected");
    }
    {
        auto bad = ParallelTopology::create(ParallelShape{1, 1, 2}, {0, 0});
        check(!bad.ok(), "duplicate device rejected");
    }
    {
        auto bad = ParallelTopology::create(ParallelShape{0, 1, 1}, {0});
        check(!bad.ok(), "zero axis rejected");
    }

    {
        ParallelTopology topology =
            must_create(ParallelShape{1, 2, 2}, {0, 1, 2, 3});
        check(topology.world_size() == 4u, "ep2tp2 world size");
        check(topology.stage_size() == 4u, "ep2tp2 stage size");
        check(topology.rank(ParallelCoord{0, 0, 0}) == 0u, "ep2tp2 coord 0,0,0");
        check(topology.rank(ParallelCoord{0, 1, 0}) == 2u, "ep2tp2 coord 0,1,0");
        check(topology.coord(3u).ep == 1u && topology.coord(3u).tp == 1u,
              "ep2tp2 rank 3 coordinate");
        check(topology.device(2u) == 2, "ep2tp2 rank 2 device");
        check(same(topology.tp_group(0u, 0u), {0u, 1u}), "ep2tp2 tp group ep0");
        check(same(topology.tp_group(0u, 1u), {2u, 3u}), "ep2tp2 tp group ep1");
        check(same(topology.ep_group(0u, 0u), {0u, 2u}), "ep2tp2 ep group tp0");
        check(same(topology.ep_group(0u, 1u), {1u, 3u}), "ep2tp2 ep group tp1");
        check(topology.ep_group(0u, 0u).axis == ParallelAxis::Expert, "ep group axis");
        check(topology.tp_group(0u, 0u).axis == ParallelAxis::Tensor, "tp group axis");
    }

    {
        ParallelTopology topology =
            must_create(ParallelShape{2, 2, 1}, {0, 1, 2, 3});
        check(same(topology.stage_ranks(0u), {0u, 1u}), "pp2ep2tp1 stage 0");
        check(same(topology.stage_ranks(1u), {2u, 3u}), "pp2ep2tp1 stage 1");
        check(same(topology.ep_group(0u, 0u), {0u, 1u}), "pp2ep2tp1 stage0 ep group");
        check(same(topology.ep_group(1u, 0u), {2u, 3u}), "pp2ep2tp1 stage1 ep group");
        check(same(topology.tp_group(0u, 0u), {0u}), "pp2ep2tp1 tp group is one rank");
    }

    {
        ParallelTopology topology =
            must_create(ParallelShape{2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
        check(topology.world_size() == 8u, "pp2ep2tp2 world size");
        check(same(topology.tp_group(0u, 0u), {0u, 1u}), "pp2ep2tp2 tp pp0 ep0");
        check(same(topology.tp_group(0u, 1u), {2u, 3u}), "pp2ep2tp2 tp pp0 ep1");
        check(same(topology.tp_group(1u, 0u), {4u, 5u}), "pp2ep2tp2 tp pp1 ep0");
        check(same(topology.tp_group(1u, 1u), {6u, 7u}), "pp2ep2tp2 tp pp1 ep1");
        check(same(topology.ep_group(0u, 0u), {0u, 2u}), "pp2ep2tp2 ep pp0 tp0");
        check(same(topology.ep_group(0u, 1u), {1u, 3u}), "pp2ep2tp2 ep pp0 tp1");
        check(same(topology.ep_group(1u, 0u), {4u, 6u}), "pp2ep2tp2 ep pp1 tp0");
        check(same(topology.ep_group(1u, 1u), {5u, 7u}), "pp2ep2tp2 ep pp1 tp1");
        check(same(topology.stage_group(1u), {4u, 5u, 6u, 7u}), "pp2ep2tp2 stage group");
    }

    {
        ParallelTopology topology = must_create(ParallelShape{1, 1, 2}, {0, 1});
        check(same(topology.tp_group(0u, 0u), {0u, 1u}), "legacy tp group");
        check(same(topology.ep_group(0u, 0u), {0u}), "legacy ep group is the rank itself");
        check(topology.shape().ep == 1u, "legacy topology has one expert axis");
    }

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
