#include <shelf/runtime/parallel/parallel_topology.h>

#include <algorithm>

namespace shelf {
namespace runtime {

Result<ParallelTopology> ParallelTopology::create(ParallelShape shape,
                                                  std::vector<int> devices) {
    if (shape.pp == 0u || shape.ep == 0u || shape.tp == 0u) {
        return Status::invalid_argument("parallel shape axes must be positive", __FILE__,
                                        __LINE__);
    }
    const std::size_t expected = static_cast<std::size_t>(shape.pp) * shape.ep * shape.tp;
    if (devices.size() != expected) {
        return Status::invalid_argument(
            "device count must equal pipeline * expert * tensor", __FILE__, __LINE__);
    }
    std::vector<int> sorted = devices;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        return Status::invalid_argument("duplicate device in parallel topology", __FILE__,
                                        __LINE__);
    }
    ParallelTopology topology;
    topology.shape_ = shape;
    topology.devices_ = std::move(devices);
    return topology;
}

std::uint32_t ParallelTopology::world_size() const noexcept {
    return static_cast<std::uint32_t>(devices_.size());
}

std::size_t ParallelTopology::stage_size() const noexcept {
    return static_cast<std::size_t>(shape_.ep) * shape_.tp;
}

std::uint32_t ParallelTopology::rank(ParallelCoord coord) const noexcept {
    return (coord.pp * shape_.ep + coord.ep) * shape_.tp + coord.tp;
}

ParallelCoord ParallelTopology::coord(std::uint32_t rank) const noexcept {
    ParallelCoord coord;
    coord.pp = rank / (shape_.ep * shape_.tp);
    const std::uint32_t within = rank % (shape_.ep * shape_.tp);
    coord.ep = within / shape_.tp;
    coord.tp = within % shape_.tp;
    return coord;
}

int ParallelTopology::device(std::uint32_t rank) const noexcept {
    return rank < devices_.size() ? devices_[rank] : -1;
}

std::vector<std::uint32_t> ParallelTopology::stage_ranks(std::uint32_t pp) const {
    std::vector<std::uint32_t> ranks;
    if (pp >= shape_.pp) return ranks;
    ranks.reserve(stage_size());
    for (std::uint32_t e = 0; e < shape_.ep; ++e) {
        for (std::uint32_t t = 0; t < shape_.tp; ++t) {
            ranks.push_back(rank(ParallelCoord{pp, e, t}));
        }
    }
    return ranks;
}

RankGroup ParallelTopology::tp_group(std::uint32_t pp, std::uint32_t ep) const {
    RankGroup group;
    group.axis = ParallelAxis::Tensor;
    if (pp >= shape_.pp || ep >= shape_.ep) return group;
    group.ranks.reserve(shape_.tp);
    for (std::uint32_t t = 0; t < shape_.tp; ++t) {
        group.ranks.push_back(rank(ParallelCoord{pp, ep, t}));
    }
    return group;
}

RankGroup ParallelTopology::ep_group(std::uint32_t pp, std::uint32_t tp) const {
    RankGroup group;
    group.axis = ParallelAxis::Expert;
    if (pp >= shape_.pp || tp >= shape_.tp) return group;
    group.ranks.reserve(shape_.ep);
    for (std::uint32_t e = 0; e < shape_.ep; ++e) {
        group.ranks.push_back(rank(ParallelCoord{pp, e, tp}));
    }
    return group;
}

RankGroup ParallelTopology::stage_group(std::uint32_t pp) const {
    RankGroup group;
    group.axis = ParallelAxis::Pipeline;
    group.ranks = stage_ranks(pp);
    return group;
}

}  // namespace runtime
}  // namespace shelf
