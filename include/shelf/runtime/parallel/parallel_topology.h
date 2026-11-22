#pragma once

#include <shelf/core/status.h>

#include <cstdint>
#include <vector>

namespace shelf {
namespace runtime {

struct ParallelShape {
    std::uint32_t pp = 1;
    std::uint32_t ep = 1;
    std::uint32_t tp = 1;
};

struct ParallelCoord {
    std::uint32_t pp = 0;
    std::uint32_t ep = 0;
    std::uint32_t tp = 0;
};

enum class ParallelAxis : std::uint8_t {
    Pipeline,
    Expert,
    Tensor,
};

struct RankGroup {
    ParallelAxis axis = ParallelAxis::Tensor;
    std::vector<std::uint32_t> ranks;
};

// How the routed experts divide. EP is the expert-ID axis; TP splits each expert's matrix
// inside the EP group.
enum class ExpertPartitionMode : std::uint8_t {
    Replicated,
    AcrossEP,
};

// Pure rank geometry: coordinate <-> rank <-> device, and the rank groups the collective
// scopes are built from. It performs no GPU work.
//
// Rank order is PP major, then EP, then TP:
//   rank = (pp * ep + e) * tp + t
class ParallelTopology {
public:
    static Result<ParallelTopology> create(ParallelShape shape,
                                           std::vector<int> devices);

    const ParallelShape& shape() const noexcept { return shape_; }
    std::uint32_t world_size() const noexcept;
    std::size_t stage_size() const noexcept;

    std::uint32_t rank(ParallelCoord coord) const noexcept;
    ParallelCoord coord(std::uint32_t rank) const noexcept;
    int device(std::uint32_t rank) const noexcept;

    // One PP stage's EP x TP ranks, ordered (ep, tp).
    std::vector<std::uint32_t> stage_ranks(std::uint32_t pp) const;
    // Every rank of one stage in the same EP group (varying TP).
    RankGroup tp_group(std::uint32_t pp, std::uint32_t ep) const;
    // Every rank of one stage in the same TP group (varying EP).
    RankGroup ep_group(std::uint32_t pp, std::uint32_t tp) const;
    // Every rank of one pipeline stage.
    RankGroup stage_group(std::uint32_t pp) const;

private:
    ParallelTopology() = default;

    ParallelShape shape_;
    std::vector<int> devices_;
};

}  // namespace runtime
}  // namespace shelf
