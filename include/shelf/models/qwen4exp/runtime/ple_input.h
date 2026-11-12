#pragma once

#include <shelf/core/status.h>
#include <shelf/inference/scheduling/scheduled_batch.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shelf {
namespace qwen4exp {

// Builds the PLE external input for one scheduled batch. The token history is kept per
// sequence slot; the caller must not enable a prefix cache that restores tokens this
// history never saw.
class PleInputRuntime {
 public:
    static std::shared_ptr<PleInputRuntime> create(
        std::shared_ptr<const PleTable> table, std::uint32_t hidden);

    Status prepare(const ::shelf::scheduling::ScheduledBatch& batch,
                   std::vector<bf16_t>& out);

    void reset_sequence(std::uint32_t slot);

    Status seed(std::uint32_t slot, std::span<const std::int32_t> tokens);

 private:
    PleInputRuntime(std::shared_ptr<const PleTable> table, std::uint32_t hidden)
        : table_(std::move(table)), hidden_(hidden) {}

    std::shared_ptr<const PleTable> table_;
    std::uint32_t hidden_ = 0;
    std::unordered_map<std::uint32_t, std::vector<std::int32_t>> history_;
};

}  // namespace qwen4exp
}  // namespace shelf
