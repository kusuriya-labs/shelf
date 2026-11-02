#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/resident/resident_client.h>
#include <phaseshift/resident/resident_model_key.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ps {
namespace resident {

struct ResidentTestRequest {
    std::string qwen_model_dir;
    qwen35::Qwen35LoadOptions qwen_options;
    std::string dflash2_model_dir;
    ps::weights::WeightLoadOptions dflash2_options;
    int device = 0;
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    std::size_t arena_bytes = 0;
};

class ResidentModelFixture {
 public:
    ResidentModelFixture() = default;
    ResidentModelFixture(const ResidentModelFixture&) = delete;
    ResidentModelFixture& operator=(const ResidentModelFixture&) = delete;
    ResidentModelFixture(ResidentModelFixture&&) noexcept = default;
    ResidentModelFixture& operator=(ResidentModelFixture&&) noexcept = default;
    ~ResidentModelFixture()
    {
        if (release_on_destroy_) {
            release();
        }
    }

    static Result<ResidentModelFixture> acquire(const ResidentTestRequest& request);

    void set_release_on_destroy(bool enabled) noexcept { release_on_destroy_ = enabled; }

    void release();

    bool enabled() const noexcept { return qwen_.valid() || dflash_.valid(); }
    bool has_qwen35() const noexcept { return qwen_.valid(); }
    bool has_dflash2() const noexcept { return dflash_.valid(); }

    std::size_t resident_bytes() const noexcept
    {
        return qwen_.resident_bytes() + dflash_.resident_bytes();
    }

    std::size_t persistent_bytes() const noexcept
    {
        std::size_t out = qwen_.persistent_bytes();
        if (dflash_.persistent_bytes() > out) {
            out = dflash_.persistent_bytes();
        }
        const std::size_t own = resident_bytes();
        return own > out ? own : out;
    }

    std::size_t arena_bytes() const noexcept
    {
        if (requested_arena_bytes_ == 0) {
            return 0;
        }
        const std::size_t persistent = persistent_bytes();
        if (persistent >= requested_arena_bytes_) {
            return 0;
        }
        return requested_arena_bytes_ - persistent;
    }

    std::size_t requested_arena_bytes() const noexcept { return requested_arena_bytes_; }

    Result<qwen35::Qwen35Model> take_qwen35() const { return qwen_.take_qwen35(); }

    Result<qwen35::dflash2::DFlash2Weights> take_dflash2(
        qwen35::dflash2::DFlash2Config* config_out = nullptr) const
    {
        return dflash_.take_dflash2(config_out);
    }

 private:
    ResidentModelAttachment qwen_;
    ResidentModelAttachment dflash_;
    std::size_t requested_arena_bytes_ = 0;
    bool release_on_destroy_ = false;
};

}
}
