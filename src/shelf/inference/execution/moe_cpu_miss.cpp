#include <shelf/inference/execution/moe_cpu_miss.h>

#include <shelf/quantization/mxfp4/mxfp4_cpu_expert.h>
#include <shelf/runtime/cpu_team.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace shelf::execution {
namespace {

constexpr std::uint32_t kRowTile = 16u;

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

class Runner {
public:
    Runner() : team_(runtime::CpuTeam::create(0u)), max_rows_(env_u32("SHELF_MOE_CPU_MISS_ROWS", 64u)) {}

    ~Runner() {
        if (pinned_ != nullptr) static_cast<void>(hipHostFree(pinned_));
    }

    Status run(const MoeCpuMissDesc& desc, hipStream_t stream, bool* handled);

private:
    bool ensure_pinned(std::size_t bytes);

    std::unique_ptr<runtime::CpuTeam> team_;
    std::uint32_t max_rows_ = 64;
    void* pinned_ = nullptr;
    std::size_t pinned_bytes_ = 0;
};

bool Runner::ensure_pinned(std::size_t bytes) {
    if (bytes <= pinned_bytes_) return true;
    if (pinned_ != nullptr) {
        static_cast<void>(hipHostFree(pinned_));
        pinned_ = nullptr;
        pinned_bytes_ = 0;
    }
    void* block = nullptr;
    if (hipHostMalloc(&block, bytes) != hipSuccess) return false;
    pinned_ = block;
    pinned_bytes_ = bytes;
    return true;
}

Status Runner::run(const MoeCpuMissDesc& desc, hipStream_t stream, bool* handled) {
    *handled = false;
    if (desc.miss_count == nullptr || desc.group_capacity == 0u) return Status::make_ok();

    const hipError_t sync_error = hipStreamSynchronize(stream);
    if (sync_error != hipSuccess) {
        return Status::hip_error("moe cpu miss sync", hipGetErrorString(sync_error), __FILE__,
                                 __LINE__);
    }

    const bool trace = env_u32("SHELF_MOE_CPU_MISS_TRACE", 0u) != 0u;
    const bool dry = env_u32("SHELF_MOE_CPU_MISS_DUMP", 0u) != 0u;
    std::vector<std::uint32_t> plan(4u + 3u * desc.group_capacity, 0u);
    std::uint32_t* host_count = plan.data();
    std::uint32_t* host_slot = host_count + 1;
    std::uint32_t* host_rows = host_slot + desc.group_capacity;
    std::uint32_t* host_base = host_rows + desc.group_capacity;

    hipError_t error = hipMemcpy(host_count, desc.miss_count, sizeof(std::uint32_t),
                                 hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("moe cpu miss count", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    const std::uint32_t groups = std::min(*host_count, desc.group_capacity);
    if (dry) {
        std::fprintf(stderr, "CPU_MISS_DRY groups=%u first_slot=%u first_base=%u\n", groups,
                     host_slot[0], host_base[0]);
        std::fflush(stderr);
    }
    if (trace) {
        std::fprintf(stderr, "CPU_MISS count=%u groups=%u\n", *host_count, groups);
        std::fflush(stderr);
    }
    if (groups == 0u) return Status::make_ok();

    error = hipMemcpy(host_slot, desc.miss_slot, groups * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("moe cpu miss slots", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    error = hipMemcpy(host_rows, desc.miss_rows, groups * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("moe cpu miss rows", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    error = hipMemcpy(host_base, desc.miss_base, groups * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("moe cpu miss base", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }

    std::uint32_t total = 0u;
    for (std::uint32_t g = 0; g < groups; ++g) total += host_rows[g];
    if (trace) {
        std::fprintf(stderr, "CPU_MISS total=%u first: slot=%u rows=%u base=%u\n", total,
                     host_slot[0], host_rows[0], host_base[0]);
        std::fflush(stderr);
    }
    if (total == 0u || total > max_rows_) return Status::make_ok();

    const std::size_t tile_bytes =
        static_cast<std::size_t>(desc.k_padded) * 16u;
    const std::size_t codes_bytes = tile_bytes;
    const std::size_t scales_bytes =
        static_cast<std::size_t>(desc.activation_scale_stride) * total;
    const std::size_t out_element = desc.output_bf16 ? 2u : 4u;
    const std::size_t out_bytes = static_cast<std::size_t>(desc.rows) * out_element * total;
    const std::size_t row_scratch = static_cast<std::size_t>(desc.rows) * 4u;
    const std::size_t needed = codes_bytes + scales_bytes + out_bytes + row_scratch + 64u;
    if (!ensure_pinned(needed)) {
        return Status::invalid_state("moe cpu miss: pinned staging allocation failed", __FILE__,
                                     __LINE__);
    }
    std::uint8_t* base = static_cast<std::uint8_t*>(pinned_);
    std::uint8_t* codes_stage = base;
    std::uint8_t* scales_stage = codes_stage + codes_bytes;
    std::uint8_t* out_stage = scales_stage + scales_bytes;
    float* row_scratch_ptr = reinterpret_cast<float*>(out_stage + out_bytes);

    quantization::mxfp4::CpuExpertPlane weights;
    weights.codes = desc.host_codes;
    weights.scales = desc.host_scales;
    weights.rows = desc.rows;
    weights.k_padded = desc.k_padded;
    weights.blocks = desc.k_padded / 32u;
    weights.codes_batch_stride = desc.codes_batch_stride;
    weights.scales_batch_stride = desc.scales_batch_stride;
    weights.scale_row_stride = desc.scale_row_stride;

    std::int64_t staged_tile = -1;

    std::uint32_t slot_index = 0u;
    std::uint32_t staged_rows = 0u;
    for (std::uint32_t g = 0; g < groups; ++g) {
        const std::uint32_t row_base = host_base[g] * kRowTile;
        if (row_base + host_rows[g] > desc.row_capacity) return Status::make_ok();
        const std::uint32_t group_rows = host_rows[g];
        for (std::uint32_t i = 0; i < host_rows[g]; ++i, ++slot_index, ++staged_rows) {
            const std::uint32_t row = row_base + i;
            const std::uint32_t tile = row >> 4u;
            std::uint8_t* scales_row = scales_stage + static_cast<std::size_t>(slot_index) *
                                                          desc.activation_scale_stride;
            if (static_cast<std::int64_t>(tile) != staged_tile) {
                error = hipMemcpy(codes_stage,
                                  desc.activation_codes + static_cast<std::size_t>(tile) *
                                                               desc.activation_code_stride * 16u,
                                  tile_bytes, hipMemcpyDeviceToHost);
                if (error != hipSuccess) {
                    return Status::hip_error("moe cpu miss activation",
                                             hipGetErrorString(error), __FILE__, __LINE__);
                }
                staged_tile = static_cast<std::int64_t>(tile);
            }
            error = hipMemcpy(scales_row,
                              reinterpret_cast<const std::uint8_t*>(desc.activation_scales) +
                                  static_cast<std::size_t>(row) * desc.activation_scale_stride,
                              desc.activation_scale_stride, hipMemcpyDeviceToHost);
            if (error != hipSuccess) {
                return Status::hip_error("moe cpu miss activation scales",
                                         hipGetErrorString(error), __FILE__, __LINE__);
            }
            error = hipMemcpy(scales_row,
                              reinterpret_cast<const std::uint8_t*>(desc.activation_scales) +
                                  static_cast<std::size_t>(row) * desc.activation_scale_stride,
                              desc.activation_scale_stride, hipMemcpyDeviceToHost);
            if (error != hipSuccess) {
                return Status::hip_error("moe cpu miss activation scales",
                                         hipGetErrorString(error), __FILE__, __LINE__);
            }

            quantization::mxfp4::CpuActivationTile activation;
            activation.codes = codes_stage;
            activation.row_scale = reinterpret_cast<const float*>(scales_row);
            activation.k_padded = desc.k_padded;
            team_->parallel_for(desc.rows, [&](std::uint32_t begin, std::uint32_t end) {
                quantization::mxfp4::gemv_rows(weights, host_slot[g], activation, row & 15u,
                                               row_scratch_ptr, 1u, begin, end);
            });
            if (trace && slot_index == 0u) {
                std::int64_t act_sum = 0;
                for (std::uint32_t b = 0; b < 64u; ++b) {
                    act_sum += codes_stage[b];
                }
                std::fprintf(stderr,
                             "CPU_MISS row=%u slot=%u act_sum=%lld out0=%.4f outN=%.4f\n",
                             row, host_slot[g], static_cast<long long>(act_sum),
                             row_scratch_ptr[0], row_scratch_ptr[desc.rows - 1u]);
                std::fflush(stderr);
            }

            std::uint8_t* out_row = out_stage + static_cast<std::size_t>(staged_rows) *
                                                    out_element * desc.rows;
            if (desc.output_bf16) {
                std::uint16_t* half = reinterpret_cast<std::uint16_t*>(out_row);
                for (std::uint32_t o = 0; o < desc.rows; ++o) half[o] = to_bf16(row_scratch_ptr[o]);
            } else {
                std::memcpy(out_row, row_scratch_ptr,
                            static_cast<std::size_t>(desc.rows) * sizeof(float));
            }
        }
        if (group_rows == 0u) {
            continue;
        }
        std::uint8_t* src = out_stage + static_cast<std::size_t>(staged_rows - group_rows) *
                                            out_element * desc.rows;
        std::uint8_t* dst = static_cast<std::uint8_t*>(desc.output) +
                            static_cast<std::size_t>(row_base) * desc.output_row_stride *
                                out_element;
        if (dry) continue;
        error = hipMemcpy2D(dst, static_cast<std::size_t>(desc.output_row_stride) * out_element,
                            src, static_cast<std::size_t>(desc.rows) * out_element,
                            static_cast<std::size_t>(desc.rows) * out_element, group_rows,
                            hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            return Status::hip_error("moe cpu miss output", hipGetErrorString(error), __FILE__,
                                     __LINE__);
        }
    }

    *handled = !dry;
    return Status::make_ok();
}

Runner& runner() {
    static Runner instance;
    return instance;
}

}  // namespace

bool moe_cpu_miss_enabled() {
    static const bool enabled = env_u32("SHELF_MOE_CPU_MISS", 0u) != 0u;
    return enabled;
}

Status run_moe_cpu_miss(const MoeCpuMissDesc& desc, hipStream_t stream, bool* handled) {
    static const bool trace = env_u32("SHELF_MOE_CPU_MISS_TRACE", 0u) != 0u;
    if (trace) {
        std::fprintf(stderr, "CPU_MISS enter rows=%u k=%u cap=%u/%u\n", desc.rows,
                     desc.k_padded, desc.group_capacity, desc.row_capacity);
    }
    Status status = runner().run(desc, stream, handled);
    if (trace) {
        std::fprintf(stderr, "CPU_MISS exit handled=%d ok=%d\n", *handled ? 1 : 0,
                     status.ok() ? 1 : 0);
        std::fflush(stderr);
    }
    return status;
}

}  // namespace shelf::execution
