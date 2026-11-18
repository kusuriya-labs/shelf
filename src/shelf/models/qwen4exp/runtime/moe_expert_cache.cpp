#include <shelf/models/qwen4exp/runtime/moe_expert_cache.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace shelf {
namespace qwen4exp {

using shelf::weights::MatrixWeight;

namespace {

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}

Status copy_async(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind,
                  hipStream_t stream, const char* what) {
    const hipError_t error = hipMemcpyAsync(dst, src, bytes, kind, stream);
    if (error != hipSuccess) {
        return Status::hip_error(what, hipGetErrorString(error), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status sync(hipStream_t stream, const char* what) {
    const hipError_t error = hipStreamSynchronize(stream);
    if (error != hipSuccess) {
        return Status::hip_error(what, hipGetErrorString(error), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

MoeExpertCache::MoeExpertCache(const Qwen4ExpRankWeights& weights) : weights_(&weights) {
    layers_ = static_cast<std::uint32_t>(weights.layers.size());
    vram_experts_ = weights.vram_experts;
    host_experts_ = weights.host_experts;
    experts_ = vram_experts_ + host_experts_;
    inserts_ = std::min<std::uint32_t>(env_u32("SHELF_MOE_CACHE_INSERTS", 4u), 64u);
    decay_at_ = std::max<std::uint32_t>(env_u32("SHELF_MOE_CACHE_DECAY", 4096u), 1u);
    log_interval_ = std::max<std::uint32_t>(
        env_u32("SHELF_MOE_CACHE_LOG_INTERVAL", 64u), 1u);

    const std::size_t route_slots =
        static_cast<std::size_t>(layers_) * experts_;
    freq_acc_.assign(route_slots, 0u);
    step_freq_.assign(route_slots, 0u);
    map_host_.assign(route_slots, -1);
    vram_expert_.assign(static_cast<std::size_t>(layers_) * vram_experts_, -1);
    host_expert_.assign(static_cast<std::size_t>(layers_) * host_experts_, -1);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            const std::int32_t location =
                expert < vram_experts_
                    ? static_cast<std::int32_t>(expert)
                    : -1 - static_cast<std::int32_t>(expert - vram_experts_);
            map_host_[static_cast<std::size_t>(layer) * experts_ + expert] = location;
        }
        for (std::uint32_t slot = 0; slot < vram_experts_; ++slot) {
            vram_expert_[static_cast<std::size_t>(layer) * vram_experts_ + slot] =
                static_cast<std::int32_t>(slot);
        }
        for (std::uint32_t slot = 0; slot < host_experts_; ++slot) {
            host_expert_[static_cast<std::size_t>(layer) * host_experts_ + slot] =
                static_cast<std::int32_t>(vram_experts_ + slot);
        }
    }
}

std::unique_ptr<MoeExpertCache> MoeExpertCache::create(
    const Qwen4ExpRankWeights& weights) {
    if (weights.host_experts == 0u || weights.vram_experts == 0u ||
        weights.expert_slot_map.ndim() == 0u || weights.expert_route_freq.ndim() == 0u) {
        return nullptr;
    }
    return std::unique_ptr<MoeExpertCache>(new MoeExpertCache(weights));
}

Status MoeExpertCache::step(hipStream_t stream) {
    if (weights_ == nullptr || layers_ == 0u || experts_ == 0u) return Status::make_ok();
    const std::size_t route_slots = static_cast<std::size_t>(layers_) * experts_;
    const std::size_t bytes = route_slots * sizeof(std::uint32_t);
    Status st = copy_async(step_freq_.data(), weights_->expert_route_freq.data<std::int32_t>(),
                           bytes, hipMemcpyDeviceToHost, stream, "moe cache freq read");
    if (!st.ok()) return st;
    st = sync(stream, "moe cache freq read");
    if (!st.ok()) return st;
    const hipError_t zero_error = hipMemsetAsync(
        weights_->expert_route_freq.data<std::int32_t>(), 0, bytes, stream);
    if (zero_error != hipSuccess) {
        return Status::hip_error("moe cache freq zero", hipGetErrorString(zero_error),
                                 __FILE__, __LINE__);
    }

    std::uint32_t peak = 0;
    for (std::size_t index = 0; index < route_slots; ++index) {
        const std::uint32_t count = step_freq_[index];
        const std::uint64_t sum = static_cast<std::uint64_t>(freq_acc_[index]) + count;
        freq_acc_[index] = sum > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(sum);
        peak = std::max(peak, freq_acc_[index]);
    }
    if (peak >= decay_at_) {
        for (std::uint32_t& count : freq_acc_) count /= 2u;
    }

    std::uint64_t routes = 0;
    std::uint64_t hits = 0;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            const std::size_t index = static_cast<std::size_t>(layer) * experts_ + expert;
            const std::uint32_t count = step_freq_[index];
            if (count == 0u) continue;
            routes += count;
            if (map_host_[index] >= 0) hits += count;
        }
    }
    routes_ += routes;
    hits_ += hits;

    struct Candidate {
        std::uint32_t layer = 0;
        std::uint32_t host_slot = 0;
        std::uint32_t vram_slot = 0;
        std::uint64_t score = 0;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(layers_);
    if (inserts_ != 0u) {
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            std::int32_t best_host = -1;
            std::uint64_t best_host_score = 0;
            for (std::uint32_t slot = 0; slot < host_experts_; ++slot) {
                const std::size_t expert_index =
                    static_cast<std::size_t>(layer) * host_experts_ + slot;
                const std::int32_t expert = host_expert_[expert_index];
                const std::size_t index =
                    static_cast<std::size_t>(layer) * experts_ + expert;
                if (step_freq_[index] == 0u) continue;
                const std::uint64_t score = freq_acc_[index];
                if (best_host < 0 || score > best_host_score) {
                    best_host = static_cast<std::int32_t>(slot);
                    best_host_score = score;
                }
            }
            if (best_host < 0) continue;
            std::int32_t best_vram = -1;
            std::uint64_t best_vram_score = UINT64_MAX;
            for (std::uint32_t slot = 0; slot < vram_experts_; ++slot) {
                const std::size_t expert_index =
                    static_cast<std::size_t>(layer) * vram_experts_ + slot;
                const std::int32_t expert = vram_expert_[expert_index];
                const std::size_t index =
                    static_cast<std::size_t>(layer) * experts_ + expert;
                if (step_freq_[index] != 0u) continue;
                const std::uint64_t score = freq_acc_[index];
                if (best_vram < 0 || score < best_vram_score) {
                    best_vram = static_cast<std::int32_t>(slot);
                    best_vram_score = score;
                }
            }
            if (best_vram < 0 || best_host_score <= best_vram_score) continue;
            Candidate candidate;
            candidate.layer = layer;
            candidate.host_slot = static_cast<std::uint32_t>(best_host);
            candidate.vram_slot = static_cast<std::uint32_t>(best_vram);
            candidate.score = best_host_score - best_vram_score;
            candidates.push_back(candidate);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.score > b.score;
                  });
    }

    const std::uint32_t count =
        std::min<std::uint32_t>(inserts_, static_cast<std::uint32_t>(candidates.size()));
    for (std::uint32_t index = 0; index < count; ++index) {
        const Candidate& candidate = candidates[index];
        st = swap(candidate.layer, candidate.host_slot, candidate.vram_slot, stream);
        if (!st.ok()) return st;
        ++swaps_;
    }

    ++step_;
    if ((step_ % log_interval_) == 0u && routes_ != 0u) {
        std::fprintf(stderr,
                     "MOE_CACHE step=%llu routes=%llu hit=%.2f%% swaps=%llu\n",
                     static_cast<unsigned long long>(step_),
                     static_cast<unsigned long long>(routes_),
                     100.0 * static_cast<double>(hits_) / static_cast<double>(routes_),
                     static_cast<unsigned long long>(swaps_));
        std::fflush(stderr);
    }
    return Status::make_ok();
}

Status MoeExpertCache::swap(std::uint32_t layer, std::uint32_t host_slot,
                            std::uint32_t vram_slot, hipStream_t stream) {
    const Qwen4ExpMoeWeights& mlp = weights_->layers[layer].mlp;
    const MatrixWeight* vram_planes[] = {&mlp.gate_experts, &mlp.up_experts,
                                         &mlp.down_experts};
    const MatrixWeight* host_planes[] = {&mlp.gate_experts_host,
                                         &mlp.up_experts_host,
                                         &mlp.down_experts_host};
    for (std::uint32_t index = 0; index < 3u; ++index) {
        const MatrixWeight& vram = *vram_planes[index];
        const MatrixWeight& host = *host_planes[index];
        if (vram.codes.ndim() == 0u || vram.scales.ndim() == 0u ||
            vram.row_ref.ndim() == 0u || host.host_codes == nullptr ||
            host.host_scales == nullptr || host.host_row_ref == nullptr) {
            return Status::invalid_state("moe cache swap plane is incomplete", __FILE__,
                                         __LINE__);
        }
        std::uint8_t* vram_codes =
            vram.codes.data<std::uint8_t>() + static_cast<std::size_t>(vram_slot) *
                                                  vram.codes_batch_stride_bytes;
        std::uint8_t* host_codes =
            const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(host.host_codes)) +
            static_cast<std::size_t>(host_slot) * host.codes_batch_stride_bytes;
        std::uint8_t* vram_scales =
            vram.scales.data<std::uint8_t>() + static_cast<std::size_t>(vram_slot) *
                                                   vram.scales_batch_stride_bytes;
        std::uint8_t* host_scales =
            const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(host.host_scales)) +
            static_cast<std::size_t>(host_slot) * host.scales_batch_stride_bytes;
        std::uint8_t* vram_ref =
            vram.row_ref.data<std::uint8_t>() +
            static_cast<std::size_t>(vram_slot) * vram.rows;
        std::uint8_t* host_ref =
            const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(host.host_row_ref)) +
            static_cast<std::size_t>(host_slot) * host.rows;

        const std::size_t codes_bytes = vram.codes_batch_stride_bytes;
        const std::size_t scales_bytes = vram.scales_batch_stride_bytes;
        const std::size_t ref_bytes = vram.rows;
        std::vector<std::uint8_t> demoted_codes(codes_bytes);
        std::vector<std::uint8_t> demoted_scales(scales_bytes);
        std::vector<std::uint8_t> demoted_ref(ref_bytes);

        Status st = copy_async(demoted_codes.data(), vram_codes, codes_bytes,
                               hipMemcpyDeviceToHost, stream, "moe cache demote codes");
        if (!st.ok()) return st;
        st = copy_async(demoted_scales.data(), vram_scales, scales_bytes,
                        hipMemcpyDeviceToHost, stream, "moe cache demote scales");
        if (!st.ok()) return st;
        st = copy_async(demoted_ref.data(), vram_ref, ref_bytes,
                        hipMemcpyDeviceToHost, stream, "moe cache demote row_ref");
        if (!st.ok()) return st;
        st = sync(stream, "moe cache demote");
        if (!st.ok()) return st;

        st = copy_async(vram_codes, host_codes, codes_bytes, hipMemcpyHostToDevice, stream,
                        "moe cache promote codes");
        if (!st.ok()) return st;
        st = copy_async(vram_scales, host_scales, scales_bytes, hipMemcpyHostToDevice,
                        stream, "moe cache promote scales");
        if (!st.ok()) return st;
        st = copy_async(vram_ref, host_ref, ref_bytes, hipMemcpyHostToDevice, stream,
                        "moe cache promote row_ref");
        if (!st.ok()) return st;
        st = sync(stream, "moe cache promote");
        if (!st.ok()) return st;

        std::memcpy(host_codes, demoted_codes.data(), codes_bytes);
        std::memcpy(host_scales, demoted_scales.data(), scales_bytes);
        std::memcpy(host_ref, demoted_ref.data(), ref_bytes);
    }

    const std::size_t vram_index =
        static_cast<std::size_t>(layer) * vram_experts_ + vram_slot;
    const std::size_t host_index =
        static_cast<std::size_t>(layer) * host_experts_ + host_slot;
    const std::int32_t promoted = vram_expert_[vram_index];
    const std::int32_t demoted = host_expert_[host_index];
    vram_expert_[vram_index] = demoted;
    host_expert_[host_index] = promoted;
    const std::size_t map_base = static_cast<std::size_t>(layer) * experts_;
    map_host_[map_base + promoted] = static_cast<std::int32_t>(vram_slot);
    map_host_[map_base + demoted] = -1 - static_cast<std::int32_t>(host_slot);

    Status st = copy_async(weights_->expert_slot_map.data<std::int32_t>() + map_base,
                           map_host_.data() + map_base,
                           static_cast<std::size_t>(experts_) * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream, "moe cache map upload");
    if (!st.ok()) return st;
    return Status::make_ok();
}

}  // namespace qwen4exp
}  // namespace shelf
