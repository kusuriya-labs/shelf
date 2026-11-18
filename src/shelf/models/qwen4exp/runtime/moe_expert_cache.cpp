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
    budget_ = env_u32("SHELF_MOE_CACHE_SWAP_BUDGET", 96u);
    min_gain_ = std::max<std::uint32_t>(env_u32("SHELF_MOE_CACHE_MIN_GAIN", 150u), 100u);
    rebalance_steps_ =
        std::max<std::uint32_t>(env_u32("SHELF_MOE_CACHE_REBALANCE_STEPS", 4u), 1u);
    rebalance_routes_ = env_u32("SHELF_MOE_CACHE_REBALANCE_ROUTES", 0u);
    decay_at_ = std::max<std::uint32_t>(env_u32("SHELF_MOE_CACHE_DECAY", 4096u), 1u);
    log_interval_ = std::max<std::uint32_t>(
        env_u32("SHELF_MOE_CACHE_LOG_INTERVAL", 64u), 1u);
    oracle_ = env_u32("SHELF_MOE_CACHE_ORACLE", 0u) != 0u;
    pcie_ = shelf::PcieBudget::from_env();

    const std::size_t route_slots =
        static_cast<std::size_t>(layers_) * experts_;
    if (rebalance_routes_ == 0u) {
        rebalance_routes_ = 16u * static_cast<std::uint64_t>(route_slots);
    }
    next_rebalance_routes_ = rebalance_routes_;
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
        for (std::uint32_t& count : freq_acc_) {
            count = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(count) * 7u / 10u);
        }
    }

    std::uint64_t step_routes = 0;
    std::uint64_t routes = 0;
    std::uint64_t hits = 0;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            const std::size_t index = static_cast<std::size_t>(layer) * experts_ + expert;
            const std::uint32_t count = step_freq_[index];
            step_routes += count;
            if (count == 0u) continue;
            routes += count;
            if (map_host_[index] >= 0) hits += count;
        }
    }
    routes_ += routes;
    hits_ += hits;

    if (pending_head_ >= pending_.size()) {
        const bool due = routes_ >= next_rebalance_routes_;
        const bool periodic = routes_ >= rebalance_routes_ &&
                              step_ - last_plan_step_ >= rebalance_steps_;
        if (due || periodic) plan();
    }
    const std::uint64_t batched_routes = static_cast<std::uint64_t>(layers_) * experts_;
    const bool batched = step_routes > batched_routes;
    const std::uint64_t move_bytes = swap_bytes();
    const std::size_t drain_limit = batched
        ? pending_.size()
        : std::min<std::size_t>(pending_.size(),
                                pending_head_ + static_cast<std::size_t>(budget_));
    while (pending_head_ < drain_limit) {
        if (!batched && !pcie_.try_acquire(move_bytes)) break;
        const Move move = pending_[pending_head_];
        st = swap(move.layer, move.host_slot, move.vram_slot, stream);
        if (!st.ok()) return st;
        ++pending_head_;
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
        if (oracle_) log_oracle();
    }
    return Status::make_ok();
}

std::uint64_t MoeExpertCache::swap_bytes() const {
    if (weights_ == nullptr || weights_->layers.empty()) return 0u;
    const Qwen4ExpMoeWeights& mlp = weights_->layers[0].mlp;
    const MatrixWeight* planes[] = {&mlp.gate_experts, &mlp.up_experts, &mlp.down_experts};
    const MatrixWeight* hosts[] = {&mlp.gate_experts_host, &mlp.up_experts_host,
                                   &mlp.down_experts_host};
    std::uint64_t bytes = 0u;
    for (std::uint32_t index = 0; index < 3u; ++index) {
        bytes += planes[index]->codes_batch_stride_bytes;
        bytes += planes[index]->scales_batch_stride_bytes;
        bytes += hosts[index]->rows;
    }
    return 2u * bytes;
}

void MoeExpertCache::plan() {
    pending_.clear();
    pending_head_ = 0;

    struct Ranked {
        std::uint32_t freq = 0;
        std::uint32_t slot = 0;
    };
    std::vector<Ranked> resident(vram_experts_);
    std::vector<Ranked> waiting(host_experts_);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const std::size_t base = static_cast<std::size_t>(layer) * experts_;
        for (std::uint32_t slot = 0; slot < vram_experts_; ++slot) {
            const std::size_t index =
                static_cast<std::size_t>(layer) * vram_experts_ + slot;
            const std::uint32_t expert = static_cast<std::uint32_t>(vram_expert_[index]);
            resident[slot].freq = freq_acc_[base + expert];
            resident[slot].slot = slot;
        }
        for (std::uint32_t slot = 0; slot < host_experts_; ++slot) {
            const std::size_t index =
                static_cast<std::size_t>(layer) * host_experts_ + slot;
            const std::uint32_t expert = static_cast<std::uint32_t>(host_expert_[index]);
            waiting[slot].freq = freq_acc_[base + expert];
            waiting[slot].slot = slot;
        }
        std::sort(resident.begin(), resident.end(),
                  [](const Ranked& a, const Ranked& b) { return a.freq < b.freq; });
        std::sort(waiting.begin(), waiting.end(),
                  [](const Ranked& a, const Ranked& b) { return a.freq > b.freq; });
        const std::size_t pairs =
            std::min<std::size_t>(vram_experts_, host_experts_);
        for (std::size_t i = 0; i < pairs; ++i) {
            const std::uint64_t admitted = waiting[i].freq;
            const std::uint64_t evicted = resident[i].freq;
            if (admitted == 0u || admitted <= evicted) break;
            if (admitted * 100u <= evicted * min_gain_) break;
            Move move;
            move.layer = layer;
            move.host_slot = waiting[i].slot;
            move.vram_slot = resident[i].slot;
            move.score = admitted - evicted;
            pending_.push_back(move);
        }
    }
    std::sort(pending_.begin(), pending_.end(),
              [](const Move& a, const Move& b) { return a.score > b.score; });
    last_plan_step_ = step_;
    next_rebalance_routes_ =
        routes_ + std::max<std::uint64_t>(rebalance_routes_, routes_ / 4u);
}

void MoeExpertCache::log_oracle() const {
    std::uint64_t total = 0;
    for (const std::uint32_t count : freq_acc_) total += count;
    if (total == 0u) return;

    std::vector<std::uint32_t> order(freq_acc_.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [this](std::uint32_t a, std::uint32_t b) {
        return freq_acc_[a] > freq_acc_[b];
    });

    const std::size_t capacity = static_cast<std::size_t>(layers_) * vram_experts_;
    std::uint64_t global_top = 0;
    for (std::size_t i = 0; i < order.size() && i < capacity; ++i) {
        global_top += freq_acc_[order[i]];
    }

    std::uint64_t layer_top = 0;
    std::vector<std::uint32_t> slice(experts_);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const std::size_t base = static_cast<std::size_t>(layer) * experts_;
        for (std::uint32_t e = 0; e < experts_; ++e) slice[e] = freq_acc_[base + e];
        const std::size_t keep = std::min<std::size_t>(vram_experts_, experts_);
        std::nth_element(slice.begin(), slice.begin() + keep, slice.end(),
                         [](std::uint32_t a, std::uint32_t b) { return a > b; });
        for (std::size_t i = 0; i < keep; ++i) layer_top += slice[i];
    }

    const double fraction[] = {0.02, 0.05, 0.10, 0.25, 0.50};
    double coverage[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::uint64_t running = 0;
    std::size_t next = 0;
    for (std::size_t i = 0; i < order.size() && next < 5u; ++i) {
        running += freq_acc_[order[i]];
        while (next < 5u &&
               static_cast<double>(i + 1u) >= fraction[next] * static_cast<double>(order.size())) {
            coverage[next] = 100.0 * static_cast<double>(running) / static_cast<double>(total);
            ++next;
        }
    }

    std::fprintf(stderr,
                 "MOE_CACHE_ORACLE step=%llu slot=%.1f%% current=%.2f%% layer_top=%.2f%% "
                 "global_top=%.2f%% cov2=%.1f%% cov5=%.1f%% cov10=%.1f%% cov25=%.1f%% "
                 "cov50=%.1f%%\n",
                 static_cast<unsigned long long>(step_),
                 100.0 * static_cast<double>(capacity) / static_cast<double>(order.size()),
                 100.0 * static_cast<double>(hits_) / static_cast<double>(routes_),
                 100.0 * static_cast<double>(layer_top) / static_cast<double>(total),
                 100.0 * static_cast<double>(global_top) / static_cast<double>(total),
                 coverage[0], coverage[1], coverage[2], coverage[3], coverage[4]);
    std::fflush(stderr);
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
    const std::int32_t evicted = vram_expert_[vram_index];
    const std::int32_t admitted = host_expert_[host_index];
    vram_expert_[vram_index] = admitted;
    host_expert_[host_index] = evicted;
    const std::size_t map_base = static_cast<std::size_t>(layer) * experts_;
    map_host_[map_base + admitted] = static_cast<std::int32_t>(vram_slot);
    map_host_[map_base + evicted] = -1 - static_cast<std::int32_t>(host_slot);

    Status st = copy_async(weights_->expert_slot_map.data<std::int32_t>() + map_base,
                           map_host_.data() + map_base,
                           static_cast<std::size_t>(experts_) * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream, "moe cache map upload");
    if (!st.ok()) return st;
    return Status::make_ok();
}

}  // namespace qwen4exp
}  // namespace shelf
