// shelf placement — 予算・所在の実装（純粋。GPU にもモデル型にも触れない）。
#include <shelf/placement/budget.hpp>

#include <algorithm>
#include <cstdio>
#include <string>

namespace shelf::placement {
namespace {

std::string bytes_str(std::uint64_t b) {
    char buf[32];
    if (b >= (std::uint64_t{1} << 30))
        std::snprintf(buf, sizeof buf, "%.2f GiB", static_cast<double>(b) / (1u << 30));
    else if (b >= (std::uint64_t{1} << 20))
        std::snprintf(buf, sizeof buf, "%.1f MiB", static_cast<double>(b) / (1u << 20));
    else
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(b));
    return buf;
}

// 予備率を掛けた上限（headroom が 0 未満・1 以上でも破綻しないよう clamp）。
std::uint64_t usable_of(std::uint64_t free_bytes, double headroom) {
    if (free_bytes == 0) return 0;
    const double h = std::min(0.9, std::max(0.0, headroom));
    return static_cast<std::uint64_t>(static_cast<double>(free_bytes) * (1.0 - h));
}

}  // namespace

std::uint64_t Footprint::max_stage() const noexcept {
    std::uint64_t m = 0;
    for (const auto& s : stages) m = std::max(m, s.total());
    return m;
}

std::uint64_t Footprint::sum() const noexcept {
    std::uint64_t t = 0;
    for (const auto& s : stages) t += s.total();
    return t;
}

Report plan_budget(const Footprint& footprint, const Demand& demand, const Limits& limits) {
    Report r;
    r.per_stage_required = footprint.max_stage();
    r.required_bytes = static_cast<std::uint64_t>(
        static_cast<double>(r.per_stage_required) * (1.0 + std::max(0.0, limits.headroom)));
    r.usable_bytes = usable_of(limits.device_free_bytes, limits.headroom);

    // ステージ数と footprint の段数が食い違うのは呼び手の誤り（推測で埋めない）。
    if (demand.stages > 1 && !footprint.stages.empty() &&
        footprint.stages.size() != demand.stages) {
        r.shortfalls.push_back("stage 数が不一致（要求 " + std::to_string(demand.stages) +
                               " / 見積り " + std::to_string(footprint.stages.size()) + "）");
    }

    if (r.usable_bytes != 0 && r.required_bytes > r.usable_bytes) {
        r.shortfalls.push_back("device: 最大ステージ " + bytes_str(r.required_bytes) +
                               " が使える " + bytes_str(r.usable_bytes) + " を " +
                               bytes_str(r.required_bytes - r.usable_bytes) + " 超過");
    }

    if (demand.host_kv_tokens != 0 && demand.kv_bytes_per_token != 0) {
        r.host_required = demand.host_kv_tokens * demand.kv_bytes_per_token;
        r.host_usable = usable_of(limits.host_free_bytes, limits.headroom);
        if (r.host_usable != 0 && r.host_required > r.host_usable) {
            r.shortfalls.push_back("host(pinned): KV " + bytes_str(r.host_required) +
                                   " が使える " + bytes_str(r.host_usable) + " を " +
                                   bytes_str(r.host_required - r.host_usable) + " 超過");
        }
    }
    (void)demand.kv_tokens;   // KV は state_bytes に含めて渡す（二重計上しない）
    return r;
}

std::string Report::describe() const {
    std::string s = "budget: 必要 " + bytes_str(required_bytes) + " / 使える " +
                    bytes_str(usable_bytes);
    if (host_required != 0 || host_usable != 0)
        s += " / host KV " + bytes_str(host_required) + " (使える " + bytes_str(host_usable) + ")";
    if (ok()) return s + " [ok]";
    s += " [不足]";
    for (const auto& f : shortfalls) s += " " + f + ";";
    return s;
}

std::uint64_t PlacementPlan::device_bytes() const noexcept {
    std::uint64_t t = 0;
    for (const auto& p : weights) if (p.kind == MemoryKind::device) t += p.bytes;
    for (const auto& p : states) if (p.kind == MemoryKind::device) t += p.bytes;
    return t;
}

std::uint64_t PlacementPlan::host_bytes() const noexcept {
    std::uint64_t t = 0;
    for (const auto& p : weights) if (p.kind != MemoryKind::device) t += p.bytes;
    for (const auto& p : states) if (p.kind != MemoryKind::device) t += p.bytes;
    return t;
}

bool PlacementPlan::all_on_device() const noexcept {
    for (const auto& p : weights) if (p.kind != MemoryKind::device) return false;
    for (const auto& p : states) if (p.kind != MemoryKind::device) return false;
    return true;
}

std::size_t PlacementPlan::count(MemoryKind kind) const noexcept {
    std::size_t n = 0;
    for (const auto& p : weights) if (p.kind == kind) ++n;
    for (const auto& p : states) if (p.kind == kind) ++n;
    return n;
}

PlacementPlan place_all_device(std::vector<Placement> weights, std::vector<Placement> states) {
    for (auto& p : weights) p.kind = MemoryKind::device;
    for (auto& p : states) p.kind = MemoryKind::device;
    PlacementPlan plan;
    plan.weights = std::move(weights);
    plan.states = std::move(states);
    return plan;
}

Status place_fitting_budget(std::vector<Placement> weights, const Limits& limits,
                            PlacementPlan* out) {
    if (out == nullptr)
        return Status::invalid_argument("place_fitting_budget: null out", __FILE__, __LINE__);
    PlacementPlan plan;
    const std::uint64_t usable = usable_of(limits.device_free_bytes, limits.headroom);
    std::uint64_t used = 0;
    for (auto& p : weights) {
        // 使える量が不明（0）なら全部 device（既定の挙動）。
        if (usable == 0 || used + p.bytes <= usable) {
            p.kind = MemoryKind::device;
            used += p.bytes;
        } else {
            p.kind = MemoryKind::pinned_host;
        }
        plan.weights.push_back(p);
    }
    *out = std::move(plan);
    if (usable != 0 && out->all_on_device() && out->device_bytes() > usable)
        return Status::insufficient_memory("placement: 重みが device に収まらない", __FILE__,
                                           __LINE__);
    return Status::make_ok();
}

}  // namespace shelf::placement
