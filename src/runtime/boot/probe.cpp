// shelf runtime/boot — probe と要求の突き合わせ（純粋）。
#include <shelf/runtime/boot/probe.hpp>

#include <string>

namespace shelf::runtime::boot {
namespace {

std::string num(std::size_t v) { return std::to_string(v); }

}  // namespace

Decision evaluate(const Probe& probe, const Requirements& req) {
    Decision d;
    d.device_count = probe.devices.size();

    // 機能表（D6）: 要求機能が probe の機能集合に含まれなければ**起動前に**拒否する。
    // probe が機能を申告しない（空）場合は従来どおり判定しない。
    if (!probe.features.empty() && !probe.features.contains_all(req.required)) {
        d.reason = "この環境では要求機能を満たせません（不足: " +
                   probe.features.missing_from(req.required).join() + " / 利用可能: " +
                   probe.features.join() + "）";
        return d;
    }
    if (!probe.features.empty()) d.missing_optional = probe.features.missing_from(req.optional);

    if (req.need_cpu_reference && !probe.cpu_reference) {
        d.reason = "参照経路（backend/cpu）が無いビルドです";
        return d;
    }
    if (!req.need_device_execution) {
        // CPU 実行: デバイスは要らない。
        d.accepted = true;
        return d;
    }
    if (probe.devices.empty()) {
        d.reason = "GPU 実行が要求されたが、使えるデバイスが 1 台も無い"
                   "（このビルドは HIP 非対応か、ドライバが初期化できない）";
        return d;
    }
    if (req.devices_needed == 0) {
        d.reason = "内部矛盾: devices_needed が 0";
        return d;
    }
    const std::size_t last = req.first_device + req.devices_needed - 1;
    if (last >= probe.devices.size()) {
        d.reason = "--gpu " + num(req.first_device) + " から " + num(req.devices_needed) +
                   " 台を使う指定だが、見えているデバイスは " +
                   num(probe.devices.size()) + " 台（必要: " + num(last + 1) + " 番まで）";
        return d;
    }
    for (std::size_t i = 0; i < req.devices_needed; ++i) {
        d.devices.push_back(probe.devices[req.first_device + i].id);
    }
    d.accepted = true;
    return d;
}

ResolvedConfig resolve(const Probe& probe, const RunRequest& req) {
    ResolvedConfig out;
    Requirements r;
    r.need_device_execution = req.device_execution;
    r.first_device = req.first_device;
    r.devices_needed = req.pipeline_parallel;
    r.required = req.required;
    r.optional = req.optional;
    out.decision = evaluate(probe, r);
    out.accepted = out.decision.accepted;
    out.reason = out.decision.reason;
    out.arena_bytes = req.arena_bytes;
    out.pipeline_parallel = req.pipeline_parallel;
    if (!out.accepted) {
        out.pipeline_parallel = 0;
        out.arena_bytes = 0;
    }
    return out;
}

std::string describe(const Probe& probe, const Decision& decision) {
    std::string s;
    s += "backend=" + (probe.backend.empty() ? std::string("?") : probe.backend);
    s += " devices=" + num(probe.devices.size());
    if (!probe.devices.empty()) {
        // 使える装置を列挙する（どれを選ぶかの判断 = 予算は D2。ここでは事実だけ出す）。
        s += " [";
        for (std::size_t i = 0; i < probe.devices.size() && i < 8; ++i) {
            const auto& d = probe.devices[i];
            if (i) s += ", ";
            s += num(i) + ":" + (d.name.empty() ? std::string("unnamed") : d.name);
            if (d.total_memory_bytes > 0)
                s += "/" + num(d.total_memory_bytes >> 20) + "MiB";
        }
        if (probe.devices.size() > 8) s += ", …";
        s += "]";
    }
    s += " cpu_reference=" + std::string(probe.cpu_reference ? "yes" : "no");
    s += " → " + std::string(decision.accepted ? "受理" : "拒否");
    if (!decision.accepted) s += ": " + decision.reason;
    return s;
}

}  // namespace shelf::runtime::boot
