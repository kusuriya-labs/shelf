// shelf apps/common — 起動の解決（D1）。HIP の有無をここで吸収する。
#include "common/boot.hpp"

#include <cstdio>
#include <utility>

#ifdef SHELF_HAVE_HIP
#include <shelf/runtime/boot/probe.hpp>

#include "common/app_support.hpp"
#endif

namespace shelf::app {
namespace {

BootOutcome rejected(BootOutcome out, std::string why) {
    out.accepted = false;
    out.reason = std::move(why);
    out.arena_bytes = 0;
    out.pipeline_parallel = 0;
    out.stage_devices.clear();
    return out;
}

}  // namespace

BootOutcome boot(const BootRequest& req, bool print) {
    BootOutcome out;
    out.pipeline_parallel = req.pipeline_parallel;
    out.arena_bytes = req.arena_bytes;

    if (req.pipeline_parallel == 0)
        return rejected(out, "--pipeline-parallel は 1 以上");
    if (req.pipeline_parallel > 1 && !req.device_execution)
        return rejected(out,
                        "--pipeline-parallel > 1 は GPU 実行（--device hip）のときだけ使える");

#ifdef SHELF_HAVE_HIP
    const auto probe = probe_devices(req.device_execution);
    ::shelf::runtime::boot::RunRequest breq;
    breq.device_execution = req.device_execution;
    breq.first_device = req.first_device;
    breq.pipeline_parallel = req.pipeline_parallel;
    breq.arena_bytes = req.arena_bytes;
    const auto resolved = ::shelf::runtime::boot::resolve(probe, breq);
    out.summary = ::shelf::runtime::boot::describe(probe, resolved.decision);
    if (print) std::printf("boot: %s\n", out.summary.c_str());
    if (!resolved.accepted) return rejected(out, resolved.reason);
    out.accepted = true;
    out.arena_bytes = resolved.arena_bytes;
    out.pipeline_parallel = resolved.pipeline_parallel;
    for (const auto id : resolved.decision.devices) out.stage_devices.push_back(id);
    return out;
#else
    if (req.device_execution)
        return rejected(out,
                        "このビルドは HIP 非対応（SHELF_ENABLE_HIP=ON で再構築してください）");
    out.summary = "backend=cpu（HIP 非対応ビルド）";
    if (print) std::printf("boot: %s\n", out.summary.c_str());
    out.accepted = true;
    return out;
#endif
}

}  // namespace shelf::app
