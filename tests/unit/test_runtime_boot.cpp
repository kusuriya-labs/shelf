// shelf unit test — 起動の受理判定（D1）。GPU 不要の純粋関数だけを見る。
#include <cstdio>
#include <string>

#include <shelf/runtime/boot/probe.hpp>

namespace {

using shelf::runtime::boot::Decision;
using shelf::runtime::boot::DeviceInfo;
using shelf::runtime::boot::Probe;
using shelf::runtime::boot::Requirements;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

DeviceInfo dev(shelf::backend::DeviceId id, std::uint64_t mem = 32ull << 30) {
    DeviceInfo d;
    d.id = id;
    d.kind = shelf::backend::DeviceKind::gpu;
    d.name = "test-gpu-" + std::to_string(id);
    d.total_memory_bytes = mem;
    return d;
}

Probe probe_with(std::size_t gpus) {
    Probe p;
    p.backend = "hip";
    for (std::size_t i = 0; i < gpus; ++i)
        p.devices.push_back(dev(static_cast<shelf::backend::DeviceId>(i)));
    return p;
}

// D6: 機能表。probe が申告した機能に、要求が含まれるかを起動前に判定する。
void test_features() {
    using shelf::backend::Feature;
    using shelf::backend::FeatureSet;

    Probe p = probe_with(2);
    p.features = shelf::backend::build_features(/*device_runtime=*/true);
    check(p.features.has(Feature::fp8_w8a8), "HIP ビルドは fp8_w8a8 を申告する");
    check(p.features.has(Feature::pipeline_parallel), "HIP ビルドは pipeline_parallel を申告する");

    Requirements req;
    req.need_device_execution = true;
    req.required.add(Feature::fp8_w8a8);
    req.required.add(Feature::device_execution);
    const auto ok = shelf::runtime::boot::evaluate(p, req);
    check(ok.accepted, "要求機能が揃っていれば受理");

    // 要求が環境に無い場合（CPU ビルド相当）
    Probe cpu;
    cpu.backend = "cpu";
    cpu.features = shelf::backend::build_features(/*device_runtime=*/false);
    Requirements need_gpu;
    need_gpu.need_device_execution = false;
    need_gpu.required.add(Feature::fp8_w8a8);
    const auto ng = shelf::runtime::boot::evaluate(cpu, need_gpu);
    check(!ng.accepted, "fp8_w8a8 が無いビルドは拒否");
    check(ng.reason.find("fp8_w8a8") != std::string::npos,
          "拒否理由に不足した機能名が入る");

    // 任意機能の不足は記録だけ（拒否しない）
    Requirements opt;
    opt.need_device_execution = true;
    opt.optional.add(Feature::mxfp4);
    opt.optional.add(Feature::device_execution);
    Probe no_mxfp4 = p;
    no_mxfp4.features = p.features;
    no_mxfp4.features.remove(Feature::mxfp4);
    const auto ro = shelf::runtime::boot::evaluate(no_mxfp4, opt);
    check(ro.accepted, "任意機能の不足では拒否しない");
    check(ro.missing_optional.has(Feature::mxfp4), "任意機能の不足を記録する");

    // 機能を申告しない probe（従来の挙動）では機能判定をしない
    Probe empty = probe_with(1);
    Requirements r2;
    r2.need_device_execution = true;
    r2.required.add(Feature::fp8_w8a8);
    check(shelf::runtime::boot::evaluate(empty, r2).accepted,
          "probe が機能未申告なら機能判定をしない（後方互換）");

    // 集合演算
    FeatureSet a;
    a.add(Feature::fp8_w8a8).add(Feature::gdn);
    FeatureSet b;
    b.add(Feature::fp8_w8a8).add(Feature::mxfp4);
    check(a.missing_from(b).has(Feature::mxfp4) && a.missing_from(b).size() == 1,
          "missing_from は不足分だけ");
    check(!a.contains_all(b), "contains_all は不足を見逃さない");
    check(a.join() == "fp8_w8a8, gdn", "join は定義順");
    check(FeatureSet{}.join() == "なし", "空集合の表示");
}

}  // namespace

int main() {
    test_features();
    // 1. GPU 実行を要求したが 1 台も無い → 理由付きで拒否
    {
        Probe p = probe_with(0);
        Requirements r;
        r.need_device_execution = true;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(!d.accepted, "GPU 0 台は拒否");
        check(d.reason.find("デバイス") != std::string::npos, "拒否理由にデバイスの話が出る");
    }
    // 2. 1 台あり・--gpu 0 → 受理
    {
        Probe p = probe_with(1);
        Requirements r;
        r.need_device_execution = true;
        r.first_device = 0;
        r.devices_needed = 1;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(d.accepted, "1 台あれば受理");
        check(d.devices.size() == 1 && d.devices[0] == 0, "使うデバイスは 0 番");
    }
    // 3. --gpu 1 だが 1 台しか見えない → 拒否（原因が分かる）
    {
        Probe p = probe_with(1);
        Requirements r;
        r.need_device_execution = true;
        r.first_device = 1;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(!d.accepted, "存在しないデバイス指定は拒否");
        check(d.reason.find("--gpu 1") != std::string::npos, "拒否理由に --gpu が出る");
    }
    // 4. PP で 2 台必要・2 台ある → 受理
    {
        Probe p = probe_with(2);
        Requirements r;
        r.need_device_execution = true;
        r.devices_needed = 2;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(d.accepted && d.devices.size() == 2, "2 台そろえば PP=2 を受理");
        check(d.devices[0] == 0 && d.devices[1] == 1, "デバイスは 0,1 の順");
    }
    // 5. PP で 2 台必要・1 台しかない → 拒否
    {
        Probe p = probe_with(1);
        Requirements r;
        r.need_device_execution = true;
        r.devices_needed = 2;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(!d.accepted, "台数不足の PP は拒否");
        check(d.reason.find("2 台") != std::string::npos, "拒否理由に必要台数が出る");
    }
    // 6. CPU 実行はデバイス不要
    {
        Probe p = probe_with(0);
        Requirements r;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(d.accepted && d.devices.empty(), "CPU 実行はデバイス無しで受理");
    }
    // 7. 参照経路が要るのに無い → 拒否
    {
        Probe p = probe_with(1);
        p.cpu_reference = false;
        Requirements r;
        r.need_device_execution = true;
        r.need_cpu_reference = true;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(!d.accepted, "参照経路が無ければ拒否");
        check(d.reason.find("参照経路") != std::string::npos, "拒否理由に参照経路が出る");
    }
    // 8. 内部矛盾（0 台要求で GPU 実行）は拒否
    {
        Probe p = probe_with(1);
        Requirements r;
        r.need_device_execution = true;
        r.devices_needed = 0;
        const Decision d = shelf::runtime::boot::evaluate(p, r);
        check(!d.accepted, "devices_needed=0 は拒否");
    }
    // 9. describe は受理・拒否と backend 名を含む
    {
        Probe p = probe_with(1);
        Requirements r;
        r.need_device_execution = true;
        const Decision ok = shelf::runtime::boot::evaluate(p, r);
        const std::string s = shelf::runtime::boot::describe(p, ok);
        check(s.find("hip") != std::string::npos && s.find("受理") != std::string::npos,
              "describe に backend と受理が出る");
        Probe none = probe_with(0);
        const Decision ng = shelf::runtime::boot::evaluate(none, r);
        const std::string s2 = shelf::runtime::boot::describe(none, ng);
        check(s2.find("拒否") != std::string::npos, "describe に拒否が出る");
    }

    std::printf("test_runtime_boot: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
