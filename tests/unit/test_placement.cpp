// shelf unit test — placement の予算算術（D2）。GPU 無しで通ること。
//
// 見ていること:
//   1. KV 1 トークンぶんの式（`engine_contract.md` §8.2）が既知の値と一致する
//   2. 予算判定が「名前付きの不足」を返す（容量不明の 0 を誤って不足にしない）
//   3. PP では**最大ステージ**で判定する（合計で判定しない）
//   4. headroom の効き方
//   5. 所在は既定で全 device 常駐（`directory_design` D2 の既定）
#include <cstdio>
#include <string>
#include <vector>

#include <shelf/placement/budget.hpp>

namespace {

using namespace shelf::placement;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

}  // namespace

int main() {
    // 1. KV の式: 9B は full 8 層・kv_heads 4・head_dim 256 → 32 KiB/token
    {
        const auto per_tok = kv_bytes_per_token(8, 4, 256);
        std::printf("9B KV/token = %llu B\n", static_cast<unsigned long long>(per_tok));
        check(per_tok == 32768u, "9B の KV は 1 トークン 32 KiB");
        check(kv_bytes_per_token(4, 2, 128) == 4096u, "小さな形でも式どおり");
        check(kv_bytes_per_token(0, 4, 256) == 0u, "full 層 0 なら KV は 0");
    }

    // 2. 予算判定: 収まる / 収まらない / 容量不明
    {
        Footprint fp;
        fp.add_stage(StageCost{/*weights*/ 10u << 20, /*states*/ 2u << 20, /*scratch*/ 1u << 20});
        Demand dem;
        dem.stages = 1;

        Limits lim;                       // 不明（0）→ 判定しない
        auto r = plan_budget(fp, dem, lim);
        check(r.ok(), "容量不明なら拒否しない（推測で止めない）");

        lim.device_free_bytes = 64u << 20;   // 十分
        r = plan_budget(fp, dem, lim);
        check(r.ok(), "収まる構成は ok");
        check(r.per_stage_required == (13u << 20), "最大ステージの必要量");
        check(r.required_bytes > r.per_stage_required, "headroom 込みの必要量は素の値より大きい");
        std::printf("  %s\n", r.describe().c_str());

        lim.device_free_bytes = 8u << 20;    // 不足
        r = plan_budget(fp, dem, lim);
        check(!r.ok(), "不足を検出する");
        check(!r.shortfalls.empty() &&
                  r.shortfalls.front().find("device") != std::string::npos,
              "不足は名前付き（device）");
        std::printf("  %s\n", r.describe().c_str());
    }

    // 3. PP は最大ステージで判定する（合計だと誤って拒否する）
    {
        Footprint fp;
        fp.add_stage(StageCost{8u << 20, 0, 0});
        fp.add_stage(StageCost{8u << 20, 0, 0});
        fp.add_stage(StageCost{8u << 20, 0, 1u << 20});   // 最大はここ
        Demand dem;
        dem.stages = 3;
        Limits lim;
        lim.device_free_bytes = 12u << 20;   // 合計 25 MiB は超えるが、最大 9 MiB は収まる
        lim.headroom = 0.0;
        const auto r = plan_budget(fp, dem, lim);
        check(fp.max_stage() == (9u << 20), "最大ステージ = 9 MiB");
        check(r.ok(), "PP は最大ステージで判定する（合計では拒否しない）");

        // ステージ数の不一致は呼び手の誤りとして報告する
        Demand wrong;
        wrong.stages = 2;
        const auto r2 = plan_budget(fp, wrong, lim);
        check(!r2.ok(), "ステージ数が footprint と食い違えば不足として報告");
    }

    // 4. headroom が効く
    {
        Footprint fp;
        fp.add_stage(StageCost{100u, 0, 0});
        Demand dem;
        Limits lim;
        lim.device_free_bytes = 105u;    // headroom 5% → 使えるのは 99（100 は入らない）
        lim.headroom = 0.05;
        const auto r = plan_budget(fp, dem, lim);
        check(!r.ok(), "headroom 5% では 105 に 100 は入らない");
        lim.headroom = 0.0;
        check(plan_budget(fp, dem, lim).ok(), "headroom 0 なら入る");
    }

    // 5. 所在: 既定は全 device 常駐
    {
        std::vector<Placement> w{{"model.layers.0.mlp.up_proj.weight", 1000u, MemoryKind::host},
                                 {"model.layers.0.mlp.down_proj.weight", 2000u, MemoryKind::host}};
        auto plan = place_all_device(w);
        check(plan.all_on_device(), "既定は全 device 常駐（入力の kind に依らない）");
        check(plan.device_bytes() == 3000u, "device 合計");
        check(plan.host_bytes() == 0u, "host に残らない");
        check(plan.count(MemoryKind::device) == 2u, "件数");

        // 予算に収まらない分だけ pinned_host へ落とす（opt-in）
        Limits lim;
        lim.device_free_bytes = 1500u;
        lim.headroom = 0.0;
        PlacementPlan fitted;
        const auto st = place_fitting_budget(w, lim, &fitted);
        check(st.ok(), "収まる範囲で device に置ける");
        check(fitted.device_bytes() == 1000u, "先頭だけ device");
        check(fitted.host_bytes() == 2000u, "残りは pinned_host");
        check(fitted.count(MemoryKind::pinned_host) == 1u, "pinned_host の件数");

        PlacementPlan none;
        Limits tiny;
        tiny.device_free_bytes = 0u;   // 不明 → 全部 device（安全側の既定）
        check(place_fitting_budget(w, tiny, &none).ok(), "容量不明なら全 device");
        check(none.all_on_device(), "容量不明の既定は全 device");
    }

    std::printf("test_placement: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
