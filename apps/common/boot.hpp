// shelf apps/common — 起動（D1）: CLI の生値から実行設定を解決する。
//
// 3 つの実行ファイル（run / validate / bench）が「probe → 受理判定 → ログ」を
// 写経していたので、その手順をここ 1 か所にまとめる。判定そのものは
// `runtime/boot`（GPU 非依存の純粋関数）が持ち、ここは HIP の有無を吸収する層。
//
// 使い方: モデルを読む前に `boot()` を 1 度だけ呼び、`accepted == false` なら
// `reason` を見せて終了する（装置初期化やモデルのロードより前に落とす）。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace shelf::app {

// 起動要求（CLI の生値）。`arena_bytes` は 0 = グラフから自動。
struct BootRequest {
    bool device_execution = false;      // --device hip
    std::size_t first_device = 0;       // --gpu
    std::size_t pipeline_parallel = 1;  // --pipeline-parallel
    std::size_t arena_bytes = 0;        // --arena-mib（0 = 自動）
};

struct BootOutcome {
    bool accepted = false;
    std::string reason;                 // 拒否理由（accepted なら空）
    std::string summary;                // probe の 1 行要約（監査証跡）
    std::size_t arena_bytes = 0;
    std::size_t pipeline_parallel = 1;
    std::vector<int> stage_devices;     // ステージ番号順の GPU 番号
};

// 装置を問い合わせ、要求と突き合わせる。`print` が真なら要約を 1 行 stdout へ。
BootOutcome boot(const BootRequest& req, bool print);

}  // namespace shelf::app
