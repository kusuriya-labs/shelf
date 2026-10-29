// shelf runtime/boot — 起動時の probe と受理判定（D1）。
//
// 役割: 「どの装置で走らせられるか」を**構造化データ**として受け取り、
//       要求機能と突き合わせて**エンジンを起こす前に**可否を決める。
//       HIP 型はここに持ち込まない（probe の実体は backend が埋める）。
//
// 依存: `shelf/backend/common/device.hpp` の POD のみ（HIP ヘッダは不可）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <shelf/backend/common/device.hpp>

namespace shelf::runtime::boot {

using backend::DeviceId;
using backend::DeviceInfo;

// backend が返す構造化された事実。判定はここから純粋に行う。
struct Probe {
    std::vector<DeviceInfo> devices;   // 実行に使える GPU（無ければ空）
    bool cpu_reference = true;         // 参照経路（backend/cpu）が使えるか
    std::string backend;               // 表示用（"hip" / "cpu"）
};

// 起動に要求する機能。足りなければ**理由付きで拒否**する。
struct Requirements {
    bool need_device_execution = false;   // GPU 実行が必要（--device=hip）
    std::size_t first_device = 0;         // 使い始めるデバイス番号（--gpu）
    std::size_t devices_needed = 1;       // 同時に使う台数（PP の K。単一なら 1）
    bool need_cpu_reference = false;      // 参照経路が同時に要る（validate の exact 等）
};

struct Decision {
    bool accepted = false;
    std::string reason;                    // 拒否理由（accepted なら空）
    std::size_t device_count = 0;          // probe が見つけた台数
    std::vector<DeviceId> devices;         // 実際に使うデバイス（accepted のとき）
};

// 要求と probe を突き合わせる（純粋関数。GPU 不要で単体テストできる）。
Decision evaluate(const Probe& probe, const Requirements& req);

// 起動要求（CLI の生値）→ 実行設定の解決。
// `device_execution` は「GPU 実行か」、`pipeline_parallel` は同時に使う台数。
struct RunRequest {
    bool device_execution = false;
    std::size_t first_device = 0;
    std::size_t pipeline_parallel = 1;
    std::size_t arena_bytes = 0;      // 0 = グラフから自動
};

struct ResolvedConfig {
    bool accepted = false;
    std::string reason;
    Decision decision;                 // 使うデバイス
    std::size_t arena_bytes = 0;       // 0 = 自動
    std::size_t pipeline_parallel = 1;
};

// probe と要求から実行設定を決める（受理判定を含む。純粋関数）。
ResolvedConfig resolve(const Probe& probe, const RunRequest& req);

// 1 行の要約（ログ用）。
std::string describe(const Probe& probe, const Decision& decision);

}  // namespace shelf::runtime::boot
