// shelf placement — 予算と所在の純粋算術（D2）。
//
// 役割: 「この構成は載るか」を **GPU に触らずに** 判定する。入力は既知量だけ
// （重みのバイト数・状態のバイト数・スクラッチ・空き容量）で、実確保は backend が行う。
// 推測はしない（radiance「既知量の算術」/ paiton の `fits` 判定に倣う）。
//
// 依存: `shelf/backend/common`（`MemoryKind`）と `shelf/foundation` のみ。
// モデル型・HIP 型は持たない（境界ルール: placement は model / runtime / hip 不可）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <shelf/backend/common/device.hpp>
#include <shelf/status.hpp>

namespace shelf::placement {

using backend::MemoryKind;

// KV 1 トークンぶんのバイト数（full_attention 層のみ。GDN は固定費として別扱い）。
//   KV_bytes(T) = T × L_full × 2 × kv_heads × head_dim × sizeof(bf16)
// `engine_contract.md` §8.2 の式そのもの（同じ式を 2 箇所に書かないため、ここが正）。
constexpr std::uint64_t kv_bytes_per_token(std::size_t full_layers, std::size_t kv_heads,
                                           std::size_t head_dim,
                                           std::size_t bytes_per_elem = 2) noexcept {
    return static_cast<std::uint64_t>(full_layers) * 2u * kv_heads * head_dim * bytes_per_elem;
}

// 使える容量（呼び手が実測して渡す。0 は「不明」= その項目は判定しない）。
struct Limits {
    std::uint64_t device_free_bytes = 0;   // 1 台あたりの空き VRAM
    std::uint64_t host_free_bytes = 0;     // pinned 予定の空き（CPU 側）
    double headroom = 0.05;                // 予備率（0.05 = 5%）
};

// 1 ステージ分の必要量。
struct StageCost {
    std::uint64_t weight_bytes = 0;    // 実体化後の重み
    std::uint64_t state_bytes = 0;     // KV・conv・recurrent（kv_capacity 依存）
    std::uint64_t scratch_bytes = 0;   // ステップのアリーナ（行数依存）

    std::uint64_t total() const noexcept { return weight_bytes + state_bytes + scratch_bytes; }
};

// 構成全体の必要量。ステージは各自のデバイスに載る（= 判定は最大ステージで行う）。
struct Footprint {
    std::vector<StageCost> stages;
    std::uint64_t host_bytes = 0;      // pinned 正本に置く分（0 = すべて VRAM）

    std::uint64_t max_stage() const noexcept;
    std::uint64_t sum() const noexcept;
    void add_stage(const StageCost& c) { stages.push_back(c); }
};

// 1 リクエスト（同時に保持する量）の要求。
struct Demand {
    std::uint64_t kv_tokens = 0;       // 同時に保持する KV 長
    std::uint64_t host_kv_tokens = 0;  // pinned 正本に置く分（0 なら見ない）
    std::uint64_t kv_bytes_per_token = 0;   // `kv_bytes_per_token(...)` の値
    std::size_t stages = 1;            // PP の K（1 台に全段を置く構成では 1）
};

// 判定結果。不足は**名前付き**で返す（起動前拒否の理由に使う）。
struct Report {
    std::uint64_t required_bytes = 0;   // 最大ステージ × (1 + headroom)
    std::uint64_t usable_bytes = 0;     // device_free − (device_free × headroom)
    std::uint64_t host_required = 0;
    std::uint64_t host_usable = 0;
    std::uint64_t per_stage_required = 0;
    std::vector<std::string> shortfalls;   // 例: "stage 1: weights+kv が 2.1 GB 不足"

    bool ok() const noexcept { return shortfalls.empty(); }
    std::string describe() const;   // 1 行要約（ログ用）
};

// 予算判定（純粋）。`Limits` の 0 は「不明」として、その項目は判定しない
// （不明を 0 として誤って拒否しないため）。
Report plan_budget(const Footprint& footprint, const Demand& demand, const Limits& limits);

// 重み・状態の所在。既定は**全 device 常駐**（`directory_design` D2 の既定）。
struct Placement {
    std::string name;      // checkpoint 名
    std::uint64_t bytes = 0;
    MemoryKind kind = MemoryKind::device;
};

struct PlacementPlan {
    std::vector<Placement> weights;
    std::vector<Placement> states;

    std::uint64_t device_bytes() const noexcept;
    std::uint64_t host_bytes() const noexcept;
    bool all_on_device() const noexcept;
    std::size_t count(MemoryKind kind) const noexcept;
};

// 全 device 常駐のプラン（既定）。candidates の kind は無視して device に置く。
PlacementPlan place_all_device(std::vector<Placement> weights,
                               std::vector<Placement> states = {});

// 予算に収まる範囲で device に置き、残りを pinned_host へ落とす（低 VRAM 用の opt-in）。
// 収まらない場合は `Status::insufficient_memory` を返し、理由を `out` に残す。
Status place_fitting_budget(std::vector<Placement> weights, const Limits& limits,
                            PlacementPlan* out);

}  // namespace shelf::placement
