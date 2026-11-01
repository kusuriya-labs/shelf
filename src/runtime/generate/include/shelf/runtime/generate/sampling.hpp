// shelf api — 生成パラメータと抽選（`engine_contract.md` §3 の実装）。
//
// 契約（そのままコードにしてある。変えるときは文書を先に直す）:
//   * パラメータは 5 つ: temperature / top_k / top_p / min_p / seed
//   * 実行順序は **top_k → top_p → min_p → temperature → 抽選**（列挙順ではない）
//     — top_p / min_p は**確率**で切るので、温度を先に掛けると生存候補が変わる
//   * `temperature == 0`（既定）は **greedy**（乱数を一切使わない = 現行 shelf と同じ）
//   * 乱数は `(seed, position)` の**純粋関数**（状態付き RNG を使わない）。
//     こうしないと結果がバッチ構成に依存する（radiance が明文化している失敗パターン）
//   * seed 未指定はエンジンが採番し、`done.seed` で返す（再現性のため）
//
// 置き場: `src/runtime/generate`（生成の進行層。`engine_contract.md` §3.3）。
// logits を受け取って index を返すだけで、backend も HIP も知らない。
// `src/api` はこれを再輸出する（Server 層は runtime を参照してよいが、
// runtime は api を参照しない = 境界ルール #12）。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <shelf/status.hpp>

namespace shelf::runtime::generate {

// 生成パラメータ（`engine_contract.md` §3 の 5 つ）。
struct Sampling {
    double temperature = 0.0;   // 0 = greedy
    std::uint32_t top_k = 0;    // 0 = off
    double top_p = 1.0;         // (0, 1]
    double min_p = 0.0;         // [0, 1]。0 = off
    std::optional<std::uint64_t> seed;   // 未指定 = エンジン採番

    // 既定のまま（= greedy）か。greedy なら乱数も温度も使わない。
    bool is_greedy() const noexcept { return temperature == 0.0; }
};

// 検証（`engine_contract.md` §3.2）。違反は `invalid_argument` で、**フィールド名**を
// message に含める（Server が `param` に写せるようにする）。
// `param` を渡すと違反したフィールド名を返す。
Status validate_sampling(const Sampling& s, std::string* param = nullptr);

// Philox4x32-10（契約 §3.4 の推奨アルゴリズム）。
// 状態を持たない: `(seed, position)` から 4 ワードを決定的に作る。
class Philox4x32 {
public:
    static std::array<std::uint32_t, 4> bits(std::uint64_t seed, std::uint64_t position) noexcept;
    // [0, 1) の一様乱数（bits の先頭ワードから。仮数 32 bit をそのまま使う）。
    static double uniform(std::uint64_t seed, std::uint64_t position) noexcept;
};

// 抽選の結果。
struct Draw {
    std::uint32_t index = 0;        // 選ばれた候補の logits 添字
    std::uint32_t candidates = 0;   // 絞り込み後に残った候補数（1 = 実質 greedy）
    bool used_random = false;       // 乱数を使ったか（greedy なら false）
};

// logits から 1 つ選ぶ。`position` は生成位置（トークン位置。乱数の鍵）。
//   * 既定（temperature 0）は argmax。NaN は無視（安全側に倒す）
//   * 同値の argmax は**添字の小さい方**（決定的）
// 検証違反は `invalid_argument`（呼ぶ前に `validate_sampling` を通すのが本来）。
Result<Draw> sample(const std::vector<float>& logits, const Sampling& s, std::uint64_t position);

// 連続バッチの行など、`logits` の部分列をそのまま抽選する版（コピーしない）。
// `n` は語彙数（範囲外アクセスはしない）。
Result<Draw> sample_range(const float* logits, std::size_t n, const Sampling& s,
                          std::uint64_t position);

// seed 未指定のときの採番（エンジン側）。`done.seed` に載せて返す。
// 0 も正規の値として扱う（契約 §3.4）。
std::uint64_t assign_seed(std::optional<std::uint64_t> requested, std::uint64_t nonce) noexcept;

}  // namespace shelf::runtime::generate
