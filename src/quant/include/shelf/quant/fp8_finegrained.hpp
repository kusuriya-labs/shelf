// shelf quant — FinegrainedFP8（ブロック単位スケール形式）の契約。
//
// 粗いスケール形式（fp8_perchannel.hpp）との違いはスケールの粒度だけ:
//   * 重みスケール: **128(出力) × 128(K)** のタイルに 1 個
//     （checkpoint の `weight_block_size: [128, 128]`）
//   * 活性化スケール: K を **128 要素**のグループにしたもの（`amax / kE4m3Max`）
//   * 数値は通常形と同じ `code(1 バイト) × scale(f32)`、コード配置も同じ断片順
//
// どちらの粒度を使うかは**checkpoint の宣言が正**で、apps 側が
// `quantization_config` から読む（決め打ちしない）:
//   9B-FP8  : strategy=token      -> 活性化 0     / 重み strategy=channel
//   27B-FP8 : blockwise dynamic   -> 活性化 128   / 重み weight_block_size=[128,128]
// 粒度の実測差（同じ実活性化に対する相対 L2 誤差 1.14 倍改善）は
// draft_docs/shelf_validate.md に記録してあるが、宣言に無い粒度は使わない。
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/quant/fp8_codec.hpp>

namespace shelf::quant::fp8::fine {

inline constexpr std::uint32_t kActGroupWholeRow = 0u;
inline constexpr std::uint32_t kActGroupK = 128u;

// 活性化スケールの個数（行あたり）。
constexpr std::uint32_t act_scale_groups(std::uint32_t k_padded, std::uint32_t group) noexcept {
    return (group == 0u) ? 1u : (k_padded / group);
}

// 重みスケールの粒度。
//   PerChannel : 出力特征ごとに 1 個（通常形）
//   Block128   : 128(出力) × 128(K) タイルに 1 個（この形式）
enum class WeightScaleGranularity : std::uint8_t {
    PerChannel,
    Block128,
};

// K 方向のスケールブロック数（128 単位。端数は切り上げ = 詰めを含むタイル数）。
constexpr std::uint32_t scale_k_blocks(std::uint32_t k_padded) noexcept {
    return (k_padded + kActGroupK - 1u) / kActGroupK;
}

// 出力方向のブロック数（Block128 のとき 128 単位、PerChannel のときは特征数そのもの）。
constexpr std::uint32_t scale_n_blocks(WeightScaleGranularity g, std::uint32_t features) noexcept {
    return (g == WeightScaleGranularity::Block128) ? ((features + kActGroupK - 1u) / kActGroupK)
                                                   : features;
}

// 重みスケール配列の要素数（f32）。
//   PerChannel : 出力特征ごと（K 方向は 1）
//   Block128   : [ceil(features/128)][ceil(k_padded/128)]
constexpr std::size_t weight_scale_count(WeightScaleGranularity g, std::uint32_t features,
                                         std::uint32_t k_padded) noexcept {
    if (g == WeightScaleGranularity::Block128)
        return static_cast<std::size_t>(scale_n_blocks(g, features)) * scale_k_blocks(k_padded);
    return features;
}

// 活性化スケール配列の要素数（f32、行優先 [rows][k_blocks]）。
constexpr std::size_t activation_scale_count(std::uint32_t rows, std::uint32_t k_padded,
                                             std::uint32_t act_group) noexcept {
    return static_cast<std::size_t>(rows) * act_scale_groups(k_padded, act_group);
}

}  // namespace shelf::quant::fp8::fine
