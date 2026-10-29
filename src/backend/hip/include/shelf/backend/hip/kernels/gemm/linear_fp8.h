// shelf backend/hip — FP8 W8A8 linear（E4M3 行列命令・gfx1201）。
//
// 数値の契約（`src/quant/include/shelf/quant/fp8_w8a8.hpp` と同一）:
//   out[m, n] = ( Σ_k a_code[m,k] · w_code[n,k] ) · a_scale[m][kb] · w_scale[nb][kb]
//   * 活性化: `act_group = 128` のとき K を 128 要素のブロックごとに 1 スケール、
//     0 のときは行（= トークン）全体で 1 スケール。**どちらかは checkpoint の宣言が正**。
//   * 重み: `weight_block128 = false` のとき出力特征ごと（この checkpoint の
//     `strategy: channel`）。true のとき 128(出力)×128(K) ブロック
//     （移植元 PhaseShift の native 形）。
//   * コードはいずれも**断片順**（`quant/fp8` の preshuffle と同じ形。一致は
//     tests/unit/test_hip_gemm_fp8.cpp が readback で検証する）。
//
// 元: PhaseShift（own work, CopenDeCamp）の `kernels/optimized/linear/fp8.hip`。
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

enum class Fp8GemmOutputDType : std::uint8_t {
    BF16,
    F32,
};

enum class Fp8GemmConfigId : std::uint8_t {
    Auto,         // rows から選ぶ（rows=1 は Decode1、それ以外は RowBlock1/2/4）
    Decode1,      // rows == 1（GEMV 形・行列命令 1 本で足し積む）
    RowBlock1,    // rows <= 16
    RowBlock2,    // rows <= 32
    RowBlock4,    // rows <= 64（kMaxStepRows）
};

struct Fp8GemmConfig {
    Fp8GemmConfigId id = Fp8GemmConfigId::Auto;
};

// スケール配置。個数の算出は quant/fp8（scale_n_blocks / scale_k_blocks /
// act_scale_groups）が正で、ここで手計算しない。
struct Fp8GemmScaleArgs {
    std::uint32_t act_group = 0u;            // 0 = 行全体、128 = K を 128 単位（呼び手が宣言から設定）
    bool weight_block128 = false;              // true: 128×128 block、false: per-channel
    std::uint32_t activation_scale_stride = 1u; // f32 個数 / 行
    std::uint32_t weight_scale_k_blocks = 1u;   // Block128 の行 stride（f32 個数）
};

// 受理される形か（rows=1 は Decode1、rows<=64、K は 32 の倍数）。
bool fp8_gemm_shape_supported(std::uint32_t rows, std::uint32_t features,
                              std::uint32_t k_padded) noexcept;

// スケール配置が受理できるか（act_group は 0/128、stride 整合）。
bool fp8_scale_args_supported(const Fp8GemmScaleArgs& sc, std::uint32_t rows,
                              std::uint32_t features, std::uint32_t k_padded) noexcept;

// rows=1 専用カーネルが使える形か（GEMV 形は 16 出力単位・K が 32 の倍数）。
bool fp8_decode1_supported(std::uint32_t rows, std::uint32_t features,
                           std::uint32_t k_padded) noexcept;

// 1 ステップ分の断片順活性化バッファに必要なバイト数（codes）。
std::size_t fp8_activation_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept;

hipError_t launch_gemm_fp8_w8a8(
    const Fp8GemmConfig& config,
    const std::uint8_t* weight_codes,      // 断片順 [features_pad][k_padded]
    const float* weight_scales,            // per-channel [features] または [n_blocks][k_blocks]
    const std::uint8_t* activation_codes,  // 断片順 [rows_pad][k_padded]
    const float* activation_scales,        // [rows][activation_scale_stride]
    void* output,
    Fp8GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t output_row_stride,       // 要素数
    const Fp8GemmScaleArgs& scales,
    hipStream_t stream);

}  // namespace shelf::hipk
