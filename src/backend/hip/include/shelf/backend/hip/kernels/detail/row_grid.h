// 行方向のブロック数を grid.y に写すカーネルの分割発射。
//
// HIP の grid.y / grid.z の上限は 65535。行を blockIdx.y に対応させるカーネル
// （elementwise 系、l2 normalize、gdn qk prepare）は、行数が上限を超えると
// 「invalid configuration argument」で発射自体が失敗する。これらのカーネルは
// 行ごとに独立で、行は stride 経由のオフセットとしてしか参照しないため、
// 基底ポインタを行分だけ進めて何回か発射すれば 1 発射と同一の結果になる。
//
// 行数が 2048 を超える例（GDN の head 方向を畳んだ行）:
//   linear_num_value_heads 48 × 2048 行 = 98,304 行（rmsnorm_gated の swiglu）
//   linear_num_value_heads 32 × 2048 行 = 65,536 行（同上、9B）
#pragma once

#include <shelf/backend/hip/kernels/elementwise/elementwise.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace shelf::hipk::detail {

constexpr uint32_t kMaxGridY = 65535u;  // HIP の grid.y 上限

// 各ポートの要素サイズ（byte）。shift_rows は stride（要素単位）を
// バイトに変換して基底ポインタを進めるため、型情報をここで伝える。
struct PortElemSize {
    std::size_t in0 = 2u;
    std::size_t in1 = 2u;
    std::size_t out0 = 2u;
    std::size_t out1 = 2u;
};

// base の各行独立ポインタを begin 行進めたコピーを作る（null ポートは触らない）。
inline void shift_rows(ElementwiseArgs& a, const ElementwiseArgs& base, uint32_t begin,
                       const PortElemSize& es) {
    const auto adv_in = [](const void* p, uint32_t stride, std::size_t esz,
                           uint32_t b) -> const void* {
        return p ? static_cast<const char*>(p) + static_cast<size_t>(b) * stride * esz : nullptr;
    };
    const auto adv_out = [](void* p, uint32_t stride, std::size_t esz,
                            uint32_t b) -> void* {
        return p ? static_cast<char*>(p) + static_cast<size_t>(b) * stride * esz : nullptr;
    };
    a.in0 = adv_in(base.in0, base.in0_row_stride, es.in0, begin);
    a.in1 = adv_in(base.in1, base.in1_row_stride, es.in1, begin);
    a.out0 = adv_out(base.out0, base.out0_row_stride, es.out0, begin);
    a.out1 = adv_out(base.out1, base.out1_row_stride, es.out1, begin);
}

// emit(args, grid_x, grid_y) を grid.y 上限の区間に分割して呼ぶ。
// 呼び出し側は grid_x（features 方向）を変えないこと。
// 発射エラーがあれば最初のエラーを返す（同期はしない）。
template <class Emit>
inline hipError_t launch_rows(const ElementwiseArgs& base, uint32_t grid_x,
                              const PortElemSize& es, const Emit& emit) {
    for (uint32_t begin = 0u; begin < base.rows; begin += kMaxGridY) {
        ElementwiseArgs args = base;
        args.rows = std::min(base.rows - begin, kMaxGridY);
        shift_rows(args, base, begin, es);
        emit(args, grid_x, args.rows);
        const hipError_t e = hipGetLastError();
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}

}  // namespace shelf::hipk::detail
