// shelf backend/hip — GDN チャンク版（prefill 用・gfx1201 FP16 WMMA）。
//
// 逐次の gated delta rule
//   S_t = d_t·S_{t-1} + k_t·delta_tᵀ,  delta_t = beta_t·(v_t − (d_t·S_{t-1})ᵀ·k_t)
// を、チャンク内で**まとめて**解く。1 チャンク = CH 行として
//   gk_t = Σ_{s≤t} g_s（対数減衰の累積。≤ 0）,  ge_t = e^{gk_t},  c_u = e^{gk_T−gk_u}
// と置くと、delta は下三角系
//   (I + tril(diag(beta)·G, −1))·delta = diag(beta)·(V − ge⊙(K·S₀)),  G[i][j] = (k_i·k_j)·e^{gk_i−gk_j}
// を解いたものになり、出力と状態は
//   O = ge⊙(Q·S₀) + tril(e^{gk_i−gk_j}⊙(Q·Kᵀ))·delta
//   S_T = e^{gk_T}·S₀ + Σ_u c_u·k_u·delta_uᵀ
// で書ける。減衰はすべて (0, 1] の**比**で作る（累積そのものを割らない）ので、
// チャンク内の減衰が大きくても f32 で溢れない。
//
// 行列積（K·Kᵀ / Q·Kᵀ / K·S₀ / Q·S₀ / QK·delta / Kᵀ·delta）は FP16 の WMMA
// （16×16×16・累積 f32）。状態 S は f32 のまま。
//
// 形の前提: head_k = 128, head_v = 128, CH = 32（`gdn_chunk_supported`）。
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gdn/gdn_recurrence.h>

namespace shelf::hipk {

// チャンク版が使える形か（カーネルは head 形状を固定で持つ）。
bool gdn_chunk_supported(const GdnRecurrenceArgs& args) noexcept;

// チャンク 1 個ぶん（rows ≤ kGdnChunk）を回す。`rows` がチャンク長より短い場合は
// 余りを 0（beta = 0, c = 0）として扱い、出力は rows 行ぶんだけ書く。
// [row_begin, row_end) を 1 発射で処理（チャンクはカーネル内ループ）。
hipError_t launch_gdn_rows(const GdnRecurrenceArgs& args, std::uint32_t row_begin,
                           std::uint32_t row_end, hipStream_t stream);

hipError_t launch_gdn_chunk(const GdnRecurrenceArgs& args, std::uint32_t row_begin,
                            std::uint32_t rows, hipStream_t stream);

inline constexpr std::uint32_t kGdnChunk = 32u;

}  // namespace shelf::hipk
