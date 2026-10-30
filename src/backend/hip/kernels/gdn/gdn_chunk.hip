// shelf GDN chunk — prefill 用のチャンク化 gated delta rule（gfx1201 FP16 WMMA）。
//
// 数式と前提は gdn_chunk.h の注記を参照。ここは実装だけ。
//
// 1 ブロック = 1 (request, v_head)。256 スレッド = 8 波。1 チャンク = 32 行。
// 波の担当は 16×16 タイル単位（fp16 WMMA の断片配置は linear_bf16 と同じ:
//   A: 行 m = lane&15, K = 8*(lane>>4)+0..7 / B: 行 n = lane&15, K 同上
//   C: 要素 (m = 8*(lane>>4)+j, n = lane&15)）。
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>

#include <shelf/backend/hip/kernels/gdn/gdn_chunk.h>

namespace shelf::hipk {
namespace {

using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));

constexpr std::uint32_t CH = kGdnChunk;   // チャンクの行数
constexpr std::uint32_t DK = 128u;        // head_k（前提）
constexpr std::uint32_t DV = 128u;        // head_v（前提）
constexpr std::uint32_t kWaves = 8u;
constexpr std::uint32_t kThreads = kWaves * 32u;

// LDS の行ピッチ（fp16 は 16 バイト境界を保つため 8 の倍数）。
constexpr std::uint32_t kPitch = DK + 8u;    // s_k  [CH][DK]  = 136
constexpr std::uint32_t kQPitch = CH;        // s_qk [CH][CH]  f16（16 B ベクトル読み → 8 の倍数）
constexpr std::uint32_t kMPitch = CH;        // s_m  [CH][CH]  f32（スカラー読みなので余白不要）
// workgroup を head_v の 64 列グループに割る。grid が num_v_heads 個だけだと
// 2048 行の prefill で 48 workgroup < 64 CU で機械を使い切っていなかった（段階 6）。
constexpr std::uint32_t DV2 = 64u;         // 1 workgroup が担う head_v 列数
constexpr std::uint32_t kVGroups = DV / DV2;
constexpr std::uint32_t kQP = 132u;        // s_q [CH][kQP] f16
constexpr std::uint32_t kDPitch = DV2;     // s_d [CH][DV2] f32（スカラー読み）
// LDS 64 KiB/CU で 2 workgroup 载せるためのピッチ。列アクセスは常に連続列なので
// ピッチ変更で銀行競合は増えない（前進代入は 1 列=1 スレッドで同じ行を読む）。

struct Smem {
    _Float16 k[CH * kPitch];      // 正規化済み k（bk の代わりに fp16 へ）
    // q も 1 回だけ LDS へ（Q·Kᵀ と Q·S で 2 回読む）。
    // ピッチは 132: 2 workgroup/CU を保つには LDS 合計を約 32,000 B 以下に抑える必要が
    // ある（32,256 B で実際に 1 workgroup/CU へ落ちて 2 倍遅くなった実測あり）。
    // 128 だと行ごとに同じ銀行へ衝突するので 4 だけ余らせる。
    _Float16 q[CH * kQP];
    _Float16 qk[CH * kQPitch];    // 減衰つき下三角 Q·Kᵀ
    float m[CH * kMPitch];        // 解くべき下三角（対角 1）
    float d[CH * kDPitch];        // 右辺 → delta
    float ge[CH], cu[CH], beta[CH], gk[CH];
};

__device__ __forceinline__ float bf16f(const shelf::bf16_t* p) {
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(p));
}

__device__ __forceinline__ float softplus_exact(float x) {
    return __logf(1.0f + __expf(x));
}

__device__ __forceinline__ f32x8 wmma16(f16x8 a, f16x8 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ f32x8 zero8() {
    return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__global__ void __launch_bounds__(kThreads) gdn_chunk_impl(
    const GdnRecurrenceArgs args, const std::uint32_t row_begin, const std::uint32_t row_end) {
    __shared__ Smem s;

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave = tid >> 5u;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t lm = lane & 15u;    // 断片の行
    const std::uint32_t kh = lane >> 4u;    // K の 8 要素組

    const std::uint32_t vg = blockIdx.x % kVGroups;
    const std::uint32_t v_head = blockIdx.x / kVGroups;
    const std::uint32_t v0 = vg * DV2;     // この workgroup の列オフセット
    const GdnRequestDescriptor req = args.requests[0];
    const std::uint32_t repeat = args.num_v_heads / args.key_heads;
    const std::uint32_t qk_head = v_head / repeat;

    const std::uint32_t head_k = args.head_k;
    const std::uint32_t head_v = args.head_v;
    if (head_k != DK || head_v != DV) return;

    float* state = args.recurrent_state +
                   static_cast<std::uint64_t>(req.slot) * args.recurrent_slot_stride +
                   static_cast<std::uint64_t>(args.state_index) * args.recurrent_layer_stride +
                   static_cast<std::uint64_t>(v_head) * DK * DV;
    const float* q_base = args.q + static_cast<std::size_t>(qk_head) * head_k;
    const float* k_base = args.k + static_cast<std::size_t>(qk_head) * head_k;

    // 状態はこの workgroup の 64 列ぶんだけグローバルで読む（LDS に張る版は
    // 段階 5b/6 で計測し、LDS 枠の関係で workgroup/CU が落ちて遅くなった）。
    // 1 発射でチャンクを順に処理する（チャンク毎の発射はしない）。
    //
    // 状態を LDS（fp16, 32 KiB）に張る版も実装したが**速くならなかった**
    // （2,561 → 2,603 µs/dis、prefill p50 は 983.7 → 985.2 ms）。理由:
    //   * per-head の状態 64 KiB は直近のレイヤぶん（48 × 64 KiB = 3 MB）しか触らず
    //     **L2 に収まる**ので、チャンク毎の往復は DRAM 帯域を消費していない
    //     （手順書の「L2/L3 に吸われる通信を節約として数えない」の該当）
    //   * そのうえ LDS 64,512 B で 1 workgroup/CU、VGPR 248 になり平行度が半減した
    // 参照（1,592 µs/dis）との差は平行度でも通信でもなく**データフローそのもの**
    // （head_v を 64 列に割って workgroup を倍にし、状態を LDS に張った形で 1 workgroup
    // が全チャンクを回す構成。`recurrence.hip` の `gdn_recurrence_wmma_body`）にある。
    for (std::uint32_t rb = row_begin; rb < row_end; rb += CH) {
        const std::uint32_t rows = (row_end - rb < CH) ? (row_end - rb) : CH;

    // ---- 1. k を LDS へ（K と Kᵀ）、ゲートを用意する -------------------------
    for (std::uint32_t e = tid; e < CH * DK; e += kThreads) {
        const std::uint32_t t = e / DK;
        const std::uint32_t i = e % DK;
        float val = 0.0f;
        if (t < rows) val = k_base[static_cast<std::size_t>(rb + t) * args.k_row_stride + i];
        const _Float16 h = static_cast<_Float16>(val);
        s.k[t * kPitch + i] = h;   // 転置版（kt）は持たない。状態更新側で転置読みする
        float qv = 0.0f;
        if (t < rows) qv = q_base[static_cast<std::size_t>(rb + t) * args.q_row_stride + i];
        s.q[t * kQP + i] = static_cast<_Float16>(qv);   // 換算は従来と同じ 1 回だけ
    }
    if (tid == 0u) {
        const float dt_bias = bf16f(args.dt_bias + v_head);
        const float neg_exp_a_log = -__expf(bf16f(args.a_log + v_head));
        float acc = 0.0f;
        for (std::uint32_t t = 0; t < CH; ++t) {
            if (t >= rows) {
                s.gk[t] = acc;
                s.beta[t] = 0.0f;
                continue;
            }
            const std::uint32_t row = rb + t;
            const float a_val =
                bf16f(args.a + static_cast<std::size_t>(row) * args.a_row_stride + v_head);
            const float b_val =
                bf16f(args.b + static_cast<std::size_t>(row) * args.b_row_stride + v_head);
            acc += neg_exp_a_log * softplus_exact(a_val + dt_bias);
            s.gk[t] = acc;
            s.beta[t] = 1.0f / (1.0f + __expf(-b_val));
        }
        const float gk_last = s.gk[rows > 0u ? rows - 1u : 0u];
        for (std::uint32_t t = 0; t < CH; ++t) {
            s.ge[t] = __expf(s.gk[t]);
            s.cu[t] = (t < rows) ? __expf(gk_last - s.gk[t]) : 0.0f;
        }
    }
    __syncthreads();

    // ---- 2. D = K·Kᵀ（32×32 = 4 タイル）→ 下三角の行列 M ---------------------
    if (wave < 4u) {
        const std::uint32_t ti = wave >> 1u, ui = wave & 1u;
        f32x8 acc = zero8();
        for (std::uint32_t kt = 0; kt < DK / 16u; ++kt) {
            const std::uint32_t kk = kt * 16u + kh * 8u;
            const f16x8 a = *reinterpret_cast<const f16x8*>(&s.k[(ti * 16u + lm) * kPitch + kk]);
            const f16x8 b = *reinterpret_cast<const f16x8*>(&s.k[(ui * 16u + lm) * kPitch + kk]);
            acc = wmma16(a, b, acc);
        }
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t t = ti * 16u + kh * 8u + j;
            const std::uint32_t u = ui * 16u + lm;
            float val = 0.0f;
            if (u < t)
                val = s.beta[t] * acc[j] * __expf(s.gk[t] - s.gk[u]);
            else if (u == t)
                val = 1.0f;
            s.m[t * kMPitch + u] = val;
        }
    }
    __syncthreads();

    // ---- 3. QK = Q·Kᵀ（32×32）に減衰と下三角マスクを掛けて LDS へ ------------
    if (wave < 4u) {
        const std::uint32_t ti = wave >> 1u, ui = wave & 1u;
        f32x8 acc = zero8();
        for (std::uint32_t kt = 0; kt < DK / 16u; ++kt) {
            const std::uint32_t kk = kt * 16u + kh * 8u;
            const f16x8 a = *reinterpret_cast<const f16x8*>(&s.q[(ti * 16u + lm) * kQP + kk]);
            const f16x8 b = *reinterpret_cast<const f16x8*>(&s.k[(ui * 16u + lm) * kPitch + kk]);
            acc = wmma16(a, b, acc);
        }
        // 減衰は fp32 の指数で入れる（比なので (0, 1]）。
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t t = ti * 16u + kh * 8u + j;
            const std::uint32_t u = ui * 16u + lm;
            const float val = (u <= t) ? acc[j] * __expf(s.gk[t] - s.gk[u]) : 0.0f;
            s.qk[t * kQPitch + u] = static_cast<_Float16>(val);
        }
    }
    __syncthreads();

    // ---- 4. KS と QS（A = K / Q、B = S₀ の列）→ 右辺と QS の保持 --------------
    //     タイル 16 個（2 行タイル × 8 列タイル）。wave w は idx = w と w + 8。
    f32x8 qs_reg = zero8();
    {
        const std::uint32_t idx = wave;        // 2 行タイル × 4 列タイル = 8
        const std::uint32_t ti = idx >> 2u;
        const std::uint32_t ji = idx & 3u;
        const std::uint32_t m0 = ti * 16u, j0 = ji * 16u;   // j0 は LDS 内の列
        f32x8 cks = zero8(), cqs = zero8();
        for (std::uint32_t kt = 0; kt < DK / 16u; ++kt) {
            const std::uint32_t kk = kt * 16u + kh * 8u;
            const f16x8 ak = *reinterpret_cast<const f16x8*>(&s.k[(m0 + lm) * kPitch + kk]);
            const f16x8 aq = *reinterpret_cast<const f16x8*>(&s.q[(m0 + lm) * kQP + kk]);
            // 状態 S₀ の fp16 読み出し。**split-fp16（hi + lo）は測って却下**:
            // 実装して unit.hip_gdn_chunk を回したが誤差は 5.0–6.3e-4 で変わらず
            // （誤差は q/k 側の丸めが支配的で、状態の寄与ではなかった）、KS/QS の
            // WMMA が 2 倍になるだけだった。精度を上げるには積の両オペランドを
            // hi/lo に分ける（WMMA 3 倍）必要がある。
            f16x8 b;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t i = kk + static_cast<std::uint32_t>(j);
                b[j] = static_cast<_Float16>(state[static_cast<std::size_t>(i) * DV +
                                                   v0 + (j0 + lm)]);
            }
            cks = wmma16(ak, b, cks);
            cqs = wmma16(aq, b, cqs);
        }
        qs_reg = cqs;
        // 右辺: rhs[t][j] = beta_t · (v[t][j] − ge_t · (K·S₀)[t][j])
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t t = m0 + kh * 8u + j;
            const std::uint32_t col = j0 + lm;
            const std::uint32_t row = rb + t;
            float vv = 0.0f;
            if (t < rows)
                vv = bf16f(args.v + static_cast<std::size_t>(row) * args.v_row_stride +
                           static_cast<std::size_t>(v_head) * DV + v0 + col);
            const float rhs = (t < rows) ? s.beta[t] * (vv - s.ge[t] * cks[j]) : 0.0f;
            s.d[t * kDPitch + col] = rhs;
        }
    }
    __syncthreads();

    // ---- 5. 前進代入（32 段・64 列を 64 スレッドで） --------------------------
    if (tid < DV2) {
        for (std::uint32_t t = 0; t < rows; ++t) {
            float acc = s.d[t * kDPitch + tid];
            const float* mrow = &s.m[t * kMPitch];
            for (std::uint32_t u = 0; u < t; ++u)
                acc -= mrow[u] * s.d[u * kDPitch + tid];
            s.d[t * kDPitch + tid] = acc;
        }
    }
    __syncthreads();

    // ---- 6. O = ge⊙(Q·S₀) + (減衰つき下三角 Q·Kᵀ)·delta -----------------------
    {
        const std::uint32_t idx = wave;        // 2 行タイル × 4 列タイル = 8
        const std::uint32_t ti = idx >> 2u;
        const std::uint32_t ji = idx & 3u;
        const std::uint32_t m0 = ti * 16u, j0 = ji * 16u;
        f32x8 acc = zero8();
        for (std::uint32_t ut = 0; ut < CH / 16u; ++ut) {
            const std::uint32_t uu = ut * 16u + kh * 8u;
            const f16x8 a = *reinterpret_cast<const f16x8*>(&s.qk[(m0 + lm) * kQPitch + uu]);
            f16x8 b;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t u = uu + static_cast<std::uint32_t>(j);
                b[j] = static_cast<_Float16>(s.d[u * kDPitch + (j0 + lm)]);
            }
            acc = wmma16(a, b, acc);
        }
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t t = m0 + kh * 8u + j;
            if (t >= rows) continue;
            const std::uint32_t col = j0 + lm;
            const float val = s.ge[t] * qs_reg[j] + acc[j];
            args.output[static_cast<std::size_t>(rb + t) * args.output_row_stride +
                        static_cast<std::size_t>(v_head) * DV + v0 + col] = val;
        }
    }
    __syncthreads();

    // ---- 7. S_T = e^{gk_T}·S₀ + Kᵀ·(c⊙delta) --------------------------------
    //     タイル 32 個（8 行タイル × 4 列タイル）。wave w は 4 個ずつ。
    const float gk_last = s.gk[rows > 0u ? rows - 1u : 0u];
    const float state_decay = __expf(gk_last);
    for (std::uint32_t idx = wave; idx < 32u; idx += kWaves) {
        const std::uint32_t it = idx >> 2u;
        const std::uint32_t jt = idx & 3u;
        const std::uint32_t i0 = it * 16u, j0 = jt * 16u;
        f32x8 acc;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t i = i0 + kh * 8u + static_cast<std::uint32_t>(j);
            acc[j] = state_decay * state[static_cast<std::size_t>(i) * DV +
                                         v0 + (j0 + lm)];
        }
        for (std::uint32_t ut = 0; ut < CH / 16u; ++ut) {
            const std::uint32_t uu = ut * 16u + kh * 8u;
            // kᵀ 断片: s.k[u][i] を転置して読む（kt を LDS に置かないことで
            // 10.2 KB 空け、workgroup あたりの LDS を下げる）。
            f16x8 a;
#pragma unroll
            for (int j = 0; j < 8; ++j) a[j] = s.k[(uu + static_cast<std::uint32_t>(j)) * kPitch + (i0 + lm)];
            f16x8 b;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t u = uu + static_cast<std::uint32_t>(j);
                b[j] = static_cast<_Float16>(s.cu[u] * s.d[u * kDPitch + (j0 + lm)]);
            }
            acc = wmma16(a, b, acc);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t i = i0 + kh * 8u + static_cast<std::uint32_t>(j);
            state[static_cast<std::size_t>(i) * DV + v0 + (j0 + lm)] = acc[j];
        }
    }
        __syncthreads();   // 次チャンクが LDS を上書く前に状態の更新を完了させる
    }
}

}  // namespace

bool gdn_chunk_supported(const GdnRecurrenceArgs& args) noexcept {
    if (args.num_requests != 1u) return false;
    if (args.requests == nullptr) return false;
    if (args.head_k != DK || args.head_v != DV) return false;
    if (args.num_v_heads == 0u || args.key_heads == 0u) return false;
    if (args.num_v_heads % args.key_heads != 0u) return false;
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr) return false;
    if (args.a == nullptr || args.b == nullptr) return false;
    if (args.dt_bias == nullptr || args.a_log == nullptr) return false;
    if (args.recurrent_state == nullptr || args.output == nullptr) return false;
    if (args.output_row_stride != args.head_v * args.num_v_heads) return false;
    return true;
}

// [row_begin, row_end) を 1 発射で処理する（チャンク長ぶんずつ LDS を使い回す）。
hipError_t launch_gdn_rows(const GdnRecurrenceArgs& args, std::uint32_t row_begin,
                           std::uint32_t row_end, hipStream_t stream) {
    if (!gdn_chunk_supported(args)) return hipErrorNotSupported;
    if (row_begin >= row_end || row_end > args.actual_rows) return hipErrorInvalidValue;
    const GdnRecurrenceArgs a = args;
    gdn_chunk_impl<<<dim3(kVGroups * args.num_v_heads), kThreads, 0, stream>>>(
        a, row_begin, row_end);
    return hipGetLastError();
}

// チャンク 1 個ぶん（rows <= CH）。単発射版のラッパ（テストと部分再実行用）。
hipError_t launch_gdn_chunk(const GdnRecurrenceArgs& args, std::uint32_t row_begin,
                            std::uint32_t rows, hipStream_t stream) {
    if (rows == 0u || rows > CH) return hipErrorInvalidValue;
    return launch_gdn_rows(args, row_begin, row_begin + rows, stream);
}

}  // namespace shelf::hipk
