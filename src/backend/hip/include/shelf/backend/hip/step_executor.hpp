// shelf backend/hip — GPU step executor for the single-token Model Graph.
//
// Executes one decode step of an ir::Graph on the device:
//   - weights pre-uploaded as bf16 (raw checkpoint bits),
//   - activations bf16 in a bump arena (per-step reset), f32 where the
//     vendored kernels demand it (post-rmsnorm-before-rope, GDN prepared
//     q/k, GDN output, logits),
//   - states device-resident (paged KV separate pools / conv / recurrent),
//     usage rows persist across steps (single sequence).
//
// Neutral by contract: ir:: + foundation:: + own kernels only. The
// composition root (apps/) wires model plans into this executor.
//
// Graph input convention (declare_input order):
//   [0] tokens   i32[1]   — the decode token id
//   [1] cos      f32[]    — unused on GPU (RoPE derives freqs on device)
//   [2] sin      f32[]    — unused on GPU
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <shelf/backend/hip/kernels/gdn/gdn_types.h>
#include <shelf/foundation/numeric.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/status.hpp>

namespace shelf::backend::hip {

struct HipStepExecutorOptions {
    int device = 0;
    // 1 ステップ分の活性化アリーナ。0 = declare_states のグラフから自動算出
    // （64 MiB + 2×Σ 出力(f32)。バンプ割当でステップ内の再利用が無いため
    //   必要量は行数に比例する: 4B 級モデルの実測で rows=16/32/64 が
    //   192/384/768 MiB 未満、既定だった 64 MiB は prefill 16 行で枯渇する）。
    // 明示指定するとそれを上限にする（枯渇時は insufficient_memory）。
    std::size_t arena_bytes = 0;           // activations per step (0 = auto)
    double rope_theta = 1.0e7;             // from model config (composition root)
    std::size_t rope_rotary_dim = 64;      // pair-kernel constraint (model config)
    std::size_t kv_block_size = 16;        // from the resolved kernel
    // 活性化の FP8 量子化粒度（K のブロック長）。apps 側が checkpoint の宣言
    // （quantization_config）から決めて渡す。0 = 行（= トークン）全体で 1 スケール。
    std::uint32_t fp8_act_group = 128u;
};

class HipStepExecutor {
public:
    explicit HipStepExecutor(HipStepExecutorOptions opts = {});
    ~HipStepExecutor();

    HipStepExecutor(const HipStepExecutor&) = delete;
    HipStepExecutor& operator=(const HipStepExecutor&) = delete;

    // Weight upload (host f32 -> device bf16, RNE).
    Status upload_weight(TensorId id, const std::vector<float>& f32);
    // Checkpoint-native bf16: raw bits straight to device (no host conversion).
    Status upload_weight_bf16(TensorId id, const bf16_t* data, std::size_t n);
    // FP8 W8A8 重み（E4M3 コードは断片順、スケールは f32）。
    // weight_block128 = false: 出力特征ごと [features]（この checkpoint の宣言）
    // weight_block128 = true : 128(出力)×128(K) ブロック [n_blocks][k_blocks]
    Status upload_weight_fp8(TensorId id, const std::uint8_t* codes, std::size_t n_codes,
                             const float* scales, std::size_t features,
                             std::uint32_t k_padded, bool weight_block128);

    // Device allocation + zero-init for every declared state.
    Status declare_states(const ir::Graph& g);
    // Zero all state buffers and reset usage rows (new sequence).
    Status reset_states();

    // One step over `n` tokens at absolute positions base_pos..base_pos+n-1.
    // The graph's batch mode must match (n == graph leading rows).
    static constexpr std::size_t kMaxStepRows = 64u;
    Status run_step(const ir::Graph& g, const std::int32_t* tokens, std::size_t n,
                    std::uint32_t base_pos);
    Status run_step(const ir::Graph& g, std::int32_t token, std::uint32_t position) {
        return run_step(g, &token, 1, position);
    }

    // pipeline-parallel の 1 ステージ実行。
    //   * 先頭ステージ（embedding を持つ）: `tokens` を渡し `hidden` は nullptr。
    //   * 中間・最終ステージ: `hidden`（前段の `hidden_out` の bf16 生ビット）を渡し
    //     `tokens` は nullptr。グラフの入力 dtype で判別する。
    // グラフの行数（prefill チャンク / decode 1 行）は n と一致させること。
    Status run_stage(const ir::Graph& g, const std::int32_t* tokens, const bf16_t* hidden,
                     std::size_t n, std::uint32_t base_pos);

    // ステージ出力（bf16 のまま）を host へ。PP の hand-off 専用
    // （f32 経由にすると丸めが入りうるので生ビットで出す）。
    Result<std::vector<bf16_t>> fetch_value_bf16(const ir::Graph& g, ir::ValueId id);

    Result<std::vector<float>> fetch_logits(const ir::Graph& g);
    // Debug/parity: read back an activation value produced by the last run_step.
    Result<std::vector<float>> fetch_value_debug(const ir::Graph& g, ir::ValueId id);

    struct Stats { double last_step_ms = 0.0; };
    const Stats& stats() const { return stats_; }

    // 実際に確保しているアリーナのバイト数。
    // auto（arena_bytes == 0）では declare_states のグラフから決まる値を返す。
    std::size_t arena_bytes() const noexcept { return arena_cap_; }

private:
    struct ValueLoc { void* dev = nullptr; bool f32 = false; std::size_t elems = 0; };
    struct StateLoc {
        void* dev = nullptr;            // conv/recurrent: single buffer
        void* k_pool = nullptr;         // kv_cache: separate pools
        void* v_pool = nullptr;
        std::uint32_t* table = nullptr; // device block table (identity, uint32)
        std::size_t rows = 0;
        std::size_t pages = 0;
        ir::StateKind kind = ir::StateKind::kv_cache;
        std::size_t elems = 0;          // elems per single buffer
    };

    Status run_primitive(const ir::Graph& g, const ir::Primitive& p,
                         std::uint32_t position);
    void* arena_alloc(std::size_t bytes);
    void arena_reset() { arena_top_ = 0; }

    // FP8 活性化の per-step scratch（断片順コード + per-token スケール）。
    // 同一ステップ内で GEMM が順次消費するため 1 本で足りる（行数が最大になる
    // 時点で確保し、以降は再利用）。
    struct Fp8Weight { std::uint8_t* codes = nullptr; float* scales = nullptr; bool block128 = false; };
    // scale_floats = rows × (活性化スケールのブロック数)。
    Status ensure_fp8_scratch(std::size_t code_bytes, std::size_t scale_floats);

    HipStepExecutorOptions opts_;
    hipStream_t stream_ = nullptr;
    void* arena_dev_ = nullptr;
    std::size_t arena_cap_ = 0, arena_top_ = 0;

    std::unordered_map<std::uint32_t, void*> weight_dev_;        // TensorId -> bf16*
    std::unordered_map<std::uint32_t, TensorShape> weight_shape_;
    std::unordered_map<std::uint32_t, Fp8Weight> weight_fp8_;    // TensorId -> fp8 (W8A8)
    std::uint8_t* fp8_codes_dev_ = nullptr;                      // 断片順コード scratch
    float* fp8_scales_dev_ = nullptr;                            // per-token スケール scratch
    std::size_t fp8_codes_cap_ = 0, fp8_scales_cap_ = 0;
    std::vector<ir::StateId> state_ids_;
    std::unordered_map<std::uint32_t, StateLoc> state_dev_;
    std::unordered_map<std::uint32_t, ValueLoc> value_dev_;
    std::unordered_set<std::uint32_t> f32_out_rmsnorms_;          // ValueIds of rmsnorms feeding rope/gdn-norm

    // per-step device scratch
    static constexpr std::size_t kPosScratch = 4096u;
    std::int32_t* token_dev_ = nullptr;     // kMaxStepRows ids
    std::uint32_t* posrep_dev_ = nullptr;   // per-rope-row positions (rows*heads)
    std::uint32_t* rowpos_dev_ = nullptr;   // per-token row positions (kv append/attn)
    std::uint32_t* slot_zero_dev_ = nullptr;  // device constant 0 (row_sequence_slots)
    std::uint32_t* errword_dev_ = nullptr;  // embedding error word
    double* inv_freq_dev_ = nullptr;        // rope table (rotary/2)
    std::size_t step_rows_ = 1;
    std::uint32_t step_base_pos_ = 0;
    hipk::GdnRequestDescriptor* req_dev_ = nullptr;
    float* logits_dev_ = nullptr;
    std::vector<std::uint32_t> posrep_host_;  // scratch positions (kMaxPosScratch)

    bool states_declared_ = false;
    Stats stats_;
};

} // namespace shelf::backend::hip
