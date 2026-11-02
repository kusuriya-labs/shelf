// shelf backend/hip — GPU step executor for the single-token Model Graph.
//
// Executes one decode step of an ir::Graph on the device:
//   - weights pre-uploaded as bf16 (raw checkpoint bits),
//   - activations bf16 in an arena whose offsets are decided in the declare
//     phase (compiler/arena_offsets.hpp): the run phase only writes through
//     them, it never allocates. f32 where the vendored kernels demand it
//     (post-rmsnorm-before-rope, GDN prepared q/k, GDN output, logits),
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
#include <shelf/quant/fp8_finegrained.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <shelf/backend/common/stage_executor.hpp>
#include <shelf/backend/hip/kernels/gdn/gdn_types.h>
#include <shelf/compiler/arena_offsets.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/status.hpp>

namespace shelf::backend::hip {

struct HipStepExecutorOptions {
    int device = 0;
    // 1 ステップ分の活性化アリーナ。0 = declare 相（compiler/arena_offsets.hpp）が
    // グラフから決めた必要量をそのまま確保する。明示指定するとそれを上限にする
    // （offsets 表が収まらなければ insufficient_memory。行数を増やしたグラフを
    //   同じ実行器で回す場合は、足りない分だけ広げる）。
    std::size_t arena_bytes = 0;           // activations per step (0 = auto)
    // device メモリの予算（D2 の入口）。この実行器が確保する合計 — 重み + 状態 +
    // アリーナ + per-step scratch — の上限で、**モデル込み**の総量として扱う。
    // 0 = 無制限（従来動作）。予算を超える確保は hipMalloc を試みる前に
    // insufficient_memory として返す（どれを増やせばよいか数字で示す）。
    // HIP コンテキスト自体（数百 MB）は計上しないので、実_device 使用量はこれより
    // 寿命の重ならない出力に同じ offset を割り当てる（D4）。ステップ後に中間値を
    // 読むデバッグ経路（`fetch_value_debug`）では false にする。
    bool arena_reuse = true;
    double rope_theta = 1.0e7;             // from model config (composition root)
    std::size_t rope_rotary_dim = 64;      // pair-kernel constraint (model config)
    std::size_t kv_block_size = 16;        // from the resolved kernel
    // 活性化の FP8 量子化粒度（K のブロック長）。apps 側が checkpoint の宣言
    // （quantization_config）から決めて渡す。0 = 行（= トークン）全体で 1 スケール。
    std::uint32_t fp8_act_group = 128u;
    // N01: 行数がこの値以下の decode は **行ごとに decode1（GEMV）** で回す。
    // バッチ GEMM と GEMV は同じ数学でも加算順が違い、greedy の生成列が分岐する。
    // 1 行ずつ 1 並列と同じ経路に揃えることで、1〜N 並列の出力を一致させる。
    // 0 = 常にバッチ経路（性能比較用）。prefill（行数が多い）は影響を受けない。
    std::uint32_t decode1_max_rows = 4u;
    // T14: KV プールを pinned host（mapped）に置く。VRAM は使わないが、
    // attention は PCIe 越しに読む（低 VRAM 用の退避経路）。
    bool kv_pinned_host = false;
    // T14: pinned 正本のうち直近 N ページを VRAM にミラーする（online KV）。
    // kv_pinned_host のときだけ有効。0 = 無効（従来どおり全部を host から読む）。
    std::size_t online_kv_pages = 0;
    // T15: 状態（KV・conv・recurrent）をスロット数分確保する。
    // conv/recurrent はスロット先頭オフセットで並べる。
    std::size_t max_sequences = 1;
    // T16: KV のグローバルブロック数（層ごと）。0 = 自動（pages × max_sequences）。
    // 明示するとブロック表を任意のグローバル block へ向けられる（prefix cache 用）。
    std::size_t kv_blocks = 0;
    // T17: VRAM の確保に失敗したら pinned host へ落として続行する（縮退）。
    // 落ちた重みは CPU で計算する（MoE）か、mapped host を GPU が読む。
    bool allow_host_fallback = false;
};

class HipStepExecutor : public backend::StageExecutor {
public:
    explicit HipStepExecutor(HipStepExecutorOptions opts = {});
    ~HipStepExecutor();

    HipStepExecutor(const HipStepExecutor&) = delete;
    HipStepExecutor& operator=(const HipStepExecutor&) = delete;

    // Weight upload (host f32 -> device bf16, RNE).
    Status upload_weight(TensorId id, const std::vector<float>& f32);
    // Checkpoint-native bf16: raw bits straight to device (no host conversion).
    Status upload_weight_bf16(TensorId id, const bf16_t* data, std::size_t n);
    // 次の `upload_weight_*` の置き場を決める（T14 の置き場接続）。
    // true = pinned host に確保し、`hipHostGetDevicePointer` のマップ済み
    // device ポインタをカーネルへ渡す（低 VRAM 用。帯域は PCIe 相当）。
    // false（既定）= VRAM。upload 1 回ごとに消費される。
    void set_weight_placement(bool pinned_host) noexcept { pending_weight_pinned_ = pinned_host; }

    // pinned host に置いた重みのバイト数（device の計上には含めない）。
    std::size_t host_weight_bytes() const noexcept { return host_weight_bytes_; }
    // pinned host に置いた KV プールのバイト数（K + V）。
    std::size_t host_kv_bytes() const noexcept { return host_kv_bytes_; }
    std::size_t promoted_weight_bytes() const noexcept { return promoted_weight_bytes_; }
    // T17: MoE で CPU が計算した / GPU が計算したエキスパート数の累計。
    // 「pinned のままの重みは CPU、VRAM の重みは GPU」という分担の実測に使う。
    std::size_t moe_cpu_experts() const noexcept { return moe_cpu_experts_; }
    std::size_t moe_gpu_experts() const noexcept { return moe_gpu_experts_; }
    // N08: ルーターの同期回数と D2H 転送バイト数（累計）。
    std::size_t moe_router_syncs() const noexcept { return moe_router_syncs_; }
    std::uint64_t moe_router_copy_bytes() const noexcept { return moe_router_copy_bytes_; }
    double moe_cpu_ms() const noexcept { return moe_cpu_ms_; }

    // pinned 正本の重みを VRAM へ移す（転送 + 再配置）。host 側は正本として残す。
    // カーネルは次ステップから device の複製を読む。
    Status promote_host_weights();

    // MXFP4（quark）重み: E2M1 ニブルは断片順、スケールは **E8M0 の U8**
    // （カーネルが `const std::uint8_t*` で受けるため f32 にしない）。
    Status upload_weight_mxfp4(TensorId id, const std::uint8_t* codes, std::size_t n_codes,
                               const std::uint8_t* scales, std::size_t n_scales,
                               std::size_t features, std::uint32_t k_padded);

    // FP8 W8A8 重み（E4M3 コードは断片順、スケールは f32）。
    // weight_block128 = false: 出力特征ごと [features]（この checkpoint の宣言）
    // weight_block128 = true : 128(出力)×128(K) ブロック [n_blocks][k_blocks]
    Status upload_weight_fp8(TensorId id, const std::uint8_t* codes, std::size_t n_codes,
                             const float* scales, std::size_t features,
                             std::uint32_t k_padded, bool weight_block128);

    // Device allocation + zero-init for every declared state.
    Status declare_states(const ir::Graph& g);
    // Zero all state buffers and reset usage rows (new sequence).
    Status reset_states() override;

    // T15: 1 スロット分の状態（KV ページ・conv/recurrent）とアリーナ scratch を
    // 0 に戻す。連続バッチングで **新しい要求をスロットへ入れるとき**に呼ぶ。
    // `kv_blocks` を明示した（共有プール）ときは KV ページを 0 にしない
    // （他のスロットが参照している可能性がある）。
    Status reset_slot(std::size_t slot);

    // T16: スロットのブロック表を書き換える（論理 page `offset` から `count` 個）。
    // `blocks[p]` はグローバル block（< kv_blocks()）。全 KV 層に同じ表を書く。
    Status assign_kv_blocks(std::size_t slot, const std::uint32_t* blocks, std::size_t count,
                            std::size_t offset = 0);
    // 層あたりのグローバル KV ブロック数（declare 後）。
    std::size_t kv_blocks() const noexcept { return kv_total_pages_; }
    std::size_t kv_pages_per_sequence() const noexcept { return kv_pages_per_seq_; }

    // T16: prefix cache 用の GDN 状態（conv/recurrent）の退避と復元。
    // KV 以外の状態を宣言順に連結した float 列（スロット 1 個分）。
    // prefix の途中で KV ブロックを共有するには、その長さでの GDN 状態が要る。
    Result<std::vector<float>> export_gdn_state(std::size_t slot) const;
    Status import_gdn_state(std::size_t slot, const std::vector<float>& data);
    // デバッグ用: スロットの KV（ブロック表の先頭 `pages` ページ）の内容を
    // FNV-1a で要約する。prefix hit と scratch の KV が同じかを切り分ける。
    Result<std::uint64_t> debug_kv_hash(std::size_t slot, std::size_t pages,
                                        std::size_t visible = 0u) const;
    // デバッグ用: conv 状態と recurrent 状態を別々に FNV で要約する。
    Result<std::pair<std::uint64_t, std::uint64_t>> debug_state_hashes(
        std::size_t slot) const;
    // デバッグ用: スロットのブロック表（先頭 count ページ）を返す。
    Result<std::vector<std::uint32_t>> debug_block_table(std::size_t slot,
                                                         std::size_t count) const;

    // 1 ステップ分の行メタデータ（T15）。slots[i] は状態スロット、positions[i] は
    // その行の絶対位置（KV の書き込み先）。nullptr なら run_stage と同じ
    // 「全行 slot 0・positions = base_pos + i」。
    struct BatchRows {
        const std::uint32_t* slots = nullptr;
        const std::uint32_t* positions = nullptr;
    };

    // 複数スロットの 1 ステップ（decode バッチ / 単一スロットの prefill チャンク）。
    // 行は **slot ごとに連続**していること（GDN のリクエスト記述子をまとめるため）。
    Status run_batch(const ir::Graph& g, const std::int32_t* tokens, const bf16_t* hidden,
                     std::size_t n, const BatchRows& rows);

    // One step over `n` tokens at absolute positions base_pos..base_pos+n-1.
    // The graph's batch mode must match (n == graph leading rows).
    static constexpr std::size_t kMaxStepRows = 2048u;
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
                     std::size_t n, std::uint32_t base_pos) override;
    // `backend::StageExecutor::RowBatch` 版（PP の各ステージへ同じ行メタデータを渡す）。
    Status run_stage_batch(const ir::Graph& g, const std::int32_t* tokens, const bf16_t* hidden,
                           std::size_t n, const RowBatch& rows) override;

    // ステージ出力（bf16 のまま）を host へ。PP の hand-off 専用
    // （f32 経由にすると丸めが入りうるので生ビットで出す）。
    Result<std::vector<bf16_t>> fetch_value_bf16(const ir::Graph& g, ir::ValueId id) override;

    Result<std::vector<float>> fetch_logits(const ir::Graph& g) override;
    // Debug/parity: read back an activation value produced by the last run_step.
    Result<std::vector<float>> fetch_value_debug(const ir::Graph& g, ir::ValueId id);

    // ステップ 1 回分の内訳。`SHELF_HIP_STEP_PROFILE=1` のときだけ
    // event を取る（既定は last_step_ms だけ。イベントは作らない）。
    //   gpu_ms   = 入力アップロード + カーネル列の GPU 時間（event elapsed）
    //   kernel_ms= カーネル列だけ（アップロードを除く）
    //   upload_ms= 入力アップロード
    //   host_ms  = wall - gpu（CPU 側の dispatch・同期待ち・ホスト処理）
    struct Stats {
        double last_step_ms = 0.0;
        double last_gpu_ms = 0.0;
        double last_kernel_ms = 0.0;
        double last_upload_ms = 0.0;
        double last_host_ms = 0.0;
        // declare 相（アリーナ・状態の確保と offsets の適用）にかかった時間。
        // 起動時に 1 回だけ。PP ではステージ毎に持つ。
        double declare_ms = 0.0;
    };
    const Stats& stats() const { return stats_; }

    // 実際に確保しているアリーナのバイト数。
    // auto（arena_bytes == 0）では declare_states のグラフから決まる値を返す。
    std::size_t arena_bytes() const noexcept override { return arena_cap_; }
    // accounting: この実行器が現在確保している device バイト合計
    // （モデルの重みを含む）。
    std::size_t device_bytes() const noexcept { return dev_bytes_; }
    double last_step_ms() const override { return stats_.last_step_ms; }

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
        std::size_t pool_bytes = 0;     // kv: 1 プールあたりのバイト数（reset 用）
        std::size_t total_pages = 0;    // kv: max_sequences を含む global page 数
        void* host_k = nullptr;         // pinned host に置いたときのホスト側 K
        void* host_v = nullptr;         // 同 V
        void* online_k = nullptr;       // T14: 直近ページの VRAM ミラー
        void* online_v = nullptr;
        std::size_t online_pages = 0;   // ミラーのページ数（スロットあたり）
        std::size_t online_bytes = 0;   // ミラー全体（max_sequences 分）のバイト数
    };

    // run_stage / run_batch の共通実装。rows が nullptr なら全行 slot 0。
    Status run_impl(const ir::Graph& g, const std::int32_t* tokens, const bf16_t* hidden,
                    std::size_t n, std::uint32_t base_pos, const BatchRows* rows);

    // 1 primitive 分の実行。`prim_index` は declare 相の一時領域を引くための添字。
    Status run_primitive(const ir::Graph& g, const ir::Primitive& p, std::size_t prim_index,
                         std::uint32_t position);

    // declare 相の offsets 表（グラフ毎に一度だけ作って使い回す）。
    const compiler::ArenaPlan& plan_for(const ir::Graph& g);

    // swiglu の出力を fp8 活性化量子化まで融合できる値の集合（グラフ毎に一度）。
    // 条件は「その値の消費者が 1 個で、その linear が fp8 重みを使う」こと。
    struct FuseCache {
        const ir::Graph* graph = nullptr;
        std::size_t primitives = 0;
        std::uint32_t first_value = 0, last_value = 0;
        std::vector<std::uint32_t> swiglu_out;
        // rmsnorm の出力を fp8 量子化まで融合できる値（消費側が全て fp8 GEMM のとき）
        std::vector<std::uint32_t> rmsnorm_out;
        // rmsnorm_gated の末尾掛け算（swiglu）を fp8 量子化まで融合できる値
        std::vector<std::uint32_t> gated_out;
    };
    const FuseCache& fuse_for(const ir::Graph& g);
    std::vector<std::unique_ptr<FuseCache>> fuses_;
    // 値の置き場・一時領域（計画に無ければ nullptr）。
    void* slot_ptr(ir::ValueId v, bool f32) const;
    void* scratch_ptr(std::size_t prim_index, std::uint8_t slot, std::size_t need) const;

    // FP8 活性化の per-step scratch（断片順コード + per-token スケール）。
    // 同一ステップ内で GEMM が順次消費するため 1 本で足りる（行数が最大になる
    // 時点で確保し、以降は再利用）。
    struct Fp8Weight { std::uint8_t* codes = nullptr; float* scales = nullptr; bool block128 = false; };
    struct Mxfp4Weight { std::uint8_t* codes = nullptr; std::uint8_t* scales = nullptr;
                         std::size_t features = 0; std::uint32_t k_padded = 0; };
    // scale_floats = rows × (活性化スケールのブロック数)。
    // 次の upload の置き場に応じて確保する（device = hipMalloc、host = mapped pinned）。
    // `src` があれば内容をコピーする。`dev_out` はカーネルが読むポインタ。
    Status alloc_weight_payload(void** dev_out, std::size_t bytes, const void* src,
                                const char* what, TensorId id, int kind);
    // 重みの付属物（スケール等）を pinned host に置く（縮退用。host_weights_ には
    // 登録しない = CPU 計算の対象にはならない）。
    Status alloc_pinned_aux(void** dev_out, std::size_t bytes, const void* src,
                            const char* what);
    // KV プールの確保（opts_.kv_pinned_host なら mapped pinned）。
    Status alloc_state_pool(void** dev_out, void** host_out, std::size_t bytes,
                            const char* what);

    Status ensure_fp8_scratch(std::size_t code_bytes, std::size_t scale_floats);
    // device 確保の計上（予算 enforcement）。`dev_reserve` は hipMalloc を試みる前に
    // 予算を見るので、超過の理由は「確保済み + 要求 > 予算」の数字で分かる。
    // 実測使用量ではなくこの実行器の合計である点に注意（コンテキストは入らない）。
    Status dev_reserve(void** out, std::size_t bytes, const char* what);
    void dev_release(void* p);
    // hipMalloc は void** を取る（型付きポインタへの reinterpret_cast は
    // strict aliasing 違反。実際に -O2 で誤コード化を観察した）。
    // void* に取ってから型付き lvalue へ代入する。
    template <class T>
    Status dev_alloc(T** out, std::size_t count, const char* what) {
        void* p = nullptr;
        SHELF_RETURN_IF_ERROR(dev_reserve(&p, count * sizeof(T), what));
        *out = static_cast<T*>(p);
        return Status::make_ok();
    }

    std::unordered_map<void*, std::size_t> dev_allocs_;  // 確保中ポインタ -> バイト数
    std::size_t dev_bytes_ = 0;                          // dev_allocs_ の合計

    HipStepExecutorOptions opts_;
    hipStream_t stream_ = nullptr;
    void* arena_dev_ = nullptr;
    std::size_t arena_cap_ = 0;
    // 新しいシーケンスの最初のステップでアリーナを 0 クリアする（下の reset_states 参照）。
    bool arena_zero_pending_ = false;
    const compiler::ArenaPlan* plan_ = nullptr;   // このステップで使う offsets 表

    // グラフ → offsets 表。ポインタが同じでも中身が入れ替わりうるので、
    // プリミティブ数と端の ValueId も鍵に入れて取り違えを防ぐ。
    // ヒープに置く（push/破棄で他の要素が動かない = `plan_` が無効化されない）。
    struct CachedPlan {
        const ir::Graph* graph = nullptr;
        std::size_t primitives = 0;
        std::uint32_t first_value = 0, last_value = 0;
        compiler::ArenaPlan plan;
    };
    std::vector<std::unique_ptr<CachedPlan>> plans_;

    std::unordered_map<std::uint32_t, void*> weight_dev_;        // TensorId -> bf16*
    std::unordered_map<std::uint32_t, TensorShape> weight_shape_;
    std::unordered_map<std::uint32_t, Fp8Weight> weight_fp8_;    // TensorId -> fp8 (W8A8)
    std::unordered_map<std::uint32_t, Mxfp4Weight> weight_mxfp4_;  // TensorId -> mxfp4
    std::uint8_t* fp8_codes_dev_ = nullptr;                      // 断片順コード scratch
    float* fp8_scales_dev_ = nullptr;                            // per-token スケール scratch
    bool pending_weight_pinned_ = false;   // 次の upload を pinned host に置く
    bool allow_host_fallback_ = false;     // device 確保失敗時に pinned へ落とす
    int fail_dev_alloc_ = 0;               // テスト用: 先頭 N 回の device 確保を失敗させる
    std::size_t host_weight_bytes_ = 0;    // pinned host に置いた重みの合計
    std::vector<std::pair<void*, void*>> pinned_allocs_;  // (host, mapped dev) の対
    // 転送（promote）用の pinned 正本。kind: 0 = bf16, 1 = fp8 codes, 2 = mxfp4 codes。
    struct HostWeight {
        TensorId id;
        int kind = 0;
        void* host = nullptr;
        std::size_t bytes = 0;
        bool promoted = false;
        // kind 2（MXFP4 codes）のとき、CPU 計算用のスケールと形。
        std::vector<std::uint8_t> host_scales;
        std::uint32_t features = 0;
        std::uint32_t k_padded = 0;
    };
    std::vector<HostWeight> host_weights_;
    std::unordered_map<std::uint32_t, std::size_t> host_weight_index_;
    // T17: CPU で計算した（pinned のままの）エキスパート数の累計。計測・検証用。
    std::size_t moe_cpu_experts_ = 0;
    std::size_t moe_gpu_experts_ = 0;
    // N08: ルーターの D2H を 1 ステップ 1 回にまとめるための logits バッファと、
    // 転送量・同期回数の実測カウンタ。
    std::uint8_t* moe_logits_dev_ = nullptr;
    std::size_t moe_logits_cap_ = 0;
    std::size_t moe_router_syncs_ = 0;
    std::uint64_t moe_router_copy_bytes_ = 0;
    // N09: 行の索引と選択重みを GPU から直接読ませるための mapped 領域。
    void* moe_rows_sel_dev_ = nullptr;   // act_rows[total] / out_rows[total]
    void* moe_rows_w_dev_ = nullptr;     // sel_w[rows*k]
    std::size_t moe_rows_sel_cap_ = 0;
    std::size_t moe_rows_w_cap_ = 0;
    Status ensure_moe_rows_sel(std::size_t total, std::size_t rows);
    double moe_cpu_ms_ = 0.0;   // CPU 分担の計算に使った時間（累計）
    std::size_t promoted_weight_bytes_ = 0;
    std::size_t host_kv_bytes_ = 0;
    std::size_t kv_total_pages_ = 0;    // 層あたりのグローバル KV ブロック数
    std::size_t kv_pages_per_seq_ = 0;  // 1 シーケンスの最大ブロック数
    bool kv_shared_pool_ = false;       // kv_blocks を明示した（共有プール）
    // mapped host の device ポインタか（hipFree してはいけない）。
    bool is_pinned_dev(void* p) const noexcept;
    // 重み 1 本の解放（device なら hipFree、pinned なら何もしない＝最後に一括）。
    void release_weight_ptr(void* p) noexcept;
    std::size_t max_sequences_ = 1;                              // T15: スロット数
    std::uint32_t* rowslot_dev_ = nullptr;                       // 行ごとの slot
    std::uint32_t* slot_window_dev_ = nullptr;                   // T14: slot ごとの online 窓
    bool rows_multi_slot_ = false;                               // このステップの行が複数スロットか
    std::vector<std::uint32_t> rowslot_host_;
    std::vector<std::uint32_t> slot_window_host_;
    std::vector<std::uint32_t> rowpos_batch_host_;
    std::uint32_t cur_requests_ = 0;                             // GDN 記述子の数
    std::uint32_t cur_max_pos_ = 0;                              // バッチ内の最大位置 + 1
    // T17: MoE ブロックの一時領域（ルーター logits・専門家の gate/up/mid/out・
    // f32 累積）。幅は層で変わるので、必要量が増えたときだけ確保し直す。
    std::uint8_t* moe_scratch_dev_ = nullptr;
    std::size_t moe_scratch_cap_ = 0;
    // MoE の down 用に中間（swiglu 出力）を量子化する scratch（W4A8 の f32 スケール）。
    std::uint8_t* moe_act_codes_dev_ = nullptr;
    std::uint8_t* moe_act_scales_dev_ = nullptr;
    std::size_t moe_act_codes_cap_ = 0;
    std::size_t moe_act_scales_cap_ = 0;
    Status ensure_moe_scratch(std::size_t bytes);
    Status ensure_moe_act_scratch(std::size_t code_bytes, std::size_t scale_bytes);
    // MoE の 1 行を計算する（bf16 / MXFP4 の重み形式で分岐）。
    // N08: ルーターの logits と入力のホスト側コピーを呼び手が用意すると、
    // この行の D2H と同期を省く（1 ステップ分をまとめて読む）。
    Status run_moe_row(const ir::Primitive& p, const bf16_t* x_row, bf16_t* out_row,
                       std::size_t H, std::size_t E, std::size_t I, std::size_t Is,
                       bool shared, const float* logits_host = nullptr,
                       const bf16_t* x_host_in = nullptr);
    Status ensure_moe_logits(std::size_t bytes);
    // N09: 同一エキスパートへ複数行をまとめて流す（重みが全部 VRAM にあるとき）。
    // 行ごとの値は per-row 経路とビット同一。
    Status run_moe_grouped(const ir::Primitive& p, const bf16_t* xb, bf16_t* ob,
                           std::size_t rows, std::size_t H, std::size_t E, std::size_t I,
                           std::size_t Is, bool shared, const float* logits,
                           const float* shared_logits);
    std::size_t fp8_codes_cap_ = 0, fp8_scales_cap_ = 0;
    // ---- CP（コマンドプロセッサ）を起こしておく heartbeat -------------------
    // fp8 活性化 scratch に現在入っている値。同じ行・同じ K 切り・同じ粒度で
    // 消費者 GEMM が使える場合は量子化パスを省略する（同じ入力を使う兄弟 GEMM、
    // および swiglu 融合で生成した場合）。
    std::uint32_t fp8_owner_value_ = 0u;
    std::uint32_t fp8_owner_rows_ = 0u, fp8_owner_kp_ = 0u, fp8_owner_group_ = 0u;
    // 活性化 scratch が **1 行ずつ**（行あたり tiled_bytes(1,kp) バイト）で並んで
    // いるか。swiglu 融合を少数並列で行ごとに発射したとき true。行ごとの decode1 は
    // この並びをそのまま使える（false のバッチ並びとは互換でない）。
    bool fp8_owner_rowwise_ = false;
    std::vector<ir::StateId> state_ids_;
    std::unordered_map<std::uint32_t, StateLoc> state_dev_;
    std::unordered_map<std::uint32_t, ValueLoc> value_dev_;
    std::unordered_set<std::uint32_t> f32_out_rmsnorms_;          // ValueIds of rmsnorms feeding rope/gdn-norm

    // per-step device scratch
    static constexpr std::size_t kPosScratch = 8192u;
    std::int32_t* token_dev_ = nullptr;     // kMaxStepRows ids
    std::uint32_t* posrep_dev_ = nullptr;   // per-rope-row positions (rows*heads)
    std::uint32_t* rowpos_dev_ = nullptr;   // per-token row positions (kv append/attn)
    std::uint32_t* slot_zero_dev_ = nullptr;  // device constant 0 (row_sequence_slots)
    std::uint32_t* errword_dev_ = nullptr;  // embedding error word
    double* inv_freq_dev_ = nullptr;        // rope table (rotary/2)
    std::size_t step_rows_ = 1;
    std::uint32_t step_base_pos_ = 0;
    hipk::GdnRequestDescriptor* req_dev_ = nullptr;
    // N01: 1 行 = 1 リクエストの記述子（行ごとの decode1 用。kMaxStepRows 個）。
    hipk::GdnRequestDescriptor* row_req_dev_ = nullptr;
    // row_req_dev_ のホスト側（非同期コピーの間、寿命を保つためメンバに置く。
    // ブロックローカルの vector から非同期コピーすると、コピー完了前に解放され
    // 内容が不定になる）。
    std::vector<hipk::GdnRequestDescriptor> row_req_host_;
    float* logits_dev_ = nullptr;
    std::vector<std::uint32_t> posrep_host_;  // scratch positions (posrep_cap_)
    // rope の「行 × ヘッド」位置。posrep_host_（ステップ先頭の rowpos アップロード元）
    // を書き換えると、非同期コピーが完了前に上書きされて内容が不定になるため分ける。
    std::vector<std::uint32_t> rope_pos_host_;
    std::size_t posrep_cap_ = kPosScratch;    // 現在の確保数（足りなければ伸ばす）

    bool states_declared_ = false;
    Stats stats_;
};

} // namespace shelf::backend::hip
