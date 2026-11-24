# HunYuanDenseV1 (`hunyuan_v1_dense`)

Tencent **Hy-MT3-1.8B / Hy-MT3-7B**（`model_type: "hunyuan_v1_dense"`、`architectures: ["HunYuanDenseV1ForCausalLM"]`）の BF16 および EXL3-INT8（mul24、body 4 bpw・`lm_head` 6 bpw）推論 contract。現在は単一 GPU（TP=1 / PP=1、speculative decode 非対応）の `shelf-compute` / `shelf-bench` / `shelf-server` で動作する。

## 対応形状と config gate

`src/shelf/models/hunyuan_dense/model/config.cpp` は `config.json` を検査し、未実装機能を tensor 読み込み前に `Status::unsupported` で拒否する。

| モデル | layers | hidden | intermediate | q_heads | kv_heads | head_dim | vocab_size | EXL3 `lm_head` 出力 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hy-MT3-1.8B | 32 | 2048 | 6144 | 16 | 4 | 128 | 120818 | 120832 |
| Hy-MT3-7B | 32 | 4096 | 14336 | 32 | 8 | 128 | 128167 | 128256 |

- `model_type == "hunyuan_v1_dense"` かつ（指定がある場合は）`architectures` に `"HunYuanDenseV1ForCausalLM"` を含むことを必須とする
- `use_cla == true`、`attention_bias == true`、`bias == true`、`mlp_bias == true`、`add_classification_head == true`、`use_qk_norm == false`、`use_rotary_pos_emb == false`、`hidden_act != "silu"`、`norm_type != "rms"` は拒否する
- `--rope-factor` の明示上書きは拒否し（`auto` のみ受理）、`max_seq_len` は `max_position_embeddings`（262144）以下に制限する
- 停止 token は `generation_config.json` の `eos_token_id`（スカラーまたは配列: 1.8B は `120020`、7B は `[127960, 127967]`）を優先し、無い場合は `config.json` の `eos_token_id` にフォールバックする

## アーキテクチャ差分と lowering 契約

1. **Post-RoPE QK RMSNorm**
   - `self_attn.q_proj` / `self_attn.k_proj` 出力に対して先に `RoPENode`（`bf16_intermediates = true`）を適用し、その後に `self_attn.query_layernorm` / `self_attn.key_layernorm`（`group_size = 128`, `OFFSET_BF16_SMALL_GROUP`）を適用する。
   - すべての RMSNorm（`input_layernorm` / `post_attention_layernorm` / `query_layernorm` / `key_layernorm` / `model.norm`）は `+1` オフセット無しの標準 RMSNorm であり、`weight_offset = 0.0f` を設定して正規化後の BF16 中間丸めと重み乗算を公式 `HunYuanDenseV1RMSNorm` に一致させる。
2. **NTK-Alpha 静的 RoPE**
   - `rope_scaling.type == "dynamic"` かつ正の `alpha`（`1000.0`）が指定されている場合、`EffectiveRopeProfile::Type::NtkAlpha` として解決する。
   - Executor 初期化時に $\theta_{\text{eff}} = \theta \cdot \alpha^{d / (d - 2)}$（$d = \text{head\_dim}$）を計算し、float32 周波数テーブル（`float32_frequencies = true`）を生成する。
   - `dynamic` で `alpha` が無い設定や他の未対応 `rope_scaling.type` は `Status::unsupported` で拒否する。
3. **128 境界パディング済み EXL3 `lm_head` と `tie_word_embeddings`**
   - 原本 BF16 チェックポイントは `tie_word_embeddings: true` で `lm_head.weight` を持たず、`model.embed_tokens.weight` を共有する。
   - EXL3-INT8 チェックポイントは `config.json` に `tie_word_embeddings: true` を残したまま独立した 6 bpw の `lm_head.{trellis,suh,svh,mul1}` を持ち、EXL3 制約のため出力次元が 128 の倍数へパディングされている。
   - Loader は `lm_head.trellis` または `lm_head.weight` が存在する場合は独立した重みとして優先ロードし（無い場合のみ `tie_word_embeddings` に従い `model.embed_tokens.weight` を共有）、行数が `vocab_size` または `[vocab_size, vocab_size + 128)` の 128 倍数であることを検証する。
   - Graph lowering は `LinearNode` 出力幅を `w.lm_head.rows`（パディング済み次元）で確保しつつ、`SamplingNode` の有効語彙幅に `c.vocab_size` を渡す。`Executor::logits` の行ストライドは `program.cpp` と同じ 16 バイト境界に揃え、`ComputeSession::copy_logits` および `shelf-bench quality` は先頭 `vocab_size` 列だけをホストへコピーする。これにより argmax / stochastic sampling / KLD 評価でパディング領域 `[vocab_size, padded_vocab)` は一切参照されない。
4. **EXL3-INT8 実行と最適化パス**
   - Layer 線形重み（`q_proj` / `k_proj` / `v_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj`）は Qwen3.5 系の標準 EXL3-INT8 mul24 経路と融合 SwiGLU / decode GEMV / HIP Graph を使用する（`plamo_exl3_linear` は無効）。
   - GDN 状態を持たないため `make_gdn_state_layout` は `declared_model_type == "hunyuan_v1_dense"` に対して空レイアウトを返す。

## サーバー統合

`src/apps/server/shelf_server.py` は `model_type == "hunyuan_v1_dense"` を非 reasoning 翻訳モデルとして扱い、起動時の reasoning template 検査をスキップして `supports_reasoning=False` で `ChatService` を初期化する。これにより `/v1/chat/completions`（通常応答および SSE ストリーミング）の出力が `reasoning_content` へ誤分離されず `content` として返される。
