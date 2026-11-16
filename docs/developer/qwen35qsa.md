# Qwen35Qsa（`model_type: qwen3_5_qsa`）

Qwen3.5 QSA checkpoint の contract。dense backbone（GDN + full attention、dense FFN）は
[qwen35.md](qwen35.md) と同じで、その上に block-sparse indexer を重ねる。
`qwen35` / `qwen35moe` は QSA を知らない。family の選択は構造キーで行う。

## identity と family

- root `model_type` は `qwen3_5_qsa`。gate は root の `model_type` だけを見る
  （[architecture.md](architecture.md) の Checkpoint識別）。
- `shelf::models::probe_model_dir` は root の `qsa` オブジェクトの有無でこの family を選ぶ。
  `qwen3_5_qsa` と宣言されていても `qsa` が無ければ dense として読まれ、その逆も同じ。
  読み方を決めるのは構造であって identity ではない。
- 対応する `ModelFamily` は `Qsa`。`LoadedModel` は `qsa_model()` で参照できる。

## config

root の `qsa` を `shelf::qwen35qsa::read_qwen35qsa_text_config` が読む。

| key | 意味 |
| --- | --- |
| `budget` | 1 query が保持する token 数。保持 block 数は `budget / block_size` |
| `block_size` | 連続する `block_size` key を 1 block にする圧縮率（2 の冪） |
| `indexer_heads` | indexer の query head 数（キーは 1 head 共有） |
| `indexer_head_dim` | indexer head 幅（select kernel の契約で 128 固定） |
| `rope_dim` | indexer head のうち partial RoPE を受けるチャネル数 |
| `multimodal_policy` | `dense` のみ実装 |

`budget % block_size == 0`、`rope_dim` は偶数かつ `indexer_head_dim` 以下を要求する。
違反は tensor を開く前に拒否する。

## weights

full-attention 層だけが indexer を持つ（GDN 層は持たない）。

| tensor | shape |
| --- | --- |
| `...self_attn._qsa_indexer.q_proj.weight` | `[indexer_heads * indexer_head_dim, hidden]` |
| `...self_attn._qsa_indexer.k_proj.weight` | `[indexer_head_dim, hidden]` |
| `...self_attn._qsa_indexer.q_norm` | `[indexer_head_dim]`（全 query head で共有） |
| `...self_attn._qsa_indexer.k_norm` | `[indexer_head_dim]` |
| `...self_attn._qsa_indexer.k_block_norm` | `[indexer_head_dim]` |

- `q_norm` は head ごとに同じ 1 本を使う。runtime では per-group norm
  （parameter 要素数 == `group_size`）として渡す。
- indexer は量子化しない。BF16/F32 payload を読み、F32 は load 時に BF16 へ widen する。
- 重みは backbone（`qwen35` の loader をそのまま使う）と並ぶ別配列に置く。
  backbone の weight struct に indexer の語彙を足さない。

## lowering

`lower_qwen35qsa_to_primitives` は層ごとに

```text
input_norm -> [indexer] -> gated attention -> residual -> post_norm -> FFN
```

を emit する。indexer は `emit_qwen35qsa` の stage で、`qwen_hybrid` の block を
`input_norm` と `attention+residual` に分けた間に挟まる。

- query: `q_proj` → per-head RMSNorm（共有 weight）→ partial RoPE（token 位置）
- key: `k_proj` → per-head RMSNorm → `QSA_KEY_CACHE` が
  「`block_size` 本の平均 → RMSNorm(`1 + parameter`) → block 先頭位置で partial RoPE」
  を行い、block key を per-sequence state に積む
- score: `relu(q·k)` を head 方向に加算（select kernel の `scale = 1/sqrt(head_dim)` は
  正の定数倍なので順位に影響しない）
- select: 観測済み block のうち上位 `min(top_k, blocks)` を bitmap にする
- attention はこの bitmap を block mask として消費する（`block_size` とセット）
- block 数 `blocks = ceil(max_context_tokens / block_size)`

## dense fallback（budget 以内）

`max_context_tokens <= budget` のとき、選択は何も削らない（全 block が残る）。
この場合は indexer を emit せず QSA state pool も確保しない。graph は dense attention と
同一になり、`--max-seq-len` が budget 以下なら QSA のコストはゼロになる。

## state

`QsaKeyStatePool` を attention 層数分確保する（`ExecutorModelInput::uses_qsa_state`）。

- `max_blocks = ceil(max_context_tokens / block_size)`
- 1 系列あたり `attention_layers * max_blocks * indexer_head_dim * 2 B` の pooled key と、
  `attention_layers * block_size * indexer_head_dim * 2 B` の partial block
- `executor_reserve_bytes` は dense の reserve に select workspace と pool を加える

## 制約

- pipeline stage と tensor parallel は未実装（明示的に `Status::unsupported`）。
- YaRN などの非 default rope profile は未実装。
- speculative decode（MTP drafter）は未配線。`LoadedModel::has_mtp_block()` は QSA
  family では false を返す（draft 棄却時に QSA state を巻き戻す経路が無いため）。
- select kernel の上限は `indexer_head_dim == 128`、`indexer_heads <= 8`、
  `top_k <= 1024`。

## dispatch

`try_launch_qsa_select` の tiled 経路（score を block tile ごとに分割し state を持ち回る）は
staged request が **全 row を覆っている**ときだけ使う。padded 行には slice が無く、
stale な position で未 score の block を選んでしまうため。覆っていない場合は
fused な `launch_qsa_select_batch` にフォールバックする。
