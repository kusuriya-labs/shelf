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

## tensor parallel

`load_qwen35qsa_weights_tensor_parallel_rank` は backbone を
`load_qwen35_weights_tensor_parallel_rank` の rank-local plan で読み、
indexer は **replicated** のまま全 rank に置く。

- indexer が replicated なのは、各 rank が同じ block mask を作る必要があるため。
  backbone の hidden は TP combine 後に rank 間で一致するので、同じ indexer weight と
  同じ入力から同じ mask が決まる。mask を all-gather / all-reduce しない。
- indexer の `q_proj` / `k_proj` は TP plan に入れない。head 数
  （`indexer_heads`）は attention の `num_attention_heads` と独立で、
  QSA select kernel も global の indexer head 数で動く。
- lowering は backbone の attention / GDN / dense FFN を
  `Qwen35TensorParallelContext` の local geometry（local heads / local intermediate）で
  emit する。`o_proj` / `out_proj` / `mlp.down_proj` は `tp_combine` を持つ。
- vocab-parallel `lm_head` は dense family と同じ条件（BF16 または EXL3 で、plan に
  axis 0 の entry がある）で使う。
- MTP block は TP rank では load しない（`unsupported`）。

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

## attention

bitmap の意味は上記のまま変わらない。実行形は 2 つある。

- **block-sparse kernel**（`kernels/optimized/attention/qsa_sparse.hip`）。
  条件は BF16 / FP8 / octave KV / F32 output / block mask あり / tree ancestor なし /
  `head_dim == 256` / `q_heads == 6 * kv_heads` または `12 * kv_heads` /
  `block_size == 4` / `page_tokens % 4 == 0`。1 CTA は `(split, kv_head, row)` で、
  3 query head ずつを warp に割り当て、`q_per_kv = 6` は 2 warp、`12` は 4 warp を持つ。
  選択 block の 4 cell を page table 経由で直接読み、
  K/V を kv head あたり 1 回だけ読む。decode region は block list を split し、
  softmax の部分状態を reduce kernel で結合する。`block_size == 4` なので
  1 block は必ず 1 page 内の連続 4 token に写る。
- **masked kernel**。上記以外（tree 行など）は従来どおり bitmap を page 間引きと
  token 可視判定に使う paged attention で実行する。

prefill は `max_visible_tokens < 12288` の間 masked tiled kernel を使い、
それ以上で block-sparse に切り替える。decode region は常に block-sparse を選ぶ。

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
- TP では rank ごとに indexer が replicated なので pool も rank ごとに持つ。
  `TpRankModel::executor_state_reserve_bytes` が rank の
  `max_scheduled_requests` 分（prefix cache の shadow を含む）を予約する。

## 制約

- pipeline stage に対応する。stage ごとに担当層の indexer と QSA state pool を
  保持し、stage 間の hidden は `hidden_size` 幅で受け渡す。
- tensor parallel は **TP=2** まで。1 rank あたりの attention は local head
  （Qwen3.8-27B では q 12 / kv 2）で、QSA の block-sparse kernel 条件
  `q_heads == 6 * kv_heads` を満たす。`tp_size > 2` は `TpCoordinator` が拒否する。
- YaRN などの非 default rope profile は未実装。
- MTP drafter は未配線。`LoadedModel::has_mtp_block()` は QSA family では false を返す
  （QSA checkpoint の MTP block を load する経路がまだ無い）。TP rank も MTP を load しない。
- DFlash2 の chain / tree verify は `shelf::state::SpecStateHistory` で QSA state を
  巻き戻す（[dflash2.md](dflash2.md) の rolling state の rollback）。PLE を持つ target の
  tree verify は未対応。
- select kernel の上限は `indexer_head_dim == 128`、`indexer_heads <= 8`、
  `top_k <= 1024`。

## dispatch

`try_launch_qsa_select` の tiled 経路（score を block tile ごとに分割し state を持ち回る）は
staged request が **全 row を覆っている**ときだけ使う。padded 行には slice が無く、
stale な position で未 score の block を選んでしまうため。覆っていない場合は
fused な `launch_qsa_select_batch` にフォールバックする。

`QSA_SELECT` を含む program は decode の HIP graph 対象外にする。
`QsaTopkArgs::block_count` は capture 時の position で固定されるため、replay で
position が進むと `visible > block_count` になり、`qsa_topk_kernel` が
未 score の index を mask に書く。graph を無効にすると dispatch ごとに
現在の `sequence_length` から block count を作り直せる。
