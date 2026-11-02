# EAGLE-4 drafter（speculative decode）

EAGLE-4 の current contract である。
chain（`--eagle4-tree-width 0|1`）と tree（`--eagle4-tree-width 2..16`）の両方を含む。
経緯は Git history と `docs/research/20261023/` の Gate 記録を参照する。

- drafter: `models/Qwen3.8-27B-EAGLE4`（BF16、1 層、sliding window 4096）
- 対象: `shelf-compute` の単一 request 経路（`--eagle4-model-dir`）
- 実装: `include/shelf/models/qwen35/eagle4/`、
  `src/shelf/models/qwen35/eagle4/`、
  `src/shelf/models/qwen35/kernels/eagle4/`

DFlash2 と同じ `DFlash2SpecDecoder`（`src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp`）
を使い、drafter の呼び出しだけを差し替える。target verify / accept / GDN state の契約は
[DFlash2](dflash2.md) と同一である。

---

## config

`read_eagle4_config()` が `config.json` から読む key は次のとおり。

| key | 意味 |
| --- | --- |
| `architectures[0]` | `LlamaForCausalLMEagle4` のみ許可 |
| `vocab_size` / `hidden_size` / `intermediate_size` / `num_hidden_layers` | 形状（`num_hidden_layers` は 1 のみ） |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | draft attention の形状。MHA のみ許可 |
| `target_hidden_size` | target hidden の次元 |
| `sliding_window` | draft attention の窓幅（ring capacity と一致させる） |
| `eagle_config.eagle_aux_hidden_state_layer_ids` | tap する target 層（昇順、3 本） |
| `efficient_lm_head_config.type` / `.rank` / `.num_candidates` | candidate head（`low_rank` のみ許可） |
| `fc_norm` / `norm_output` | input 側の per-tap norm / 出力 norm の有無 |
| `rms_norm_eps` / `rope_parameters.rope_theta` | RMSNorm eps と RoPE theta（`rope_type == "default"` のみ） |
| `tie_word_embeddings` | **false のみ許可** |

不変条件（違反は load 時 error）:

- `num_hidden_layers == 1`
- `num_attention_heads == num_key_value_heads`
- `num_attention_heads * head_dim == hidden_size`
- `head_dim == 128`（`eagle4` kernel の前提）
- `sliding_window ∈ [1, 4096]`
- `aux_layer_ids` は狭義単調増加
- `candidate_rank > 0`、`0 < num_candidates <= vocab_size`

`fc_norm` / `norm_output` は config から読み、runtime が分岐する。27B checkpoint は両方 true。

---

## weight

`eagle4_expected_tensors()` が唯一の tensor 定義である。19 tensor、BF16 のみ。

| tensor | shape |
| --- | --- |
| `fc.weight` | `[hidden, 3 * target_hidden]` |
| `fc_norm.{0,1,2}.weight` | `[target_hidden]` |
| `norm.weight` | `[hidden]` |
| `embed_tokens.weight` | `[vocab, hidden]` |
| `lm_head.weight` | `[vocab, hidden]` |
| `efficient_lm_head.weight` | `[vocab, rank]` |
| `efficient_lm_head_down_proj.weight` | `[rank, hidden]` |
| `layers.0.input_layernorm.weight` / `hidden_norm.weight` / `post_attention_layernorm.weight` | `[hidden]` |
| `layers.0.self_attn.{q,k,v}_proj.weight` | `[hidden, 2 * hidden]` |
| `layers.0.self_attn.o_proj.weight` | `[hidden, hidden]` |
| `layers.0.mlp.{gate,up}_proj.weight` | `[intermediate, hidden]` |
| `layers.0.mlp.down_proj.weight` | `[hidden, intermediate]` |

missing / unexpected / shape 不一致 / dtype 不一致はすべて error。error message に tensor 名を含める。

`embed_tokens` は常に target から借りる（`load_embed_tokens = false`）。27B の checkpoint では
embed table が target と byte 一致するため、共有しても数値は変わらず 2.4 GiB を節約できる。
借りた tensor は所有も複製もしない。shape と BF16 であることを create 時に検証する。

`lm_head` は `--eagle4-lm-head` で選ぶ。

| 値 | 意味 |
| --- | --- |
| `draft` | checkpoint の `lm_head` を load する（candidate head の再評価と同一の head。既定の意味論） |
| `target`（既定） | target の `lm_head` を借りる（VRAM を 2.4 GiB 節約する） |

`target` を選ぶと、候補集合を作る `efficient_lm_head` の factor（draft の head を近似）と、
候補を再評価する head（target の head）が一致しない。draft の受理数はこの設定に依存する。
`--tp 2` では target の lm_head が vocab 分割されるため `target` は shape 検証で拒否され、
`draft` が必須である。単 GPU（`--tp 1`）では 32 GiB の target と同居すると `draft` の
2.4 GiB が memory plan に収まらないため既定を `target` としている。

drafter の backbone（fc / layer / candidate factors）は BF16 のみを受け付ける。

---

## draft context state

`Eagle4ContextState` は draft 専用の KV ring である。

| buffer | shape | dtype |
| --- | --- | --- |
| `k_ring` | `[1, capacity, kv_features]` | BF16 |
| `v_ring` | `[1, capacity, kv_features]` | BF16 |

- `capacity == config.sliding_window`（27B は 4096）、`kv_features = num_key_value_heads * head_dim`。
- physical slot は `absolute_position % capacity`。27B の実 allocation は
  `2 * 4096 * 5120 * 2 = 83,886,080 bytes`（80 MiB / sequence）。
- ring に入るのは drafter が forward した row だけである。commit-only で rollback は無い。
- `next_position` は drafter が次に forward する absolute position。
  draft step は `next_position` を進めない（commit されていない row のため）。

ring の slot `p` が表す値は「position `p` の hidden と position `p + 1` の token」の組から
作った K/V である。したがって row `p` の候補は position `p + 2` の token になる。

---

## drafter execution

### extend

`eagle4_extend_rows()` は 1 回の forward で次を行う。

1. `fc_norm.{0,1,2}` で 3 本の target tap をそれぞれ RMSNorm し、feature 軸へ連結する
   （`launch_eagle4_tap_norm_concat_bf16`）。
2. `fc` で `[rows, 3 * target_hidden] -> [rows, hidden]`。
3. token embedding を引き、`cat(input_layernorm(embed), hidden_norm(fc_out))` を作る。
4. q/k/v projection、RoPE、ring append、ring attention、o projection、residual、
   post attention norm、MLP（SwiGLU）、residual、final norm。
5. 最終 row の hidden を root hidden に、最終 row の normed を candidate head に渡し、
   `num_root_candidates` 本の root token を書く。

入力は target tap（`rows` 行）と token id（`rows` 行）である。呼び出し側は

- 各 row `p` の token に「position `p + 1` の token」を渡す
- position は `position_start + row`

を守る。

### draft step

`eagle4_draft_steps()` は `num_drafts - 1` 回の 1-row forward を行う。

- step `i` の position は `first_position + i`。
- step 0 の token は `drafts[0]`（extend が書いた root token）、hidden は root hidden。
- step `i` の hidden は step `i-1` の出力（`norm_output` が true なら final norm 後）。
- step `i` の候補 token を `drafts[i + 1]` に書く。

したがって `drafts[0]` は position `position_start + rows + 1`、`drafts[i]` は
`position_start + rows + 1 + i` の候補である。verify は `[pending, drafts[0..K-1]]` を
1 回の target forward で検証する。

### candidate head

`eagle4` の candidate head は 2 経路を持つ。`--eagle4-candidate-head` で選ぶ。

```
u       = normed · P1                         P1 = efficient_lm_head_down_proj [rank, hidden]
ŝ       = u · P2                              P2 = efficient_lm_head [vocab, rank]
C       = top_{num_candidates}(ŝ)             (降順、同値は token id 昇順)
logit_c = w_c · normed                        w_c = lm_head の候補 row（f32 積）
```

- `lowrank`（既定）: `launch_eagle4_rank_project_bf16` → `launch_eagle4_rough_scores_bf16`
  → `launch_dflash2_radix_topn`（pool = `num_candidates`）→ `launch_eagle4_candidate_logits`
  → `launch_eagle4_candidate_topk`。
- `gemv`: `lm_head` の full GEMV と argmax（`k == 1` のみ）。比較・検証用である。
  logits は BF16 GEMM の出力をそのまま使うため、`lowrank` の f32 再評価とは丸めが異なる
  （差は約 0.4%）。2 経路の一致は「同じ候補を同じ値で再評価できること」で検証する。

`u` と `ŝ` は BF16 に丸める（torch の BF16 matmul と同じ丸め位置）。候補の再評価だけを f32 で行う。
top-k の ordering は logit 降順、同値は token id 昇順である。

### attention

`launch_eagle4_ring_attention_bf16()` は 1 workgroup = 1 (query row, head) で ring を読む。

```
visible(p, q) = q <= p  ∧  p - q < sliding_window
```

ring は append 済みの row だけを読む。workgroup は 128 thread、head_dim = 128 固定。
出力は BF16、softmax は f32。

### fallback GEMM

drafter の linear は `dflash2_linear`（optimized BF16 GEMM selector）を使う。selector が
geometry を拒否した場合（test 用の小さい形状など）だけ
`launch_eagle4_gemm_fallback_bf16`（1 thread = 1 出力特徴の単純な f32 積和）へ落ちる。
通常の 27B draft 形状はすべて selector の allowlist にあるため fallback は使われない。

### primitive dispatch

extend / draft step は各 logical primitive を独立した kernel launch として実行する。
対象は tap norm + concat、embedding、layer input norm、q/k/v projection、RoPE、ring append、
attention、o projection、residual、post attention norm、gate/up、SwiGLU、down、residual、
final norm、candidate head である。融合は行わない。

### hot path

`eagle4_extend_rows` / `eagle4_draft_steps` は非同期 enqueue API である。
`hipStreamSynchronize` / D2H memcpy / allocation を含まない。draft の token はすべて device 上で
決まり、host へ戻らない。host への同期は verify と accept の decision だけで起きる。

---

## decoder integration

`DFlash2SpecDecoder` に次の field が加わる。

| field | 意味 |
| --- | --- |
| `eagle` / `eagle_context` | EAGLE-4 drafter と draft ring（非所有） |
| `eagle_draft_tokens_device` | 1 round の draft token 列（提案出力） |
| `eagle_extend_tokens_device` | commit した row の token（extend 入力） |
| `eagle_num_candidates` / `eagle_candidate_topk` | candidate head の形状 |
| `eagle_root_ready` | root hidden と `drafts[0]` が ready か |

`create_eagle4_spec_decoder()` が decoder を作り、`dflash2_spec_decoder_shutdown()` が
field を落とす。drafter と ring の所有は呼び出し側にある。

### prefill

`dflash2_spec_prefill()` は EAGLE-4 のとき次を行う。

- prompt を target の chunk 単位で prefill し、chunk ごとに drafter の extend を行う。
- extend は `kEagle4ExtendRowsMax`(256) 行ずつに分割して実行される
  （1 回の extend の最大行数のため）。
- 非最終 chunk の extend rows は `n`、token は `prompt[offset + 1 .. offset + n]`。
- 最終 chunk の extend rows は `n - 1`、続いて pending token 1 row を position
  `prompt_count - 1` で extend する。これが最初の round の root になる。
- prefix cache の restore では draft ring（KV / `next_position`）も保存・復元される
  （`prefix_cache_spec_ring_bytes`）。

### round

`dflash2_spec_step()` の EAGLE-4 経路。

1. proposal: `eagle4_draft_steps()` で `drafts[0..K-1]` を生成する。
   `heagle_root_ready` が false なら error（silent fallback しない）。
2. verify: `[pending, drafts...]` の `K + 1` 行を target が 1 回で検証する（DFlash2 と同一）。
3. accept: `spec_greedy_accept()`（DFlash2 と同一）。
4. commit: `[pending, drafts...]` の先頭 `accepted + 1` token と、その row の target tap で
   `eagle4_extend_rows()` を呼び、draft ring を更新して次の root を作る。
   `K == 0`（`run_single_target`）の経路も 1 row の extend を行う。

verify の行数は `K + 1`、`eagle_candidate_topk` は 1 である。draft の proposal は
1 round に 1 回、commit の extend も 1 round に 1 回である。

---

## CLI

`shelf-compute` の option。

| option | 意味 |
| --- | --- |
| `--eagle4-model-dir PATH` | EAGLE-4 draft model directory（指定で EAGLE-4 経路） |
| `--eagle4-drafts N` | 1 round の draft 数（既定 7、上限 15） |
| `--eagle4-candidate-head lowrank\|gemv` | candidate head の経路（既定 lowrank） |
| `--eagle4-lm-head draft\|target` | candidate head の再評価に使う lm_head（既定 target） |
| `--eagle4-stats 0\|1` | EAGLE-4 統計行の出力（既定 1） |

無効な組み合わせは model load 前に exit 2 とする。

- `--dflash2-model-dir` との同時指定
- `--dump-logits`
- `--prefix-cache 1` は `--serve-stdio` なしでは不可
- `temperature != 0`（greedy のみ）

`shelf-launcher` は EAGLE-4 の option を child に透過する。
`shelf-server` も `--eagle4-model-dir` / `--eagle4-drafts` / `--eagle4-lm-head` を
`shelf-compute` へ転送する。

`draft_bytes`（memory plan の draft weights）は checkpoint の実ファイルサイズから
共有する embed / lm_head の分を除いた値である。

---

## tensor parallel

`--tp 2` で動作する。serve 経路（`--serve-stdio`）は drafter も target と同じ TP 形状で
**rank ごとに 1 基** 置く。`shelf-compute` の direct 経路は drafter を rank0 に 1 基だけ
置き、target のみ TP2 で実行する。

- draft 重みは `build_eagle4_tensor_partition_plan` で分割する。q/k/v/gate/up は出力行、
  o/down は入力列、`embed_tokens` / `lm_head` / `efficient_lm_head` は vocab 行を分割する。
  `fc`、`fc_norm.*`、`norm.weight`、layer norm は replicated。
- rank ごとに `Eagle4Executor` / `Eagle4ContextState` / stream / device を持つ
  （`Eagle4RankDraft`）。executor と context は **その rank の device scope 内**で生成する。
  `Eagle4Executor::device` は生成時の `hipGetDevice()` であり、rank の device と一致しなければ
  ならない。
- TP の契約は「各 rank の EAGLE4 kernel には、その rank が所有する device pointer だけを
  渡す」である。rank をまたぐ device pointer を kernel 引数にしない。
  - 入力 token と tree metadata（`tokens`, `row_nodes`, `src_rows`,
    `ancestor_offsets`, `ancestor_nodes`）は host 配列で job に渡す。各 rank が
    `Eagle4Executor` の staging buffer へ自分の stream で H2D する。
  - hidden tap は各 rank の target `Executor::dflash_target_hidden` から取る。
    連続行は `tap_row_offset` で選び、tree commit のように行が飛ぶ場合は
    `launch_eagle4_gather_rows_bf16` で rank ごとに staging へ gather する。
- embedding は rank ごとの local vocab を担当する。`vocab_offset` で global token を
  local 行へ写像し、担当外の token は zero を書く（`partial_vocab`）。直後に
  `TpSumTarget` で全 hidden を sum する。
- attention は head 並列（q/k/v は出力行、o は入力列）、MLP は intermediate 並列
  （gate/up は出力行、down は入力列）。o_proj と down_proj の後で sum する。
- candidate head は local vocab 上で候補を選び、rank0 が両 rank の候補 logits を merge して
  global top-k を決める。rank1 の候補 id には rank0 の local vocab を足す。候補 pool は
  rank ごとの local top-k の和になるため、TP2 の候補集合は TP1 と一致するとは限らない
  （pool は TP2 の方が広い）。
- draft job は `TpBatchHook::on_eagle4_draft(job)` で rank ごとの worker に渡す。worker は
  自分の device を current にした状態で `run_eagle4_rank_job(job, rank)` を実行する。
- verify は `TpBatchHook::on_execute(batch, per_rank)` に同じ batch を渡して両 rank で
  実行する。`ExecuteBatchOptions` は rank ごとの GDN spec history / compact を持つ。
- GDN の snapshot / restore / commit は `DFlash2GdnRankBinding` のリスト（rank 順）を
  遍歴して全 rank 対象で行う。
- sequence は rank0 に置き、`on_sequence_created` / `on_sequence_released` で
  rank1 の mirror を管理する。mirror の適用は次の `on_execute` 前の sync で行われる。
- target lm_head が vocab 分割される構成では `--eagle4-lm-head draft` が必須である
  （`target` は shape 検証で拒否される）。

---

## 現状の制約

- greedy のみ（`temperature != 0` は model load 前に拒否）。
- direct 経路（`run_eagle4_shot()`）は単発生成のみで、`--prefix-cache` は併用できない
  （`--serve-stdio` と併用したときのみ有効）。
- `--serve-stdio` の serve 経路は対応する（`--tp 2` と併用可、prefix cache 併用可）。
- `--model-host` は対応する。target は resident attach、drafter は compute process が
  local load して保持する。

---

## tests

- `test_eagle4_config`: config の受理・拒否。
- `test_eagle4_weight_contract`: tensor 集合・shape・dtype の契約、共有 tensor の skip。
- `test_eagle4_kernels`: tap norm concat、ring attention（causal + window）、rank project、
  rough scores、candidate 選択、candidate 再評価・top-k を CPU 参照と比較する。
- `test_eagle4_forward`: 小さな checkpoint を生成し、extend（tap → fc → 1 層 → ring）と
  draft step を CPU 参照と比較する（hidden は max_abs/rms、token は完全一致）。
- `test_eagle4_tp_forward`（2 GPU）: 小さな checkpoint を TP1 と TP2 で実行し、
  embedding partial + sum / attention row-parallel / MLP row-parallel / final normed を
  stage ごとに比較して最初に発散した stage を報告する。候補 id は「両 rank の候補 logits を
  merge した結果」と比較し、TP2 の最良候補が TP1 より悪化しないことを確認する。
- `test_eagle4_candidate_paths`（`SHELF_EAGLE4_MODEL_DIR` 必須）: 実 checkpoint で
  `lowrank` の候補再評価が full-vocab GEMV の logits と一致すること、および argmax の一致数を
  報告する。
