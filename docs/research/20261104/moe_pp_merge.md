# MoE 作業と pipeline parallel の統合（merge）

`feature/moe-weights`（MoE weight / executor 配線）と `main` へ入った
pipeline parallel ポート（`0928674`）の統合記録。確定した contract は
`docs/developer/pipeline_parallel.md` §4 と `docs/developer/qwen35moe.md` §8 に置き、
ここには経緯と採否判断だけ残す。

## 競合の実測

`git merge-tree` で見た競合は 2 ファイルだけだった。

| ファイル | 競合の中身 |
| --- | --- |
| `src/apps/compute/compute_runtime.h` | member と `model()` の返り型。PP 側は `const Qwen35Model&`（stage0）を返し、MoE 側は `const LoadedModel&` を返す |
| `src/shelf/models/qwen35/runtime/executor.hip` | PP 側は `create_executor` 内で env 解析＋lowering を行い、MoE 側はそれを `create_model_executor` へ移している。同じ領域に `lower_options.stage` が足されていた |

意味的に直さなければならなかったのは他に 4 箇所（git は auto-merge したが型が合わない）。

1. PP 側が足した stage 判定 2 箇所が `executor.model->text_config()` を読んでいた。
   `Executor::model` は `RuntimeModelView`（値）なので `executor.model.config->num_hidden_layers`
   にした
2. `make_qwen35_gdn_state_layout_stage` / `qwen35_stage_gdn_layers` /
   `qwen35_stage_attention_layers` / `qwen35_stage_layer_count` の引数を
   `Qwen35TextConfig` から `QwenHybridConfig` へ広げた（`Qwen35TextConfig` は
   `QwenHybridConfig` の派生なので dense 側の呼び出しはそのまま通る）。executor core が
   持つのは hybrid config なので、core から呼べる形にすることが必要だった
3. `qwen35_executor_reserve_bytes` は `d92c9e3` で `(hybrid config, ffn_intermediate_size, ...)`
   に変わっていた。PP の auto reserve 呼び出しを新署名へ寄せる（PP は dense なので
   `tc.ffn.intermediate_size` を渡す）
4. `lower_options.stage` は dense wrapper（`create_model_executor`）で下ろす

## 判断

### accessor は `LoadedModel` に統一し、PP は stage0 を借用する

`Qwen35ComputeRuntime::model()` の返り型を `const shelf::models::LoadedModel&` のままにし、
PP 経路では `LoadedModel::borrow(pp_coordinator_->stage(0).model())` を保持する
（`pp_model_`）。apps は family を選ばない、という `867c139` の形を PP でも保つためである。
stage0 の dense model へ借用で乗るだけなので、重みの所有権は coordinator のまま。
所有していない resource を指す member は先に離す規則に従い、shutdown 順序は
`pp_model_ = nullptr` → `pp_coordinator_.reset()` と決定的にした。

### MoE + pipeline parallel は起動時に `unsupported`

拒否は apps 層（`initialize_pipeline_parallel`）に置いた。理由は module 境界で、
family 判定の `probe_model_dir` は `shelf_models` にあり、coordinator は
`shelf_qwen35_runtime` にある。`shelf_qwen35_runtime` から `shelf_qwen35moe` へ link 辺を
足すと `test_architecture_boundaries` が source と link の両方でこれを拒否する
（qwen35 と qwen35moe を互いに参照させない）。両 family を link できるのは apps と
`shelf_models` だけなので、gate は apps に置くのが唯一の素直な形になった。

stage 分割自体も dense 前提である。weight load の stage 字段
（`HybridLoadOptions` / `Qwen35LoadOptions`）と executor footprint は dense の
`ffn.intermediate_size` から平面量を積み上げるもので、group 行数容量と plan scratch の
stage 別配分を持たない。MoE の expert 平面を stage でどう分けるかは未設計のまま
placeholder を作らない、という方針で拒否にした。

## 統合で新たに見つかった欠陥（main 側の変更由来）

`validate_qwen35_weights` はもともと共有 helper `qhy::validate_hybrid_model_level` を呼んでいた
が、`0928674` がそれを inline に置き換えており、model level の検査が **partition を読まない形**
（`head.rows != config.vocab_size`）になっていた。共有 helper は `full_rows()` で partition の
global 行数に張り戻すので、TP rank の weight が通らなかった。

```text
test_qwen35_tp_weight_load: FAIL rank0 / rank1 weights pass the model invariants (partition aware)
  invalid argument: Qwen35Model: lm_head geometry mismatch
```

`full_rows` を持つ共有 helper 側へ stage 範囲を渡す形に戻した（inline の複製をやめ、single
source にする）。範囲は `validate_hybrid_model_level(model, config, loaded_layer_lo,
loaded_layer_hi)` で受け、

- 範囲が layer 0 から始まる（または最終 stage で head が tied）→ embedding を要求
- 範囲が最終层で終わる → lm_head と final norm を要求
- 途中 stage → どちらも要求しない
- 行数は partition の global 値で照合（TP rank が通る）

呼び出し側は dense が `(lo, hi)`（stage 無効なら `(0, num_layers)`）、MoE は `(0,
layers.size())` に加えて `layers.size() == config.num_hidden_layers` を自分で行う（共有 helper
から layer 数検査を外したので、検査の置き場所を呼び出し側に明示した）。

`test_qwen35_pipeline_stage`（cpu;required）に model level の検査を追記した。途中 stage が
model level tensor を持たないことを ok とし、embedding を partition 付きで通すこと、partition を
消すと `embed_tokens vocab size mismatch` で落ちること、最終 stage で head が欠けると落ちること、
範囲が model を超える・空の範囲を拒否することを張る。

tamper 実測: 共有 helper の embed 検査を `model.embed_tokens.rows`（partition を読まない形）へ
戻すと `test_qwen35_pipeline_stage` が FAIL（`first stage: sharded embedding passes` と
`first stage: unpartitioned sharded embedding is rejected`）。戻して PASS。

## 統合で寄せる必要のあったその他

- `docs/user/compute.md` の CLI 表が main の手で書いた値と実 help に食い違った
  （`pipeline parallel` / `pipeline-parallel`）。`tools/gen_cli_docs.py` で再生成して同期した
- `shelf-compute --help` の見出しが "for Qwen 3.5 Dense" のままだったので、
  dense と grouped expert の両方を走らせる build の文言へ直した

## 既存機能の実走（統合前 / 統合後の A/B）

`shelf-compute --tp 2` の production 経路を、統合前 binary（`b0669ff` の build）と統合後で
同一条件（`models/Qwen3.8-27B-FP8`、prompt 8 token、`--max-new-tokens 8 --seed 42
--device 0,1`）で走らせた。

| build | GENERATED_IDS | DECODE_TOKENS_PER_SEC |
| --- | --- | ---: |
| 統合後（`merge/moe-pp`） | `5,74,44,12,14248,74,44,12` | 29.84 |
| 統合前（`feature/moe-weights`） | `5,74,44,12,14248,74,44,12` | 29.91 |

生成列は一致、速度差は run 間の揺らぎの範囲（0.2%）。apps・executor・共有 validation を同時に
変えた統合で、dense TP=2 の production 経路は同じ結果を出している。

`test_qwen35_tp_e2e` は既定 arena が TP1 leg で 24 GiB のため、`Qwen3.8-27B-FP8`（weight 約
27 GB）では alloc 失敗になった。

```text
FAIL tp1 run: tp1 model load: insufficient memory: quantized matrix codes allocation failed:
model.language_model.layers.54.linear_attn.out_proj.weight
```

これは test の既定値と現在の primary target のサイズが合っていないだけで、統合とは無関係
（load 段階の失敗で GPU 上の byte 数問題）。`SHELF_TP_ARENA_GIB=28` を与えて実行した結果:

```text
arena GiB tp1=28 tp2=28
TP_TOKPS tp1_decode=18.48
TP_TOKPS tp2_decode=29.31
PASS
```

TP=1 と TP=2 の greedy 列一致・logits argmax 一致・rank-local geometry・staging capture の
同一性まで通っている（test が張っている check 全体）。`tensor_parallel_execution.md` の test 表
に arena の要求を追記した。上の tok/s は test 内蔵の値で、`docs/perf/current.md` の計測口では
ないため、ここへのみ記録する。

MoE 側も統合前後で同じ条件を走らせた（`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`、prompt 6
token、`--max-new-tokens 8 --max-num-batched-tokens 64 --max-concurrent-requests 1 --seed 42`）。

| build | GENERATED_IDS | DECODE_TOKENS_PER_SEC | ARENA_USED_GB |
| --- | --- | ---: | ---: |
| 統合前（`b0669ff`） | `3966,51,271,248068,271,248069,271,760` | 8.05 | 30.223 |
| 統合後（`merge/moe-pp`） | `3966,51,271,248068,271,248069,271,760` | 8.02 | 30.223 |

生成列と arena が一致。統合で validation と executor の共有部を変えたが、dense TP=2 も MoE
40 层も同じ結果を出している。**これは生成列の一致であって、HF との数値照合ではない**（40 层
E2E の数値 gate は未実施のまま）。

## 実測

- merge 後の build（fresh configure）: 286 targets、error なし
- required acceptance: Passed 156  Failed 0  Errors 0  Skipped 0
  （統合前の `feature/moe-weights` は 154。main が足した `test_qwen35_pipeline_stage` と
  `test_qwen35_pipeline_lowering` で 156 になり、`test_qwen35_pipeline_stage` には model level
  の検査を追記した）
- MoE checkpoint へ `--pipeline-parallel 2 --device 0,1` を向けると load 前に拒否される
  （下の command で確認）

```sh
./build/shelf-compute --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 \
  --input-ids-file /tmp/opencode/ids_moe.txt \
  --device 0,1 --tp 1 --pipeline-parallel 2 \
  --max-num-batched-tokens 64 --max-concurrent-requests 1 --max-new-tokens 1
```

```text
runtime create failed: src/apps/compute/compute_runtime.hip:526: unsupported:
pipeline parallel is not implemented for grouped expert models
```

load（約 26 s）より前の起動段階で拒否される。

## 統合後に残る確認事項

- PP の E2E 採否（PP=1 と PP=2 の生成列一致）は `pipeline_parallel_port.md` の手順に従う。
  MoE 側の 40 层 E2E 数値 gate（HF 照合）とは独立で、どちらも未実施
- `merge/moe-pp` は `.worktrees/moe-pp-merge` に作成した。`feature/moe-weights` への反映は
  `docs/homework.md` の未 commit 編集を片付けてから fast-forward する
