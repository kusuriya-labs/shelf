# `shelf-bench pp` / `batch` を `LoadedModel` へ寄せる（MoE が load できない）

対象: `src/apps/bench/pp.hip`、`src/apps/bench/batch.hip`。目的は MoE（grouped expert
checkpoint）の prefill / continuous batching を計測口から実行できるようにこと。
変更の経緯であり、現在の contract は `docs/developer/qwen35moe.md` に書く。

## 状況

`tg`（decode）と `quality` は `shelf::models::LoadedModel::load` を使うようになっていたが、
`pp`（prefill bench）と `batch`（continuous batching bench）は
`shelf::qwen35::Qwen35Model::load_from_safetensors` を直接呼んでいた。`Qwen35Model` は dense の
weight set なので、grouped expert checkpoint を渡すと load 段階で失敗する。つまり
**MoE の prefill を測る口が無かった**（`shelf-compute` は通るが、prefill throughput を
chunk で掃引する計測口ではなかった）。

同じ理由で executor も `shelf::qwen35::create_model_executor(model, ...)` を呼んでおり、
`LoadedModel::create_executor` が選ぶ family 別の executor 生成口を通っていなかった。

## 変更

- `pp.hip` / `batch.hip` の load を `LoadedModel::load(model_dir, arena, stream)` へ、
  executor 生成を `loaded.create_executor(seq_pool, gdn_pool, kv_pool, arena, exec_config)` へ寄せ、
  幾何参照を `text_config()` から `hybrid_config()` へ変えた。両ファイルの
  `count_attention_layers` は `Qwen35TextConfig`（dense の config、`QwenHybridConfig` の derived）
  を受けていたが、`hybrid_config()` が返す基底型を受け取るようにした
  （`count_layer_topology` は基底だけを見る）
- `pp.hip` は vocab 検査を `model.weights().embed_tokens.rows` でやっていた。weight を
  family 別に開かなくても決まる値なので `hybrid_config().vocab_size` へ替えた
- TP 経路（`pp --tensor-parallel 2`）は `TpCoordinator` が dense のみのため、`quality` と同じ
  起動時拒否を入れた: `probe_model_dir` が Dense でなければ
  `pp --tensor-parallel 2 runs dense checkpoints only` を返す。TP + MoE が深い地点で
  落ちるのを、load 前の判定にした

`shelf-server` は `shelf-compute --serve-stdio` を起動する側で、model を自分で load しない。
`compute_runtime.hip` は `LoadedModel::load` を既に使うので、server 側に配線は無い。

## 実測（device 1）

`pp --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --prompt-tokens 8192 --runs 3 --warmup 1`
（`--prefill-chunk` の掃引。数値の意味づけは
`moe_group_row_elementwise_envelope.md` の grid.z 分割の節）:

| `--prefill-chunk` | pp tok/s |
| ---: | ---: |
| 1024 | 3648.15 |
| 2048 | 3759.43 / 3769.14 |
| 4096 | 3683.91 |
| 8192 | 3569.17 |

dense の対照として `pp --model-dir models/Qwen3.8-27B-FP8 --prompt-tokens 8192` は
`--prefill-chunk 4096 / 8192` とも `insufficient memory: arena capacity exceeded:
capacity 30.78 GiB used 28.10 GiB requested 3.45 GiB` になった（27B の weight + KV で arena が
足りない。MoE の prefill 計測には影響しないが、dense の 8192 token prefill をこの device で
測るには `--arena-gib` を増やす必要がある）。

## 受け入れ

- 既存の required acceptance で `pp` / `batch` はビルド対象としてカバーされる。実行時の
  受理（MoE を load できること）は E2E の `pp` 実行で確認した（上の表がその実行そのもの）
- TP + MoE の起動時拒否は `probe_model_dir` の条件で、`quality` と同じ形。既存の
  `test_qwen35moe_executor` 等は dense fixture を使うのでこの経路を覆わない
