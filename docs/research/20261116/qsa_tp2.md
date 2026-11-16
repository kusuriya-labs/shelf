# QSA sparse attention の TP=2

対象: `models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`（QSA indexer、block_size 4、
budget 2048、indexer 4 head × 128）。

## 1. 初期状態

`lower_qwen35qsa_to_primitives` は `backbone_options.tp->tp_size > 1` を
`unsupported: qwen35qsa: tensor parallel is not implemented` で拒否していた。
`probe_model_dir` は QSA を `ModelFamily::Qsa` と判定するが、
`select_tp_rank_model` は QSA を dense builder に落としていたため、
TP=2 では QSA indexer が使われず、dense attention として動いていた。

## 2. 接続

- `load_qwen35qsa_weights_tensor_parallel_rank` を追加した。
  backbone は `load_qwen35_weights_tensor_parallel_rank` の rank-local plan、
  indexer は TP plan に入れずに replicated で全 rank へ load する。
- `Qwen35QsaModel::load_tensor_parallel_from_safetensors` と
  `Qwen35QsaTpRankModel` を追加し、`select_tp_rank_model` の `Qsa` 分岐と
  `TpGroup::create` の `LoadedModel::borrow(qsa)` に接続した。
- `TpRankModel` に `uses_qsa_state()` を追加し、`qwen35_executor_reserve_bytes` の
  QSA select workspace と `executor_state_reserve_bytes` の QSA state pool を
  rank ごとに予約する。Qwen4Exp は同じ virtual を既存の
  `qwen4exp_state_pool_layouts` で返す。
- lowering は backbone を `Qwen35TensorParallelContext` の local geometry
  （q 12 / kv 2、local intermediate 8704、local GDN key/value heads 8/24）で emit する。
  GDN `out_proj` / attention `o_proj` / `mlp.down_proj` に `tp_combine` を付け、
  BF16 lm_head は dense と同じ vocab-parallel 条件で shard する。
- indexer は replicated のまま。hidden は TP combine 後に rank 間で一致するので、
  各 rank が同じ indexer 入力から同じ block mask を作る。mask の collective は不要。
- MTP block は TP rank では load しない（`unsupported`）。

## 3. 検証中に見つけた shape 欠落

TP=2 の GDN `in_proj_qkv` は segmented ranges の合計が local 幅になる。
Qwen3.8-27B（qk 2048 / value 6144 / TP=2）では

```text
{0,1024} + {2048,1024} + {4096,3072} = 5120
```

であり、local shape は `5120 x 5120` になる。最初 `7168` を allowlist に
入れていたため、48 GDN 層の qkv linear がすべて correctness path に落ち、
2203 token prefill が TP1 2.1 s / TP2 21.2 s になっていた。
`{5120, 5120, kFamilyMxfp4}` を追加すると miss は 0 になり、
同じ prefill は TP2 1.46 s になった。この shape は dense MXFP4 の TP=2 にも
共通である（[mxfp4_tp2_dense.md](mxfp4_tp2_dense.md)）。

## 4. 検証結果

モデルは `Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`、`--max-seq-len 4096`。
2203 token（BetterBench corpus、budget 2048 を超えるので pruning が動く）と
3500 token（corpus を連結）で TP=1 / TP=2 の greedy token を比較した。

| 条件 | TP=1 | TP=2 |
| --- | --- | --- |
| 2203 token, BF16 KV, prefill | 2.11 s | 1.46 s |
| 3500 token, BF16 KV | 23, 19 | 23, 19（一致） |
| 3500 token, Octave `c5v5` | 23, 19 | 23, 19（一致） |
| 3500 token, Octave + DFlash2 6 drafts | 23, 19, 13, 14748 | 23, 19, 13, 14748（一致） |
| 3500 token, Octave prefill | 2.97 s | 1.96 s |
| 13000 token, BF16 KV（sparse prefill 経路） | 15, 15 | 15, 15（一致） |
| 13000 token, BF16 prefill | 7.87 s | 5.07 s |

`MODEL_FAMILY=qsa`、`MODEL_INTERMEDIATE_SIZE=17408`、
`--max-seq-len 2000`（budget 以下）では indexer を emit しない dense fallback も
TP=1 / TP=2 で同じ token になった（1500 token、`7497`）。

`shelf-compute --tensor-parallel 2` の 3500 token prefill は
`SHELF_LINEAR_DEBUG=1` で `LINEAR_SELECT_MISS` が 0 であることを確認した。

`./run.sh`（prefix cache slot、`--max-seq-len 262144`、Octave KV、DFlash2 6 drafts）は
`SHELF_SERVE_READY` / `Ready.` まで起動した。QSA 有効時の rank の reserve は
executor footprint 0.98 GiB、prefix cache の family shadow 1.00 GiB だった。

## 5. 結論

ACCEPT。QSA は TP=2 で indexer / block mask / sparse attention を各 rank が実行し、
budget を超える文脈でも TP=1 と同じ greedy token を返す。
indexer は replicated なので state pool と select の計算は rank ごとに重複するが、
mask の collective を持たない。
