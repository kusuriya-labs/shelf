# QSA decode の qsa_topk メモリフォルト

実験 ID: `QSA-TOPK-1`

`models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4` を `--max-seq-len 4096 --max-new-tokens 128`
で decode すると、`qsa_topk_kernel` が
`HSA_STATUS_ERROR_MEMORY_FAULT` で落ちる。変更前の commit でも同じ条件で再現した。

## 1. 再現

```sh
./build/shelf-compute \
  --model-dir models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4 \
  --device 2 --tensor-parallel 1 --max-seq-len 4096 \
  --kv-cache-dtype bf16 --input-ids-file /tmp/opencode/prompts/en_db.txt \
  --max-new-tokens 128 --temperature 0 --prefix-cache 0
```

`--max-seq-len 512`（budget 2048 以下の dense fallback）では落ちない。
`--max-new-tokens 1` でも落ちず、decode が数 step 進むと落ちる。

## 2. 原因

QSA select の tiled 経路は、1 回の decode step で
`QsaTopkArgs::block_count` に現在の sequence_length から求めた block 数を入れ、
score と top-k を block tile ごとに実行する。

decode の HIP graph は、最初の decode step の `QsaTopkArgs` を値として capture する。
graph replay では `positions` だけが device で進み、`block_count` は capture 時の
値（短い context の 3 など）に固定される。その結果、kernel 内の

```text
visible = (position + 1) / compress_ratio
usable  = min(visible - block_begin, block_count)
```

で `visible > block_count` になり、`select_once` が false になる。この経路は
state の first half を `-1` で初期化してから新しい `usable` 個だけを radix select
するが、`block_max = min(visible, top_k)` は `visible` のままなので、
merge 後の `-1` を mask の block として `atomicOr` してしまう。

```text
block = indices[i] = -1
words[block / 32u]  // 0xFFFFFFFF / 32 -> mask 外
```

## 3. 修正

`QSA_SELECT` を含む program を decode graph の対象外にした。

- `src/shelf/models/qwen35/runtime/executor.hip` の `graph_eligible` に
  `!program_needs_qsa_workspace(*program)` を追加。
- graph を使わない dispatch では、毎 step の `sequence_length` から
  `block_count` を作り直すため、`visible <= block_count` が保たれる。
- QSA が dense fallback のときは `QSA_SELECT` が program に無いので、
  graph の有無は変わらない。

QSA の選択は 16 layer × decode で launch 数が多くないため、graph を外す影響は
限定的である。QSA 自体は現状 dense より遅く、速度より正しさを優先した。

## 4. 検証

- `--max-seq-len 4096 --max-new-tokens 128` が完走する。
- `DECODE_TOKENS_PER_SEC=25.96`（R9700 1 枚、TP1、bf16 KV）。
- `--max-new-tokens 1` と `--max-seq-len 512` も従来どおり完走する。
- 生成 ID は graph 無効化後の baseline と一致した
  （同じ checkpoint、同じ prompt、同じ 128 token）。
- `test_qsa_select`、`test_qsa_state_pool`、`test_qsa_sparse_attention`、
  `test_qsa_key_path`、`test_qsa_spec_state_replay` を ctest で確認する。

## 5. 残課題

- graph を維持したまま QSA select を動かすには、block count を host の
  `sequence_length` ではなく device の position から計算し、score/topk の
  tile 数も device scalar で決める必要がある。現状は未実装。
- QSA の長 context decode は dense より遅い。graph 無効化のコストを含めた
  再計測は `docs/perf/current.md` の対象外である。
