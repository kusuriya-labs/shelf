# qwen4exp の MTP と継続バッチ verify（2026-11-08）

実験 ID: `Q4EXP-MTP-1`

対象: `models/Qwen3.8-Flash-Next-exl3`（qwen4exp、EXL3）+ 同 checkpoint の MTP block。

結論:

- **ACCEPT**: MTP drafter を qwen4exp へ移植した。MTP block は rank0 だけが load し、
  drafter の入力は final norm ではなく pre-mixer の wide residual（`hc * hidden`）にする。
- **ACCEPT**: speculative verify は target の `PleConvStatePool` と `QsaKeyStatePool` の
  partial 状態を row 単位の history から復元する。これがないと partial accept 後の
  greedy 列が target-only と一致しない。
- **ACCEPT**: 融合 verify の分岐原因は EXL3 split-K の分割数が `rows` 依存だったこと。
  verify の行容量で分割数を固定すると、同居する request 数に関係なく同じ生成列になる。
- **REJECT**: 分割数を実行行数から選ぶこと（verify の数値が round の構成に依存する）。

## 1. 直列 MTP の移植

qwen4exp の MTP block は `load_mtp` で rank0 のみ load し、
`lower_qwen4exp_mtp_to_primitives` で 1 layer の graph にする。`MtpDrafterInput` は

- `hidden_width = hc * hidden`（wide residual）
- `wide_hidden = true`
- `token_row_intermediate = shared_expert_intermediate_size`
- `grouped_executor_bytes = mtp_grouped_executor_bytes(...)`

を渡す。drafter の external hidden input / normed hidden output は wide 幅で作る。

この時点で 2 token prompt、8 token 生成の greedy 列は

```text
target-only : 60,271,550,220,97036,128158,271,247252
MTP         : 60,271,550,4558,10586,220,247074,95905
```

となり、3 token 目で分岐した。partial accept は起きており、GDN の restore / compact replay
だけでは足りない。

## 2. PLE conv と QSA partial の復元

`PleConvStatePool` は 9 要素の running window（dilation 3）を持ち、verify の全 row で
進む。`QsaKeyStatePool` は `compress_ratio` ごとの partial と pooled / partial count を
持つ。どちらも partial accept で巻き戻す必要がある。

- `PleConvArgs` に pre-row の window を row 単位で保存する `state_history_store` を追加。
- `QsaKeyCacheBatchArgs` に partial と 2 つの count を row 単位で保存する
  `partial_history` / `count_history` を追加。
- `ExecuteBatchOptions` に `ple_conv_history` / `qsa_spec_history` を追加し、
  executor が kernel へ渡す。
- accept の partial 分岐で `restore_ple_conv_spec_state` / `restore_qsa_spec_state` を
  `committed` 行に対して呼ぶ。

これで直列 MTP の greedy 列は target-only と一致した（8 token、`MTP_ACCEPTED_DRAFTS=4`）。

## 3. 融合 verify の分岐（原因と修正）

2 request を同時に流すと、次の 2 経路で結果が変わる。

- 両 request を同じ最初の round に入れる: 両方一致。
- 1 request が solo round を 1 回実行した後にもう 1 つが加わる: 先着 request が分岐
  （同じ入力でも約 50 %、分岐する token 列は常に同じ）。

融合 verify の前後で QSA の pooled / partial checksum、`pool_count` / `partial_count`、
PLE の window を採取したところ、**分岐する run と一致する run で state は同一**だった。
状態の巻き戻しは正しく、forward の数値だけが変わっていた。

分岐する row の logits を採取すると、pos=4 で top1 220（16.2129）と top2 4558（16.0078）、
差 0.205 の near-tie だった。同じ request / 同じ位置で、verify の行数だけを変えると

| verify rows | top1 | top2 | 差 |
| ---: | ---: | ---: | ---: |
| 5 | 16.2129 | 16.0078 | 0.205 |
| 10 | 16.6415 | 16.6329 | 0.009 |

となり、10 行では 4558 が勝つ。`SHELF_EXL3_SPLIT=1` で分割数を固定すると 5 行と 10 行の
logits が完全一致したため、差は **EXL3 split-K の分割数が `rows` から選ばれている**こと
による K 縮約順の違いである。

`make_launch_plan` は `rows == 1` / `rows <= 8` / それ以外で分割数を変える。verify は
同居する request 数で `rows` が変わるため、同じ request が別の分割数で走っていた。

修正は `Exl3GemmArgs::splitk_tuning_rows` を追加し、`ExecutorConfig::
max_scheduled_output_rows`（verify の行容量）で分割数と scratch 見積を計算する。
実行行数ではなく容量で決まるので、1 request でも 2 request でも同じ分割数になる。
1 request の serve では容量 = `drafts + 1` なので従来と同じ分割数であり、
decode（`rows == 1`）と prefill は変更していない。

## 4. 再現

```bash
# 直列（target-only と一致）
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-exl3 \
  --max-new-tokens 8 --max-seq-len 4096 --device 2,3 --tensor-parallel 2 \
  --prefix-cache 0 --temperature 0 --mtp --mtp-drafts 4

# 2 request 同時（serve-stdio、request ごとに verify）
#   request 11: [248041,77091]  request 22: [248041,77091,279,2614]
```

## 5. 検証

- 直列 MTP の greedy 列が target-only と一致（両 model）。
- qwen4exp 2 request 同時（同一 prompt）を 4 回、dense 27B 2 request を 1 回、
  融合 verify のまま target-only と一致。
- `ctest -L required` 191 件 pass。
