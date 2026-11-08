# qwen4exp の MTP と継続バッチ verify（2026-11-08）

実験 ID: `Q4EXP-MTP-1`

対象: `models/Qwen3.8-Flash-Next-exl3`（qwen4exp、EXL3）+ 同 checkpoint の MTP block。

結論:

- **ACCEPT**: MTP drafter を qwen4exp へ移植した。MTP block は rank0 だけが load し、
  drafter の入力は final norm ではなく pre-mixer の wide residual（`hc * hidden`）にする。
- **ACCEPT**: speculative verify は target の `PleConvStatePool` と `QsaKeyStatePool` の
  partial 状態を row 単位の history から復元する。これがないと partial accept 後の
  greedy 列が target-only と一致しない。
- **ACCEPT（限定）**: QSA / PLE を持つ target では verify を request ごとに実行する。
  融合 verify（複数 request を 1 forward）は dense / MoE の 27B で一致するが、
  qwen4exp では admission のタイミングによって片方の request が分岐する。
- **REJECT**: qwen4exp の融合 verify を現状のまま qualified とする。

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

## 3. 融合 verify の限界

2 request を同時に流すと、次の 2 経路で結果が変わる。

- 両 request を同じ最初の round に入れる: 両方一致。
- 1 request が solo round を 1 回実行した後にもう 1 つが加わる: 先着 request が分岐
  （同じ入力でも約 50 %、分岐する token 列は常に同じ）。

融合 verify の前後で QSA の pooled / partial checksum、`pool_count` / `partial_count`、
PLE の window を採取したところ、**分岐する run と一致する run で state は同一**だった。
融合 verify の forward 自体が経路依存で変わるため、共有 history の row 対応だけでは
説明できない。admission のタイミングを固定すると再現性が変わるので、実行順に依存する
要因が残っている。

このため QSA / PLE を持つ target では `MtpSpecBatchContext::solo_verify` を立て、
verify を request ごとに実行する。dense / MoE の 27B は融合 verify のままで、
2 request 同時の greedy 列が target-only と一致することを確認している。

## 4. 再現

```bash
# 直列（target-only と一致）
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-exl3 \
  --max-new-tokens 8 --max-seq-len 4096 --device 2,3 --tensor-parallel 2 \
  --prefix-cache 0 --temperature 0 --mtp --mtp-drafts 4

# 2 request 同時（serve-stdio、request ごとに verify）
#   request 11: [248041,77091]  request 22: [248041,77091,279,2614]
```

## 5. 残り

- 融合 verify が経路依存で変わる原因の特定（executor / TP の batch 実行）。
- 特定後に `solo_verify` を外し、qwen4exp でも 1 forward verify にする。
