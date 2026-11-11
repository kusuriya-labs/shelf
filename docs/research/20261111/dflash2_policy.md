# DFlash2 batch の単純な投機ポリシー

## 条件

R9700（gfx1201）、Qwen3.8 27B EXL3 4bpw mul24、DFlash2 K=7、FP8 KV、
greedy、各 request 最大 160 token、prefix cache 無効。全 request が生成中の共通区間の
合計 token/s を用いる。損益分岐は benchmark と日本語を各 2 回測る。比較は前回の
6 種の入力（mixed の同時 1 は混在できないため対象外）で行う。
各実行は `/tmp/r9700.lock` を取得する。回数は依頼どおり 1〜2 回であり、
通常の 5 回中央値ではない。順序は非投機、固定、ポリシーの固定順で、
反復間で逆順にしていないため、clock 変動の影響が残る。日本語・同時 2 の
非投機は 52.92 と 49.09 tok/s で 7% 以上変動し、3% 境界の判定は特に不確かである。単一 request は serial verify、複数の
DFlash2 request は exact chunk verify を使う。exact は bit 一致を意味しない。

## ポリシー

K は 7、batch あたりの負荷上限は暫定 12（`SHELF_DFLASH2_POLICY_MAX_SESSIONS`
で変更可）、受理 draft 数 / round の EMA 初期値 2、係数 0.03、停止閾値は 0.6。
既定の DFlash2 セッションプールは 8 件なので、投機候補だけでは 12 を超えない。
したがって枠不足で通常 decode に入った request も負荷に数える。これは多人数時に
投機と通常 decode が混在する実測を基にした保守的な判定である。
停止中は 64 round ごとに 1 回の試行を行い、同時に複数 request が試行期限に
到達した場合は順番に試す。短い生成残量で draft 数を減らした round は EMA に
含めない。これらの閾値は本測定の結果により評価する。

## 損益分岐と 35 条件の比較

測定中。未完了の欄は未測定を示す。暫定閾値 1.2 / 係数 0.15 では
benchmark 同時 1 が 30.68 tok/s（固定 45.25）、同時 2 が 97.79 tok/s
（固定 118.46）と未達だった。閾値と係数を下げ、同時 1 は 45.28、同時 2 は
118.36 に改善した。単一 request の serial round 約 58.9 ms と通常 decode 約
36 ms から、単独の損益分岐は約 0.6 accepted/round と見積もった。閾値と
係数の設定には他の入力での追加評価が必要である。

同時 16 の初期測定は、投機候補だけを数えたため 8 セッションの上限に達せず、
ポリシー 274.85 と非投機 288.33 tok/s で未達だった。セッション枠外の通常
request を数える修正後は 279.55 tok/s（非投機比 -3.05%）だった。
反復は非投機 288.38、固定 273.13、ポリシー 279.50 tok/s で、
ポリシーは非投機比 -3.08% と未達が再現した。同時 32 は非投機 423.20、ポリシー
407.71 tok/s（-3.66%）で未達。再開に必要な target tap の保持などが負荷に
関与する可能性はあるが、内訳は未測定。

再現時は `/tmp/venv-so` を変更せず、環境変数 `SHELF_MODEL_DIR`、
`SHELF_DFLASH2_MODEL_DIR`、`SHELF_BENCH_BINARY`、`LD_LIBRARY_PATH`、
`ROCR_VISIBLE_DEVICES` を指定する。以下の各 driver は実行単位で GPU ロックを
取得する。損益分岐の driver は `SHELF_GDN_VERIFY_CHUNKED=1` と
`SHELF_GDN_VERIFY_CHUNKED_EXACT=1` を明示する。比較 driver は本 branch の
既定動作を使用する。比較対象が 1 件だけの場合は serial を使用する。

```sh
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/prepare_prompts.py /tmp/dflash2-policy-prompts
/tmp/venv-so/bin/python docs/research/20261111/dflash2_policy_scripts/breakeven.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-breakeven
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/run_comparison.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-comparison
/tmp/venv-so/bin/python docs/research/20261111/dflash2_policy_scripts/tables.py /tmp/dflash2-policy-breakeven /tmp/dflash2-policy-comparison
```

### 損益分岐（tok/s、各 2 回）

| prompt | 同時数 | 非投機 | 固定 K=7 |
| --- | ---: | --- | --- |
| benchmark | 2 | 53.34 / 53.16 | 118.56 / 118.42 |
| benchmark | 4 | 98.72 / 98.28 | 222.56 / 222.49 |
| benchmark | 8 | 175.35 / 174.14 | 295.06 / 294.85 |
| benchmark | 12 | 239.39 / 240.34 | 259.32 / 258.19 |
| benchmark | 16 | 288.27 / 287.72 | 273.35 / 274.65 |
| benchmark | 20 | 327.22 / 325.47 | 291.42 / 294.12 |
| benchmark | 24 | 362.66 / 365.45 | 303.62 / 303.35 |
| benchmark | 32 | 426.79 / 423.06 | 334.60 / 333.59 |
| japanese | 2 | — / — | — / — |
| japanese | 4 | — / — | — / — |
| japanese | 8 | — / — | — / — |
| japanese | 12 | — / — | — / — |
| japanese | 16 | — / — | — / — |
| japanese | 20 | — / — | — / — |
| japanese | 24 | — / — | — / — |
| japanese | 32 | — / — | — / — |

### 比較（tok/s、各 1 回）

| prompt | 同時数 | 非投機 | 固定 K=7 | ポリシー | 判定 | greedy 一致 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| benchmark | 1 | 27.25 | 45.25 | 45.28 | 通過 | 一致 |
| benchmark | 2 | 52.23 | 118.46 | 118.36 | 通過 | 不一致 |
| benchmark | 4 | 97.73 | 223.51 | 223.18 | 通過 | 不一致 |
| benchmark | 8 | 173.84 | 295.97 | 292.02 | 通過 | 不一致 |
| benchmark | 16 | 288.33 | 274.93 | 279.55 | 未達 | 不一致 |
| benchmark | 32 | 423.20 | 333.10 | 407.71 | 未達 | 不一致 |
| japanese | 1 | 27.60 | 33.25 | 33.19 | 通過 | 一致 |
| japanese | 2 | 52.49 | — | — | 未測定 | 未確認 |
| japanese | 4 | — | — | — | 未測定 | 未確認 |
| japanese | 8 | — | — | — | 未測定 | 未確認 |
| japanese | 16 | — | — | — | 未測定 | 未確認 |
| japanese | 32 | — | — | — | 未測定 | 未確認 |
| shell | 1 | — | — | — | 未測定 | 未確認 |
| shell | 2 | — | — | — | 未測定 | 未確認 |
| shell | 4 | — | — | — | 未測定 | 未確認 |
| shell | 8 | — | — | — | 未測定 | 未確認 |
| shell | 16 | — | — | — | 未測定 | 未確認 |
| shell | 32 | — | — | — | 未測定 | 未確認 |
| code | 1 | — | — | — | 未測定 | 未確認 |
| code | 2 | — | — | — | 未測定 | 未確認 |
| code | 4 | — | — | — | 未測定 | 未確認 |
| code | 8 | — | — | — | 未測定 | 未確認 |
| code | 16 | — | — | — | 未測定 | 未確認 |
| code | 32 | — | — | — | 未測定 | 未確認 |
| reasoning | 1 | — | — | — | 未測定 | 未確認 |
| reasoning | 2 | — | — | — | 未測定 | 未確認 |
| reasoning | 4 | — | — | — | 未測定 | 未確認 |
| reasoning | 8 | — | — | — | 未測定 | 未確認 |
| reasoning | 16 | — | — | — | 未測定 | 未確認 |
| reasoning | 32 | — | — | — | 未測定 | 未確認 |
| mixed | 2 | — | — | — | 未測定 | 未確認 |
| mixed | 4 | — | — | — | 未測定 | 未確認 |
| mixed | 8 | — | — | — | 未測定 | 未確認 |
| mixed | 16 | — | — | — | 未測定 | 未確認 |
| mixed | 32 | — | — | — | 未測定 | 未確認 |

## 関連テストとサーバー確認

指定どおり configure と Release ビルドを実施。関連 ctest 19 件は通過し、
最後の batch 上限修正後も CPU の `test_dflash2_adaptive` が通過した。
`tests/server/` の DFlash2 関連 5 本は 99 / 26 / 7 / 8 / 21 項目が通過し、
失敗はなかった。`--all` は実行していない。

`shelf-server` で日本語・英語・コード・shell の混在 8 件を同時送信し、
8 件すべて完了、エラーなし、合計 1280 completion token を確認した。
全員生成中の通知は 209.3 chunk/s、全体の completion token / 壁時計は
137.63 token/s。chunk 数と token 数はこの実行でともに 1280 だったが、
通知単位による値と stdio の token/s を同一視しない。
