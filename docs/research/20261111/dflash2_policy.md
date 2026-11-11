# DFlash2 batch の単純な投機ポリシー

## 条件

R9700（gfx1201）、Qwen3.8 27B EXL3 4bpw mul24、DFlash2 K=7、FP8 KV、
greedy、各 request 最大 160 token、prefix cache 無効。全 request が生成中の共通区間の
合計 token/s を用いる。損益分岐は benchmark と日本語を各 2 回測る。比較は前回の
6 種の入力（mixed の同時 1 は混在できないため対象外）で行う。
各実行は `/tmp/r9700.lock` を取得する。単一 request は serial verify、複数の
DFlash2 request は exact chunk verify を使う。exact は bit 一致を意味しない。

## ポリシー

K は 7、batch あたりの投機候補上限は暫定 12（`SHELF_DFLASH2_POLICY_MAX_SESSIONS`
で変更可）、受理 draft 数 / round の EMA 初期値 2、係数 0.03、停止閾値は 0.6。
停止中は 64 round ごとに 1 回の試行を行い、同時に複数 request が試行期限に
到達した場合は順番に試す。短い生成残量で draft 数を減らした round は EMA に
含めない。これらの閾値は本測定の結果により評価する。

## 損益分岐と 35 条件の比較

測定中。未完了の欄は未測定を示す。

### 損益分岐（tok/s、各 2 回）

| prompt | 同時数 | 非投機 | 固定 K=7 |
| --- | ---: | --- | --- |
| benchmark | 2 | 53.34 / 53.16 | 118.56 / 118.42 |
| benchmark | 4 | 98.72 / 98.28 | 222.56 / 222.49 |
| benchmark | 8 | 175.35 / 174.14 | 295.06 / 294.85 |
| benchmark | 12 | 239.39 / 240.34 | 259.32 / 258.19 |
| benchmark | 16 | 288.27 / 287.72 | 273.35 / 274.65 |
| benchmark | 20 | 327.22 / — | — / — |
| benchmark | 24 | — / — | — / — |
| benchmark | 32 | — / — | — / — |
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
| benchmark | 16 | — | — | — | 未測定 | 未確認 |
| benchmark | 32 | — | — | — | 未測定 | 未確認 |
| japanese | 1 | — | — | — | 未測定 | 未確認 |
| japanese | 2 | — | — | — | 未測定 | 未確認 |
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

