# DFlash2 の lossy chunk verify と投機ポリシー

## 条件と方針

R9700（gfx1201）、Qwen3.8 27B EXL3 4bpw mul24、DFlash2 K=7、FP8 KV、
greedy、各 request 最大 160 token、prefix cache 無効。全 request が生成中の共通区間の
合計 token/s を用いる。損益分岐は benchmark と日本語を同時 2 / 4 / 8 / 12 / 16 /
20 / 24 / 32 で各 2 回測る。比較は benchmark / japanese / shell / code /
reasoning / mixed、同時 1 / 2 / 4 / 8 / 16 / 32 のうち mixed の同時 1 を除く
35 条件とし、各 1〜2 回測る。各実行は `/tmp/r9700.lock` を取得する。
依頼どおりの回数は通常の 5 回中央値ではなく、方式の実行順も固定である。

ユーザー判断により、serve の DFlash2 compact verify は単一 request を含み
lossy chunk を既定とする。serial は `SHELF_GDN_VERIFY_CHUNKED=0`、exact chunk は
`SHELF_GDN_VERIFY_CHUNKED_EXACT=1` で選ぶ。ユーザーから提示された固定 corpus
4 種の平均 mean KL は lossy 0.00103、exact 0.00228、top-1 一致率は 98.4〜99.8%。
これは今回の性能実行とは別の測定である。chunk は単一 request でも非投機 greedy
との生成列の完全一致を保証しない。単発 decoder は全受理時の state store と replay の
変更が必要であり、今回は serial のままである。指定された外れ値の追加調査は行わない。

適応ポリシーは既定 K=7、負荷上限 12（`SHELF_DFLASH2_POLICY_MAX_SESSIONS` で
上書き可能）、各 request の受理 draft 数 / round の EMA 初期値 2、係数 0.03、
停止閾値 0.6、停止中の再試行間隔 64 round。複数 request が同時に再試行期限に
達した場合は順番に試す。既定の DFlash2 セッションプールは 8 件なので、枠不足で
通常 decode 中の request も上限判定に含める。短い残り長による打ち切り round は
EMA に含めない。損益分岐と閾値は lossy 既定の測定で評価する。

途中まで実施した exact chunk・単一 serial 前提の測定は、方針変更により中断した。
旧実行では benchmark の損益分岐が同時 12 と 16 の間、日本語では 4 と 8 の間にあり、
単一の同時数上限だけでは両入力に最適化できなかった。受理閾値を 1.5 に上げた試作は
日本語・同時 8 で 142.13 → 102.44 tok/s に低下し、採用せず元の閾値に戻した。
これらは exact の旧条件であり、lossy の結果には使わない。

## 再現方法

`SHELF_MODEL_DIR`、`SHELF_DFLASH2_MODEL_DIR`、`SHELF_BENCH_BINARY`、
`LD_LIBRARY_PATH`、`ROCR_VISIBLE_DEVICES` を対象環境に設定する。
`SHELF_GDN_VERIFY_CHUNKED`、`SHELF_GDN_VERIFY_CHUNKED_EXACT`、
`SHELF_GDN_RECURRENCE_EXACT`、`SHELF_GDN_RECURRENCE_LOSSY` は未設定にし、
`/tmp/venv-so` は読み取りのみで使用する。入力を作った後、各 driver は実行単位で
GPU ロックを取得する。

```sh
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/prepare_prompts.py /tmp/dflash2-policy-prompts
/tmp/venv-so/bin/python docs/research/20261111/dflash2_policy_scripts/breakeven.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-breakeven
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/run_comparison.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-comparison
/tmp/venv-so/bin/python docs/research/20261111/dflash2_policy_scripts/tables.py /tmp/dflash2-policy-lossy-breakeven /tmp/dflash2-policy-lossy-comparison
```

## 損益分岐と比較

lossy 既定の再測定中。未完了の欄は未測定を示す。

### 損益分岐（tok/s、各 2 回）

| prompt | 同時数 | 非投機 | 固定 K=7 |
| --- | ---: | --- | --- |
| benchmark | 2 | 53.32 / 52.57 | 118.92 / 119.03 |
| benchmark | 4 | 97.72 / 97.92 | 212.77 / 212.79 |
| benchmark | 8 | 173.71 / 173.60 | 295.70 / 295.79 |
| benchmark | 12 | 238.84 / — | — / — |
| benchmark | 16 | — / — | — / — |
| benchmark | 20 | — / — | — / — |
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
| benchmark | 1 | 27.64 | 46.06 | 48.49 | 通過 | 不一致 |
| benchmark | 2 | 52.33 | 118.87 | 119.25 | 通過 | 一致 |
| benchmark | 4 | 97.61 | 212.65 | 212.84 | 通過 | 不一致 |
| benchmark | 8 | 173.97 | 295.36 | 295.55 | 通過 | 不一致 |
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

## テストとサーバー確認

関連 ctest 20 件はすべて通過し、exact override の専用実行も通過した。
server の DFlash2 関連 5 本も 99 / 26 / 7 / 8 / 21 項目が通過し、失敗はない。
非投機との完全一致を検証する次の 3 テストは、serial を明示して比較する。

- `tests/server/test_compute_dflash2.py`
- `tests/server/test_compute_dflash2_unaligned.py`
- `tests/server/test_compute_dflash2_prefix_long.py`

`test_compute_dflash2_octave.py` と `test_server_dflash2.py` は既定の chunk で動作を確認する。
`test_gdn_compact_log_commit` は serial 明示と、単一 request の既定 chunk 選択を検証する。
実サーバーでは英語・日本語・コード・shell が混在する 8 件すべてが完了し、
合計 1280 completion token、エラーなし。全員生成中 200.8 chunk/s、
全体の completion token / 壁時計は 138.23 token/s だった。通知 chunk/s と
stdio の token/s は測定単位が異なる。全回帰 `--all` は実行していない。
