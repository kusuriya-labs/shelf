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
EMA に含めない。lossy 既定の損益分岐は benchmark で同時 12 と 16 の間、
日本語では同時 4 と 8 の間だった。上限 12 は benchmark の同時 8 / 12 の利得を
維持する選択だが、日本語・同時 8 の未達を解決しない。

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
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat benchmark 16 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat benchmark 32 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat japanese 2 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat japanese 16 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat shell 8 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat shell 16 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_adaptive_scripts/repeat_case.py /tmp/dflash2-policy-prompts /tmp/dflash2-policy-lossy-repeat shell 8 1
/tmp/venv-so/bin/python docs/research/20261111/dflash2_policy_scripts/tables.py /tmp/dflash2-policy-lossy-breakeven /tmp/dflash2-policy-lossy-comparison /tmp/dflash2-policy-lossy-repeat
```

## 損益分岐と比較

lossy 既定の再測定中。未完了の欄は未測定を示す。日本語・同時 8 の初回は
非投機 173.94、固定 126.67、ポリシー 126.83 tok/s で、最良値に対して
27.08% 未達だった。ポリシーの選択は K=7 が 544 回、K=0 が 4 回で、
受理 draft 数の平均 1.253 / request-round は停止閾値 0.6 を上回った。
損益分岐の 2 回測定でも日本語・同時 8 は非投機 173.09 / 173.62、固定
128.91 / 126.16 tok/s と非投機が優位だった。一方、同じ同時数の benchmark は
固定 295.70 / 295.79、非投機 173.71 / 173.60 tok/s であり、同時数上限だけでは
両方を選び分けられない。shell・同時 8 の初回も固定 232.38、ポリシー
222.66 tok/s（-4.18%）で未達だった。ポリシーは 320 回すべて K=7 を選び、
固定との生成列も一致したため、選択以外の overhead または変動が候補だった。
反復は固定 222.54、ポリシー 222.71 tok/s で、2 回平均では固定 227.46、
ポリシー 222.69 tok/s（-2.10%）となり基準内だったが、初回は未達である。
shell・同時 16 も境界に近かったため反復し、非投機 287.06 / 286.73、
ポリシー 278.59 / 280.15 tok/s、2 回平均では約 -2.62% だった。benchmark・同時 16 は
2 回平均で非投機 288.20、固定 265.10、ポリシー 279.68 tok/s、非投機比
-2.96% で形式上は基準内だが、3% 境界に近い。比較表は反復のある条件について
各方式の平均を表示する。同時 16 の固定は 274.79 / 255.41 tok/s と
変動しており、境界に近い判定は不確かである。日本語・同時 2 の固定は
比較で 68.97、損益分岐の初回で 74.95 tok/s と差があり、生成列は同一だった。
日本語・同時 16 の非投機も 288.20 / 279.36 tok/s と約 3.1% 変動したが、
生成列は同一だった。
日本語・同時 16 は非投機 286.40 / 285.89、ポリシー 278.05 / 272.90 tok/s で、
2 回平均では非投機 286.14、ポリシー 275.48 tok/s（-3.73%）の未達だった。同時 32 の 2 回平均は非投機
421.48、固定 339.31、ポリシー 405.39 tok/s で、非投機比 -3.82% の未達だった。

### 損益分岐（tok/s、各 2 回）

| prompt | 同時数 | 非投機 | 固定 K=7 |
| --- | ---: | --- | --- |
| benchmark | 2 | 53.32 / 52.57 | 118.92 / 119.03 |
| benchmark | 4 | 97.72 / 97.92 | 212.77 / 212.79 |
| benchmark | 8 | 173.71 / 173.60 | 295.70 / 295.79 |
| benchmark | 12 | 238.84 / 240.09 | 259.42 / 259.21 |
| benchmark | 16 | 287.92 / 288.23 | 275.17 / 274.00 |
| benchmark | 20 | 323.82 / 323.76 | 292.56 / 292.23 |
| benchmark | 24 | 359.94 / 359.45 | 304.11 / 304.71 |
| benchmark | 32 | 420.47 / 420.54 | 331.04 / 331.61 |
| japanese | 2 | 52.19 / 52.49 | 74.95 / 69.04 |
| japanese | 4 | 97.46 / 97.38 | 108.34 / 108.33 |
| japanese | 8 | 173.09 / 173.62 | 128.91 / 126.16 |
| japanese | 12 | 237.63 / 239.12 | 123.84 / 124.56 |
| japanese | 16 | 288.20 / 279.36 | 147.17 / 146.78 |
| japanese | 20 | 311.84 / 324.44 | 162.61 / 159.79 |
| japanese | 24 | 360.84 / 361.35 | 175.24 / 178.93 |
| japanese | 32 | 414.75 / 422.67 | 208.75 / 202.40 |

### 比較（tok/s、各 1 回）

| prompt | 同時数 | 非投機 | 固定 K=7 | ポリシー | 判定 | greedy 一致 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| benchmark | 1 | 27.64 | 46.06 | 48.49 | 通過 | 不一致 |
| benchmark | 2 | 52.33 | 118.87 | 119.25 | 通過 | 一致 |
| benchmark | 4 | 97.61 | 212.65 | 212.84 | 通過 | 不一致 |
| benchmark | 8 | 173.97 | 295.36 | 295.55 | 通過 | 不一致 |
| benchmark | 16 | 288.20 | 265.10 | 279.68 | 通過 | 不一致 |
| benchmark | 32 | 421.48 | 339.31 | 405.39 | 未達 | 不一致 |
| japanese | 1 | 27.60 | 36.78 | 36.67 | 通過 | 不一致 |
| japanese | 2 | 52.66 | 69.08 | 69.12 | 通過 | 不一致 |
| japanese | 4 | 97.60 | 108.78 | 108.41 | 通過 | 不一致 |
| japanese | 8 | 173.94 | 126.67 | 126.83 | 未達 | 不一致 |
| japanese | 16 | 286.14 | 144.19 | 275.48 | 未達 | 不一致 |
| japanese | 32 | 408.43 | 206.79 | 400.34 | 通過 | 不一致 |
| shell | 1 | 27.17 | 73.33 | 73.94 | 通過 | 不一致 |
| shell | 2 | 52.08 | 125.87 | 126.13 | 通過 | 不一致 |
| shell | 4 | 97.51 | 172.46 | 181.83 | 通過 | 不一致 |
| shell | 8 | 171.25 | 227.46 | 222.69 | 通過 | 不一致 |
| shell | 16 | 286.89 | 221.09 | 279.37 | 通過 | 不一致 |
| shell | 32 | 417.89 | 279.70 | 406.76 | 通過 | 不一致 |
| code | 1 | 27.34 | 76.29 | 76.22 | 通過 | 一致 |
| code | 2 | 51.42 | 129.78 | 129.76 | 通過 | 不一致 |
| code | 4 | 98.01 | 186.31 | 186.21 | 通過 | 不一致 |
| code | 8 | 172.02 | 236.34 | — | 未測定 | 未確認 |
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
