# Executor の keepalive thread が multi-row batch の per-dispatch latency を悪化させていた

実験 ID `EXEC-KEEPALIVE`、結論 **ACCEPT（executor の keepalive thread を削除）**。

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）、device 0 / TP=1。
動機は [./moe_mtp.md](./moe_mtp.md) §6 で残った未切り分けの課題、つまり MoE MTP の
verify round が forward 単体より約 40 ms 高いという実測の局所化である。

## 1. 着手時点の状態

- MoE MTP（drafts=1、2 行 verify）の verify round は 62.2 ms。同じ model の 2 行 forward 単体は
  `shelf-bench pp --prefill-chunk 2` で 22.5 ms、1 行 decode step は 16.2 ms。
- `--max-num-batched-tokens 16 / 64 / 256 / 2048` で verify 時間に差が無い（§6）。
  「row bucket ぶん elementwise を walk している」仮説は否定済み。
- 40 ms が round 側の費用（descriptor・staging・sampling・state save）なのか、
  multi-row batch 自体の遅さなのかが未切り分けだった。
- `Executor` の keepalive thread はこの branch 以前（`main` の `a1eb24c` 時点）から既定 on で、
  `SHELF_KEEPALIVE=0` で止まるだけの形だった。したがってこれまでに測られた MoE の multi-row
  数値はすべて keepalive が効いた状態で測られていたことになる

## 2. round 内の費用を hip event で切る

`mtp_drafter.cpp` の round に hip event を入れて分解した（一時的な計装、本記録の測定後に取り除いた）。

| 内訳 | 値 |
| --- | ---: |
| GDN snapshot | 0.00 ms |
| verify sample の readback | 0.12 ms |
| `run_target(... SPEC_VERIFY ...)` の host 時間 | 62.4 ms |
| 同じ区間の hip event span（GPU 時間） | 62.4 ms |

つまり費用は `run_target` の内側で、host 時間と GPU 時間が一致している。
round 周辺の host 処理（snapshot・readback・accept 判定）は 0.2 ms 未満だった。

## 3. kernel trace で pass 単位に分解する

`SHELF_OP_DUMP` は 1 dispatch ごとに fprintf するため、1355 dispatch / pass の MoE では
host 費用が支配的になって使えない。`rocprofv3 --kernel-trace`（grid / workgroup / scratch /
VGPR 付き）で pass を `prepare_batch_descriptor_kernel` の境界から割り、
span（最初から最後の kernel）、busy（kernel 時間合計）、idle（空隙）を出した。

| pass | kernels / pass | span | busy | idle | busy / dispatch | idle / dispatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tg 1 行 decode（`shelf-bench tg`） | 1366 | 20.9 ms | 14.3 ms | 6.6 ms | 10.5 µs | 4.8 µs |
| pp 2 行 prefill（`shelf-bench pp --prefill-chunk 2`） | 1364 | 26.7 ms | 19.5 ms | 7.1 ms | 14.3 µs | 5.2 µs |
| MoE MTP verify 2 行（`shelf-compute --mtp`） | 1369 | **65.8 ms** | 32.6 ms | **33.2 ms** | 23.8 µs | **24.3 µs** |

同じ 2 行・同じ kernel・同じ launch shape なのに、`shelf-compute --mtp` だけ
idle / dispatch が 24 µs（他は 5 µs 程度）だった。同じプロセス内の 1600 token prefill pass
ですら idle / dispatch は 38 µs あり、`shelf-bench tg` の prefill pass（同条件）の 12 µs と
合わなかった。
**pass の中身ではなくプロセスの設定の問題**だということがここで分かった。

## 4. 候補を潰す

timing のみを目的にした実験（生成列は壊れる可能性があることを承知で、時間だけを比較した）。

| 変更 | verify round |
| --- | ---: |
| 何もしない（`--mtp-drafts 3`、4 行） | 63.9 ms |
| `ExecutionRole` を Verify → Decode に固定 | 67.3–70.6 ms（悪化） |
| `gdn_spec_history` を渡さない（history capture 無し） | 63.8 ms（変化なし） |
| `--max-num-batched-tokens 16 / 64 / 256 / 2048` | 変化なし（§6） |
| **`SHELF_KEEPALIVE=0`** | **19.8 ms** |

`Executor` は `create_executor` の末尾で keepalive thread を起動していた。
`SHELF_KEEPALIVE` が未設定のとき**有効**で、専用 stream へ 1 ms 間隔で `hipEventRecord` を投げる
（CP を醒ましたままにする目的、`hipEventDisableSystemFence`）。`shelf-bench` 側の
`--gap-keepalive` は既定 off の別物で、executor 側は常に on だった。

## 5. keepalive の費用を実測で押さえる

record 間隔と回数を変えて verify round を測った（同一条件、`--mtp-drafts 3`、4 行、3 round）。

| keepalive 設定 | verify round（gpu event span） |
| --- | ---: |
| 既定（1 ms 間隔） | 62.3–63.9 ms |
| 10 ms 間隔 | 26.5–44.5 ms |
| 100 ms 間隔 | 19.9–21.9 ms |
| thread は生存、record は 1 回だけ | 18.7–19.8 ms |
| 無効（`SHELF_KEEPALIVE=0`） | 18.8–20.0 ms |

thread が生きているだけで record をほぼしないケースが無効時と同じなので、
**費用は thread 自体ではなく `hipEventRecord` の回数に比例**する。1 回あたり数十 µs の
GPU 側の損失になる。

kernel trace で同じ pass を keepalive on / off で比べると、kernel・grid・workgroup・scratch・
VGPR は同一のまま、小さな kernelの時間だけが変わった（2 verify pass の平均）。

| kernel | keepalive on | keepalive off | 倍率 |
| --- | ---: | ---: | ---: |
| `gemm_bf16_exact_rows_kernel`（n=391） | 12899 µs | 8572 µs | 1.50 |
| `activation_quantize_mxfp4_rowref_kernel`（n=80） | 2600 µs | 1368 µs | 1.90 |
| `gemm_mxfp4_w4a8_grouped_kernel`（n=80） | 2376 µs | 2007 µs | 1.18 |
| `shelf_swiglu_bf16`（n=80） | 1959 µs | 763 µs | 2.57 |
| `shelf_rmsnorm_bf16_pf_oneplus`（n=82） | 1308 µs | 245 µs | 5.34 |
| `shelf_residual_add_bf16`（n=80） | 1270 µs | 158 µs | 8.04 |
| `shelf_l2_normalize_bf16_f32`（n=60） | 945 µs | 135 µs | 7.00 |
| `shelf_sigmoid_bf16_f32`（n=50） | 810 µs | 107 µs | 7.57 |
| `moe_combine_kernel`（n=40） | 652 µs | 208 µs | 3.13 |
| `moe_slot_scatter_kernel`（n=40） | 634 µs | 132 µs | 4.80 |
| `shelf_attention_paged_bf16`（n=10） | 933 µs | 927 µs | 1.01 |
| `shelf_gdn_recurrence_wmma_verify_lossy`（n=30） | 702 µs | 685 µs | 1.02 |

小さな kernel に固定値（約 13 µs）が載る形であり、50 µs 級以上の kernel は影響を受けていない。
clock 変化ではなく per-dispatch の固定費用である。pass 集計では busy / dispatch が
23.8 → 12.4 µs、idle / dispatch が 24.3 → 4.3 µs で、1 行 decode や pp chunk 2 と同じ帯に戻った。

| pass | keepalive on | keepalive off |
| --- | ---: | ---: |
| MoE MTP verify 2 行 span | 65.8 ms | **22.8 ms** |
| MoE MTP prefill 1600 token idle / dispatch | 38.1 µs | 10.2 µs |
| MoE MTP prefill busy / dispatch | 283.9 µs | 280.5 µs |
| MoE `tg --context 1600 --tokens 32`（2 run） | 61.16 / 61.30 t/s | 60.57 / 60.49 t/s |

1 行 decode の dense な連続実行では on / off の差は測定誤差以内だった。
差が出るのは dispatch 数が多く 1 kernel が短い batch（multi-row forward・verify）と
prefill の空隙である。

## 6. dense 27B 側への影響（keepalive を消して良いか）

MoE だけの結論で既定を変えるのは危険なので、`models/Qwen3.8-27B-Quark-AWQ-MXFP4`（dense）で
同じ A/B をした。実行順が結果に効くので、on / off を交互に並べたペアで測った。

| 条件 | keepalive on | keepalive off |
| --- | ---: | ---: |
| pp prompt 8192 chunk 2048（交互 5 ペア、runs 3 warmup 1 の median） | 3519.9 / 3565.1 / 3569.7 / 3575.9 / 3576.6 ms | 3626.8 / 3632.9 / 3728.9 / 3741.4 / 3774.1 ms |
| 同上を別セッションで交互 2 ペア（runs 2 warmup 1） | 3560.1 / 3571.0 ms | 3560.5 / 3575.2 ms |
| pp prompt 32768 chunk 2048（交互 2 ペア） | 19143.6 / 19220.2 ms（1711.69 / 1704.87 tok/s） | 19218.3 / 19237.0 ms（1705.04 / 1703.39 tok/s） |
| MoE pp prompt 8192 chunk 2048（交互 3 run） | 2158.5 / 2175.6 / 2176.8 ms | 2168.8 / 2170.4 / 2171.3 ms |
| dense tg context 1024 / 2048（tokens 32） | 25.97 / 26.27 t/s | 26.02 / 26.25 t/s |
| dense MTP drafts=7 prompt 1024（8 round） | verify 431.4 / 755.6 / 754.1 ms、81.56 / 57.08 / 57.18 t/s | verify 349.6 / 350.8 / 350.2 ms、91.88 / 92.89 / 92.69 t/s |

- dense prefill の差は**再現しなかった**。交互 5 ペアでは on が 1.4〜5.9 % 速かったが、
  同じ条件（prompt 8192・chunk 2048・runs 2・交互）の別セッションでは 0.01 % / 0.12 %、
  prompt 32768 でも 0.4 % だった。run 数の違いや heuristics では説明できず、
  **small effect は GPU 共有下で信用できない**ということなので、keepalive の benefit としては
  採用しない（harm 側の再現性は高い。下表・下 §6-1 参照）
- MoE prefill・dense / MoE の 1 行 decode は差が無い（測定誤差以内）
- dense MTP は keepalive off が 1.6 倍速く、しかも run 間の振れが消える
  （on は 57.18 / 57.08 / 81.56 と振れ、off は 91.88 / 92.89 / 92.69）

### 6-1. 費用は thread 数と record 数に比例する

executor は 1 台につき 1 つ thread を持つため、MTP run は target と drafter で **2 本**になる。
thread を device 単位で共有する（refcount）一時パッチで測ると、MoE MTP verify round は
27.2 ms（1 本）まで下がり、無効時の 20.9 ms に近づいた（on 2 本では 64.3 ms）。
同様に record 間隔を空けると費用は比例して消えた（MoE MTP drafts=4 の verify 合計と t/s）。

| record 間隔 | verify 合計 | t/s |
| --- | ---: | ---: |
| 1 ms（既定） | 581.42 ms | 45.39 |
| 20 ms | 300.99 ms | 78.99 |
| 100 ms | 199.84 ms | 109.25 |
| 無効 | 188.08 ms | 112.74 |

つまり原因は thread の存在ではなく **別 stream への signal 投入回数**で、1 回あたり
dispatch 1 本ぶん（約 13 µs）に相当する費用が乗る。共有化や間隔の緩和は部分的にしか効かず、
`hipEventRecord` を使う限り費用は残る。driver 側の clock / PM 設定で代替できるかは、
`power_dpm_force_performance_level` が root 専用で確認できなかった（未解決）。

## 7. 採否判断

- **ACCEPT（削除）**: executor の keepalive thread を**削除**した。`KeepAlive` /
  `keepalive_start` / `keepalive_stop` と `SHELF_KEEPALIVE` を `Executor` から取り除き、
  executor は追加の host thread と追加 stream を持たない契約にした。
  - 費用は再現する（MoE MTP verify round 62.2 → 19.2 ms、dense MTP verify round 94.5 → 43.9 ms、
    kernel trace の busy / idle per dispatch は §5 と §8 の表）
  - 利点は再現しなかった（§6。dense prefill で 1.4〜5.9 % 出たセッションもあるが、同一条件の
    別セッションで 0.01 % / 0.12 %、prompt 32768 で 0.4 %）
  - `hipEventRecord` を使う限り費用は残るので、thread 共有や間隔の緩和は採用しない（§6-1）
- driver / firmware 側の clock・PM 設定（`power_dpm_force_performance_level` など）で
  代替できる可能性は、benefit 自体が再現しなくなったため追わない。設定は root 専用で
  本環境では検証できない
- idle 復帰を測る口は `shelf-bench` の `--gap-keepalive` が app 側に opt-in で残っている。
  executor 側に同種の activity を戻さないこと
- **削除による生成列・数値の変化は無い**（同じ kernel を同じ引数で launch するだけ）。
  MoE target-only の `GENERATED_IDS`、MoE `tg` の `GREEDY_TOKEN_SUM`（context 64 で 976467、
  context 2048 で 975006）、dense `tg` の `GREEDY_TOKEN_SUM`（context 1024 で 153624、
  context 2048 で 184095）、dense MTP の `GENERATED_IDS` sha1 は keepalive on / off で一致した

## 8. MoE MTP への影響

keepalive 無効化後の MoE MTP（同一条件: 1600 token prompt、`--max-new-tokens 32`、
`--max-seq-len 4096`、bf16 KV、device 0、TP=1）。数値の正本は
[../../perf/current.md](../../perf/current.md) の MoE MTP 節に置く。

| 条件 | t/s | verify ms / round | draft ms / round | round ms / round |
| --- | ---: | ---: | ---: | ---: |
| target-only | 61.56 / 61.44 | - | - | - |
| `--mtp-drafts 1`（2 行 verify） | 84.80 | 19.2 | 2.6 | 21.8 |
| `--mtp-drafts 4`（平均 4.8 行 verify） | **115.83** | 20.6 | 9.1 | 29.7 |
| `--mtp-drafts 7`（7 行 verify） | 93.37 | 22.5 | 14.4 | 36.9 |

verify 時間 / round は 62.2 / 64.3 / 66.8 ms から 19.2 / 20.6 / 22.5 ms になった（3.0〜3.3 倍）。
[./moe_mtp.md](./moe_mtp.md) §6 の「round の 62 ms のうち forward が約 22 ms、残り約 40 ms は
forward 以外の費用」という内訳は**誤り**で、実際は forward 単体と同じ費用まで multi-row batch が
削られていなかった。keepalive を止めると verify は 2 行 forward 単体の既知の費用
（`pp --prefill-chunk 2` の 22.5 ms）と同じになり、行数に対して緩やかに増えるようになった。
生成列（`GENERATED_IDS`）は keepalive の on / off で変化しない。

MoE の MTP 有効化（性能主張）は [./moe_mtp.md](./moe_mtp.md) §5 / §6 の REJECT から
**ACCEPT に訂正**する。受理率は 35 % のままですが break-even（drafts=4 で accepted ≥ 0.82）を
超えるためである。break-even の計算を §6 に訂正して残した。

## 8-1. 削除後 build での確認

keepalive thread を削除した build で、同じ条件を再測した（device 0 / TP=1、1600 token prompt、
`--max-new-tokens 32`、`--max-seq-len 4096`）。

| 条件 | t/s | verify ms / round |
| --- | ---: | ---: |
| target-only | 61.59 / 60.87 | - |
| `--mtp-drafts 1` | 83.83 | 19.4 |
| `--mtp-drafts 4` | 113.18 | 20.8 |
| `--mtp-drafts 7` | 90.93 | 22.6 |

opt-in 化して無効にした build（上表）と同じ帯で、差異は GPU 共有下の run 間振れの範囲内
（1.3 〜 2.6 %）。`GENERATED_IDS` は target-only と一致したままだ。
`shelf-bench tg` の MoE anchor も保持された（context 64 で 499.63 / 500.15 ms、
`GREEDY_TOKEN_SUM=976467`、context 2048 で 512.11 / 512.04 ms、`GREEDY_TOKEN_SUM=975006`）、
`shelf-bench pp --prompt-tokens 8192 --prefill-chunk 2048` は 3809.14 / 3762.39 tok/s。

dense 27B 側も `GENERATED_IDS` の sha1 が記録した anchor と一致した
（1024 `6c07c6fe…`、2048 `08e9e2c9…`。target-only と MTP で同一）。

## 9. 再現条件

- build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo` +
  `ninja -C build shelf-compute shelf-bench`
- trace: `/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace
  --output-format csv --output-file X -d DIR -- <cmd>`
  pass 分割は `prepare_batch_descriptor_kernel`（1 batch に 1 回）を使った。
  kernel 時間合計（busy）と span の差が idle 空隙で、`Grid_Size_*` / `Workgroup_Size_X` /
  `Scratch_Size` / `VGPR_Count` が同じかを必ず確認する（launch shape が変わっていないことを
  示さないと「kernel が遅い」を主張できない）。
- on / off の切替は削除前 build の `SHELF_KEEPALIVE=0`（既定 on）で行った。削除後の build には
  on 側が存在しないので、上記の on 数値を再現するには thread を戻す必要がある
