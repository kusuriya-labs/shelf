# Executor の keepalive thread が multi-row batch の per-dispatch latency を悪化させていた

実験 ID `EXEC-KEEPALIVE`、結論 **ACCEPT（keepalive thread を既定は無効に変更）**。

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

## 6. 採否判断

- **ACCEPT**: executor の keepalive thread を既定は無効にした。`SHELF_KEEPALIVE=1` を
  明示的に設定したときだけ有効になる（`keepalive_requested()`）。
  既定時の計測可能なオーバーヘッドが大きい（MoE MTP verify round で 3.2 倍）ため。
  idle 復帰を測る口は `shelf-bench` の `--gap-keepalive` が opt-in で残っている。
- **無効化による生成列・数値の変化は無い**（同じ kernel を同じ引数で launch するだけ）。
  MoE target-only の `GENERATED_IDS`、`tg --context 1600` の `GREEDY_TOKEN_SUM=1005760` は
  on / off で一致した。
- keepalive が CP の idle 復帰に与える効果は本 build では実測していない（benefit の証拠が無い
  ことと、harm が実測できたことで判断は確定できる）。戻るなら tg の
  `--gap-keepalive` と同一の条件で、GPU idle を意図的に作って比較する。

## 7. MoE MTP への影響

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

## 8. 再現条件

- build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo` +
  `ninja -C build shelf-compute shelf-bench`
- trace: `/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace
  --output-format csv --output-file X -d DIR -- <cmd>`
  pass 分割は `prepare_batch_descriptor_kernel`（1 batch に 1 回）を使った。
  kernel 時間合計（busy）と span の差が idle 空隙で、`Grid_Size_*` / `Workgroup_Size_X` /
  `Scratch_Size` / `VGPR_Count` が同じかを必ず確認する（launch shape が変わっていないことを
  示さないと「kernel が遅い」を主張できない）。
