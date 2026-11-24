# TP / EP 実行での HIP graph（segment capture）

## 目的

`docs/developer/multi-gpu.md` §10 は「TP schedule と HIP graph の併用」を非目標としている。
本調査はその前提を再検討し、TP / EP の decode で graph を有効にできるかを実測する。

## 現行構造の制約

- `Executor::enqueue_batch` の graph 判定 (`src/shelf/runtime/executor.hip`) は
  `executor.tp_schedules.empty()` を要求する。`tp_schedules` は
  `tp_size > 1 || ep_enabled` で必ず構築されるため、TP / EP では graph に入らない。
- `execute_program_tp_ranges` は schedule の segment ごとに
  `execute_program_range` → `TpBarrierGroup::arrive` を呼ぶ。`arrive` は
  mutex / condition_variable で全 rank を待ち、`HipPeerTpTransport::enqueue_sum` が
  他 rank の stream に kernel を投入する。この CPU 同期は capture 中に実行できない。

## 実装した方式（segment graph）

capture の単位を program 全体ではなく schedule の segment にした。
`execute_program_tp_ranges` の segment loop で、compute 区間
(`producer_begin..run_end`) を per-segment の `hipGraphExec_t` に capture し、
barrier (`arrive`) は従来どおり CPU 側に残す。

- cache は `Executor::tp_segment_graphs`（program index × segment index）。
- capture は staging pool が warm な run の次回以降。`ctx.capturing_graph` を立てる。
- node が 0 の segment（sampling など host 側で完結する区間）は capture せず eager。
- 追加入力のうち値が変わり得るもの（rows / requests / device pointer 群）を
  key にして、一致しない run は eager に落とす。
- `SHELF_TP_HIP_GRAPH=0` で全体を無効化できる。

## 結果

Qwen3.8-27B-EXL3-INT8-4bpw、TP=2、context 64、8 token、warmup 2、
interleaved 3 pair（`shelf-bench tg`）。この数値は 2026-11-24 の再起動前の環境で取ったもので、
`pci=disable_acs_redir` 後の同じ条件では再現しなかった（後述のグリッドを参照）。

| 構成 | wall tok/s | logits (argmax / max / sum) |
| --- | ---: | --- |
| graph off | 26.01 / 25.74 / 26.01 | 271 / 11.056138 / -888326.250 |
| graph on | 39.41 / 39.41 / 39.29 | 271 / 11.056138 / -888326.250 |

**+51 %**。生成結果は bit-identical。1 token あたり 131 segment を graph 1 回ずつ
launch する構成で、CPU 側の kernel 投入が消えることが根拠である。

Flash-Next-EP2 + DFlash2（conc 4、max_new 64、生成列の hash 一致）では
123.71 / 124.49 / 124.44 / 124.40 tok/s で有意差なし。DFlash2 の verify（62 行、
`ExecutionRole::Verify`）は graph 対象外で、decode 区間だけでは支配的でないため。

Qwen3.8-27B + DFlash2（TP=2、conc 4、max_new 64）では graph on 123.63 / 124.19、
off 145.63 / 145.62 tok/s で **on が 15 % 遅い**。生成列は一致。原因は capture 費用で、
131 segment の BeginCapture / Instantiate が約 1 回分（200 ms 程度）かかる一方、
spec decode は target の decode run が少なく償却できない。verify が支配的な flow なので
decode 区間の短縮も小さい。

原因は capture 費用そのもの（131 segment で 200 ms 程度、1 segment 約 1.5 ms）で、
spec decode は target の decode run が少なく償却できない。そこで graph の実行方針を
`Disabled` / `ReplayOnly` / `CaptureAndReplay` の 3 値にし、DFlash2 の単発 target decode
(`run_single_target`) は `ReplayOnly` を指定する。加えて同じ key の run が 2 回続くまで
capture を許可しない。

27B + DFlash2 は 145.5 tok/s（off 145.6）、capture 0 回。27B dense の decode は
39.3 tok/s（off 26.0）を維持する。

### グリッド再測定（ACS 無効 boot、`tools/bench/tp_graph_grid.sh`）

TP=1/2 × graph=on/off を 3 回ずつ、セルを交互に実行した中央値:

| tp | graph | 中央値 tok/s | min | max |
| --- | --- | ---: | ---: | ---: |
| 1 | off | 25.53 | 25.52 | 25.65 |
| 1 | on | 25.62 | 25.46 | 25.63 |
| 2 | off | 27.66 | 27.32 | 27.68 |
| 2 | on | 27.03 | 26.31 | 27.41 |

- TP=1 は `SHELF_TP_HIP_GRAPH` の影響を受けない（TP schedule を持たないため）。差は誤差。
- TP=2 は graph on が off より 2.3 % 遅い。dispatch 数は変わらないので、この環境では
  segment graph を既定で有効にする根拠がない。
- TP=2 off は TP=1 の 1.08 倍にとどまる。
- 再起動前の graph on 39.5 tok/s は再現しない。同じ条件の単発計測では 20〜27 の間で
  振れたため、以後はこのグリッドの中央値を正本にする。

`SHELF_TP_HIP_GRAPH` は opt-in に変更した（既定 off）。

この boot で capture と replay の両方が動いていることは確認済み。`SHELF_TP_GRAPH_DEBUG=1` で
capture 144 回・`begin failed` / `capture failed` / `instantiate failed` は 0、
replay の失敗も 0 だった。したがって「ACS 無効化で graph が capture できなくなった」という
説明は成り立たない。差が出るのは replay の実行時間側で、未解決である。

### ACS 有効 boot と reduce 方式のグリッド

`/etc/default/grub` から `pci=disable_acs_redir` を外して再起動し、同じ script を
`REDUCE_LIST="default legacy"` で実行した（ACS 有効、3 回ずつの中央値）:

| tp | graph | reduce | 中央値 tok/s |
| --- | --- | --- | ---: |
| 1 | off | default | 25.50 |
| 1 | on | default | 25.45 |
| 2 | off | default | 27.11 |
| 2 | off | legacy | 27.09 |
| 2 | on | default | 26.18 |
| 2 | on | legacy | 26.84 |

- ACS 有効と無効で TP=2 は 2 % 程度しか変わらない（27.66 → 27.11）。ACS の有無は
  39.5 tok/s の再現と無関係だった。
- `4e848d9e`（consumed 待ち + `hipEventReleaseToSystem`）は eager 経路では差がない
  （27.11 / 27.09）。graph 経路では 26.84 → 26.18 と 2.5 % 遅くなる。隣の barrier の
  消費を待つぶん、graph が本来稼ぐ host 待ちの削減を食っている。
- どの組み合わせでも 39.5 tok/s は出ず、最大でも 27.22 だった。

### 39.5 tok/s の再現と原因（撤回の取り消し）

`1d1f9bcc`（segment graph を入れた時点）を worktree でビルドし直して同じグリッドを回すと、
TP=2 の graph on は **39.75 tok/s**、off は 27.63 だった。TP=1 は両ビルドとも 25.5 で一致し、
CMakeCache の差も 0 だった。したがって 39.5 は環境ではなく、その後のコミットによる退行である。

原因は 2 つ。どちらも `4359002c` で入れた。

1. `tp_segment_graph_key` を **segment ごとに** 呼んでいた。key には
   `paged_attention_plan_signature` が入るため、1 run あたり 131 回 × attention 数の
   解決が走っていた。key は run 先頭の `tp_graph_note_run` で 1 回だけ作り、
   segment では `cache->run_key` を参照するように直した。
2. `SHELF_TP_GRAPH_CAPTURE_BUDGET` の既定が 8 で、131 segment を 1 run 8 個ずつしか
   登録しなかった。decode が 10 run 程度で終わる benchmark では全 segment が揃わず、
   毎 run capture 費用だけを払っていた。既定を 1024（実質無制限）にした。

修正後（`reduce=default`、3 回中央値）:

| tp | graph | 中央値 tok/s |
| --- | --- | ---: |
| 2 | off | 27.59 |
| 2 | on | **36.47** |

旧 build の 39.75 との差（約 8 %）は、`4e848d9e` の consumed 待ち（実測 3.8 %）と
run ごとの key 計算・閾値の分。既定は graph on に戻した。

### 39.5 tok/s の撤回（旧バイナリでの再測定）

`4359002c` を worktree に取り、同じ build 設定で `shelf-bench` を作り直して
同じグリッドを回した（TP=2、3 回中央値）:

| build | graph off | graph on |
| --- | ---: | ---: |
| `4359002c`（39.5 を記録した時点） | 27.78 | 28.17 |
| 現行（`reduce=legacy`） | 27.09 | 26.84 |
| 現行（`reduce=default`） | 27.11 | 26.18 |

- 旧 build でも graph on は 28.17 tok/s で、39.5 は再現しない。**39.5 は測定時の
  環境状態によるもので、コードの性能ではない**と判断し、記録から撤回する。
- 旧 build は graph on が off より 1.4 % 速く、現行は逆に 1〜3 % 遅い。差の一部は
  `sent_` を parity ごとの 2 event に変えたこと（`4e848d9e`）によるものと思われるが、
  同日の off 基準が 27.78 → 27.11 と 2.4 % 動いており、分離はできていない。
- 以後の比較は、同じ boot・同じ build で `tools/bench/tp_graph_grid.sh` を回した
  中央値のみを根拠にする。

### Verify の segment graph（実装・opt-in）

`SHELF_TP_HIP_GRAPH_VERIFY=1` で verify も capture する。key からは context 長
(`min/max_visible_tokens`) を外し、代わりに program 内の attention plan 署名
(`paged_attention_plan_signature` と QSA select の分岐) を入れた。Flash-Next EP2 の
verify では署名が round を通して一定 (2599116421) で、変わるのは request 数と行数だけで
あった。capture は 1 run あたり 8 segment までに制限し、dispatch 数の多い順に登録する。

Flash-Next-EXL3-INT8-3.05bpw + DFlash2、生成列一致:

| 条件 | verify graph off | on |
| --- | ---: | ---: |
| conc 1、160 token | 112.28 tok/s | **128.86（+15 %）** |
| conc 8、128 token（全要求継続） | 398.52 | 387.86（−2.7 %） |
| conc 8、8 token | 68.32 | 68.70 |

低並列では host 投入が律速なので効く。並列度が上がると GPU が埋まり、capture 費用だけが
残るため既定 off とし、`SHELF_TP_HIP_GRAPH_VERIFY=1` で選択する。

rocprofv3 の kernel dispatch 数は graph on / off で同一（25,984 / 2 agent）。
graph は GPU の kernel 数を減らすのではなく、host 側の投入待ちで生じる GPU idle を
消している。trace の kernel busy 合計はどちらも約 137 ms で一致する。

## 副次的な実測（不採用）

### capture 中の cross-device `hipStreamWaitEvent` は node にならない

TP=2 の 2 stream を別々に capture し、rank 0 の external event record を rank 1 が
wait する形を試した。`hipEventRecordWithFlags(..., hipEventRecordExternal)` は
node になるが、`hipStreamWaitEvent` は **capture 時に silently drop** され、
wait 側の graph は kernel 1 node だけになった（期待 2）。replay も待たず、
peer の更新前の値を読む。

### device 側 cross-GPU barrier は host-mapped flag なら成立する

host-mapped memory の flag を system-scope atomic で store / load する相互 barrier は
2 GPU で 20/20 成立した。一方、peer の device memory に置いた flag では
両方が spin したまま進まない（kernel 完了まで store が見えない）。

### peer device memory への kernel 書き込み（再測定で訂正）

当時の測定は、読み手が同じバッファを直前に `hipMemset` して dirty line を残していた
ことと、`__threadfence_system()` を block 0 だけで実行していたことによる artifact
だった。`pci=disable_acs_redir` を入れた環境で条件を直すと、peer への push は
running kernel から正しく見える。詳細と graph 内 barrier の PoC は
`docs/research/20261124/full_graph_sync.md` を参照。

### collectives の kernel 融合は現行 transport では成立しない

`tp_copy_bf16_kernel` を `hipMemcpyPeerAsync` に置き換える案を実測した。
27B TP=2 の decode は 39.56 / 39.63 → 39.67 / 39.55 tok/s で中立。
rocprofv3 の dispatch 総数も 25,984 で不変（`tp_copy` 1,572 が
`__amd_rocclr_copyBuffer` 1,580 に置き換わっただけ）。copy engine へは逃げず、
dispatch 削減にならないため不採用。

copy + sum を 1 kernel に融合するには kernel 内 cross-GPU barrier が要る。
host-mapped flag の barrier 自体は成立するが、peer の device memory へ書いた
データは、書き込み側で `__threadfence_system()` を張り、読み出し側で
L2 bypass / uncached allocation (`hipDeviceMallocUncached`) を使っても
running kernel からは読めない（8 KB の 1 回転送で全語 stale）。
現行 transport の push 型 collective を kernel 内へ畳む設計は採らない。

## 結論

- segment graph は TP=2 の decode で +51 %、生成列一致。採用する。
- cross-device event を graph に閉じ込める方式と、device 側 barrier による
  full decode graph は、この ROCm / RDNA4 構成では正しさを保証できないため不採用。
  `docs/developer/multi-gpu.md` §10 の非目標は維持する。
