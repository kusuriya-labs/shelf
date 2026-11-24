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
interleaved 3 pair（`shelf-bench tg`）:

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

Verify の segment graph 化は行っていない。verify 1 round は約 10 ms で、131 segment の
capture に 200 ms かかるため、64 token 程度の生成では償却できない。capture を複数 run に
分割して 1 run あたり数 segment ずつ登録する案は残っているが、本調査では扱わない。

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

### peer device memory への kernel 書き込みは可視性が保証されない

rank 0 が rank 1 の device buffer へ 8 KB を書き、host-mapped flag を立て、
rank 1 が flag を見てから読む方式を 50 回試した。読み出しの L2 bypass
(plain / system-scope atomic load / nontemporal) を変えても一致しない。
方向依存もあり、device 0 → device 1 の書き込みは全語が stale のままだった。
したがって「graph 内に閉じた collective」は現行 transport では正しさを保証できない。

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
