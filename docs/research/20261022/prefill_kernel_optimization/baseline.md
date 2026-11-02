# Prefill 個別最適化 — Phase A ベースライン

対象: Qwen3.8-27B FP8 / R9700 ×2 / TP=2 / PP=2048 の E2E Prefill。
ゴールは 663 ms → 500 ms（≥163 ms 削減）。融合は禁止、既存カーネル内部の改善のみ。

## 0. 参照文書の状況

| 指定文書 | 状態 |
|---|---|
| `kernel_optimization_and_fusion.md` | リポジトリ直下には無い。実在は `docs/developer/kernel_optimization_and_fusion.md`（235 行）。最適化原則・律速分類・算定式はこれを参照。融合専用節（§4 ②b / tier / matcher / topology）は今回の対象外 |
| `docs/perf/methodology.md` | 存在（115 行）。測定原則の正本 |
| `docs/perf/current.md` | 存在（459 行）。ただし HF FP8 / BF16 の E2E 再計測は未実施と明記あり（L14）。663 ms 相当は未収録 |
| `docs/developer/kernels.md` | 存在（63 行） |
| `docs/developer/testing.md` | 存在（256 行）。required acceptance の正本は `tests/run_required_acceptance.py`。skip は fail |
| `docs/rnd/optimization_findings.md` | **存在しない**（`docs/rnd/` 自体が無い）。`docs/perf/current.md` / `methodology.md` / `roadmap.md` から `rnd/reference_gap_analysis.md` が参照されているがこれも不在 |
| `docs/rnd/kernel_baseline_2026-10.md` | **存在しない**。occupancy 算定式は `kernel_optimization_and_fusion.md` §「gfx1201 での制約と算定式」と本ゴール文書の同一式で代替する（§7.88 実測表そのものは参照不能） |

## 1. 実行環境（A-1）

- GPU: AMD Radeon AI PRO R9700 ×4（DeviceID 0x7551）。TP=2 は GPU0 (`--device 0`) と GPU1 を使用。gfx1201
- CPU: Ryzen 7 7800X3D
- HIP: 7.15.26333-0000000 / AMD clang 23.0.0git（ROCm SDK バンドルの clang++）。distro 版 ROCm のバージョンファイルは `/opt/rocm` に無い
- VRAM: 34,208,743,424 B/枚。計測時の GPU0/1 使用は各プロセス起動時のみ（後述の汚染記録を除き空き）
- GPU clock/power: **計測中は未取得**（idle 時 sclk level 0 = 56 MHz, mclk 0 = 96 MHz、温度 30–47 ℃ のみ）。run 中クロックの記録は次工程で追加する
- PCIe: **Broadcom PEX890xx（PEX89048）Gen5 スイッチ配下に 4 GPU**。各カードの upstream port（Navi 10 XL Upstream Port、`05/08/0d/10:00.0`）は **Gen5 x8（32 GT/s ×8、理論 31.5 GB/s）**、GPU レベルと downstream は x16。GPU 間 P2P は両端の x8 を通る → **P2P 上限 = Gen5 x8**。tp_copy の in-situ 実効 29.1 GB/s は理論値の 92%（→ [opportunity.md](opportunity.md) で飽和と判定）
- commit: `54bc68e`（bench(pp): --tp 1|2 対応を追加）
- モデル: `models/Qwen3.8-27B-FP8`（HF FP8 配布物）。revision = `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`（`.cache/huggingface/download/*.metadata` 由来）
- FP8 設定: weight は 128×128 ブロック scale、activation は per-token-group 128（動的）。KV は計測時 BF16（pp 既定）
- TP: `TpCoordinator`（tp_size=2 の正式サポート）、transport は `SHELF_TP_TRANSPORT` 未設定のため自動選択。`hipDeviceCanAccessPeer` は全 GPU 対で 1 → **HipPeer (p2p)** が選ばれる
- モデル寸法（config.json）: hidden 5120 / intermediate 17408 / layers 64（うち attention 16 層・GDN 48 層）/ heads 24 / kv heads 4 / head_dim 256 / vocab 248320

### 測定コマンド全文

```sh
# 基準値（ゴールの 663ms に相当する経路）
./build/shelf-compute --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 0 \
  --prefix-cache 0 --input-ids-file /tmp/pp2048.txt --max-new-tokens 1 --max-seq-len 2304

# 主指標（5-run p50）
./build/shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --prompt-tokens 2048 \
  --tp 2 --device 0 --arena-gib 16 --warmup 1 --runs 5

# trace
rocprofv3 --kernel-trace --output-format csv -d /tmp/roc_e2e -- \
  ./shelf-bench pp ... --runs 1 --warmup 0
```

- `/tmp/pp2048.txt` = tokenizer.json で固定生成した 2048 トークンの ID 列（以後の比較で不変）
- warmup: pp は `--warmup 1`（初回 clock ramp の除外）。compute は初回 run を基準から除外しないが 2 回取得
- **PREFILL_MS の区間**: prefill step の開始から同期完了まで（モデルロード・デコードは除外。`single_request.hip`）
- **pp の区間**: prefill チャンク実行ループの hipEvent（GPU 時間）と wall clock（setup・モデルロードは含まない）
- `SHELF_OP_DUMP=1` は不使用

### 指標の決定

ゴールの「663 ms / 3,089 tok/s」は `shelf-compute --tp 2` の PREFILL_MS（2048/663ms=3089 tok/s）に一致する。

- **主指標**: `shelf-bench pp` の `wall_ms_median`（5-run p50、methodology の 5-run p50 に合致、モデルロードを含まない反復が速い）
- **突合指標**: `shelf-compute` の PREFILL_MS（ゴール基準の再現確認に毎回は使わないが、節目に確認する）
- 両者は約 1% の系統差（pp がやや低い）。比較は必ず同一指標内で行う

## 2. 基準値（A-2 再現）

| 指標 | 値 | 条件 |
|---|---|---|
| compute PREFILL_MS | **663.4 / 669.9 ms** | 2 回、model load 20.9 s は区間外 |
| pp wall p50 | **657.75 ms**（runs 656.68 / 657.67 / 657.75 / 658.19 / 658.60、stddev 0.71、ばらつき 0.3%） | `--warmup 1 --runs 5` |
| pp gpu p50 | 657.73 ms | 同上（wall と 0.02 ms 差 → host 時間は wall に露出していない） |
| pp throughput | 3,113.6 tok/s | 2048 / 0.65775 |
| トークン gate | `GENERATED_IDS=47410`（2 回一致） | compute, greedy 1 token |

ゴールの 663 ms は再現された（663.4 ms）。run 間ばらつきは 0.3%（30% 限度の far 以内）。

### GPU 汚染記録

- 09:34–09:49 頃、PID 1423963 `shelf-compute --device 1 --tp 2 --eagle4-…`（timeout 900s、VRAM 66.3 GB = GPU1+2）を検出。GPU1/2 を使う測定は実施せず、KFD プロセスが空（`No KFD PIDs currently running`）を確認してから計測した
- 以後も E-1 に従い各測定ペア直前に KFD プロセスを確認し、占有があれば除外・記録する

## 3. Trace（A-3）

収集: `rocprofv3 --kernel-trace`、pp TP2 p2048 ×1 run（profiler 下 wall 667.8 ms = 無測定 657.75 に対し +1.5%）。4,306 dispatch、Agent 1（GPU0/rank0）と Agent 2（GPU1/rank1）の両方を取得。

### Kernel family 別集計（Agent 1 = rank0。Agent 2 はほぼ同値）

| Kernel family | Agent1 合計 | share* | n | 平均 | VGPR | LDS | Scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm_fp8_w8a8_wmma_prefill_2d | 383.82 ms | 61.1% | 400 | 959 µs | 184 | 0** | 0 |
| tp_copy_bf16 | 92.36 ms | 14.7% | 140 | 660 µs | 8 | 0 | 0 |
| gdn_recurrence_wmma_lossy | 55.62 ms | 8.9% | 48 | 1159 µs | 224 | 35840 | 0 |
| PagedAttention prefill | 15.70 ms | 2.5% | 16 | 982 µs | 176 | 29696 | 0 |
| fused_resid_rmsnorm_quantize | 14.91 ms | 2.4% | 127 | 117 µs | 32 | 512 | 0 |
| tp_add_peers_bf16 | 14.82 ms | 2.4% | 140 | 106 µs | 8 | 0 | 0 |
| swiglu_bf16 | 11.09 ms | 1.8% | 64 | 173 µs | 32 | 0 | 0 |
| __amd_rocclr_copyBuffer | 7.31 ms | 1.2% | 736 | 10 µs | 16 | 0 | 0 |
| activation_quantize（bf16→u8） | 6.16 ms | 1.0% | 112 | 55 µs | 32 | 0 | 0 |
| fused_gdn_conv_silu | 5.57 ms | 0.9% | 48 | 116 µs | 24 | 0 | 0 |
| paged kv/act 関連（lds8192） | 5.49 ms | 0.9% | 96 | 57 µs | 72 | 8192 | 0 |
| fused_gdn_post | 5.46 ms | 0.9% | 48 | 114 µs | 40 | 512 | 0 |
| gemm_bf16_exact_rows_1（lm_head） | 2.02 ms | 0.3% | 1 | 2020 µs | 24 | 0 | 0 |
| fused_gdn_qk_prepare | 1.68 ms | 0.3% | 48 | 35 µs | 16 | 0 | 0 |
| その他（rope/rmsnorm/split/attention post quant 等） | 計 55.6 ms | 8.9% | — | — | — | — | — |
| **合計** | **628.4 ms** | 100% | 2153 | | | | |

\* share = GPU 上の kernel duration 合算に対する比率。E2E wall への寄与とは別（§5）。
\*\* `LDS_Block_Size=0` は rocprof の表示値。ソースは動的 shared（as stage 1 KB）を使うため実 LDS 使用は 1024 B の可能性がある（B で要確認）。

Agent 2 の合計は 636.5 ms。家族別比率も同一（最大差 3 ms）→ 両 GPU は対称。

### GEMM の shape 履歴（Agent 1、M=2048 = grid.y 8）

| Grid_Size_X（= grid.x×256） | grid.x → out | n | 合計 | 平均 | 対応（推定） |
|---:|---:|---:|---:|---:|---|
| 34816 | 136 → 8704 | 128 | 178.92 ms | 1398 µs | gate / up（64 層 × 2、TP2 分割後） |
| 20480 | 80 → 5120 | 176 | 161.93 ms | 920 µs | down(64) + o(16) + GDN 出力系(96) |
| 12288 | 48 → 3072 | 48 | 23.90 ms | 498 µs | GDN 系（48 層 × 1） |
| 24576 | 96 → 6144 | 16 | 15.87 ms | 992 µs | attention 系（16 層） |
| 2048 | 8 → 512 | 32 | 3.20 ms | 100 µs | k/v（TP2 分割後 512） |

（対応づけの一部は dispatch 順での確定が必要 → B で裏取り）

### 時間線・critical path

- Agent 1 prefill burst: span 681.5 ms / kernel 合算 620.2 ms / host gap 合計 61.4 ms
- Agent 2 prefill burst: span 736.8 ms / kernel 合算 629.0 ms / host gap 合計 107.8 ms
- 両 agent の**活動時間の和集合 = 643.4 ms**（純粋な合算 1249.2 ms の 51.5% しか被らない → 両 GPU はほぼ並列）
- union span = 736.8 ms（Agent 2 が全体制約）。profiler 下 wall 667.8 ms は rank0 イベント計測
- **GPU が両方止まっている時間（活動しない区間）= 736.8 − 643.4 ≈ 93 ms**（同期待ち・host）
- 各 GPU の kernel 合算 620–636 ms が wall 657.75 ms の大半を占める。**④ launch/host 律速の露出は小さい**（pp の gpu−wall 差 0.02 ms。ただし rank 間の待ち 60–108 ms は wall に含まれる）

結論（概要）: wall の削減余地は「片側 rank の chain（kernel + 同期待ち）」からの削減で評価する。GPU 別合算を wall に足さない。

## 4. ISA（A-4）

手順: `.hip.o` の `.hip_fatbin`（clang offload bundle）を ELF パーサで抽出し、`llvm-objdump-22 -d --mcpu=gfx1201` で静的解析。PMC カウンタは gfx1201 で 0 を返すため不使用。スクリプトは `/tmp/isa.py`（本ディレクトリ外・単発）。

| kernel | text 大きさ | 静的命令数（約） | wmma | s_barrier ベータ | ds | global load | scratch |
|---|---:|---:|---:|---:|---:|---|---|
| gemm_fp8_w8a8_wmma_prefill_2d\<false\> | 17,516 B | 2,932 | 64 (`fp8_fp8`) | 1 拠点 | 17 | b64×48, b32×5 | 0 |
| tp_copy_bf16 | 280 B | 55 | 0 | 0 | 0 | `global_load_d16_b16`×1 + `global_store_b16`×1 | 0 |
| gdn_recurrence_wmma_lossy | 30,676 B | 5,152 | 42 (`f16`) | 9 拠点 | — | b32×224, d16×27, u16×10 | 0 |

- GEMM: 1 barrier 拠点（g2 ループ内の as stage 後。K=5120 で動的 40 回）。`v_fmac_f32` 112 + `v_dual_fmac_f32` 144、`s_wait_alu` 425 / `s_delay_alu` 277。`s_branch` 系 68×2
- **tp_copy**: 要素（uint16）1 件ずつ `out[i]=in[i]` の stride ループ（`hip_peer_tp_transport.hip:63`）。2 B 粒度、grid 上限 1024 ブロック。**ISA から vector 化・ILP 余地が極めて大きい**（ただし B-2 で帯域律速かどうかの判定が先）
- gdn lossy: 9 拠点の barrier（LDS 35.8 KB のステージング同期）。VGPR 224

## 5. 次工程（Phase B へ）

1. 各 family の実効帯域 / TFLOP モデル（bytes・FLOP の静的算定）と律速4分類
2. occupancy 算定式（47,104 dwords/SIMD）による資源表
3. PCIe リンク世代の取得（tp_copy 判定に必須）
4. host gap 60–108 ms の内訳（同期待ち vs host）
5. Amdahl による 163 ms 削減計画 → `opportunity.md`
