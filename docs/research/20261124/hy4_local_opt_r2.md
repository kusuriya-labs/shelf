# Hy4-Flash 単機最適化 R2 実験記録

基準コミットは `4651499d`（Step 3 完了時点）。手順・環境規則は [docs/perf/methodology.md](../../perf/methodology.md) に従う。ROCm 10.2、gfx1201（単一 Radeon AI PRO R9700、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`）、Release、`SHELF_HY4_ONLY=OFF`、GPU 排他ロック `/tmp/r9700.lock` を使用した。`test_hy4_graph prefillbench` は実チェックポイントと FP8 Engram を使用し、42 層の routed expert は第 1 MoE 層の重みを共有する。入力系列は `100 + (37 * i) % 128`、既定チャンクは 1024 とした。通常計測と `rocprofv3` プロファイルは分離した。

## 初期プロファイル分析

`4651499d` の `32768` token prefill（壁時計 `37.336s` / 非計測基準 `36.898s`、`rocprofv3 --kernel-trace` GPU 実行時間合計 `36.092s`）および `stepbench B=16, DFly2 ON`（GPU 実行時間合計 `18.826s`）の上位カーネルは次の通りだった（前回の記録は [hy4_deep_opt.md](hy4_deep_opt.md)）。

| カーネル | Prefill 32k 時間（秒） | 割合 | 呼び出し回数 | 主な改善余地 |
|---|---:|---:|---:|---|
| `expert_batch<true,true,true>` | 6.418 | 17.78% | 2,624 | MoE 重み帯域律速 |
| `exl3_mul24_prefill_wmma_kernel<2,2,4>` | 6.090 | 16.87% | 14,400 | `MT=2` CTA の A-frag 再利用不足 |
| `recall_values` | 4.140 | 11.47% | 6,384 | 単一行処理で sliding/sparse KV 再利用なし |
| `recall_scores` | 3.231 | 8.95% | 6,384 | 単一行処理で key ベクトル再利用なし |
| `dense`（BF16 線形層） | 1.987 | 5.51% | 720 | スカラ反復で重み・入力再利用不足 |
| `project_value` / `project_query` | 1.944 / 1.243 | 5.39% / 3.44% | 1,344 / 1,344 | 単一行投影、LDS タイリング未適用 |
| `index_topk` / `index_scores` / `index_bounds` | 1.238 / 0.848 / 0.418 | 6.94% 計 | 23,040 / 21,504 / 21,504 | 大規模キー列での単一 CTA / 単一 wave 走査 |
| `rotate_output` / `rotate_output_swiglu_quant` / `gather_rotate_pair` | 1.217 / 0.600 / 0.328 | 5.94% 計 | 15,713 / 1,312 / 1,312 | 1 行 1 CTA 構成とスカラロード |
| `normalize`（MLA） | 1.019 | 2.82% | 6,384 | 2 回バリアと過剰スレッド数 |
| `hc_mix` / `hc_expand` | 0.861 / 0.402 | 3.50% 計 | 2,689 / 2,688 | 1 行 1 CTA、ベクタ幅不足 |
| `unpack_quant` | 0.857 | 2.37% | 5,400 | スカラ byte/half ロード・ストア |

## 実験結果（採用）

### 実験 1: MLA prefill カーネルのタイリングと 1-barrier 正規化（`mla_prefill.hip`）

**採用**（コミット `1a16ee1e`）。`project_query_tiled`（`rows > 16`）、`project_value_tiled`（`MT=4, NT=2`, `w_lds[128][40]` / `v_lds[128][40]`）、`recall_scores_tiled<HeadGroups>`（`heads=32, 64` 向け `HeadGroups=2, 4`、`Q` 複数行＋`K` タイル再利用）、`recall_values_tiled<HeadGroups>`（`s_v[32][264]` LDS ステージング）、および `normalize` の 1-barrier 化（`reduce<Max>`、`parts[2][8]`、2 wave 構成）を実装した。

- 単体効果（実験 1 時点、`task-147`）: `8192` token prefill `8.644s -> 6.937s`、`4097` (`chunk=1024`) `4.047s -> 3.309s`、末尾 logits は基準とバイト完全一致。
- `rocprofv3` 32k 内訳（中間トレース）: `recall_values` `4.140s -> 1.630s`（-60.6%）、`recall_scores` `3.231s -> 1.238s`（-61.7%）、`project_value` `1.944s -> 0.832s`（-57.2%）、`project_query` `1.243s -> 0.490s`（-60.6%）。

### 実験 2: BF16 dense・HyperConnection・RMSNorm・RoPE 最適化（`ops.hip`）

**採用**（コミット `c32f23a8`）。

- `dense_tiled<8, 8, 2, 4>` / `<2, 8, 4, 2>`: `x.rows >= 8 && (x.cols % 64) == 0` に対し、入力の計算順を基準 FMA と揃えたままレジスタタイリングを適用した（`rocprofv3` 32k `dense`: `1.987s -> ~0.27s` 級、`dense_tiled<8,8,2,4>` は `0.198s`）。
- `hc_mix_multi<4>`: `x.rows >= 256` において 1 CTA あたり 4 行を処理し（`rocprofv3` 32k `hc_mix`: `0.861s -> 0.405s`、-53.0%）、小バッチ（`B=16` 96 行等）の占有率低下を回避した。
- `hc_expand_vec`（`uint4` ベクタ転送、`0.402s -> 0.212s`）およびレジスタキャッシュ版 `normalize`（`x.cols <= 3072`）。
- `rotary_tiled<Width>`: `Width == 256 && (heads & 3) == 0` に加え、`Width == 128 && (heads & 7) == 0`（indexer `q`: `heads=16, head_dim=128` 等）を追加。後者単独で `8192` token prefill を `5.46s -> 5.413s` へ短縮し、末尾 logits はバイト完全一致した（`task-1024`）。

### 実験 3: DSA Indexer と KV `unpack_quant`・`wht` 最適化（`attention.hip`）

**採用**（コミット `cf14b015`）。

- `index_bounds`: 8 wave の並列 `wave_hist[8][256]` + `hy4_scan_256` + `uint4` ピーク走査により、`0.418s -> 0.199s`（-52.4%）。
- `index_topk`: `wave_hist[8][256]` による並列バケット集計・prefix sum および `count <= 2048` での LDS compaction を導入し、`1.238s -> 0.405s`（-67.3%）。
- `index_scores_refine`: LDS 上でのアクティブ列圧縮と `kg[4]` キャッシュにより、`index_scores<true>` `0.848s -> 0.416s`（-50.9%）。
- `wht`（64/256 点）の warp-shuffle 化と、`unpack_quant` の `uint32_t` `s_src` ステージング + `uint2` ロード／ストア化（`0.857s -> 0.659s` 以降さらに短縮）。

### 実験 4: EXL3 `prefill_wmma`・LDS 削減・`linear.hip` ベクタ化・`wmma.hip` 調整

**採用**（コミット `361f36a1`）。

- `include/shelf/kernels/fused/exl3/group_body.h`: 非融合経路（`FUSED == false`）で未使用だった `__shared__ float epi[...]` を 1 要素に削減（`FUSED ? MT * 16 * 128 : 1`）して occupancy を改善し、`if constexpr (PIPE)` による不要変数参照の整理と `afrag[MT <= 3 ? MT : 1]` の事前ロードを明確化した。
- `src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill_wmma.hip`: `rows >= 256u` で `launch_tiles<4u, 2u, kWm, kSwizzle>`（`256x64` CTA、重み再利用 2 倍）を選択し、`8192` prefill を `5.945s -> 5.451s` へ短縮した（バイト一致）。
- `src/shelf/models/hy4_flash/wmma.hip`: `Bits == 6 && rows > 48 && rows <= 96` に対し `batch<Bits, 3, false, false>` を選択し、`lm_head_argmax rows=96` を微短縮した。
- `src/shelf/models/hy4_flash/linear.hip`: `gather_rotate_pair`（`uint2` + `uint32_t`）、`rotate_output_swiglu_quant`（`float4` / `uint2` / `uint32_t`）、および `rows >= 4` 向けマルチウェーブ `rotate_output<4>`（128 スレッドで 4 行/CTA 処理）を実装し、`8192` prefill を `5.413s -> 5.391s`（`4097 chunk=1024`: `2.677s`、`task-1045`）、`stepbench B=16, DFly2 ON` を `89.519ms` へ改善した。すべてバイト完全一致。

### 実験 5: Engram token ID 再利用（`ops.h`, `engram.hip`, `generate.hip`）

**採用**（コミット `b780b887`）。Engram を持つ 4 層（layers 1, 11, 21, 31）の prefill において、`Ops::prefill_ids` バッファと `Ops::~Ops()`、および `gather_engram_into` を導入してホスト〜デバイス間の入力トークン ID 準備を再利用した。

---

## 棄却した実験

| 実験 | 内容 | 結果と棄却理由 |
|---|---|---|
| 棄却 1: `dense_wmma` | BF16 線形層・indexer 射影へ WMMA 適用 | 演算順変化により 1-ULP タイ判定が反転し `8k rel_l2=0.2807` (`greedy=138/138`)、`4097 rel_l2=0.2728` (`greedy=137/137`)（`task-167`）。FMA 版 `dense_tiled` を採用 |
| 棄却 2: `hc_mix_multi<4>` の小バッチ適用 | `rows=96`（`B=16, DFly2 ON`）への適用 | 64 CU に対して 24 CTA のみとなり `stepbench B=16` が悪化。`x.rows >= 256` に限定 |
| 棄却 3: `group_body.h` `MT=3` インループ `afrag` | レジスタ圧低減を狙いループ内で `afrag` ロード | `lm_head rows=96`: `1403 -> 1529.28 us`、`8k`: `6.130s -> 6.155s` と悪化（`task-349`） |
| 棄却 4: `recall_flash`・過大 `MT` (`Bits=6 MT=6`, `Bits=4 MT=3`) | 47.4 KB LDS で `P*V` を 1 パス融合、デコード `MT` 増加 | `recall_flash` で CTA/CU=1 制限とシリアライズにより `8k`: `6.130s -> 6.932s`、`Bits=6 MT=6` でスピルにより `lm_head_argmax rows=96`: `1350 -> 2054.64 us`（`task-418`） |
| 棄却 5: Engram 別ストリーム非同期 H2D | `hipHostMalloc` + `engram_stream` + `engram_consumed` / `engram_ready` 先行転送 | チャンク間上書き競合を解消しバイト一致達成後も `8k` が `5.451s / 5.487s -> 6.422s / 6.432s`（+0.95s）と悪化（`task-612`, `task-709`）。8k 全体で約 51.4MB しか転送しないため同期・WC 書込みコストが勝った |
| 棄却 6: `prefill_wmma` `<2u, 4u, 4u, 8u>` | `n % 128 == 0` 向け出力タイル拡大 | バイト一致のまま `8k`: `5.472s -> 6.023s` と悪化（`task-772`）。`<4u, 2u, 4u, 8u>` を維持 |
| 棄却 7: `plan_prefill` `batch=64` + `expert_batch<Prefill> MT=4` | MoE prefill バッチ上限と `MT` 拡大 | バイト一致のまま `8k`: `5.461s -> 5.775s` と悪化（`task-980`）。VGPR 増と小バッチの空行で占有率低下 |
| 棄却 8: `recall_scores_tiled_pair` / `recall_values_tiled_pair` | K/V ロードの 2 倍化（ペアヘッド統合） | `8k`: `5.461s -> 5.549s` と悪化し `8K_DIFF` / `4097_DIFF` が発生（`task-1002`）のため差し戻し |

---

## 最終ベンチマーク比較（基準 `4651499d` 対 本実装）

### Prefill（`test_hy4_graph prefillbench`、`chunk=1024`、単一 R9700）

2k / 8k / 32k は最終コードで 2 回計測し平均および各実測値を示す。128k は要件通り 1 回計測した。2k / 8k / 32k の基準値は [hy4_deep_opt.md](hy4_deep_opt.md) の Step 3 値、8k / 32k / 4097 のバイト比較は同機上の `4651499d` 出力バイナリと直接比較した。

| トークン長 (`chunk`) | Step 3 基準（秒） | R2 最終（秒） | 変化率 / 倍率 | 一致検証 |
|---:|---:|---:|---:|---|
| 2048 (`1024`) | 1.839 | 1.276（1.276026 / 1.275551） | -30.6% (1.44x) | 2 試行とも確定動作 |
| 8192 (`1024`) | 8.565（同機実測 8.644） | 5.447（5.435563 / 5.458908） | -36.4% (1.57x) | 末尾 120,832 logits バイト完全一致 |
| 32768 (`1024`) | 36.898 | 23.114（23.132452 / 23.095513） | -37.4% (1.60x) | 末尾 120,832 logits バイト完全一致 |
| 131072 (`1024`) | 167.246 | 103.258（103.258141） | -38.3% (1.62x) | `len=120832 finite=True argmax=17881 max=10.108031` |
| 4097 (`16 / 512 / 1024`) | 4.047 (`chunk=1024`) | 16.729 / 3.082 / 2.703 | -33.2% (`chunk=1024`) | 基準 (`chunk=1024`) および全チャンク間バイト完全一致 |

### Decode `microbench`（`test_hy4_graph microbench`、Step 3 ハッシュ完全一致確認）

| カーネル | 行数 | Step 3 基準（us） | R2 最終（us） | 変化率 |
|---|---:|---|---|---|
| `linear` | 1 / 6 / 16 / 96 | 15.37 / 15.92 / 21.28 / 39.26 | 15.164 / 15.945 / 21.193 / 39.013 | -1.3% / +0.2% / -0.4% / -0.6% |
| `fixed_ffn` | 1 / 6 / 16 / 96 | 31.04 / 36.13 / 49.82 / 119.34 | 30.895 / 35.989 / 49.768 / 118.416 | -0.5% / -0.4% / -0.1% / -0.8% |
| `mla_qb` | 1 / 6 / 16 / 96 | 21.89 / 26.77 / 43.32 / 85.85 | 21.517 / 26.611 / 43.004 / 85.727 | -1.7% / -0.6% / -0.7% / -0.1% |
| `mla_o` | 1 / 6 / 16 / 96 | 57.42 / 96.22 / 121.33 / 261.11 | 56.684 / 96.035 / 120.671 / 259.572 | -1.3% / -0.2% / -0.5% / -0.6% |
| `lm_head` | 1 / 6 / 16 / 96 | 476.12 / 502.09 / 529.71 / 1403.11 | 475.493 / 500.911 / 529.164 / 1397.294 | -0.1% / -0.2% / -0.1% / -0.4% |
| `lm_head_argmax` | 1 / 6 / 16 / 96 | 481.34 / 499.99 / 519.33 / 1358.63 | 480.410 / 500.086 / 518.179 / 1349.978 | -0.2% / +0.0% / -0.2% / -0.6% |
| `experts` | 1 / 4 / 8 / 16 / 6 / 24 / 48 / 96 | - | 118.859 / 399.448 / 769.138 / 1516.401 / 583.900 / 2264.076 / 3036.207 / 3140.108 | 全ハッシュ一致 |

- `indexbench`（`prefillbench 8192 1024 1 index`、`rows=16`）: `start=2045/4096/8192/32768/131072` で `scalar_ms=0.053/0.088/0.162/0.773/1.630`、`wmma_ms=0.054/0.057/0.062/0.114/0.263`、全項目 `match=8176/8176`（`cancellation=1 start=8192` も `match=8176/8176 max_error=0`）。
- `routerbench`: `router_exact / router_wmma` `rows=32` `101.120 / 52.881 us`、`rows=96` `275.339 / 54.339 us`、`rows=1024` `2766.471 / 278.353 us`、全 `router_match`・`router_edge_cases=pass`。
- `moebench`: `rows=1/4/8/16/6/24/48/96` = `241.209 / 503.079 / 786.593 / 1108.873 / 797.886 / 1290.397 / 1542.851 / 1868.369 us`（全ハッシュ一致）。

### Decode `stepbench` 中央値（5 試行、`EP=1`、`test_hy4_graph stepbench`）

| 構成 | Graph | Step 3 基準（ms） | R2 最終（ms） | 変化率 |
|---|---|---:|---:|---:|
| B=1, EP=1, DFly2 OFF | OFF / ON | 26.68 / 25.52 | 26.629 / 25.649 | -0.2% / +0.5% |
| B=1, EP=1, DFly2 ON | OFF / ON | 55.55 / 56.78 | 54.673 / 56.207 | -1.6% / -1.0% |
| B=16, EP=1, DFly2 OFF | OFF / ON | 40.87 / 39.90 | 40.087 / 39.390 | -1.9% / -1.3% |
| B=16, EP=1, DFly2 ON | OFF / ON | 96.28 / 96.13 | 89.616 / 89.519 | **-6.9% / -6.9%** |

---

## 正しさ・総合検証結果

- **CPU / 単体テスト**: `ctest --test-dir build -L "cpu" -LE "optional"` 133/133 PASS（15.23s）、`test_hy4_routes`・`test_hy4_session` PASS、`python3 tests/unit/test_hy4_flash.py` 9/9 PASS（1.522s）。
- **GPU 統合・イベント再利用検証**:
  - `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph`: 既定（論理 EP=1 / 3 / 4、Target 7/7・DFly2 7/7・`capture_fallback=pass`）、`batch`（EP=1 B=2..16・EP=2/3/4 B=2/4/8/16、Graph・DFly2 ON/OFF）、`transport`（EP=3/4 `push` / `peer` / `host`、DFly2 OFF/ON）すべて PASS。
  - `test_hy4_reduce`・`test_hy4_peer_transfer`（`push` / `peer` のエラー注入・データ破損注入フォールバック含む）すべて PASS。
  - `8k / 32k / 4097`（`chunk=16 / 512 / 1024`）の末尾 120,832 logits は基準 `4651499d` とバイト完全一致、`128k` は `len=120832 finite=True argmax=17881` で正常終了した。
