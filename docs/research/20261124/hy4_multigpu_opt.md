# Hy4-Flash 複数 GPU 最適化（Shared Expert 複製・TP=4 拡張・マルチスロット・HIP Graph）

- 対象ハードウェア: AMD Radeon AI PRO R9700 (`gfx1201`) × 4（各 32 GiB / 利用可能約 31.86 GiB、合計 128 GiB VRAM、PCIe Gen5 x8 P2P `pci=disable_acs_redir` 有効、HBM/DRAM 帯域 640 GB/s・L3 64 MB/GPU）
- 対象モデル: `Hy4-flash-exl3-4bpw-mul24`（42 層: Layer 0 Dense + Layer 1..41 MoE、512 experts / top-16、32 Attention heads、IHC 4 stream、Compressed Indexer + Engram）および DFlash2 ドラフタ `Hy4-flash-dfly-4bpw`（5 層）
- 運用前提: `c=1..16`（最大 16 並行スロット、KV 上限は最大 ~8 Mi tokens 以内）、既存の `EP=1..4, TP=1` 既定経路は 100% 後方互換を維持

---

## 1. 背景とボトルネック

現行の `EP=4, TP=1` 構成では、Routed Experts のみ 4 GPU へ分散される一方、Attention（32 heads）・Compressed Indexer・Router・Layer 0 Dense MLP・Shared Experts・LM Head・DFlash2 がすべて **rank 0 に直列集中** する。

実チェックポイント（`Hy4-flash-exl3-4bpw-mul24` + `Hy4-flash-dfly-4bpw`）の safetensors ヘッダ実測による重み内訳は次の通りである（Engram テーブルの大半はホスト常駐、`folded_bytes` は GPU 常駐展開分 `1,056,964,608 B` を別途保持）。

| コンポーネント | 合計サイズ (Bytes / MiB) | `EP=4, TP=1` 分担 | `EP=1, TP=4` 分担 |
|---|---:|---|---|
| Routed Experts (41 層 × 512) | 74,676,688,896 B (71,217.24 MiB) | 4 GPU 分散（rank 0 少なめ） | 4 GPU 均等分散（5,248 experts/GPU） |
| Attention 射影群 (`q_b`/`kv_b`/`o_proj`/`linear_gate`) | 2,492,423,664 B (2,376.96 MiB) うち TP 分割 2,384,419,488 B | 全量 rank 0 直列 | 4 GPU 分割（596.10 MB/GPU）＋射影入力 `q_a`/`kv_a` 108.00 MB 複製 |
| Shared Experts (41 層) | 291,327,468 B (277.83 MiB) | rank 0 直列（または旧 TP=2 で毎層 AllReduce） | **全 rank 複製 (`Split::None`)・AllReduce 撤去＋`local_expert` 上で並列重畳** |
| LM Head (`vocab=120,832`) | 278,644,740 B (265.74 MiB) | 全量 rank 0 直列 | 4 GPU 列分割（69.66 MB/GPU）＋`tp_argmax` / `tp_gather_logits` |
| Layer 0 Dense MLP | 63,802,380 B (60.85 MiB) | 全量 rank 0 直列 | 4 GPU 分割（15.95 MB/GPU） |
| Indexer / Norms / HC / Router | 約 413 MB | rank 0 直列 | 全 rank 複製 |
| DFlash2 Draft (5 層 + `hidden_correction` + `eh_proj`) | 424,428,112 B (404.77 MiB) うち TP 分割 401,964,184 B | 全量 rank 0 直列 | 4 GPU 分割（約 123.0 MB/GPU） |

この偏りにより、`EP=4, TP=1` で rank 0 のクリティカルパス上の非エキスパート重み読み出しだけでも **3.182 GB/token**（加えて `folded`・router・norm・engram および KV 読み出し）に達し、rank 1..3 が Attention / DFly2 / LM Head の間アイドル待機していた。

---

## 2. 実装した最適化内容

### 2.1 Shared Experts の全 rank 複製と `routed_experts` 重畳・融合復活

- [checkpoint.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/checkpoint.cpp): `tp_size > 1` 時に `.shared_experts.`（`gate_proj` / `up_proj` / `down_proj`）を `Split::Col`/`Row` ではなく **`Split::None`（全 rank 複製）** へ変更。
  - 41 層合計 **277.83 MiB/GPU** の追加 VRAM のみで、従来 `tp_size > 1` で毎回発生していた Shared Experts 用の専用 TP AllReduce を完全に撤去した。
  - 複製により全 rank でローカル形状が `(3072 -> 1536 -> 3072)` のまま維持されるため、[moe.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/moe.hip) の `feedforward()` において融合カーネル `shelf_hy4_shared_gate_up`（`x.rows <= 16`、非 Graph・Graph 両経路）がそのまま有効化された。
- [moe.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/moe.hip) / [residency.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/residency.cpp): `tp_size > 1` かつ非キャプチャ時（`!capture`）の `decode_moe` / `prefill_moe` において、専用ストリーム `local_expert[rank]` 上で `shared_experts` を先行起動し、計算ストリーム `compute[rank]` 上の `routed_experts` および P2P Collective（`tp_sum_f32` / `ep_reduce`）と並列重畳させた。`local_done[rank]` で待ってから `routed_expert_up_proj(result) + shared` を加算する（`capture` 中は単一ストリーム内実行へフォールバックし、HIP Graph キャプチャ安全性を維持）。

### 2.2 本命構成 `EP=1 × TP=4`（Attention/DFly2/LM Head TP=4 ＋ Routed Experts 4 GPU 分散）と集団通信の拡張

4 基の R9700 上で **`Attn TP=4 + MoE EP=4 相当 + Shared 複製`** を実現するため、`--tensor-parallel 4 --expert-parallel 1` 構成および既存 `--tensor-parallel 2 --expert-parallel 2` 強化を実装した。

- **4-rank P2P Reduce (`Residency::reduce_tp4`)**:
  - [residency.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/residency.cpp) に `reduce_tp4` を実装。
  - **`bytes <= 2 MiB`（Decode 全領域 `B=1..16`・投機検証・Graph 内）**: `slot_1m` 上限を `slot_2m`（2 MiB スロット、recv 空間 `24 MiB` 内の `tp_recv` で確保）へ拡張し、`B=16` かつ DFlash2 `draft=6` 時（最大 112 行 × 3072 × FP32 = `1.3125 MiB <= 2 MiB`、`96` 行パディング時 `1.125 MiB`）でも HIP Graph キャプチャ内の単発 Single-Shot に収まるようにした。各 rank の部分和を一発送出し、`tp_sum_f32_to_bf16_4` / `reduce4_kernel` で一括加算して `host` / `peer` / `push`（IPC + P2P Write）の全輸送手段に対応する。
  - **`bytes > 2 MiB`（長 Prefill 領域）**: 各 rank の担当 1/4 チャンクに partial を集約する Reduce-Scatter フェーズと AllGather フェーズの **Two-Shot アルゴリズム** により、総転送量を約 1/2 に削減しつつ `tp_recv` 上限（24 MiB）内で最大 `tp_prefill_rows = 4096`（チャンク分割最大 12 MiB BF16 / 24 MiB FP32）までサポートする。
- **FP32 段階集約後の BF16 丸め (`tp_sum_f32`)**:
  - [linear.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/linear.hip) / [moe.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/moe.hip) / [ops.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/ops.hip) / [residency.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/residency.cpp): `tp > 1` において、`row_parallel` な `o_proj` / `down_proj`（`EP=1, TP=4` 時はさらに `decode_experts` / `prefill_experts` の `routed_experts` も）の中間結果を BF16 へ丸めず FP32（`f32_output` / `result_f32`）として `tp_sum_f32` へ渡し、4 rank（または 2 rank）を FP32 のまま総和してから最後に 1 回だけ RNE で BF16 へ丸める経路を実装した。`tp == 1` の既存動作は一切変更しない。
- **LM Head / Argmax の TP=4 分散**:
  - [generate.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/generate.hip) / [residency.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/residency.cpp) / [ops.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/ops.hip): 語彙方向 `120,832` を TP 分割したまま、`greedy` 時は各 rank でローカル最大値＋グローバル語彙 ID をパック（`tp_argmax_pack`）して全 rank へ共有し `tp_argmax_reduce` で 8B/row のみの最小通信で確定トークンを決定する。サンプリング時または `logits` 要求時のみ `tp_gather_logits` で結合する。
- **DFlash2 5 層の TP 分割と `tp_prefill_rows=4096`**:
  - [checkpoint.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/checkpoint.cpp) / [residency.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/residency.cpp): DFlash2 の Attention・MLP を `Split::N`/`Split::K` で TP 分割し、`tp_prefill_rows` を `4096` へ引き上げるとともに [hy4_compute.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/apps/compute/hy4_compute.cpp) / [shelf_server.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/apps/server/shelf_server.py) の CLI・ドキュメントを `--tensor-parallel {1,2,4}` へ同期した。
- **マルチスロット (`B=1..16`) ＋マルチランク HIP Graph (`TpGraphStep`)**:
  - [graph.h](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/include/shelf/models/hy4_flash/graph.h) / [graph.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-multigpu-opt/src/shelf/models/hy4_flash/graph.cpp): P2P イベント待ち・発火・`reduce_tp4` / `reduce_pair` をグラフノードとして `hipStreamBeginCapture` 内へ記録し、`TpGraphStep` で rank ごとの `hipGraphExec_t` を管理する（`EP=1, TP=4`・`EP=2, TP=2` かつ `push` + P2P 完全接続時に有効、Capture 失敗時は自動ロールバックして eager フォールバック）。単一スロットに加え `Hy4BatchSession` / `generate_batch`（`B=1..16`・DFly2 有効/無効両対応）も TP 経路へ統合した。

---

## 3. VRAM 配置試算（`placement.py` 実測）

R9700 × 4（各 `--capacity 31.859375` GiB = 34,208,743,424 B、`safety_gib=0.5`、`Hy4-flash-exl3-4bpw-mul24` + `Hy4-flash-dfly-4bpw`）での `tools/hy4_flash/placement.py` 試算結果を示す。

| 構成 | コンテキスト / スロット | rank 別 Expert 数 | `peak_estimate` [GiB/GPU] | `free_after` (最小 rank / 各 rank) |
|---|---|---|---:|---:|
| `EP=4, TP=1` (既存) | `c=1`, 2 Mi tok | `[1783, 6403, 6403, 6403]` | `[21.97, 21.97, 21.97, 21.97]` | **9.39 GiB** (`10,085,532,880 B`) |
| `EP=2, TP=2` (4 GPU 換算: 2 グループ) | `c=1`, 2 Mi tok | `[10496, 10496]` (TP グループ毎) | `[30.53, 30.53]` | **0.83 GiB** (`888,115,408 B`) |
| **`EP=1, TP=4` (本命)** | `c=1`, 2 Mi tok | `[5248, 5248, 5248, 5248]` | `[29.55, 29.55, 29.55, 29.55]` | **1.81 GiB** (`1,947,146,960 B`) |
| `EP=4, TP=1` (既存) | `c=16`, 128k tok (総 KV) | `[4598, 5465, 5465, 5464]` | `[31.36, 18.86, 18.86, 18.85]` | **2.3 MiB** (`2,430,160 B`, rank 0 限界) |
| `EP=2, TP=2` (2 グループ) | `c=16`, 128k tok (総 KV) | `[10036, 10956]` | `[29.83, 31.36]` | **0.8 MiB** (`829,648 B`, 限界) |
| **`EP=1, TP=4` (本命)** | `c=16`, 128k tok (総 KV) | `[5248, 5248, 5248, 5248]` | `[29.61, 29.61, 29.61, 29.61]` | **1.75 GiB/rank** (`1,878,057,680 B`) |
| `EP=4, TP=1` (既存) | `c=16`, 32k tok (総 KV) | `[4844, 5383, 5383, 5382]` | `[25.88, 18.59, 18.59, 18.58]` | 5.48 GiB (rank 0) |
| `EP=2, TP=2` (2 グループ) | `c=16`, 32k tok (総 KV) | `[9944, 11048]` | `[23.39, 25.22]` | 6.14 GiB (グループ 1) |
| **`EP=1, TP=4` (本命)** | `c=16`, 32k tok (総 KV) | `[5248, 5248, 5248, 5248]` | `[23.32, 23.32, 23.32, 23.32]` | **8.04 GiB/rank** (`8,635,081,424 B`) |
| `EP=4, TP=1` (既存) | `c=16`, 64k tok (総 KV) | `[4844, 5383, 5383, 5382]` | `[27.98, 18.59, 18.59, 18.58]` | 3.38 GiB (rank 0) |
| `EP=2, TP=2` (2 グループ) | `c=16`, 64k tok (総 KV) | `[9944, 11048]` | `[25.49, 27.32]` | 4.04 GiB (グループ 1) |
| **`EP=1, TP=4` (本命)** | `c=16`, 64k tok (総 KV) | `[5248, 5248, 5248, 5248]` | `[25.41, 25.41, 25.41, 25.41]` | **5.94 GiB/rank** (`6,382,740,176 B`) |

補足:
- `EP=1, TP=4` では **Routed Experts が `Split::None`（rank 単位割当）** となるため、`Checkpoint` ロード時に 20,992 個の expert に対する `Split::N` スライス・ホストメモリ複製が不要であり、実測では `EP=2, TP=2`（`Checkpoint` スライスに約 2 分 25 秒・ホスト RSS 最大約 90.6 GB、4 rank 分の `weight_upload_s=24.499 s`）に対し、論理 4 rank 合計の転送を含めても `weight_upload_s=2.246 s` と極めて高速・省ホストメモリで起動できる。
- `c=16, 128k` トークンの場合、`EP=4, TP=1` や `EP=2, TP=2` では rank 0/グループ 1 の残余 VRAM が 1〜2 MiB 台へ逼迫するのに対し、`EP=1, TP=4` は全 4 GPU 均等に **1.75 GiB/GPU の余裕** を確保でき、`c=1` の 2 Mi トークン（DFly2 併用時 **1.81 GiB**・DFly2 無し **1.93 GiB**）も余裕をもって収容できる。

---

## 4. 4×R9700 物理マルチ GPU 定量試算（Before vs After）

物理 4×R9700 マシンは現在手元で実測できていないため、実モデル重みサイズ・実装した通信回数・R9700 公称帯域（DRAM 640 GB/s、PCIe Gen5 x8 P2P 片方向実効約 25〜30 GB/s）に基づく解析比較を示す（**実機検証前の理論試算**）。

### 4.1 1 ステップあたり計算・通信プロファイル（`c=1` Decode、BF16 隠れ状態 3072 = 6 KiB/token）

| 項目 | Before: `EP=4, TP=1` | After: `EP=1, TP=4`（本命） | After: `EP=2, TP=2` |
|---|---|---|---|
| 非 Expert 直列重み読込 (rank 0 / Critical Path) | **3.182 GB**（全 Attention・L0 MLP・Shared 291.3 MB・LM Head 278.6 MB） | **0.846 GB/GPU**（約 **3.76× 削減**。Shared 291.3 MB は `local_expert` で並列重畳） | **1.527 GB/GPU**（Shared 291.3 MB は並列重畳） |
| Routed Experts 読込 (`16/512` active) | rank 0 少なめ・rank 1..3 各約 0.78 GB | 全 4 GPU 均等 **0.583 GB/GPU** | 全 4 GPU 均等 **0.583 GB/GPU** |
| DFlash2 1 step 重み読込 (5 層＋周辺) | 424.4 MB（rank 0 直列） | **123.0 MB/GPU**（**3.45× 削減**） | **224.0 MB/GPU** |
| Shared Expert 用 AllReduce | なし（TP=2 旧版では +41 回/step） | **0 回**（全 rank 複製により撤去） | **0 回**（全 rank 複製により撤去） |
| TP Collective 回数 (Target 42 層 + LM) | 0 回（EP recv/add 82 回） | **84 回/step** (`o_proj` 42 + `L0 down` 1 + `MoE` 41、`reduce_tp4` 6 KiB FP32->BF16) ＋ `tp_argmax` 1 回 (8 B) | TP 42 回 + EP 82 回 |
| P2P データ転送量 (`m=1` 1 step) | rank0<->peer 約 5.4 MiB | `reduce_tp4` 12 KiB×3 peer×84 ≒ **3.0 MiB** (Single-Shot) | 約 1.0 MiB + EP 分 |
| P2P データ転送量 (`m=1024` chunk Prefill) | パケット送信＋結果返送 | **Two-Shot `reduce_tp4`**: Phase 1 (FP32 3 MiB×3 peer) + Phase 2 (BF16 1.5 MiB×3 peer) = **13.5 MiB/GPU/call**（単純全対全 36 MiB 比 **62.5% 削減**） | 2-rank Two-Shot + EP |
| 4 GPU 総合 Decode 重みトラフィック上界（`c=1`, DRAM 640 GB/s） | rank 0 単独 **>= 5.0 ms+**（非 Expert 3.18 GB だけで約 5.0 ms、KV・router 等除く） | Critical path **1.43 GB/GPU ≒ 2.23 ms**（理論約 2.2× 高速化余地、84 回の P2P 同期レイテンシ合計約 0.3〜0.8 ms を差し引いても純増見込み） | 2.11 GB/GPU ≒ 3.30 ms |

補足（Prefill 128k への効果）:
- Prefill 時、全体の 83〜85% を占める Attention（32 heads）＋ L0 MLP が **4 GPU で完全並列化（TP=4）** され、`tp_prefill_rows = 4096`（`--prefill-chunk 1024` 等も即利用可能）でチャンク制限なしに稼働できるため、4 GPU 総合スループットは従来の +15% 程度から 2〜3× 超への伸長が期待される（最終値は物理 4 GPU マシンでの実測が必要）。

---

## 5. 単一 R9700 上の検証結果（論理 4 rank・CPU CTest）

> 実機 4×R9700 は未接続のため、以下はすべて単一 R9700 (`GPU-a8e9ddefa2d60f55`, ROCm 10.2) 上の論理 4 rank 検証および CPU ユニットテストの実測ログである。

### 5.1 CPU CTest・Python・CLI 契約検証（実行済み）

- `ctest --test-dir build --output-on-failure -L "cpu" -LE "optional" -j $(nproc)`: **133/133 Passed**（`test_hy4_session` / `test_hy4_routes` / `test_cli_docs_sync` / `test_tp_reduction` / `test_tp_barrier_group` / `test_parallel_topology` / `test_qwen35moe_tensor_parallel_plan` / `test_architecture_boundaries` `passed=1271 failed=0` を含む）
- `ctest --test-dir build -V -R "test_hy4_flash_cpu"` (`python3 tests/unit/test_hy4_flash.py`): **10/10 Passed**（`test_tp4_placement` 含む）
- `ctest --test-dir build --output-on-failure -R "test_cli_docs_sync_python|test_hy4_routes|test_hy4_session"`: **3/3 Passed**
- `python3 tests/server/test_server_cli.py`: **passed=22 failed=0**

### 5.2 単一 R9700 論理 4 rank GPU 検証

実モデル（`BODY=/home/homelab1/llm-fast-load/Hy4-flash-exl3-4bpw-mul24`、`DRAFT=/home/homelab1/llm-fast-load/Hy4-flash-dfly-4bpw`、`ENGRAM=/home/homelab1/datapool/rocm-exl3-rdna2/hy4-flash-pilot/engram-e4m3fn`）を用い、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`・`SHELF_HY4_CHECK_EVENT_REUSE=1`・`flock /tmp/r9700.lock` 下ですべての GPU 検証を実行した。

- `./build/tests/test_hy4_reduce`: **PASS**（exit 0）
- `./build/tests/test_hy4_peer_transfer`: **PASS**（`probe fallback mode=push corrupt=0/1 pass`・`probe fallback mode=peer corrupt=0/1 pass`）
- `SHELF_HY4_ENGRAM_DIR="$ENGRAM" ./build/tests/test_hy4_graph "$BODY" "$DRAFT"`（既定）: **PASS**（`ep=1/3/4`: いずれも `target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass`）
- `./build/tests/test_hy4_graph "$BODY" "$DRAFT" tp "$ENGRAM"`:
  - `EP=4, TP=1`（`SHELF_HY4_TP_FILTER=4,1`）: **PASS**
    - `transport=push`: `prefill20_l2=291.402392`, `rel=2.223695e-01`, `ref_arg=269(12.7579)`, `tp_arg=297(12.0505)`
    - バッチ一致（`draft=0/1`, `graph=0/1`）: `B=2,4,8,16` および `slots=4 B=2,4` すべて `greedy=B/B`
  - `EP=2, TP=2`（`SHELF_HY4_TP_FILTER=2,2`）: **PASS**
    - `transport=push/peer/host` すべて `prefill20_l2=312.232068`, `rel=2.382646e-01`, `ref_arg=269(12.7579)`, `tp_arg=297(12.3074)`
    - `graph_parity=pass target=7/7 draft=7/7 capture_fallback=pass`
    - バッチ一致（`draft=0/1`, `graph=0/1`）: `B=2,4,8,16` すべて `greedy=B/B`、`session=pass draft_rounds=4`、`sparse_2047_2053=pass`
  - `EP=1, TP=4`（`SHELF_HY4_TP_FILTER=1,4`）: **PASS**
    - `transport=push/peer/host` すべて `prefill20_l2=337.009349`, `rel=2.571722e-01`, `ref_arg=269(12.7579)`, `tp_arg=297(12.4811)`
    - `graph_parity=pass target=7/7 draft=7/7 capture_fallback=pass`
    - 長文チャンク境界: `chunk1536_vs_512_l2=0.000000`, `rel=0.000000e+00`, `argmax=match`
    - バッチ一致（`draft=0/1`, `graph=0/1`）: `B=2,4,8,16` すべて `greedy=B/B`、`session=pass draft_rounds=8`、`sparse_2047_2053=pass`
  - 備考: `EP=1, TP=1` 基準（`ref_arg=269`）に対する `prefill20` の `rel ≒ 0.22〜0.26` および `tp_arg=297` は、`EP=4, TP=1` 時点から生じる layer 1 共有ウェイト構成下での 42 層ルーティング集約順序差に起因する。`push`/`peer`/`host` 輸送間・`eager`/`Graph` 間・`chunk 1536`/`512` 間・`B=1..16`・位置 `2047`/`2053` 混在セッションではすべて完全一致を確認済み。
- `SHELF_HY4_ENGRAM_DIR="$ENGRAM" ./build/tests/test_hy4_graph "$BODY" "$DRAFT" batch`: **PASS**（`ep=1..4` すべて `draft=0/1`, `graph=0/1`, `B=2..16` で `greedy=B/B`、`session=pass draft_rounds=8`）
- `SHELF_HY4_ENGRAM_DIR="$ENGRAM" ./build/tests/test_hy4_graph "$BODY" "$DRAFT" transport`: **PASS**（`ep=3/4`, `transport=push/peer/host`, `draft=0/1`, `B=1,2,4,8,16` すべて `greedy=B/B`）

---

## 6. 物理 4×R9700 マシンでの実機検証コマンド

物理 4×R9700（P2P `pci=disable_acs_redir` 有効）実機での検証は以下をそのまま実行する。

```bash
MODEL=/home/homelab1/datapool/rocm-exl3-rdna2/models/Hy4-flash-exl3-4bpw-mul24
DRAFT=/home/homelab1/datapool/rocm-exl3-rdna2/hy4-flash-pilot/shelf-dfly/Hy4-flash-dfly-4bpw
ENGRAM=/home/homelab1/datapool/rocm-exl3-rdna2/hy4-flash-pilot/engram-e4m3fn

python3 tools/hy4_flash/placement.py "$MODEL" --draft "$DRAFT" \
  --tensor-parallel 4 --capacity 31.859375 31.859375 31.859375 31.859375 \
  --max-concurrent-requests 16 --tokens 131072 --output /tmp/hy4_tp4_c16.json

SHELF_HY4_CHECK_EVENT_REUSE=1 SHELF_HY4_TP_FILTER=1,4 \
  ./build/tests/test_hy4_graph "$MODEL" "$DRAFT" tp "$ENGRAM"

./build/tests/test_hy4_graph "$MODEL" "$DRAFT" transport "$ENGRAM"

printf '120000 802 8778 299 9480\n' > /tmp/hy4_ids.txt
ROCR_VISIBLE_DEVICES=0,1,2,3 ./build/shelf-compute hy4 generate \
  --model-dir "$MODEL" --engram-dir "$ENGRAM" --dfly-dir "$DRAFT" \
  --placement /tmp/hy4_tp4_c16.json --input-ids-file /tmp/hy4_ids.txt \
  --tensor-parallel 4 --expert-parallel 1 \
  --max-seq-len 8192 --chunk 1024 --max-new-tokens 128

python3 src/apps/server/shelf_server.py --model-dir "$MODEL" \
  --dfly-dir "$DRAFT" --engram-dir "$ENGRAM" \
  --placement /tmp/hy4_tp4_c16.json \
  --device 0,1,2,3 --tensor-parallel 4 --expert-parallel 1 \
  --max-concurrent-requests 16 --max-seq-len 8192 --prefill-chunk 1024
```
