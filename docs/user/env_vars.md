# 環境変数リファレンス

Shelf のビルド・実行・ベンチマーク・量子化ツール・テストで有効な環境変数（主に `SHELF_*`）の一覧である。各機能の詳細な使い方は [compute.md](compute.md)・[server.md](server.md)・[bench.md](bench.md)・[quantize.md](quantize.md) を参照する。

CLI オプションと環境変数の両方が用意されている項目（`--moe-*` / `--ple-*` / `--pcie-*` / `--lm-head-int8` / `--dflash2-prefill-rows` / `--dflash2-radix-topn` / `--tp-transport` 等）は、CLI 引数が指定された場合に CLI 側が優先される。

## 1. ビルド・ROCm 関連

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `ROCM_PATH` | 未設定 | build | `rocm-sdk` CLI が無い環境で ROCm root を指定する（`-DSHELF_ROCM_ROOT` の代替） |
| `ROCR_VISIBLE_DEVICES` | 未設定 | runtime / quantize | 使用する GPU を UUID または index で絞り込む |
| `HIP_VISIBLE_DEVICES` | 未設定 | runtime / quantize | HIP 側に見える GPU index を指定する |
| `HIPBLASLT_USE_ROCROLLER` | 未設定 | runtime | `1` を指定すると EXL3-INT8 prefill の A 経路（hipBLASLt）を避け、自動で B 経路（自前 WMMA）へ切り替える |

CMake オプション（`SHELF_ROCM_ROOT` / `SHELF_LIBCXX_LLVM_ROOT` / `SHELF_LOAD_LIBSTDCXX_FIRST` / `SHELF_HIP_GRAPH` 等）は [build.md](build.md) を参照する。

## 2. 実行・メモリ・通信（compute / server / bench / runtime）

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_ARENA_VMM` | `0` | runtime | `1` で GPU arena に VMM 割り当て経路を使用する |
| `SHELF_KV_POOL_PAGES` | 未設定（自動計算） | compute | 物理 KV pool のページ数を明示指定する（PP 等で全 stage のページ数を揃える用途） |
| `SHELF_LM_HEAD_INT8` | `1` | compute / bench / runtime | `1` で BF16 の `lm_head` をロード時に INT8 W32 へ量子化する（`--lm-head-int8` と同等） |
| `SHELF_TP_TRANSPORT` | `auto` | compute / server / bench | TP 通信の transport を強制選択する（`auto` / `p2p` / `host`、`--tp-transport` と同等） |
| `SHELF_TP_FUSE` | `1` | runtime | `0` で HIP peer TP transport の融合を無効にする |
| `SHELF_PCIE_FRAC` | `0.25` | compute / server / bench | ランタイムが使用してよい PCIe 帯域の割合（`--pcie-frac` と同等） |
| `SHELF_PCIE_GBPS` | `25` | compute / server / bench | 想定する PCIe 帯域 GB/s（`--pcie-gbps` と同等） |
| `SHELF_NO_WORKSPACE_REUSE` | `0` | runtime | `1` で executor の workspace 再利用を無効にする（メモリ調査用） |

## 3. EXL3 / EXL3-INT8（mul24）経路制御

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_EXL3_MUL24_PREFILL_AB` | `A`（gfx1201） | runtime | EXL3-INT8（mul24）の 256 行以上 prefill 経路を切り替える。`A`（既定）は hipBLASLt INT8 経路、`B` は自前 gfx12 iu8 WMMA 経路（K128 group scale、行再量子化なし）。PLaMo 3 は未指定時に独自 prefill 経路を維持する |
| `SHELF_EXL3_INT8_DENSE` | `256` | runtime | EXL3-INT8（mul24）で dense prefill 経路（A / B）を使う最小行数。`0` で無効化し全行を融合 decode 経路で実行する |
| `SHELF_EXL3_INT8_FLAG` | `0.5` | runtime | A 経路（hipBLASLt INT8）で行再量子化を避けて group scale 融合経路へ退避する massive activation 行のエネルギー比しきい値 |
| `SHELF_EXL3_INT8_TUNE` | `0` | runtime | `1` で A 経路の hipBLASLt INT8 候補（最大 16 個）を初回実行時に実測選択する（行数ごとに約 0.15 s 増加） |
| `SHELF_EXL3_INT8_PREFILL_ALGO` | `0` | runtime | `1` で特定形状（`rows=2048` の一部形状・gfx1201）向けの固定 hipBLASLt algorithm index 選択を有効にする |
| `SHELF_EXL3_INT8_PREFILL_ALGO_LOG` | 未設定 | runtime | 設定すると固定 algorithm 選択時に選ばれた index と workspace サイズを stderr へ出力する |
| `SHELF_EXL3_INT8_MOE` | `0` | runtime | grouped EXL3（MoE routed experts）の mul24 活性型。`0`（既定）は FP16 活性、`1` は INT8 活性（iu8 WMMA） |
| `SHELF_EXL3_INT8_ROTATE_F4` | 未設定 | runtime | EXL3-INT8 出力回転の f4 / 列分割経路の上書き制御（設定時は汎用 projection group 統合も無効化される） |
| `SHELF_EXL3_DENSE` | `256` | runtime | EXL3 W16A16（mul1）で dense recon + hipBLASLt 経路を使う最小行数。`0` で無効 |
| `SHELF_EXL3_DENSE_TUNE` | `1` | runtime | EXL3 W16A16（mul1）の hipBLASLt 候補実測選択。`0` で先頭候補に固定 |
| `SHELF_EXL3_PRE` | `1` | runtime | EXL3 W16A16（mul1）の事前入力回転経路。`0` で融合経路へ固定 |
| `SHELF_EXL3_SPLIT` | `0`（自動） | runtime | EXL3 W16A16（mul1）の K 分割数を強制指定する |
| `SHELF_EXL3_GROUPS` | `0`（自動） | runtime | EXL3 W16A16（mul1）の row group 数を強制指定する |
| `SHELF_EXL3_FAST_DECODE` | `1` | runtime | EXL3 の高速 decode 経路（`ka == 4` の 2-word 抽出等）。`0` で汎用復号を使う |
| `SHELF_EXL3_GROUPED_FUSE` | `2` | runtime | grouped EXL3（MoE）の融合 group 数（1 または 2） |
| `SHELF_EXL3_GROUPED_NTILES` | `0`（自動） | runtime | grouped EXL3（MoE）の n-tile 数を上書きする（出力幅を割り切れない値はエラー） |

## 4. HIP Graph・GDN・Octave KV・PLaMo 3 制御

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_GRAPH_VERIFY` | `1` | runtime | `0` で DFlash2 / MTP の verify HIP graph 実行を無効にする |
| `SHELF_PLAMO_HIP_GRAPH` | `1` | runtime | `0` で PLaMo 3 の c=1 decode HIP graph を無効にする |
| `SHELF_GDN_RECURRENCE_EXACT` | `0` | runtime | `0`（既定）で lossy、`1` で exact な GDN recurrence 経路を使う |
| `SHELF_GDN_VERIFY_CHUNKED` | `1` | compute / server / runtime | `0` で DFlash2 verify の chunked GDN を無効にし、serial verify に戻す |
| `SHELF_GDN_VERIFY_CHUNKED_EXACT` | `0` | runtime | `1` で chunked GDN verify に exact 経路を使う |
| `SHELF_GDN_TREE_CHUNKED` | `1` | runtime | `0` で DFlash2 tree verify の chunked GDN を無効にする |
| `SHELF_GDN_CONV1D_TILE` | 自動 | runtime | GDN causal conv1d の tile 選択を上書きする |
| `SHELF_OCTAVE_PREFILL_LEGACY` | `0` | runtime | `1` で Octave KV prefill を旧 10-row 回転空間経路へ戻す |
| `SHELF_OCTAVE_DECODE_LEGACY` | `0` | runtime | `1` で Octave KV decode を旧経路へ戻す |
| `SHELF_PAGED_ATTENTION_SPLIT_REDUCE_QH` | 自動 | runtime | paged attention split-KV reduce の query head 処理経路を上書きする |
| `SHELF_GEMM2D_GROUP` | 自動 | runtime | FP8 `Prefill2D` GEMM のグループ幅を上書きする |
| `SHELF_PLAMO_EXL3_LINEAR` | `1` | runtime | `0` で PLaMo 3 の EXL3-INT8 専用線形・統合経路を無効にする |
| `SHELF_PLAMO_EXL3_PREFILL` | `1` | runtime | PLaMo 3 EXL3-INT8 の prefill モード（`0`: 無効、`1`: 既定、`3`: キャッシュ強制） |
| `SHELF_PLAMO_EXL3_PREFILL_ROWS` | `65` | runtime | PLaMo 3 EXL3-INT8 prefill 経路を使う最小行数（下限 65） |
| `SHELF_PLAMO_EXL3_NORM_ROTATE` | `1` | runtime | `0` で PLaMo 3 の `RMS_NORM` + EXL3 入力回転融合を無効にする |
| `SHELF_PLAMO_EXL3_PRODUCER_ROTATE` | `1` | runtime | `0` で PLaMo 3 の `SWIGLU` + EXL3 入力回転融合を無効にする |

## 5. DFlash2 投機デコード

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_DFLASH2_PREFILL_ROWS` | モデル依存（`256` または `8`） | compute / server | TP=1 の DFlash2 prefill context 構築行数（`--dflash2-prefill-rows` と同等。`8` で従来処理に戻す） |
| `SHELF_DFLASH2_POLICY_MAX_SESSIONS` | `12` | runtime | `--dflash2-adaptive` で投機を停止する batch 負荷上限 |
| `SHELF_DFLASH2_RADIX_TOPN` | 自動 | compute / server / runtime | `0` で従来 top-n、`1` で radix top-n（`--dflash2-radix-topn` と同等） |
| `SHELF_DFLASH2_SERVE_STATS` | `0` | compute (`--serve-stdio`) | `1` で終了時に `DFLASH2_STAGE` / `DFLASH2_POSITION` 等の内訳を stderr へ出力する |
| `SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE` | `1` | runtime | `0` で legacy host token bridge に戻す |
| `SHELF_DFLASH2_HOST_PROPOSAL_D2H` | `0` | runtime | `1` で proposal を host へ D2H して可視化する（timing 調査用） |
| `SHELF_DFLASH2_TOPK_REFERENCE` | `0` | runtime | `1` で DFlash2 top-k に参照実装を使う |
| `SHELF_DFLASH2_ATTENTION_REFERENCE` | `0` | runtime | `1` で DFlash2 ring attention に参照実装を使う |
| `SHELF_DFLASH2_TOKEN_TIMESTAMPS` | 未設定 | server | 指定パスへ DFlash2 トークンごとの到着時刻を記録する |

## 6. MoE・PLE（Qwen4Exp）オフロード・キャッシュ

すべて `shelf-compute` / `shelf-server` / `shelf-bench` の同名 CLI オプションと対応する。

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_MOE_HOST_EXPERTS` | `0` | compute / server / bench / runtime | 層あたり host メモリに退避する routed expert 数（`--moe-host-experts`） |
| `SHELF_MOE_CACHE_SWAP_BUDGET` | `96` | compute / bench / runtime | expert キャッシュの swap 予算（`--moe-cache-swap-budget`） |
| `SHELF_MOE_CACHE_MIN_GAIN` | `150` | compute / bench / runtime | swap を発火する最小頻度利得（`--moe-cache-min-gain`） |
| `SHELF_MOE_CACHE_REBALANCE_STEPS` | `4` | compute / bench / runtime | 再配置の間隔 step 数（`--moe-cache-rebalance-steps`） |
| `SHELF_MOE_CACHE_REBALANCE_ROUTES` | `0`（自動） | compute / bench / runtime | 再配置の間隔 route 数（`--moe-cache-rebalance-routes`） |
| `SHELF_MOE_CACHE_DECAY` | `4096` | compute / bench / runtime | expert 頻度を半減する回数（`--moe-cache-decay`） |
| `SHELF_MOE_CACHE_LOG_INTERVAL` | `64` | compute / bench / runtime | expert キャッシュ hit 率のログ間隔（`--moe-cache-log-interval`） |
| `SHELF_MOE_CACHE_ORACLE` | `0` | compute / bench / runtime | `1` でキャッシュ oracle 判定をログ出力する（`--moe-cache-oracle`） |
| `SHELF_PLE_ARTIFACT` | `1` | compute / bench / runtime | `1` で `<model_dir>/quanted_ple.bin` を読み書きする（`--ple-artifact`） |
| `SHELF_PLE_STREAM_IO` | `mmap` | compute / bench / runtime | PLE table の読み込み方式（`pread` / `mmap` / `direct`、`--ple-stream-io`） |
| `SHELF_PLE_HOT_CACHE_GIB` | `0`（全量常駐） | compute / server / bench / runtime | PLE 用 pinned host ブロックキャッシュ容量 GiB（`--ple-hot-cache-gib`） |
| `SHELF_PLE_PREFETCH` | `1` | compute / server / bench / runtime | `1` で prefill 中に次の PLE 行を先読みする（`--ple-prefetch`） |
| `SHELF_PLE_CACHE_LOG_INTERVAL` | 未設定 | runtime | PLE キャッシュ統計の出力間隔 |

## 7. Server・量子化ツール・テスト

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_COMPUTE` | 自動探索 | server | `shelf-server` が起動する `shelf-compute` 実行ファイルのパス（`--compute` 未指定時） |
| `SHELF_SERVER_LIB` | 自動探索 | server | `shelf-server-lib` ディレクトリのパスを上書きする |
| `SHELF_COMPUTE_LOG` | 未設定 | server | `shelf-compute` との JSONL 通信ログを書き出すファイルパス |
| `SHELF_SERVER_REQUEST_LOG` | 未設定 | server | 受信した HTTP リクエストログを書き出すファイルパス |
| `SHELF_SERVER_TOKEN_LOG` | 未設定 | server | 生成トークンログを書き出すファイルパス |
| `SHELF_NGRAM_LIB` | `build*/libshelf_ngram_quant_c.so` | quantize (`ngram_quantize.py`) | `libshelf_ngram_quant_c.so` の絶対パスを明示指定する |
| `SHELF_QUANTIZE_SEED` | `0` | quantize (`shelf_quantize.py`) | 量子化キャリブレーションの乱数 seed（通常は `--seed` で指定） |
| `SHELF_MODEL_DIR` | 未設定 | tests | 外部モデルを使うテスト（Qwen3.5 系・`tests/server/`）のモデルディレクトリ |
| `SHELF_MODEL_DIR_MTP` | 未設定 | tests | MTP テスト用モデルディレクトリ |
| `SHELF_MODEL_DIR_DFLASH2` | 未設定 | tests | DFlash2 draft モデルディレクトリ |
| `SHELF_MODEL_DIR_DFLASH2_TARGET` | 未設定 | tests | DFlash2 target モデルディレクトリ |
| `SHELF_TP_MODEL_DIR` | 未設定 | tests | TP テスト用モデルディレクトリ |
| `SHELF_QWEN4EXP_MODEL_DIR` | 未設定 | tests | Qwen4Exp テスト・microbench 用モデルディレクトリ |
| `SHELF_BUILD_DIR` | `build` | tests | Python テストランナーが参照するビルドディレクトリ |
| `SHELF_TEST_DEVICE` | `0` | tests | テストで使う GPU device index |

## 8. 診断・計測・デバッグ（開発者・切り分け用）

通常利用では設定しない。ベンチマークや不具合の切り分けに使う。

| 変数名 | 既定値 | 対象範囲 | 用途 |
| --- | --- | --- | --- |
| `SHELF_QWEN35_KERNEL_MODE` | `optimized` | runtime | `correctness` を指定すると参照実装（correctness path）で実行する |
| `SHELF_QWEN35_PERF_STATS` | `0` | bench / runtime | `1` と `--perf-stats-output` の併用で dispatch 統計 JSON を出力する |
| `SHELF_QWEN35_PERF_TIMING` | `0` | bench / runtime | `1`（`PERF_STATS=1`・TP=1 併用）で `semantic_timing.csv` に GPU event 時間を記録する |
| `SHELF_PP_HOST_TIMING` | `0` | bench (`pp`) | `1` で `shelf-bench pp` の CPU 側区間時間を stderr へ出力する |
| `SHELF_TG_MEASURED_DECODE` | 未設定 | bench (`tg`) | `shelf-bench tg` の decode 計測制御 |
| `SHELF_BATCH_TRACE` | 未設定 | compute | 設定すると step ごとの batch・`arena_used` トレースを stderr へ出力する |
| `SHELF_BATCH_STEPWALL` | 未設定 | runtime | 設定すると continuous batcher の step 壁時計時間を出力する |
| `SHELF_GRAPH_DEBUG` | 未設定 | runtime | 設定すると HIP graph の capture / replay 詳細を stderr へ出力する |
| `SHELF_MEM_DEBUG` | 未設定 | compute | 設定するとメモリ確保・解放の診断ログを出力する |
| `SHELF_WS_DEBUG` | 未設定 | runtime | 設定すると program workspace 割り当て詳細を出力する |
| `SHELF_PFXDBG` | 未設定 | runtime | 設定すると prefix cache の検索・保存・復元トレースを出力する |
| `SHELF_OP_DUMP` / `SHELF_OP_SYNC` / `SHELF_OP_FALLBACK_DEBUG` | 未設定 | runtime | 演算単位のダンプ・同期・fallback 診断（設定時は各種融合が無効化され速度が低下する） |
| `SHELF_LINEAR_INPUT_DUMP`（`_MAX_ROWS`, `_MAX_TENSORS`） | 未設定 | runtime | linear 入力テンソルをダンプする |
| `SHELF_KV_CALIB_DUMP`（`_MAX_ROWS`） | 未設定 | runtime | KV キャリブレーション用データをダンプする |
| `SHELF_DEBUG_HIDDEN` / `SHELF_DEBUG_HIDDEN_FILE` | 未設定 | compute | `--hidden-taps` で取得した中間 hidden の出力先制御 |
| `SHELF_LAYER_DUMP_DIR` / `SHELF_LAYER_INPUT_DEBUG` / `SHELF_LAYER_INPUT_OVERRIDE` | 未設定 | runtime | 層単位の入出力ダンプ・上書き |
| `SHELF_LINEAR_DEBUG` / `SHELF_TP_SUM_DEBUG` / `SHELF_PLE_TIMING` | 未設定 | runtime | linear selector・TP reduce・PLE 実行時間の詳細ログ |
| `SHELF_DISABLE_*` | 未設定 | runtime | 個別の融合カーネルを無効化して切り分ける（`FP8_BOUNDARY`, `FP8_FINAL`, `FP8_SWIGLU`, `MXFP4_PRODUCER`, `ATTN_Q_FUSION`, `ATTN_KV_FUSION`, `ATTN_RQ`, `GDN_RQ`, `SWIGLU_RQ`, `FINAL_RQ`, `LUCEBOX_FUSION`, `BOUNDARY_I1`, `BOUNDARY_I8`, `BOUNDARY_I9`。また `SHELF_BOUNDARY_I9_SERIAL` で I9 を直列実行） |
| `SHELF_DFLASH2_COST_TRACE` / `SHELF_DFLASH2_BATCH_PHASE_TRACE` / `SHELF_DFLASH2_BATCH_PHASE_SYNC` / `SHELF_DFLASH2_VERIFY_TIMING_CSV` / `SHELF_DFLASH_PREFILL_TIMING` / `SHELF_DFLASH2_MEASURE_VERIFY_DRAFTS` / `SHELF_DFLASH2_TREE_*` | 未設定 | runtime | DFlash2 のフェーズ別時間・tree verify 検証・詳細トレース出力 |

> [!NOTE]
> `SHELF_EXL3_MUL24_FUSED_DECODE` および `SHELF_GDN_RECURRENCE_LOSSY` は現在の `main` のランタイムには存在しない（`SHELF_EXL3_MUL24_FUSED_DECODE` は `tools/quality/kld_vs_bf16.py` に旧バイナリ互換チェックのみ残っている）。
