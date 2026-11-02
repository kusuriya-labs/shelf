# Static YaRN 2x / 4x コンテキスト拡張の調査記録

- ID: YARN-CONTEXT-EXTENSION
- 実施日: 2026-10-23（Gate 0: 調査・仕様固定、コード変更なし）
- 対象: Qwen3.8-27B / Qwen3.8-27B-FP8、`qwen35` 系推論ランタイム
- 状態: **Gate 0 完了（ACCEPT）** — Gate 1（モデル設定・CLI・エラー契約）へ

## 0. 対象チェックポイントの設定（実測）

`config.json` の SHA256:

| checkpoint | SHA256 (config.json) |
| --- | --- |
| Qwen3.8-27B-FP8 | `74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc` |
| Qwen3.8-27B | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Qwen3.8-27B-DFlash2 | `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980` |
| Qwen3.8-27B-EAGLE4 | `fd0c8d32a7b9133530a849f3b2411c7e00b6197b6141b8f2007e0f87356dab1d` |

Target（Qwen3.8-27B / -FP8 共通、`text_config`）:

| 項目 | 値 |
| --- | --- |
| `rope_parameters.rope_type` | `default` |
| `rope_parameters.rope_theta` | 10000000 |
| `rope_parameters.partial_rotary_factor` | 0.25 |
| `rope_parameters.mrope_interleaved` | true |
| `rope_parameters.mrope_section` | [11, 11, 10] |
| `factor` / `original_max_position_embeddings` / `beta_fast` / `beta_slow` / `truncate` / `attention_factor` | **設定に存在しない**（YaRN 指定時のみ必要） |
| `head_dim` | 256 → rotary_dim = 256 × 0.25 = **64** |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 |
| `num_hidden_layers` | 64（`full_attention_interval=4`、`layer_types` で full_attention は16層） |
| `max_position_embeddings` | 262144 |
| `mtp_num_hidden_layers` | 1 |

Drafter:

| checkpoint | head_dim | kv head | rope_parameters |
| --- | --- | --- | --- |
| DFlash2 | 128 | 8 | `rope_theta=10000000, rope_type=default` のみ |
| EAGLE4 | 128 | 40 | `rope_theta=10000000.0, rope_type=default` のみ |

> Drafter には `partial_rotary_factor` も `original_max_position_embeddings` も存在しない。
> Target の YaRN 設定を Drafter へ転用できないのは設定面からも確認できた。

## 1. Qwen 公式モデルカードの案内（2026-10-23 取得）

- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.
- 超長テキストには YaRN の利用を推奨。公式の設定上書き例は
  `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144`,
  `--max-model-len 1000000`（vLLM / SGLang / TokenSpeed の3例）。
- 公式注意書き（要約）:

  > 有名なオープンソースフレームワークはすべて static YaRN を実装しており、
  > factor は入力長に関係なく一定である。**短いテキストの性能に影響しうる。**
  > 長いコンテキストを処理する必要がある場合にのみ `rope_parameters` の変更を推奨する。
  > `factor` は必要に応じて変更することが望ましい。例えば典型的なコンテキスト長が
  > 524,288 トークンであれば `factor=2.0` の方がよい。

- 公式上書き例には `beta_fast` / `beta_slow` / `truncate` / `attention_factor` は
  含まれない → Transformers のデフォルト（32 / 1 / truncate=true、
  attention_factor は factor から導出）が使われる。

## 2. YaRN 標準値の検証（CPU、2026-10-23）

参照: Hugging Face Transformers 5.14.1 の `_compute_yarn_parameters`
（`transformers/modeling_rope_utils.py`、実物のソースを確認）。
実装は `find_correction_range` で `truncate=true` のとき
`low=floor`, `high=ceil`、ramp は `clamp((i-low)/(high-low),0,1)`、
`inv_freq = base/f * ramp + base * (1-ramp)` であり、手順書 2.1 の式と一致。

Qwen3.8（`d=64, theta=10,000,000, L=262,144`）:

| 量 | 値 |
| --- | --- |
| `correct(beta_fast=32)` | 14.240960 → `low = 14` |
| `correct(beta_slow=1)` | 21.121646 → `high = 22` |
| `attention_factor` (factor=1) | 1.0 |
| `attention_factor` (factor=2) | 1.0693147180559945 |
| `attention_factor` (factor=4) | 1.138629436111989 |

Transformers 5.14.1 が出力した float32 の inv_freq（32要素、抜粋）:

| idx | factor=1 / 2 | factor=4 |
| --- | --- | --- |
| 13 | 1.4330126577988267e-03 | 1.4330126577988267e-03 |
| 14 | 8.659643353894353e-04 | 8.659643353894353e-04 |
| 18 | 1.1547819303814322e-04（f1）/ 8.660864841658622e-05（f2） | 7.217386882985011e-05 |
| 21 | 2.5482966520939954e-05（f1）/ 1.4334168554341886e-05（f2） | 8.759770025790203e-06 |
| 22 | 1.5399264157167636e-05（f1）/ 7.699632078583818e-06（f2） | 3.849816039291909e-06 |
| 31 | 1.6548170833630138e-07（f1）/ 8.274085416815069e-08（f2） | 4.1370427084075345e-08 |

- 生成手順: `_compute_yarn_parameters(stub_config)` に
  `rope_type=yarn, factor, rope_theta=10000000, partial_rotary_factor=0.25,
  original_max_position_embeddings=262144, truncate=True, beta_fast=32, beta_slow=1,
  head_dim=256` を与えた。CPU double oracle（Python `math`）とは
  idx 一致・有効数字10桁程度まで一致（float32 演算の丸めの範囲）。
- Transformers の `base ** (arange(0,dim,2)/dim)` は float32 で計算される。
  double oracle とビット一致はしない。**golden は Transformers 出力を正とし、
  double oracle を別参照として併記する**（手順書 2.1 の方針どおり）。
- `attention_factor` の適用契約: Transformers の各モデル実装は
  `cos = emb.cos() * attention_scaling`, `sin = emb.sin() * attention_scaling`
  である（`modeling_rope_utils.py` の `rope_init_fn` が返す値を
  `attention_scaling` として利用）。角度を計算した後に cos/sin へ掛ける。

## 3. 環境（実測）

| 項目 | 値 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 × 4（GPU0-3、各 34,208,743,424 B ≒ 31.86 GiB）、GPU4 は 512 MiB の非対象デバイス |
| 現状の VRAM 使用 | GPU0-3 とも約 60 MB（アイドル、13:10 時点） |
| ROCm ツール | `rocm-smi` / `hipcc` あり（`/opt/zen/.venv/bin`） |
| Python | transformers 5.14.1、torch 2.13.0+rocm10.0.0 |
| build | `build/`（Release、`CMAKE_HIP_ARCHITECTURES=gfx1201`、`PHASESHIFT_HIP_GRAPH=OFF`） |

> 手順書は「R9700 32GiB 一枚」を前提にしているが、この環境は **32 GiB × 4 枚**である。
> TP=2/4 とカード分割による KV 分割の可否は Gate 4 以降で再確認する。

## 4. 現行実装の調査

### 4.1 RoPE 計算経路（2026-10-23 コード確認）

| 経路 | 周波数の出所 | 角度の型 | 備考 |
| --- | --- | --- | --- |
| Optimized pair | GPU の `inv_freq` double テーブル参照 | `static_cast<double>(pos) * double(freq)` + double sincos | 形状固定（head_dim=256, rotary=64, features%256==0） |
| Optimized generic | **theta から `pow` 直接計算**（`inv_freq` を受け取らない） | double | 行ごとに pow 再計算（rows×half 回） |
| Correctness | **theta から `pow` 直接計算**（テーブル参照なし） | double | 要素ごと pow |

- 逆周波数テーブル: `launch_rope_inv_freq_init`
  （`src/phaseshift/kernels/optimized/attention/rope.hip`）。GPU kernel で
  `inv_freq[i] = 1.0 / pow((double)theta, i/half)` を **double 精度**で生成。
  `half ≤ 32` 制約あり → **常に 32 double / 256 バイト固定**（rotary_dim=64 前提の
  暗黙結合）。
- 生成タイミング: `create_model_executor`
  （`src/phaseshift/models/qwen35/runtime/executor.hip`）。`host_model_state`
  構築（block_table / gdn / kv_* の設定）**より後**、`create` の1回のみ。
  `rope_theta ≤ 0` なら 1e7 にフォールバック。
- 伝播経路: GPU ポインタは `ModelDispatchStateView` ではなく
  **`HostExecutionContext.rope_inv_freq`** に設定される
  （`program_executor.h` の `build_host_execution_context`）。
  `ModelDispatchStateView` に RoPE フィールドは**存在しない**。
  Correctness の `exec_rope`（`kernels/correctness/detail/attention.inc`）は
  `env.qwen_state` を参照せず、binding（rotary_dim/group_size/scalar_b）と
  positions と workspace のみで動く。
- pair と generic の分岐: `resolve_rope_physical`
  （`src/phaseshift/inference/execution/rope_dispatch.hip`）。条件は
  `rope_inv_freq != nullptr && rope_pair_shape_supported(...)`。
  `rope_pair_shape_supported` は head_dim==256 / rotary==64 / features%256==0 /
  stride 偶数 / ptr アラインを要求。
- **selector が先に効く**: `rope_selector.cpp` が
  features ∈ {512, 1024, 3072, 4096, 6144}・head_dim=256・rotary=64・
  rows ∈ [1, 8192] のときだけ Optimized。外れると correctness に落ちる。
  features の許可値は TP1（Q6144/K1024）と TP2（Q3072/K512）に対応する値。
- `RopeF32Bf16PairArgs`（`rope.h`）: `sizeof == 56` / `alignof == 8` の
  static_assert あり。offsetof: input=0, input_row_stride=8, output=16,
  output_row_stride=24, positions=32, inv_freq=40, features=48。
  **padding は 12-15 / 28-31 / 52-55 の3箇所**（計画書が想定する offset 52 への
  float 追加は物理的には可能）。kernel は構造体を値渡しで受け、HIP Graph capture
  では値がそのまま焼き込まれる。
- positions は全経路 uint32、角度は Target 系（pair / generic / correctness）が
  すべて `double(position)`。**float(position) で角度を計算するのは
  DFlash2 / EAGLE4 Drafter 経路のみ**（`dflash2/detail/rope_device.h`、
  `eagle4/ops.hip`、inv_freq も float の powf）。
- lowering: Target Q/K の `RoPENode` は共通 Qwen Hybrid lowering
  （`qwen_hybrid/attention_lowering.h` の `emit_gated_attention`）。
  `rotary_dim = (uint32_t)((double)head_dim * partial)`、`rope_theta ≤ 0` なら 1e7、
  `partial ≤ 0` なら 0.25 フォールバック（`lower_to_primitives.cpp`）。
  TP>1 では local heads に差し替えてから geometry を作る → **RoPE の features は
  local heads 分**になる。node→binding は `program.cpp` で
  theta → `b.scalar_b`、head_dim → `b.group_size`。
- MTP: `lower_qwen35_mtp_to_primitives` に同じ theta/partial/rotary 計算と
  Q/K の RoPENode があるが、**呼び出しは tests のみで production 経路に接続されて
  いない**（現状 MTP program は実行されない）。
- HIP Graph: `PHASESHIFT_HIP_GRAPH` は cmake デフォルト OFF、現ビルドも OFF。
  さらに `graph_eligible` は **`tp_schedules.empty()` を要求** → **TP 設定時は
  graph が使われない**。inv_freq は create で確定し capture より前、input/output は
  workspace or 固定テンソル、`ctx.row_positions` は arena 固定テンソル。
  毎バッチ変わる `rope_positions`（spec tree）は graph 対象外。

### 4.2 現行実装で判明した計画外のリスク・事実

1. **TP=4 では RoPE が常時 Correctness に落ちる**: この tp4 チェックアウトの
   Qwen3.8-27B（Q24/K4/head_dim256/partial0.25）で TP=4 すると local heads は
   Q=6/K=1 → features は Q=1536/K=256 となり selector の許可値
   （`rope_selector.cpp` の登録表: features ∈ {512, 1024, 3072, 4096, 6144}、
   head_dim=256、rotary=64 固定。行数上限は `selector_limits.h` の
   `kOptimizedMaxRows = 8192`）に該当しない。optimized pair/generic は実行されない
   可能性が高い（実測は未確認）。**ただし現行 CLI は `--tp` を 1|2 のみ受理するため、
   TP4 は CLI からは起動できない**（divisibility 検証の
   `make_qwen35_tensor_parallel_context` のみ tp4 を通す）。TP2 の features は
   Q=3072/K=512 で登録値に一致する。
2. **Correctness 経路の positions が tree 位置を参照できない**:
   `DeviceBatchContext::rope_positions`
   （`include/phaseshift/runtime/batch/device_batch_context.h:62`、定義のみで
   初期値 nullptr）は、**src/ 全体で代入箇所が存在しない**ことを grep で確認した
   （書かれているのは `HostExecutionContext.rope_positions`
   （`program_executor.h:51`）と DFlash2 の tree 位置生成のみ）。
   correctness の `exec_rope` は `env.context->rope_positions`（= device 側、常に
   nullptr）を見るため `row_positions`（連番）にフォールバックする。
   一方 optimized は `HostExecutionContext.rope_positions`（spec tree 位置）を使う。
   → spec verify で optimized が外れた場合、tree 位置ではなく連番位置で RoPE される
   **不整合の可能性**。YaRN 導入時に同じ経路を統一する際には解消対象。
3. generic は `correctness_fallback` としても計上される一方、
   `record_physical_launch` / `record_optimized_op` は variant を問わず加算。
   統計の解釈に注意。
4. `docs/developer/qwen35.md` は inv_freq テーブルを「Executor / MtpExecutor が
   生成」と書くが、**`MtpExecutor` はコードに存在しない**（生成は
   `create_model_executor` のみ）。docs の誤り（ Gate 後に docs/developer を更新）。
5. `launch_rope_f32_bf16`（pair/generic 自動分岐の統合 entry）は production 未使用。
6. TP 経路の `row_positions`（`prefix_length + t`）は uint32 加算。100万トークン域では
   uint32 の範囲内だが、将来の上限検証はこの加算順序も対象。

### 4.3 設定・起動オプションの伝播経路（2026-10-23 コード確認）

**entrypoint の構成**: ユーザー向け entrypoint は `phaseshift-compute` /
`phaseshift-launcher` / `phaseshift-server` / `phaseshift-bench` の4つ。
`phaseshift-launcher`（Python）は `--model-dir` / `--model-host` / `--device` /
`--tp`（choices 1|2）を独占（`RESERVED_FLAGS`）し、他は `parse_known_args` で
子へ透過転送する。**compute のオプション定義は `compute_cli.{h,cpp}` に残存**。
`phaseshift-model-host` は launcher が rank 数だけ起動する非 user-facing プロセス。

**モデル設定の読み取り**:

- `read_qwen35_text_config()`（`qwen35_config.cpp`）が読む rope フィールドは
  `rope_theta` と `partial_rotary_factor` の**2つだけ**。
  `rope_type` / `factor` / `original_max_position_embeddings` / `beta_*` /
  `truncate` / `attention_factor` / mrope 系は**読まず、検証もしない**
  （yarn でも黙って default 扱い）。フィールド自体が `Qwen35TextConfig` に無い。
  検証は `vocab_size` / `hidden_size` / `num_hidden_layers` のゼロ検査のみ。
  `partial_rotary_factor` の値域検査（0 < f ≤ 1）は lower 時
  （`qwen_hybrid/attention_geometry.h`）。
- **非対称**: Target は rope_type を無視する一方、DFlash2 / EAGLE4 は
  非 `default` を `invalid_argument` で**明示拒否**
  （`dflash2/config.cpp` / `eagle4/config.cpp`、
  `test_dflash2_config.cpp` が "yarn" 拒否を固定）。Drafter は
  `rope_theta` のみ保持（partial / L なし）。

**Resident 経路**:

- archive が直列化する `Qwen35TextConfig` は 21 フィールド（vocab_size から
  output_gate_silu、`rope_theta` と `partial_rotary_factor` を含む）。
  **新規 rope フィールドは archive に無い**。
- `kManifestVersion = 1`、archive は host プロセスのメモリ内のみ（ディスク永続化
  なし）。末尾バイト数の trailing bytes 検査が不整合の唯一の安全網。
- **config.json を読むのは host プロセスだけ**: launcher は読まず（dir 確認のみ）、
  model-host の warmup が `read_qwen35_text_config` して archive 化、
  **compute 子は archive から復元し config.json を読み直さない**
  （`ModelSource::create` → `acquire_resident_model` → `build_qwen35_model`）。
  host なしの local load のみ compute 側で読む。
- host の Acquire キャッシュ fingerprint は (model_dir, kind, load_flags, tp, device)
  のみで **config.json の内容を含まない**（ロングランニング host で設定変更が
  検出不能。launcher は毎回新規 host のため実害は限定的）。

**TP**:

- `TpCoordinatorConfig` / `TpRankRuntimeConfig`（`tensor_parallel.h`）。
  CLI → `compute_runtime.hip`（1|2 検証）→ `initialize_tensor_parallel`
  （devices は `{device, device+1}` ハードコード）→ `TpCoordinator::initialize`
  で単一設定から rank ごとに複製。**rank 間の text_config 比較・不一致検知は
  存在しないが、単一プロセス内で同一設定から複製されるため不一致は構造上発生しない**。
- TP4 は launcher argparse / compute parse / runtime initialize /
  TpCoordinator の**4段階すべてで拒否**（`tensor_parallel_context.cpp` の
  divisibility 検証のみ tp4 を通すが実行層は全て拒否）。
- KV head 分割は `local_kv_heads = num_key_value_heads / tp_size`
  （割り切れなければ unsupported）。各 rank が local_kv_heads で
  独自 `PagedKVPool` を保持。
- **TP 経路には KV 予算の事前チェックが無い**（weights 超過検査のみ）。
  さらに **TP 分岐には working-set / prefix の自動縮小処理が無い**
  （単一 GPU 経路の縮小ロジックは tp==1 の initialize 内のみ）。

**PrefixCache**: キーは**トークン列のみ**の radix tree。checkpoint 識別子・
rope パラメータ・dtype はキーに含まない。プロセスローカル（GPU hot + cold +
pinned host）、永続化なし。起動中 factor 固定なら実害はないが、factor を変える
手段が今後追加される場合・永続化を足す場合は**キーへの rope 識別子追加が必要**。

**Server**: `build_compute_argv` が `--max-seq-len`（既定 **32768**、compute 単体
既定 512 と異なる）/ `--kv-cache-dtype` / `--tp`（>1 のとき）/ `--model-host` 等を
compute へ転送。**`--prefix-cache` 系は転送しない**（compute 既定が使われる）。
入力長検証は `chat_service.py`（`context length exceeded` + max_new_tokens クランプ）と
stdio レイヤの二重防御。YaRN factor オプションは server / launcher / compute の
いずれにも**存在しない**（完全に新規）。

### 4.4 手順書との差分・要補正点

1. **MTP は production 未接続**: `lower_qwen35_mtp_to_primitives` の呼び出しは
   tests のみ。手順書の完了定義 2「MTP経路も整合する」は、現状では
   **lowering テストの範囲**（RoPENode の rotary_dim/theta 検証）に限定して
   定義し直す必要がある。
2. **HIP Graph は TP 設定と排他**（`graph_eligible` が `tp_schedules.empty()` を
   要求）。現ビルドは OFF。Gate 3 の「Graph OFF/ON 検証」は **TP=1 でのみ有意味**。
3. **手順書 7.3 の使用例が `--kv-cache-dtype fp8` を使う**が、
   `docs/perf/current.md` に FP8 KV 品質ゲート FAIL（ppl +28〜30%、2026-10-20）の
   実績がある。方針は §7.1-6 に更新された（長文の載せ台は FP8 KV + TP2、
   YaRN 品質比較は BF16 KV）。
4. **`--tp` は 4 段階の拒否経路**。手順書の「TP=1/2」前提は正しいが、
   TP4 拡張には launcher / runtime / coordinator の全面変更が必要。
5. rank 間の RoPE 設定不一致 fail-fast（手順書 Gate 1）は、単一プロセス内
   複製という構造上、**設定伝播を TpCoordinatorConfig 経由に限定すれば
   事実上保証される**。追加の検証コードは最小限でよい。
6. `docs/developer/qwen35.md` の「inv_freq は Executor / MtpExecutor が生成」は
   誤り（`MtpExecutor` は存在しない）。Gate 1 で docs を修正する。
7. target 側 config の rope_type 無視は「黙って default 扱い」ではなく、
   YaRN 対応時には**未対応 rope_type の明示的な失敗**へ変更する（手順書 §2 の方針）。

## 5. KV 予算の事前計算（実装確認）

計算の正本は `PagedKVPool::page_bytes()`
（`include/phaseshift/inference/state/paged_kv_pool.h`、
`src/phaseshift/inference/state/paged_kv_pool.cpp`）:

- `layer_page = page_tokens × kv_heads × head_dim`
- `body = 2（K/V）× num_attention_layers × layer_page`
- BF16: `body × 2` バイト（scale なし）
- FP8 E4M3: `body × 1` ＋ K/V 各々の per-(layer, page, token, head) FP32 scale
  （`2 × num_attention_layers × page_tokens × kv_heads × 4` バイト）

Qwen3.8-27B（full attention 16層、kv head 4、head dim 256、page_tokens 既定 16）:

| dtype | 1ページ | 1トークン換算 |
| --- | --- | --- |
| BF16 | 1,048,576 B = 1 MiB | 64 KiB/token |
| FP8 | 532,480 B = 520 KiB | 32 KiB + 512 B/token |

文脈長ごとの **KV 本体のみ**の下限（リクエスト1件、ideal、ページ切り上げ前）:

| コンテキスト長 | BF16 | FP8（scale込み） |
| --: | --: | --: |
| 262,144 | 16.000 GiB | 8.125 GiB |
| 524,288 | 32.000 GiB | 16.250 GiB |
| 1,000,000 | 61.035 GiB | 30.994 GiB |
| 1,048,576（演算境界参考） | 64.000 GiB | 32.500 GiB |

（手順書第6節の表と一致することを式から再確認した。）

TP2 では kv_heads が半分（4→2）になるため **per-rank は上表の半分**、合計は同じ:

| コンテキスト長 | TP2 BF16 per-rank | TP2 FP8 per-rank | per-rank 合計（FP8 + 重み ~12.6 GiB 概算） |
| --: | --: | --: | --: |
| 524,288 | 16.001 GiB | 8.125 GiB | ~20.7 GiB（31.86 GiB 枚に収まる） |
| 1,000,000 | 30.519 GiB | 15.498 GiB | ~28.1 GiB（手一杯、成立を前提にしない） |

起動時のチェック（`src/apps/compute/compute_runtime.hip`）:

- `--max-seq-len` 既定 **512**（`compute_cli.h`、`compute_runtime.h`）、
  `--page-tokens` 既定 16、`--kv-cache-capacity-tokens` 既定 0、`--arena-gib` 既定は auto。
- `required_pages = ceil_div(max_seq_len + 1, page_tokens) + 1`（`compute_runtime.hip`）。
  `max_seq + 1u` は **uint32 のまま**（2^32-1 近傍でのみオーバーフローしうる）。
  既定 512 のとき required_pages = 34（BF16 で 34 MiB/rank）。
- 不足時は起動前に拒否する。auto arena 時は
  `"free VRAM after weights cannot hold the required KV and workspace ..."`、
  明示 capacity 時は `capacity < max_seq_len+1` を invalid_argument、
  ページ不足は GiB 換算付き `insufficient_memory`（`compute_runtime.hip`）。
- **TP 経路には単一 GPU 経路那样的 KV 予算の事前チェックがない**
  （`src/phaseshift/models/qwen35/runtime/tensor_parallel.cpp` の
  `TpRankRuntime::create`）。arena 割り当て失敗がそのまま伝播する。
  → TP での容量拒否メッセージは Gate 4 の対象。
- TP の KV head 分割は実装済み: `local_kv_heads = num_key_value_heads / tp_size`
  （割り切れなければ unsupported で拒否、`tensor_parallel_context.cpp`）。
  TP2 で kv_heads=2 → 容量は半分。
- **CLI の `--tp` は 1|2 のみ受理**（`compute_cli.cpp`、
  `"bad --tp (expected 1 or 2)"`）。context 層は 4 まで対応するが CLI は拒否する。
- server 経由の `--max-seq-len` 既定は **32768**（`docs/user/server.md`）で、
  compute 単体の既定 512 と異なる。起動ログでの確認対象。

> 実測時の注意: `docs/perf/current.md` に **fp8 KV は品質ゲート FAIL 実績
> （ppl +28〜30%）**の注記がある。§7.1-6 のとおり、FP8 KV は長さを取るための
> 載せ台として扱い、YaRN の品質比較は BF16 KV で行う。

## 6. テスト基盤・対象テスト一覧（実装確認）

- required の正本: `cmake/tests.cmake` の `required` label と
  `phaseshift-required-tests` カスタムターゲット。現行 build で
  `ctest -L required -N` = **133 件**、`ctest -L cpu -N` = **73 件**。
- `phaseshift_add_test` は `SKIP_RETURN_CODE 77`。`tests/run_required_acceptance.py`
  は JUnit の `<skipped>` を検出すると **exit 1（失敗扱い）**。
  AGENTS.md のテスト受入ルールと一致。
- RoPE 関連 CTest:
  - `test_rope`（`gpu1;required`、`tests/kernels/optimized/test_rope.hip`）
    — correctness 経路と optimized を同一入力で比較（rel > 2e-2 で fail）、
    inv_freq テーブルとインライン式の bit 一致、generic と pair の bit-exact 比較。
    **golden ファイルではなく相互比較**、GPU 無しは exit 77。
  - `test_dflash2_rope`（`gpu1;required`）— CPU 参照と比較（rel_l2 ≤ 1e-3）。
  - `test_qwen35_rope_selector`（`cpu;required`、GPU 不要）— selector ルール固定。
  - `test_bench_help_rope`（`cpu;required`、bench help smoke）。
  - 関連: `test_qwen35_mtp_lowering` / `test_qwen35_lowering_contract`
    （`gpu1;required`、RoPENode の rotary_dim=head_dim/4 と theta を検証）、
    `test_eagle4_forward`（`gpu1;required`、内部に CPU 参照 RoPE）。
- MTP / TP / prefix / server / resident:
  - MTP: `test_qwen35_mtp_weight_load` / `test_qwen35_mtp_lowering`（required）、
    `test_qwen35_mtp_weight_real`（optional;external_files）。
  - TP: cpu=plan/context/reduction（required）、`test_qwen35_tp_weight_load`
    （gpu1;required）、`test_tp_transport` / `test_qwen35_tp_execution`（gpu2;optional）、
    `test_qwen35_tp_e2e` / `test_dflash2_tp_e2e`（gpu2;optional;external_files）。
  - prefix: `test_prefix_cache` / `test_prefix_cache_batcher`（gpu1;required）。
    server 側は `tests/server/test_compute_prefix_contract.py`
    （`run_server_regression.py` 経由、PHASESHIFT_MODEL_DIR 必須）。
  - resident: `test_resident_model_ipc` / `test_resident_timeout_worker`
    （optional;external_files;resident、実行中 model-host が必要）。
  - optional 一式は `PHASESHIFT_BUILD_OPTIONAL_TESTS=ON` のときのみ build。
    **現行 build は OFF**。
- RoPE 専用 golden は `tests/data/` / `tests/fixtures/` に**存在しない**
  （現状は相互比較と CPU 参照）。YaRN の Transformers golden は新規用意が必要。
- CLI 文書同期: `tools/gen_cli_docs.py`（`--help` から docs/user の
  `<!-- cli:name -->` マーカー間を更新、`--check` で検査）。
  CTest: `test_cli_docs_sync`（cpu;required、cpp のみ）。
- 性能正本: `docs/perf/methodology.md`（5-run p50、run 間 30% 超なら FAIL 再調査、
  E2E と micro の混同禁止、未計測値の更新禁止）。
  `docs/perf/current.md` の再計測済み値は HF FP8 Prefill（TP2/pp2048: 538.67 ms、
  3803 tok/s）と HF FP8 decode（2026-10-23: TP1 非投機 19.10、TP2 非投機 31.15、
  TP2+DFlash2 67.16、TP2+EAGLE4 62.36 tok/s）。**BF16 は未計測**。
  → Gate 0 の Native baseline はこの正本の値を引用し、再計測はしない。

## 7. Gate 0 判定

**判定: ACCEPT（Gate 1 に進んでよい）**

数式・標準値・Runtime の設定責務・Resident 形式の採用案が決定し、未確定点は
下のチケットへ落とした。Native 既定の変更を伴う変更は Gate 1 以降で行う。

### 7.1 決定事項

1. **数式**: 手順書 2.1 の YaRN 計算式を採用する。Transformers 5.14.1 の
   `_compute_yarn_parameters` と一致することを実装ソースで確認した（§2）。
   `low=14, high=22`（`d=64, theta=1e7, L=262144, beta_fast=32, beta_slow=1,
   truncate=true`）。
2. **標準値**: `attention_factor` は factor=2 で 1.0693147180559945、
   factor=4 で 1.138629436111989。32 要素の inv_freq golden を
   Transformers 5.14.1 の float32 出力から固定した（§2 の表）。
   golden は double oracle と混同せず、**Transformers 出力を正**とする。
3. **Runtime の設定責務**: 手順書 3.1 の `EffectiveRopeProfile` を採用する。
   Target Executor 生成時に確定し shutdown まで不変。伝播は
   `HostExecutionContext.rope_inv_freq` 経由（`ModelDispatchStateView` ではない）
   と、現状の実装に合わせる。32 double のテーブルは起動時1回だけ生成する
   既存の `Executor::rope_inv_freq` をそのまま使う。
4. **Resident 形式**: **案B（実効 RoPE メタデータを Resident バイナリとは独立に
   target runtime で解決する）を採用する**。根拠:
   (a) archive は `kManifestVersion=1`・メモリ内のみ・同一バイナリ間で、
   writer/reader の同時更新漏れ時に値が既定値へ化ける失敗モードがあり、
   (b) config.json を読むのは model-host プロセス1回だけで、compute は archive
   から復元するため、`Qwen35TextConfig` へフィールドを足すと
   **local load と resident load で挙動が乖離しやすい**（§4.3）。
   方針: checkpoint 由来の YaRN メタデータは checkpoint から解決した
   `EffectiveRopeProfile` を正とし、Resident 経路でも同一の解決結果になることを
   **Resident/local 起動比較テストで固定する**。`Qwen35TextConfig` への
   フィールド追加は行わない。将来 archive に含める場合は案Aとして
   manifest バージョンを上げ直す。
5. **CLI**: `--rope-factor auto|1|2|4`（既定 `auto`）を `compute_cli.{h,cpp}` に
   新設する。launcher は `parse_known_args` で透過転送するため、
   `RESERVED_FLAGS` に含めない。server は `build_compute_argv` へ転送を追加する。
   `max_seq_len` 上限: factor=2 は 524,288、factor=4 は **1,000,000**
   （Qwen 公式の案内値。1,048,576 は演算境界であり運用値にしない）。
6. **KV dtype と長文実行の載せ台**: 長コンテキスト実行（Gate 6）は
   **FP8 KV + TP2** を第一候補とする（2026-10-23 ユーザ判断。
   「fp8 の KV で tp=2 であれば 262144×2 = 524,288 トークンは取れた」）。
   容量計算とも整合する: TP2/FP8 で 524,288 トークンは **per-rank 8.125 GiB**
   （重み ~12.6 GiB/rank と併せて ~21 GiB/rank、31.86 GiB 枚に収まる）。
   BF16 KV は同条件で per-rank 16 GiB（合計 ~29 GiB/rank）と手一杯。
   一方 `docs/perf/current.md` に **FP8 KV は品質ゲート FAIL 実績（ppl +28〜30%）**
   があるため、**YaRN の効果測定（Native vs 2x の品質比較）は BF16 KV で行い、
   FP8 KV は「長さを取るための載せ台」として区別して記録する**。
7. **Drafter**: Target YaRN 対応時に DFlash2 / EAGLE4 は**変更しない**。
   2x/4x 起動時の speculative は明示的に reject する（drafter の rope_type 拒否は
   既存実装にあるが、Target の factor と独立であることを起動ログで明示する）。
8. **x4 (1,000,000) は検査なし・実装のみ**（2026-10-23 ユーザ判断）。
   4x 相当の長文 E2E（1M 実行・品質・性能）は**実施しない**。
   4x の正しさは Gate 2/3 の範囲（32 周波数 golden、RoPE 単体の GPU 照合、
   起動時解決・エラー契約）で担保し、それ以上は行わない。
   長文 E2E の検証対象は **2x（524,288）に限定**する。
   1M は FP8 KV + TP2 でも per-rank ~15.5 GiB + 重み ~12.6 GiB ≒ 28 GiB と
   31.86 GiB 枚に対して手一杯で、成立を前提にしない。

### 7.2 Gate 0 で確定した現行実装の制約（Gate 1-4 が継承する事実）

- Target の3経路（pair / generic / Correctness）で周波数の出所が異なる
  （§4.1）。Gate 3 の統一対象。
- Correctness 経路は tree positions を参照できない（§4.2-2）。統一時に解消。
- HIP Graph は TP 設定と排他、現ビルド OFF（§4.4-2）。Graph 検証は TP=1。
- TP 経路に KV 予算の事前チェックと working-set/prefix 自動縮小が無い（§4.3）。
- PrefixCache キーに rope 識別子が無い（§4.3）。起動時固定なら現状で安全。
- `--max-seq-len` 既定は compute 512 / server 32768（§4.3）。

### 7.3 未確定点（チケット化）

| ID | 内容 | 担当 Gate |
| --- | --- | --- |
| T-1 | target config の rope_type 無視を「未対応なら失敗」へ変更する契約の設計（現状は黙殺） | Gate 1 |
| T-2 | resident/local 起動比較テストの設計（案B の固定手段） | Gate 1 / 4 |
| T-3 | TP 経路の KV 不足時のエラーメッセージ未採取。TP 時の容量拒否は事前チェック無しで伝播 | Gate 4 |
| T-4 | server 経由（--max-seq-len 32768 既定、prefix cache 系を転送しない）での factor 伝播検証 | Gate 4 |
| T-5 | host の Acquire fingerprint が config.json 内容を含まない（ロングランニング host の陳腐化） | Gate 4（現状は launcher 毎回新規のため限定的） |
| T-6 | `docs/developer/qwen35.md` の `MtpExecutor` 記述が誤り。Gate 1 で docs を修正 | Gate 1 |
| T-7 | MTP lowering は production 未接続。「MTP 経路整合」の完了定義を lowering テスト範囲に限定 | Gate 3 |
| T-8 | 100万トークン域の `max_seq_len + 1` の uint32 加算と `row_positions` の uint32 加算順序の上限検証 | Gate 4 |

### 7.4 Gate 0 で実施しなかったこと（記録）

- Native の短文品質・RoPE 単体時間・Decode tok/s の**再計測は未実施**。
  `docs/perf/current.md` の再計測済み値（HF FP8 Prefill / decode）を baseline として
  引用した。BF16 の値は正本に未計測のまま存在する。
- 2x の長文 E2E 実行（524,288）は Gate 6。RoPE 単体の GPU 検証は Gate 2/3。
- **4x の 1M E2E は実施しない**（§7.1-8、2026-10-23 ユーザ判断）。
  4x は Gate 2/3 の RoPE 単体検証と起動経路の実装・エラー契約まで。
- 2x の 524,288 は FP8 KV + TP2 で容量成立を見込む（§7.1-6）。
  BF16 KV + TP2 は ~29 GiB/rank で成立を前提にしない。

## 8. Gate 1〜3 実施記録（2026-10-23）

作業ブランチ: `feat/yarn-rope-factor-gate1`（base `a71a83e`）。対象は Target の text 推論経路のみ。

### 8.1 実装変更

| パス | 変更 |
| --- | --- |
| `include/phaseshift/models/qwen35/model/rope_profile.h`（新規） | `Qwen35RopeParameters` / `EffectiveRopeProfile` / 解決関数 / 上限関数 |
| `src/phaseshift/models/qwen35/model/rope_profile.cpp`（新規） | `config.json` の検証付き読み取りと profile 解決 |
| `src/apps/compute/compute_cli.{h,cpp}` | `--rope-factor auto\|1\|2\|4` の解析と help |
| `src/apps/compute/main.hip` | 起動時の解決・`--max-seq-len` 上限拒否・speculative 拒否・ROPE profile ログ |
| `src/apps/compute/compute_runtime.{h,hip}` | `EffectiveRopeProfile` の保持と executor / TP への伝播 |
| `src/apps/server/phaseshift_server.py` | `--rope-factor` の argparse と compute への転送 |
| `include/phaseshift/runtime/graph/primitive_node.h` | `RoPENode.attention_factor` |
| `src/phaseshift/runtime/program/program.cpp` | `attention_factor` → `DispatchBinding.scalar_a` |
| `include/phaseshift/models/common/qwen_hybrid/attention_lowering.h` | geometry の `rope_attention_factor` を Q/K 両方へ |
| `src/phaseshift/models/qwen35/model/lower_to_primitives.cpp` | profile を geometry / MTP RoPENode へ伝播 |
| `include/phaseshift/kernels/optimized/attention/rope.h` | `attention_factor`（offset 52、sizeof 56 維持）、`launch_rope_inv_freq_yarn_init`、generic/pair の引数追加、`kRopeInvFreqTable` |
| `src/phaseshift/kernels/optimized/attention/rope.hip` | YaRN table 生成 kernel、generic の table 参照、pair/generic の cos/sin 係数適用 |
| `src/phaseshift/inference/execution/rope_dispatch.hip` | `attention_factor` を `scalar_a` から取得して両 variant へ |
| `include/phaseshift/inference/execution/physical_launch.h` | `PhysicalRopeLaunch.attention_factor` |
| `include/phaseshift/runtime/batch/device_batch_context.h` / `src/.../device_batch_context.hip` | `DeviceBatchContext.rope_inv_freq` |
| `src/phaseshift/models/qwen35/runtime/executor.hip` | YaRN 時の table 初期化、lowering への profile 伝播、batch context への table 設定 |
| `src/phaseshift/kernels/correctness/detail/attention.inc` | correctness の RoPE を table + `scalar_a` 参照へ（pow 直計算は table 未設定時のみ） |
| `include/phaseshift/models/qwen35/runtime/executor.h` | `ExecutorConfig.rope_profile` |
| `include/phaseshift/models/qwen35/runtime/tensor_parallel.h` / `src/.../tensor_parallel.cpp` | TP の coordinator / rank 設定へ profile |
| `tests/unit/test_qwen35_rope_profile.cpp`（新規） | 解決優先順位・異常系の CPU テスト |
| `tests/kernels/optimized/test_rope_yarn.hip`（新規） | GPU の 32 周波数を Transformers golden と照合 |
| `cmake/tests.cmake` | 上記テスト登録、CLI 契約テスト 3 件 |
| `docs/developer/qwen35.md` | §2.5 に current contract、`MtpExecutor` 記述の誤りを修正 |
| `docs/user/{compute,server}.md` | `--rope-factor` 行（`tools/gen_cli_docs.py` で生成） |

### 8.2 検証結果

| 検証 | 結果 |
| --- | --- |
| `test_qwen35_rope_profile`（cpu;required） | PASS |
| `test_compute_rope_factor_*`（cpu;required、3件） | PASS |
| `test_rope_yarn`（gpu1;required） | PASS — 32 周波数が Transformers 5.14.1 golden と最大相対誤差 1e-6 以内、Native は double 恒等式と 3.0e-16 |
| `test_rope`（gpu1;required） | PASS — correctness に inv_freq table を渡した状態でも optimized と bit 一致（39/40 ケース → 全 40 PASS） |
| `run_required_acceptance.py` | **138 件 PASS / 0 fail / 0 skip**（ACCEPTANCE PASSED） |
| 実機 `auto` と `--rope-factor 1` | `GENERATED_IDS` が完全一致（Native 回帰なし） |
| 実機 `--rope-factor 2` / `4` | 起動・生成とも成功。factor 2 の `attention_factor=1.069314718`、factor 4 は 1.138629436 |
| `--max-seq-len 1000001` + factor 4 | 起動拒否（`exceeds the rope factor limit 1000000`） |
| factor 2 + `--dflash2-model-dir` | 起動拒否（static YaRN は speculative 非対応） |

### 8.3 既知の制限・未実施

- 実機確認は `--max-seq-len 256` の短文のみ。524,288 / 1,000,000 の長文 E2E は未実施（Gate 6。4x は方針により実施しない）。
- `test_rope` の full-rotary（rot=256）ケースでは table を使わず theta から計算する（table 長 32 の前提）。rotary_dim=64 の本番 geometry では影響しない。
- correctness 側の `rope_positions`（spec tree 位置）は依然として未接続。optimized と correctness で tree 位置が食い違う既存問題は未解消（今回の変更で table/係数は一致）。
- gate 3 の「Graph OFF/ON」検証は未実施（build は `PHASESHIFT_HIP_GRAPH=OFF`、かつ TP 時は graph 無効）。
- MTP lowering は profile を受ける引数を追加したが、production 経路に接続されていない点は変わらない（lowering テストのみ）。

### 8.4 採否

ACCEPT。Target text 経路の Native / YaRN 2x / YaRN 4x が同一の table と係数を参照し、Native は bit 一致で回帰なし、YaRN の 32 周波数は Transformers golden と一致した。

## 9. Gate 4 実施記録（容量・TP・Resident・PrefixCache・Server）

作業は main 上のマージ後に実施。対象は Target text 経路。

### 9.1 追加したテスト

| テスト | 内容 |
| --- | --- |
| `test_compute_rope_cap_native_over` | `--rope-factor 1 --max-seq-len 262145` → 上限 262144 で拒否 |
| `test_compute_rope_cap_2x_over` | `--rope-factor 2 --max-seq-len 524289` → 上限 524288 で拒否 |
| `test_compute_rope_cap_4x_over` | `--rope-factor 4 --max-seq-len 1000001` → 上限 1000000 で拒否 |

いずれも `cpu;required`。上限判定は GPU 取得・モデルロードより前に走る。fixture は
`tests/fixtures/rope_cap/config.json`（rope パラメータのみを持つ最小設定）。

### 9.2 実測結果

| 検証 | 結果 |
| --- | --- |
| 容量の独立性 | Native と `--rope-factor 2` で `--max-seq-len 4096` の `MEMORY_PLAN` が**完全一致**（KV budget 0.25 GiB / 4112 tokens、workspace 3.01 GiB、arena 30.78 GiB） |
| 物理 KV 不足（BF16/TP1/524288） | `free VRAM after weights cannot hold the required KV and workspace` で起動拒否 |
| **2x 全長（524,288）** | **`--rope-factor 2 --kv-cache-dtype fp8 --tp 2` で起動・生成成功**。rank あたり arena 30.63 GiB、`ARENA_USED_GB=23.652`、`used 23.11 GiB` |
| TP 経路の容量拒否 | TP2 + BF16 + 1,000,000 → `tp rank arena is too small: free 15.73 GiB, KV 30.52 GiB` で拒否（事前チェックあり） |
| Resident / local 同一性 | launcher + model-host 経由と local load で **ROPE profile 行が完全一致**、同一プロンプトの生成 ID が**完全一致** |
| PrefixCache（2x） | 保存は「prefill 位置が page_tokens の倍数」のときのみ発生。復元は「キャッシュより長いプロンプト」で発生。miss（restored=0）と hit（restored=320）で**生成 ID が完全一致** |
| Server 伝播 | `phaseshift-server --rope-factor 2` の子 compute ログに `ROPE profile: type=yarn factor=2 ... source=cli` を確認 |

### 9.3 判明した仕様（docs へ反映する契約）

- `--rope-factor` は KV 容量計画に一切影響しない（`required_pages` / `kv_page_bytes` は factor 非依存）。
- TP 経路の KV 予算チェックは main の `95a9ba4` で追加済みで、rank 単位で拒否メッセージを出す（旧チケット T-3 は解消）。
- PrefixCache はプロセスローカルで、キーはトークン列のみ。factor はプロセス起動時に固定されるため、
  異なる factor 間で KV が流用される経路は存在しない。
- prefix 保存は page 境界でのみ発生し、復元は保存済み prefix より長いプロンプトに限る。

### 9.4 残課題

- 長文 E2E の品質・性能測定は Gate 6（2x のみ。4x は方針により実装のみ）。
- 2x 全長は FP8 KV で容量が成立したが、FP8 KV の品質劣化（既知の FAIL）を含むため、
  品質評価は BF16 KV の短文比較と分離して行う。

### 9.5 採否

ACCEPT。容量・TP・Resident・PrefixCache・Server の各経路で、factor の解決結果が一致し、
容量拒否が安全に働くことを実測で確認した。
