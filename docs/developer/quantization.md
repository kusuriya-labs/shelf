# Quantization

source of truth: `include/shelf/quantization/**`,
`src/shelf/quantization/**`, `include/shelf/weights/**`,
`src/shelf/weights/**`, `src/shelf/io/**`。

canonical payload の stream / metadata / size formula / native preshuffle は
[canonical_soa_payload.md](canonical_soa_payload.md) を正本とする。
本ドキュメントは format の概要と runtime の経路を示す。

N-gram / Engram / PLE table 向けの別形式の契約は
[ngram_quantization.md](ngram_quantization.md) を参照。Qwen3.8-Flash-Next の PLE は
同形式（`ngram-q`）を直接ロードする。

実行時ランタイムは量子化 artifact の読込専用であり、HF の FP8 / Quark MXFP4 / EXL3 checkpoint および
self-contained quantized safetensors を直接ロードする（load 時の `lm_head` の INT8 W32 変換を除く）。
EXL3-INT8（mul24）および Ngram4 / Ngram5 のオフライン変換・検査ツールは
`tools/quantize/shelf_quantize.py`・`tools/quantize/ngram_quantize.py` として提供する
（利用手順は [../user/quantize.md](../user/quantize.md) を参照）。

## フォーマット

- `QuantSpecId` / `ComputeSpecId` — 量子化仕様ID（`quantization/quantization_types.h`）。
- `QuantFormatId` — weight storage / compute formatのID（`quantization/quant_format.h`）。
  numeric valueはin-memory専用で、file / manifestには保存しない。
  on-diskのencodingは `shelf_quantization.json` の文字列名で管理する。
- 対応 encoding: BF16 / FP8 K128N128 / MXFP4 E2M1 + E8M0 group 32 /
  EXL3-INT8 trellis（mul24）。
  `manifest/quantized_manifest.h` の `QuantizedEncoding` が唯一の on-disk encoding enum である。
  各formatの canonical stream、metadata、preshuffle は
  [canonical_soa_payload.md](canonical_soa_payload.md) を参照。
- self-contained quantized safetensors — `shelf_quantization.json` manifest 付きの
  quantized safetensors。`shelf-compute` が直接ロード可能。
  各tensorに `QuantSpecId` / `ComputeSpecId` / CRC32 payloadを保持する。
  manifest は V4（`format_version = 4`）で、tensors[x] は
  `domain` / `logical_shape` / `quant`（encoding + scale の `block_shape`）/
  `layout` / `payloads[]`（role ベース: codes / scale / data）で表現する。
  `metadata1..4` の位置依存を持たない。V3 manifest（`format_version = 3`）は
  互換 parser で読み込める。scale の block は既定（FP8 = 128x128）以外でも
  format 受理し、非対応 group は
  runtime の kernel capability で拒否する。
- codec（e4m3 / ue4m3）の reference math は `shelf_quant_reference`（pure CPU）。

## Weight loading（generic layer）

重みロードは model非依存の `shelf_weights`（`weights/weight_loader.h`）が担当する。

```text
Qwen35 model（tensor name -> Qwen35ModelWeights field の bindingのみ）
    ↓
shelf::weights loader
    safetensors / quantized safetensors の解決
    BF16 / FP8 の format解釈
    canonical payload validation / native preshuffle / GPU upload
    MatrixWeight（quant_spec / compute_spec 含む）の生成
    ↓
quant format / quant bundle / quant reference / io / core / gpu
```

- model側はencodingの分岐を持たない。
- tensor partition（weights 層）と TP execution は
  [tensor_partition.md](tensor_partition.md) / [multi-gpu.md](multi-gpu.md) を参照。
  canonical payload / preshuffle / quantization math は TP の影響を受けない。

### Tensor partition

`WeightLoadOptions.partition_plan` は load 時の logical TP partition を与える。
順序は常に

```text
global canonical → logical partition → rank-local canonical → preshuffle → upload
```

であり、`partition_plan == nullptr` では現行の global load path がそのまま走る
（追加の host copy なし）。manifest（`format_version = 4`。V3 読み込み互換を維持）は global canonical の
ままであり、TP degree を disk artifact へ焼き込まない。
詳細と validation は [tensor_partition.md](tensor_partition.md) を参照。
現行 Qwen35 の top-level load path（`load_qwen35_weights_from_*`）は
partition plan を受け付けない（multi-GPU lowering 未対応のため
`Status::unsupported` で停止する）。
- `tie_word_embeddings` は config.json の root → `text_config` の順に読み、
  どちらにも無ければ false とする（`read_qwen35_tie_word_embeddings` /
  `include/shelf/models/qwen35/model/qwen35_config.h`）。
  tie のとき `Qwen35ModelWeights.lm_head_tied` が true になり、`lm_head` は
  `embed_tokens` を参照する。
- BF16もquantized model内のweight formatの一種として同一経路でロードする
  （量子化されていないことを特別扱いしない）。
- `WeightLoadOptions.preshuffle` が load-time preshuffle のスイッチである。
  format別の native layout と `preshuffle = false` の制約は
  [canonical_soa_payload.md](canonical_soa_payload.md) を参照。

### HF 由来 real-quantized safetensors の直接ロード

`shelf_quantization.json` を持たない model dir のうち、`config.json` の
`quantization_config` が次の条件を満たす場合、Shelf は HF 由来の
real-quantized safetensors を manifest 非依存で直接ロードできる。

- `quant_method == "fp8"` — `fmt == "e4m3"` かつ `weight_block_size == [128,128]`
- `quant_method == "quark"` — `global_quant_config.weight` が
  `dtype == "fp4"` / `qscheme == "per_group"` / `group_size == 32` /
  `scale_format == "e8m0"` / `is_dynamic == false` かつ
  `export.weight_format == "real_quantized"`。
  これ以外（group ≠ 32、`scale_format` が e8m0 以外、fake quant export）は
  `Status::unsupported` で拒否する。
- `quant_method == "exl3"` — `<prefix>.trellis` と `<prefix>.mul1`（値 `0x00DCD12D`）を
  持つ linear を EXL3-INT8（mul24）として読む。`.mcg` や `0x00DCD12D` 以外の marker、
  half bit rate は `Status::unsupported` で拒否する。

合成は `quantization/bundle/hf_manifest.{h,cpp}` が行う。HF の tensor naming
（codes = `<name>.weight`、FP8 scales = `<name>.weight_scale_inv`（BF16）、
MXFP4 scales = `<name>.weight_scale`（U8 E8M0））から
`QuantizedManifest` を合成し、
[canonical_soa_payload.md](canonical_soa_payload.md) §6 と同じ
canonical payload contract へ流し込む。payload の bit 列は canonical 定義と一致するため、
preshuffle / runtime 経路は self-contained 量子化 model と同一である。

- FP8 の BF16 scale はロード時に F32 へ widen される（metadata1 は F32 contract のまま）。
- MXFP4 の scale は変換せず U8 E8M0 のまま metadata1 へ入れる（staging しない）。
  E8M0 byte 255（MX spec の NaN）を含む tensor は load 時に拒否する。
- MXFP4 の logical K は codes の 2 倍として復元する。K が 32 の倍数でない、または
  64 の倍数でない tensor（activation の 64 K fragment に載らない）は拒否する。
  同様に native の 16 行 tile に載らない rows は拒否する。
- EXL3 の trellis / suh / svh / marker は 4 つの payload plane として manifest に載る。
  dense 重みの trellis はロード時に 16×16 タイル単位で `[n/16, k/16]`（n-major）へ並べ替えて
  GPU へ転送し、MoE expert plane は `[k/16, n/16]` のまま転送する。codebook 復号と Hadamard は
  kernel 内で行う。
- 量子化されていない tensor（norm / embed / lm_head / dt_bias / A_log /
  conv1d.weight など）は BF16 entry として同一経路でロードされる。F16 で保存された
  unquantized tensor は load 時に BF16 へ widen する（EXL3 checkpoint の
  `linear_attn.in_proj_a/b` など）。
- HF payload の K が block size（FP8: 128）の倍数でない場合、
  canonical の padded K contract を満たせないため `Status::unsupported` で拒否する。
- 合成 manifest は `architecture = "qwen35_dense"` を立て、CRC32 は実データから計算される。
  `is_quantized_model_dir` がこの経路を検出し、
  `load_qwen35_weights_from_quantized_safetensors` が合成 manifest 経路へ入る。
- `architecture` の値域は **Shelf の loader family 名**である（現在 `qwen35_dense` と
  `dflash2_draft`）。loader は自分の expected 値と照合する（dense は
  `model_weights.cpp`、drafter は `dflash2/weights.cpp`）。config.json の `model_type` を
  写してはいけない。checkpoint の識別は `qwen_hybrid::check_config_identity` の gate が担う
  （[architecture.md](architecture.md) の Checkpoint識別 節）

### axis-slice ロード

`[E, N, K]` の rank-3 tensor から任意の unit 範囲 `[unit_begin, unit_begin + unit_count)`
だけを取り出して rank-2 の `MatrixWeight` としてロードできる。全体を VRAM に載せない
部分ロードのための API で、unit は tensor の最上位 axis である。

```cpp
Result<MatrixWeight> load_bf16_matrix_axis_slice(
    const SafetensorsCollection& collection, const std::string& name,
    std::uint64_t unit_begin, std::uint64_t unit_count,
    gpu::GpuArena& arena, hipStream_t stream);

Result<MatrixWeight> load_fp8_matrix_axis_slice(
    const SafetensorsCollection& collection, const std::string& codes_name,
    const std::string& scale_name, std::uint64_t unit_begin, std::uint64_t unit_count,
    gpu::GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options = {});
```

- BF16 は rank-3 BF16 tensor のみ受け付ける。結果は `rows = unit_count * N` /
  `cols = K` / `k_padded = K`。
- FP8 は codes が rank-3 の F8_E4M3 / U8、scale が BF16 rank-3
  `[E, ceil(N/128), K/128]`（`<codes_name>_scale_inv` naming）であることを要求する。
- span 解決は `SafetensorsCollection::axis_span` が行う。
- FP8 は slice 後の rank-2 canonical 経路（validation → preshuffle → upload）をそのまま使う。
- **FP8 の rank-3 (N-D) canonical partition / materialization は unsupported**
  （[tensor_partition.md](tensor_partition.md) §5）。
- HF manifest 合成経路（`synthesize_hf_quantized_source`）は rank-2 codes のみ受理する。
  rank-3 tensor はこの loader を直接使う。

現在の production model（Qwen3.8-27B Dense）は rank-3 weight を持たないため、
この経路は generic weights layer の API としてのみ存在し、model 側からの呼び出し元は無い。

MoE routed experts を layer あたり rank-3 の MXFP4 native SoA へ集約するロード経路
（`shelf::qwen35moe`、`MatrixWeight` の `experts` / batch stride フィールドを含む）は
[qwen35moe.md](qwen35moe.md) を参照。

## Runtime contract

runtime は weight を `MatrixEncoding`（`weights/matrix_weight.h`）として扱い、
`WeightSlot.encoding` で linear launcher を選ぶ。linear の
encoding → compute_spec → KernelId → activation の対応は下節の
contract resolver が正本であり、launch 側の対応は
[qwen35.md](qwen35.md) を参照。

- `QuantSpecId` と `ComputeSpecId` の互換性は `quantization/quantization_types.h` の
  `is_gemm_compute_compatible` が定義し、量子化 storage ごとに実行契約が1つに定まる。
- 量子化 linear は対応 weight が native layout に preshuffle 済みであることを
  dispatch 時に検証する（未 preshuffle なら失敗する）。
- canonical payload を production GPU kernel が直接参照することはない。

### Activation quantization contract

production の Program Builder は linear の activation 量子化方針を
`runtime/program/linear_quant_contract.h` の `resolve_linear_quant_contract`
1箇所で決める。weight encoding と `compute_spec` の不一致は従来どおり
`invalid_argument` で失敗する。

| weight encoding | compute_spec | linear kernel | activation | activation K padding | activation workspace |
| --- | --- | --- | --- | --- | --- |
| `Bf16` | `BF16_F32_BF16` | `LINEAR_BF16` | なし | 32（未使用） | なし |
| `Fp8Finegrained` | `FP8_FINEGRAINED_W8A8_F32_BF16` | `LINEAR_FP8_FINEGRAINED` | E4M3 | 128 | あり |
| `Mxfp4` | `MXFP4_W4A8_F32_BF16` | `LINEAR_MXFP4_W4A8` | E4M3（group E8M0 を行基準へ fold） | 64 | あり |
| `Exl3` | `EXL3_MUL24_INT8_W8A8_F32_BF16` | `LINEAR_EXL3_INT8` | kernel 内で int8 化（EXL3 workspace） | 128（未使用） | EXL3 workspace |

- MXFP4 linear は weight storage が 4-bit でも activation を 4-bit にしない。
  checkpoint が宣言する dynamic MXFP4 activation は使わず、上記の E4M3 spec を
  そのまま共有する（W4A8）。A4 を採用しないことは数値 class の差であり、
  品質比較では [testing_f64_oracle.md](testing_f64_oracle.md) の FP8 と同じ
  扱いで tolerance に反映する。
- MXFP4 weight の scale は 32 K ごとに付く。accumulation は 32 K ごとの
  partial sum に weight scale を掛けて合算し、activation scale は K ループで
  適用しない（下の行基準 fold 形のため）。
  weight scale group が 32 でない weight は kernel capability で拒否する。
- `LINEAR_MXFP4_W4A8` は optimized kernel 群で実行する。weight は native layout
  （[canonical_soa_payload.md](canonical_soa_payload.md) §11）で常駐し、correctness path
  も同じ native layout を読む。E2M1 -> E4M3 展開と E8M0 rescale は register 内で行う。
  variant 選択は `selectors/linear_selector.cpp` が hold し、geometry は allowlist 制
  （`(out_features, k)` ごとに実測値を伴って追加する）。production が使う variant は
  `Gemv`（rows <= 4、rows 5〜8 は out_features <= 2560、かつ weight が L3 以下）、
  `SplitK`（rows <= 64 かつ out_features <= 1024）、`Prefill2D`（rows >= 512 で
  weight tile の再利用が見合う形状）、`RowBlock1 / 2 / 4`（それ以外）。
  `Gemv` は 1 lane = 1 output row の decode 形で、WMMA tile の 16 row のうち
  1〜8 row しか使わない帯を FP8 dot4 で処理する。
- `WeightSlot` の native contract: `codes_row_stride_bytes == (k_padded / 32) * 256`、
  `scale_row_stride_bytes == storage_scale_stride_bytes == k_padded / 32`、
  `preshuffled == true`。満たさない slot は dispatch 側で失敗する。
  native payload は付随平面として `row_ref`（行単位の E8M0 最大値、`rows` byte）と
  フラグ `row_ref_exact`（全行の E8M0 幅が 8 以内 = operand fold が厳密）を持つ。
  Prefill2D は `row_ref_exact` が true の payload でのみ fold を使い、false の場合は
  per-block rescale 側へ戻る。

- activation spec は `kFp8FinegrainedActivationQuantSpec`（codes = `FP8E4M3`、`group_k = 128`、
  scale = `F32`）に固定され、CLI / config による切替は存在しない。平面の**形**は
  family で決まり、spec の切替ではない。
- activation K の padding は block-scaled weight の tile に合わせる。
  FP8 weight は weight tile が 128 幅のため activation も 128 単位で揃える。
  codes の物理配置は WMMA fragment block 順で、block 幅は 64 K である
  （[canonical_soa_payload.md](canonical_soa_payload.md) §11）。
- activation quantize の kernel id は family ごとに 1 つで、Program Builder から
  到達できない代替経路は持たない。
  `Fp8Finegrained` weight は `KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED`、
  `Mxfp4` weight は `KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF` を受け取る。
  両者の codes は E4M3、scale 平面の dtype と layout は同一である。
- MXFP4 の activation 平面は**行基準 fold 形**で、`launch_activation_quantize_mxfp4_rowref`
  （correctness path は `exec_activation_quantize_mxfp4_rowref`）が作る。group 128 の
  scale は E8M0（`448 * 2^e >= group peak` を満たす最小の e）で、その指数を行の最大
  指数 `ref` 基準へ codes 側へ畳む。

  ```text
  codes[i]   = e4m3_rne( e4m3_rne(a[i] / 2^e_g) * 2^-(ref - e_g) )
  scales[row][0] = 2^ref        # scale 平面の column 0 だけを使う
  ```

  GEMM は K ループで activation scale を一切適用せず、epilogue で `2^ref` を 1 回掛ける。
  fold は 2 段丸め（group 量子化してから行基準へ掛け直す）をそのまま再現するのが契約で、
  reference 実装と optimized kernel の作る平面は bit 一致する
  （`test_mxfp4_linear` の producer 比較）。設計と実測は
  [../research/20261103/mxfp4_gate_a_prime.md](../research/20261103/mxfp4_gate_a_prime.md) §13-§16。
- FP8 family の scale は group 128 幅あたり 1 本（`ceil(k_padded / 128)` 本）を row-major に
  書き、linear も同じ位置を読む。MXFP4 の行基準形では column 0 だけを使い、他 column は
  0 で埋める。`Int8ActivationWorkspaceLayout` の scale row stride は
  `align16(ceil(k_padded / 128) * 4)` で、両 family 共通である。
  workspace のバイト数・stride は `ActivationQuantSpec` から導出せず、
  `runtime/program/activation_workspace_layout.h` の
  `make_activation_workspace_layout` が返す layout を正本とする。
  この adapter は production の E4M3 spec に対して
  `Int8ActivationWorkspaceLayout::make(k_padded, max_rows)` と同一の
  `codes_offset` / `scales_offset` / `total_bytes` / `k_padded` /
  `code_row_stride_bytes` / `scale_row_stride_bytes` を返し、
  E4M3 以外（codes dtype 違い、group 指定、scale dtype 違反）は明示的に失敗する。
- activation workspace は入力 ValueId 単位で1つで、同一入力を使う複数 linear は
  1回の quantize と1つの workspace range を共有する。共有の検証は
  K / `k_padded`、activation spec、および producer kernel id（平面の形）の一致で行い、
  食い違う場合は構築時に失敗する。

### FP8 activation producer fusion

FP8 の activation quantize は、次の producer 連鎖を producer と同一 launch で実行する。

- `RESIDUAL_ADD` + `RMS_NORM` + `ACTIVATION_QUANTIZE_FP8_FINEGRAINED`
- `RMS_NORM` + `ACTIVATION_QUANTIZE_FP8_FINEGRAINED`
- `OUTPUT_GATHER` + `RMS_NORM` + `ACTIVATION_QUANTIZE_FP8_FINEGRAINED`
- `SWIGLU` + `ACTIVATION_QUANTIZE_FP8_FINEGRAINED`

RMS norm を含む形は hidden 5120、BF16、RMS norm の `PER_FEATURE` / `ONE_PLUS`、
activation K 5120 のときだけ成立する。`SWIGLU` 側は activation K が down_proj の
入力幅（Qwen3.8-27B は 17408）で、SWIGLU 出力の消費者が activation quantize だけの
ときに成立する。SWIGLU の出力値は単独 kernel と同じ F32 計算と BF16 丸めを使う。

`RMS_NORM` 起点の形は、TP combine が residual add を吸収した後に使われる。
TP 実行中も segment 内で完結する限り適用する。`OUTPUT_GATHER` 側は decode の
rows / outputs が 16 以下で、batch context と output row mapping が使えるときに限る。
いずれも producer の出力を global に materialize せず、residual 出力と gather 出力は
従来どおり書く。生成する codes / scales は単独の
`ACTIVATION_QUANTIZE_FP8_FINEGRAINED` と bit 一致する。

dispatch の無効化は `SHELF_DISABLE_FP8_BOUNDARY` / `SHELF_DISABLE_FP8_FINAL` /
`SHELF_DISABLE_FP8_SWIGLU` で行う。`SHELF_OP_DUMP`、`SHELF_OP_SYNC`、
`SHELF_LINEAR_INPUT_DUMP`、value trace、layer override では融合しない。実装は
`src/shelf/kernels/fused/fp8_activation_producer.hip` と
`try_launch_fp8_residual_norm_quantize` / `try_launch_fp8_norm_quantize` /
`try_launch_fp8_swiglu_quantize` / `try_launch_fp8_final_gather`
（`optimized_dispatch.hip`）にある。

### EXL3-INT8（mul24）

EXL3 は HF checkpoint の `quantization_config.quant_method == "exl3"` を検出し、
`<prefix>.trellis` / `.suh` / `.svh` / `.mul1` から 1 つの
`MatrixWeight`（`MatrixEncoding::Exl3`、`EXL3_MUL24_INT8_V1` /
`EXL3_MUL24_INT8_W8A8_F32_BF16` / `LINEAR_EXL3_INT8`）を作る。重みは trellis のまま常駐し、
INT8 の重み行列は常駐しない。

- コードブックは mul24 のみ。marker（`<prefix>.mul1`、I32 scalar）の値が `0x00DCD12D` の
  checkpoint のみ受理し、それ以外の multiplier、`.s0` tensor、および half bit rate
  （1.5 / 2.5 / 3.5 bpw）はロード時に `Status::unsupported` で拒否する。
- trellis は `<k/16, n/16, 16*ka>` の I16（`ka` は 1..8）、`suh` は `[k]`、`svh` は `[n]` の
  F16。`k` と `n` は 128 の倍数、`k_padded == k`。
- 計算式は `Y = (X diag(suh)) H128 W_hat H128 diag(svh)`（`H128` は 128 要素ブロックの
  正規化 Sylvester Hadamard）。入力回転・INT8 量子化・codebook 復号・出力回転はすべて
  GPU 側で行い、出力は `svh` を掛けてから BF16 / F32 で書く。
- 出力行が 0 の linear（MTP の `compute_logits=false` 経路など）は launch せずに成功として扱う。
- tensor partition（TP）は EXL3 の trellis / `suh` / `svh` / marker を 128 倍数境界で軸ごとに
  分割できる（`materialize_rank_local_canonical`）。各 rank が同じ経路を実行し、hipBLASLt の
  handle と algorithm cache は thread_local かつ device ごとに保持する。
- DFlash2 drafter の量子化 bundle は対象 linear に EXL3-INT8（mul24）または FP8 を受理する。
  DFlash2 が共有する target `lm_head` も EXL3-INT8（mul24）を受理し、INT8 経路で logits を計算する。

実装は以下の 3 層に分かれる。

- `src/shelf/kernels/optimized/linear/exl3_family/` — `mul24/`（INT8 の decode /
  prefill / prefill_super64 / projection / projection_prefill / split_plan / workspace）および
  variant 制約（`variant.cpp`）。
- `src/shelf/kernels/fused/exl3/` — decode 融合（`decode.hip`, `producer_rotate.hip`,
  `producer_rotate_rows.hip`, `rotate_quantize.hip`, `rotate_out.hip`, `group_gemm.hip`）および
  MoE 用 grouped EXL3（`grouped.hip`）。
- `src/shelf/inference/execution/` — dispatch 選択（`optimized_dispatch.hip`, `moe_dispatch.hip`,
  `exl3_dispatch_decision.h`, `exl3_dense_gemm.hip`, `hipblaslt_abi.h`）。

#### 復号と常駐レイアウト

- 復号（GEMM 内、読むたびに）: 重み 1 個の state は、タイルの循環ビット列（word 内は MSB 先頭）から
  `(t * bits + bits - 16 + 256 * bits) mod (256 * bits)` の位置の 16 bit。`q = (B - 509) >> 2`、`B` は
  `state * 0xDCD12D`（uint32、24 bit 定数なので `v_mul_u32_u24` 1 命令）の 4 byte の和で、範囲は −128..127。
  実数値は `q * kExl3Int8GridStep`（`4 * 887 * 2^-17`）。タイル内の並びは WMMA の B フラグメント順で、
  ストリーム位置 `t` は行 `k = 8 * (t / 128) + t % 8`、列 `n = (t / 8) % 16`。lane `l` が復号する 8 個
  （`t = 8l..8l+7`）がそのまま iu8 WMMA の B フラグメントになる。CPU 参照は `exl3_decode_int8`。
- 常駐形式: dense 重みの 16x16 タイルはロード時に CPU で `[kt][nt]` から `[nt][kt]`（n-major）へ
  並べ替える（`exl3_tiles_to_nmajor`）。MoE の expert plane は canonical の `[kt][nt]` のまま常駐する。

#### カーネル選択条件（dense prefill A / B と decode 融合）

`try_launch_linear_exl3_mul24`（`optimized_dispatch.hip`）は workspace layout
（`exl3_detail::mul24::make_workspace_layout`）から chunk 行数と `dense_ok` を計算し、
`exl3_mul24_use_dense(dense_ok, n_rows, dense_min_rows, rowsum_bytes)`（既定 `dense_min_rows = 256`、
`SHELF_EXL3_INT8_DENSE` で変更、`0` で無効）で prefill 経路と decode 経路を分岐する。

| 条件 | 選択経路 | 構造・launch 構成 |
| --- | --- | --- |
| `use_dense && n % 256 == 0 && k % 64 == 0` | **Prefill dense（自前 gfx12 iu8 WMMA）** | `rotate_quantize` → `rowscale`（行最大 scale へ再量子化、エネルギー比が `SHELF_EXL3_INT8_FLAG`（既定 0.5）超の行は最大 `rows/8` 行まで flagged 退避）→ `recon_super64`（64-K supertile のフラグメント平面へ INT8 展開）→ `prefill_super64`（出力回転・H128・`svh`・BF16 書き出しを融合）→ flagged 行の融合 GEMM → flagged 行 fixup。出力が BF16 でない場合は `prefill_super64` の int32 出力と `rotate_out_i32` を使う |
| `use_dense && (n % 256 != 0 \|\| k % 64 != 0)` | **Prefill dense（int32 経路）** | 同じ tile を使えない形状は `decode` 経路の `mul24_gemm` で計算する |
| `!use_dense`（decode・verify・小行数） | **Decode / 小・中行数（融合・K-split）** | 3 launch: `rotate_quantize` → `mul24_gemm`（`exl3_mul24_gemm_splits` が `rows <= 64` の K 分割数を決定）→ `rotate_out`。さらに下述の projection group 統合と producer / boundary 融合が条件成立時に適用される |

- **prefill tile の重み平面**: `recon_super64` が `(n tile, 64-K)` ごとに 1 KiB を書き、1 lane の 32 B が
  4 個の k16 フラグメントになる。GEMM は重みを LDS に置かずレジスタへ直読みし、活性だけを 64-K
  tile で LDS に段取りする。
- **Projection group 統合（`try_launch_exl3_projection`）**。BF16 入出力で、同じ入力 slot と bit 幅を持つ
  連続した `LINEAR_EXL3_INT8` を最大 3 個まとめる。4-bit は 64 行以下、3-bit は HIP Graph の capture 中の
  8 行以下で適用する。入力回転・量子化、GEMM、出力回転をそれぞれ 1 回起動し、符号ベクトルと scale は
  projection ごとに扱う。workspace 不足、入出力の重なり、出力列の部分書き込み、debug override、
  A 経路 prefill 対象時は個別に実行する。
- **Decode 融合（`src/shelf/kernels/fused/exl3/`）**: `num_prefill_requests == 0` かつ `rows <= 16` の
  条件を満たす箇所で、前段演算と入力回転・量子化（または後段の `rotate_out` と残差加算・正規化）を
  1 launch にまとめる。対象は `SWIGLU` → `down_proj`、`SIGMOID + MUL` → `o_proj`、GDN gated norm →
  `out_proj`、`OUTPUT_GATHER + RMS_NORM` → `lm_head`（`rows <= 16`）、および boundary I8/I9
  （`rotate_out + RESIDUAL_ADD + RMS_NORM + projection rotate`）。
  個別の無効化環境変数は [../user/env_vars.md](../user/env_vars.md#8-診断計測デバッグ開発者切り分け用) を参照。
- **出力回転と小・中行数 GEMM タイル**: 整列した buffer / stride を満たす小・中行数の出力回転は 128 列単位の
  wave 分割を使い、512 行以上は f4 経路を使う。active rows の device 指定が無い一部 4 / 5-bit 形状
  （`N=512 / 1024`, `K >= 1024`）は専用タイルを選ぶ。
- **workspace**: xrot / dense / accum / rowsum の領域は、EXL3 linear を実際に実行する program に
  対してだけ execution workspace から切り出す（`program_needs_exl3_workspace`）。
  xrot 領域に int8 codes / scales（A 経路ではさらに flagged 行の slot / codes / scales）、
  accum 領域に F32 出力（A 経路では int32 累積器 + flagged 行 F32 出力）を配置し、入らない行数は
  chunk 分割で実行する。EXL3 重みを持たない model では確保せず、arena の残りを KV pool へ回す
  （executor footprint の見積もりは `qwen35_executor_reserve_bytes` の `exl3_linear_workspace` 引数で制御）。

#### MoE routed experts（grouped EXL3）

rank-3 の grouped EXL3 は `LINEAR_EXL3_INT8_GROUPED`（`moe_dispatch.hip` →
`launch_gemm_exl3_int8_grouped` → `src/shelf/kernels/fused/exl3/grouped.hip`）で実行する。

- **INT8 活性（iu8 WMMA）**: 回転後の活性（`suh` と H128）を 128 要素ごとに int8 化し、
  iu8 WMMA で int32 累積して group scale と `kExl3Int8GridStep` を適用する。
- 回転と INT8 化は GEMM kernel に融合するため grouped 用 workspace を持たない。融合 group 数は
  1 または 2（`SHELF_EXL3_GROUPED_FUSE`）。`SHELF_EXL3_GROUPED_NTILES` が出力幅を割り切れない場合は
  launch を拒否する。
- loader は expert plane 内の全 expert で marker が `0x00DCD12D` かつ整数 bit rate であることを
  検査し、それ以外は `Status::unsupported` で拒否する。

## KV quantization

推論時のKVキャッシュは `PagedKVPool` の storage dtype として選択する。CLI では
`--kv-cache-dtype` で指定し、weight 量子化とは独立。

| dtype | code | scale / 構造 | 備考 |
| --- | --- | --- | --- |
| `bf16` | bf16 | なし | 既定。全モデル・TP / PP 対応 |
| `fp8` | fp8_e4m3 | per (token, kv_head) f32 | 長 context decode で split-KV（LDS LUT 経由 dequant） |
| `octave:cKvV` / `octave:bRnNvV` | Lloyd-Max bit-plane packed | 回転空間正規化 + codebook | 単一 GPU のみ（TP / PP / MoE は起動時拒否）。容量・性能は [../perf/kv-octave.md](../perf/kv-octave.md) |

- `bf16` / `fp8` とも長 context decode で KV split/reduce（flash-decoding 型
  2 kernel、`max_visible >= 2048`）を使う。
- 未定義の `--kv-cache-dtype` 文字列は起動時に明示エラーで拒否する。

### fp8 KV の paged attention 数値契約

`kPagedAttentionPrefillMinRows` 以上の行数をまとめて処理する prefill は、
QK^T と PV を fp8 WMMA（e4m3）で計算する。契約は次のとおり。

- Q は行単位の `amax/448` を scale として e4m3 化し、score 復元時に scale を掛ける。
  行 `amax` は bf16 を float へ変換して求める（整数化した値を scale にしてはならない）。
- K は保存済みの code と per (token, kv_head) の `k_scale` をそのまま使い、
  score は `q_scale * k_scale * softmax_scale` で復元する。
- V は LDS へ `[head_dim][page_tokens]`（dim 優先）で staging し、PV の B fragment を
  dim 行から読む。KV pool 上の `[token][head_dim]` のまま B fragment を作ってはならない。
- P は page 内の最大値で正規化して e4m3 化し、PV 出力に `p_scale` を掛けて復元する。
  page の最大値が `kPQuantMinScale`（1e-30）未満のときは寄与 0 として扱う。
  逆数が inf になると `0 * inf` が NaN になるため、この下限が必要である。
- Softmax の指数和は量子化せず、そのまま最終正規化に使う。
- 診断は `shelf-bench quality` の `--score-start-position` で採点位置を固定して行う
  （[testing.md](testing.md)）。
