# Quantization

source of truth: `include/shelf/quantization/**`,
`src/shelf/quantization/**`, `include/shelf/weights/**`,
`src/shelf/weights/**`, `src/shelf/io/**`。

canonical payload の stream / metadata / size formula / native preshuffle は
[canonical_soa_payload.md](canonical_soa_payload.md) を正本とする。
本ドキュメントは format の概要と runtime の経路を示す。

本 repository は **量子化 artifact の読込専用**である。量子化 artifact を生成する
offline pipeline / CLI は含まない。HF の FP8 checkpoint、および外部で生成した
self-contained quantized safetensors を直接ロードする。

## フォーマット

- `QuantSpecId` / `ComputeSpecId` — 量子化仕様ID（`quantization/quantization_types.h`）。
- `QuantFormatId` — weight storage / compute formatのID（`quantization/quant_format.h`）。
  numeric valueはin-memory専用で、file / manifestには保存しない。
  on-diskのencodingは `shelf_quantization.json` の文字列名で管理する。
- 対応 encoding: BF16 / FP8 K128N128。
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
ps::weights loader
    safetensors / quantized safetensors の解決
    BF16 / FP8 の format解釈
    canonical payload validation / native preshuffle / GPU upload
    MatrixWeight（quant_spec / compute_spec 含む）の生成
    ↓
quant format / quant bundle / quant reference / io / core / gpu
```

- model側はencodingの分岐を持たない。
- tensor partition（weights 層）と TP execution は
  [tensor_partition.md](tensor_partition.md) / [tensor_parallel_execution.md](tensor_parallel_execution.md) を参照。
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

合成は `quantization/bundle/hf_manifest.{h,cpp}` が行う。HF の tensor naming
（codes = `<name>.weight`、FP8 scales = `<name>.weight_scale_inv`（BF16））から
`QuantizedManifest` を合成し、
[canonical_soa_payload.md](canonical_soa_payload.md) §6 と同じ
canonical payload contract へ流し込む。payload の bit 列は canonical 定義と一致するため、
preshuffle / runtime 経路は self-contained 量子化 model と同一である。

- FP8 の BF16 scale はロード時に F32 へ widen される（metadata1 は F32 contract のまま）。
- 量子化されていない tensor（norm / embed / lm_head / dt_bias / A_log /
  conv1d.weight など）は BF16 entry として同一経路でロードされる。
- HF payload の K が block size（FP8: 128）の倍数でない場合、
  canonical の padded K contract を満たせないため `Status::unsupported` で拒否する。
- 合成 manifest は `architecture = "qwen35_dense"` を立て、CRC32 は実データから計算される。
  `is_quantized_model_dir` がこの経路を検出し、
  `load_qwen35_weights_from_quantized_safetensors` が合成 manifest 経路へ入る。

### Fused rank-3 expert tensor の axis-slice ロード

`[E, N, K]` の fused expert tensor（例: `mlp.experts.gate_up_proj`
`[512, 1280, 2560]`）から任意の Expert 範囲 `[unit_begin, unit_begin + unit_count)`
だけを取り出して rank-2 の `MatrixWeight` としてロードできる。
モデル全体を VRAM に載せず、指定 Expert 分の byte span だけを GPU へ転送する。

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

- BF16 は rank-3 BF16 tensor のみ受け付ける。結果の `rows = unit_count * N`、
  `cols = K`、`k_padded = K`。
- FP8 は codes が rank-3 の F8_E4M3 / U8、scale が BF16 rank-3
  `[E, ceil(N/128), K/128]`（`<codes_name>_scale_inv` naming）であることを要求する。
  K が 128 の倍数でなければ拒否する。
- span 解決は `SafetensorsCollection::axis_span`（G1 の読み取り専用 API）を使い、
  mmap payload への直接ポインタから slice を HIP へ転送する。
- FP8 は slice 後に既存の rank-2 canonical 経路（`Fp8BlockCanonicalView` validation →
  `preshuffle_fp8_native` → upload）をそのまま使う。`options.preshuffle == false` では
  canonical codes + F32 scale のまま GPU へ載せる。
- **FP8 の rank-3 (N-D) canonical partition / materialization は unsupported のまま**
  （[tensor_partition.md](tensor_partition.md) §5）。この経路は partition を使わず、
  axis-0 単位の slice を rank-2 として扱う。
- HF manifest 合成経路（`synthesize_hf_quantized_source`）は rank-2 codes のみ受理する
  制約を変更しない。rank-3 fused expert はこの loader で直接扱う。

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
| `Fp8` | `FP8_W8A8_F32_BF16` | `LINEAR_FP8` | E4M3 | 128 | あり |

- activation spec は `kE4m3ActivationQuantSpec`（codes = `FP8E4M3`、`group_k = 0`、
  scale = `F32`）に固定され、CLI / config による切替は存在しない。
- activation K の padding は block-scaled weight の tile に合わせる。
  FP8 weight は weight tile が 128 幅のため activation も 128 単位で揃える。
- 量子化 weight の activation quantize は `KernelId::ACTIVATION_QUANTIZE_W4A8`
  を受け取る。名前は既存の fusion match 条件とテストが参照するため据え置きで、
  production で意味する実体は E4M3 activation である。
- `group_k = 0` は「row あたり scale が1本」を意味する。E4M3 quantize kernel は
  row 先頭に F32 scale を1本だけ書き、linear も同じ位置を読む。
  一方 `Int8ActivationWorkspaceLayout` の scale row stride は
  `align16((k_padded / 32) * 4)` であり、I8 経路と共有するレイアウト上の
  予約量である（E4M3 は先頭1本のみを使う）。
  そのため workspace のバイト数・stride は `ActivationQuantSpec` から導出せず、
  `runtime/program/activation_workspace_layout.h` の
  `make_activation_workspace_layout` が返す layout を正本とする。
  この adapter は production の E4M3 spec に対して
  `Int8ActivationWorkspaceLayout::make(k_padded, max_rows)` と同一の
  `codes_offset` / `scales_offset` / `total_bytes` / `k_padded` /
  `code_row_stride_bytes` / `scale_row_stride_bytes` を返し、
  E4M3 以外（I8、group 指定、scale dtype 違反）は明示的に失敗する。
  I8 の既存低レベル経路は従来どおり `Int8ActivationWorkspaceLayout` を直接使う。
- activation workspace は入力 ValueId 単位で1つで、同一入力を使う複数 linear は
  1回の quantize と1つの workspace range を共有する。共有の検証は
  K / `k_padded` と activation spec の一致で行い、
  食い違う場合は構築時に失敗する。
- 通常の Program Builder から到達できない経路として
  `KernelId::ACTIVATION_QUANTIZE_FP8`（`PhysicalActivationQuantizeKind::I8Row` /
  `launch_activation_quantize_i8_row`）がある。kernel / physical launch /
  correctness の経路は実在するが、production の選択肢には入っていない。
  「通常推論で未使用」であることと「未参照」であることを区別して扱う。
  利用状況と採否の記録は
  `rnd/quantization/activation_quant_contract_phase1.md` を参照。

## KV quantization

推論時のKVキャッシュは `PagedKVPool` の storage dtype として選択する。CLI では
`--kv-cache-dtype` で指定し、weight 量子化とは独立。

| dtype | code | scale | head_dim |
| --- | --- | --- | --- |
| `bf16` | bf16 | なし | 任意 |
| `fp8` | fp8_e4m3 | per (token, kv_head) f32 | 任意 |

- `bf16` / `fp8` とも長 context decode で KV split/reduce（flash-decoding 型
  2 kernel、`max_visible >= 2048`）を使う。fp8 の split は dequant を
  LDS LUT 経由で行い、reduce は dtype 非依存の variant を流用する。
- `--kv-cache-dtype` の受理値は `bf16` / `fp8` のみで、旧値は明示エラーで拒否する。

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
