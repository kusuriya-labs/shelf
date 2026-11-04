# Canonical SoA Payload / GPU Native Preshuffle Specification

この文書は quantized weight の **canonical file payload** と、GPU 実行時の
**native preshuffle contract** を定義する。

量子化アルゴリズムそのもの（codebook、scale 探索、丸め、誤差関数）は本仕様の対象外。
実装の正本は `include/shelf/quantization/**`、`src/shelf/quantization/**`、
`include/shelf/weights/**`、`src/shelf/weights/**` である。

対象 format は BF16 / FP8 K128N128 の 2 つ。

`canonical` は「GPU が直接読む形式」を意味しない。

```text
canonical = backend 非依存・冗長性なし・CPU で解釈可能な SoA
native    = kernel の load/gather と bank mapping に合わせて preshuffle 済みの SoA
```

GPU kernel が canonical payload を直接読む production path を禁止する。

---

## 1. 設計原則

優先順位は以下とする。

1. **常駐 VRAM を増やさない**
2. **GPU 実行前の preshuffle を必須とする**
3. **file payload は必ず SoA とする**
4. **GPU native payload も SoA とする**
5. canonical file には backend / kernel 固有の shuffle を保存しない
6. metadata を複製しない
7. 不要な reserved byte / padding を追加しない
8. canonical payload だけで CPU dequantize 可能にする
9. ロード時間の増加は許容する

---

## 2. 用語

### Stream

canonical レイアウトは次の stream からなる。

```text
codes
metadata1
metadata2
metadata3
metadata4
```

- `codes`：重みコード配列（常に 1 本）。format 別の code bytes/Block。
- `metadata1..4`：descriptor の `meta[4]` に対応する metadata 配列。
  未使用 slot は size-0／省略（0 byte ダミー領域は予約しない）。

format ごとに使う stream が異なる。FP8 は
`codes` と `metadata1` を使い、`metadata2..4` は size-0 である。

### Metadata mode

metadata の配置は `quantization/quant_format.h` の `MetaCountMode` が決める。

| mode | 要素数 |
|---|---|
| `None` | なし |
| `PerBlock` | (row, K block) ごと |
| `PerSuperBlock` | (row, superblock) ごと |
| `MatrixBlock` | (N block, K block) tile ごと。tile 内の全 row で共有する |

`QuantMetaDesc` は `element_bytes`（1 item の byte 数）、`count_mode`、
`MatrixBlock` 用の `block_n` / `block_k` を持つ。現在使用する format が
`PerBlock` / `MatrixBlock` / `None` を使う。

---

## 3. 共通 shape 規則

各 row の logical K を `logical_k` とする。

```text
padded_k       = align_up(logical_k, block_elements)
blocks_per_row = padded_k / block_elements
```

- padding weight は 0 として量子化する。
- `block_elements` は format 定義（BF16 = 16、FP8 = 128）。
- metadata は row boundary をまたがない。row 間で scale / exponent を共有しない。

---

## 4. Canonical file layout

量子化 tensor の file payload は必ず SoA とする。

```text
BF16
  data
  (codes / metadata なし)

FP8
  codes
  metadata1   (FP32 scales)
```

AoS は禁止。

```text
禁止:
  [block0 code][block0 meta][block1 code][block1 meta]...
```

- stream はそれぞれ独立した連続領域として扱う。
  Safetensors 等の外側コンテナでは各 stream を独立 tensor として格納する。
- 外側コンテナが region alignment を要求することは許容するが、その alignment は
  quantization payload の情報量には含めない。block / tile ごとの padding は追加しない。
- file の `codes` / `metadata1..4` は `QuantizedTensorMetadata` の同名 ref に対応する。

---

## 5. BF16 canonical format

BF16 は量子化ではないが、weight storage / compute format として同じ descriptor 機構で扱う。

```text
block_elements       = 16
code_bytes_per_block = 32   (= 16 element * 2 bytes)
metadata             = なし
```

- on-disk では `data` ref に row-major BF16 tensor を格納する。
  `codes` / `metadata1..4` は持たない。
- `k_padded == logical_k`。
- preshuffle を行わず、row-major BF16 のまま GPU へ配置する。
- `codes` stream の BF16 descriptor は compute view 用で、file 上に canonical
  codes stream を持つわけではない。

---

## 6. FP8 K128N128 canonical format

### 6.1 Parameters

```text
block_elements (K)    = 128
code format           = OCP E4M3
codes bytes/block     = 128   (1 byte / weight)
metadata              = metadata1, FP32, MatrixBlock 128 x block_k
block_k               = 128（既定） / 64（Manifest V4 の宣言 block）
block_n               = 128（固定）
```

codes と `padded_k = align_up(logical_k, 128)` の契約は block_k の値に
よらない。block_k を変えた artifact は Manifest V4 の
`quant.scale.block_shape` で宣言され、既定（128）と異なる group は
runtime の kernel capability 検証を経て受け付けられる。

### 6.2 codes stream

codes は row-major の E4M3（1 byte / weight）を logical weight 順に格納する。

```text
codes[row][k] = e4m3(q[row * padded_k + k])
padded_k      = align_up(logical_k, 128)
```

### 6.3 metadata1 stream

metadata1 は **FP32 scale** で、`MatrixBlock`（`block_n = 128`、
`block_k = 128 または 64`）を使う。1 個の scale は 128 出力 row × block_k K の
tile 内の全 weight が共有する。

```text
scale_count = ceil(rows / 128) * (padded_k / block_k)
4 bytes / scale
```

- 2D matrix `[N, K]` では `ceil(N / 128) * (padded_k / block_k)` 個。
- batch 次元を持つ場合、scale tile は matrix 境界をまたがない。
  batch ごとに別 tile を確保する。
- 復元:

```text
w[row][k] ~= e4m3_decode(codes[row][k]) * scale[row / 128][k / block_k]
```

### 6.4 Payload density

codes は 1 byte / weight。scale は 128 x 128 tile あたり 4 bytes を tile 内の
16384 weight で割るため、weight あたり 1/4096 byte に相当する。

---

## 6bis. MXFP4 E2M1 / E8M0 group 32 canonical format

OCP Microscaling FP4（Quark `mxfp4` の real-quantized export）を使う。

### 6bis.1 Parameters

```text
block_elements (K)    = 32
code format           = OCP E2M1（2 weight を 1 byte に packed）
codes bytes/block     = 16    (32 weight * 0.5 byte)
metadata              = metadata1, E8M0, PerBlock（row と K block の組ごと）
block_n               = 1
block_k               = 32（固定）
```

E2M1 code 値は `0, 0.5, 1, 1.5, 2, 3, 4, 6`（bit3 = sign）。

### 6bis.2 codes stream

```text
codes[row][j] = low nibble  = weight[row][2 * j]
                high nibble = weight[row][2 * j + 1]
packed width  = padded_k / 2
padded_k      = align_up(logical_k, 32)
```

nibble 順は生成元 packer（`evens | (odds << 4)`）と一致する。逆に packed た
artifact は受理しない。

### 6bis.3 metadata1 stream

metadata1 は **E8M0**（1 byte の肩付き指数）で、`PerBlock`（`block_n = 1`、
`block_k = 32`）を使う。FP8 のような N 方向 tile 共有は無い。

```text
scale_count = rows * (padded_k / 32)
1 byte / scale
scale value = 2^(byte - 127)
```

- byte 0 は 2^-127、byte 255 は MX spec の NaN であり **受理しない**。
- 復元:

```text
w[row][k] ~= e2m1_decode(nibble(codes[row][k / 2], k % 2)) * 2^(scales[row][k / 32] - 127)
```

### 6bis.4 Payload density

codes は 0.5 byte / weight、scale は 1 byte / 32 weight。
raw で weight あたり 17/32 byte（FP8 の 17/16 byte の半分）。

### 6bis.5 Scale の生成規則（reference math）

artifact の生成側（テスト fixture 生成にも使う）は次で確定する。

```text
scale_byte(amax) = biased_exp(round_to_1_mantissa_bit(amax)) - 2   // clamp [-127, 127]
amax == 0        -> byte 1（2^-126）
```

code 側は 0.5 ulp 未満へ丸める **half-away-from-zero** である（MX の
round-half-to-even ではない）。0.25 / 1.25 / 2.5 / 5 の境界で code が変わるため、
生成元 packer との bit 一致をテストで固定する。

---

## 7. Stream size formulas

`rows` を row 数、`blocks = rows * (padded_k / block_elements)` とする。

### BF16

```text
data bytes = rows * logical_k * 2
```

### FP8

```text
codes bytes     = rows * padded_k
metadata1 bytes = ceil(rows / 128) * (padded_k / 128) * 4
```

batch 次元を持つ場合は `ceil(rows / 128)` を batch ごとに計算する。

### MXFP4

```text
codes bytes     = rows * padded_k / 2
metadata1 bytes = rows * padded_k / 32
```

scale は row を共有しないため、batch / rank-3 では各行列分を積み上げる。その expert 間の
stride は `include/shelf/quantization/mxfp4/mxfp4_native.h` の
`mxfp4_native_codes_batch_stride_bytes` / `mxfp4_native_scale_batch_stride_bytes` が単一源で、
native preshuffle・weight plan・weight table 検証・grouped GEMM はいずれもこの式を読む。

---

## 8. File stream naming / descriptor contract

format は固定数の Meta stream を持たない。
必要な stream だけを descriptor / manifest で参照する。

実装の descriptor（`shelf/quantization/quant_format.h`）:

```cpp
enum class QuantFormatId : uint8_t {
    None = 0, Bf16 = 1, Fp8Finegrained = 2, Mxfp4Finegrained = 3,
};

static_assert(static_cast<uint8_t>(QuantFormatId::None) == 0);
static_assert(static_cast<uint8_t>(QuantFormatId::Bf16) == 1);
static_assert(static_cast<uint8_t>(QuantFormatId::Fp8Finegrained) == 2);
static_assert(static_cast<uint8_t>(QuantFormatId::Mxfp4Finegrained) == 3);

enum class MetaCountMode : uint8_t {
    None = 0, PerBlock = 1, PerSuperBlock = 2, MatrixBlock = 3,
};

struct QuantMetaDesc {
    uint32_t element_bytes = 0;   // 1 item あたりの byte 数（可変: 1 / 2 / 4）
    MetaCountMode count_mode = MetaCountMode::None;
    uint32_t block_n = 1;         // MatrixBlock の N 方向 tile（他は 1）
    uint32_t block_k = 0;         // MatrixBlock の K 方向 tile（format block は 0）
};

struct QuantFormatDesc {
    uint32_t block_elements = 0;        // Block の weight 数
    uint32_t blocks_per_superblock = 1; // 現行 format はすべて 1
    uint32_t code_bytes_per_block = 0;  // BF16 = 32, FP8 = 128
    QuantMetaDesc meta[4];              // metadata1..4（未使用は None / 0 byte）
};
```

format descriptor:

```text
BF16
  block_elements = 16, code_bytes_per_block = 32, meta なし

FP8
  block_elements = 128, code_bytes_per_block = 128
  metadata1 : element_bytes = 4, MatrixBlock, block_n = 128, block_k = 128

MXFP4
  block_elements = 32, code_bytes_per_block = 16
  metadata1 : element_bytes = 1, PerBlock, U8 (E8M0)
```

numeric value は in-memory の identifier のみで、file / manifest には保存しない。
on-disk の encoding は `shelf_quantization.json` の文字列名で管理される。

```text
bf16                    / Bf16
fp8_e4m3_block128_f32   / Fp8
mxfp4_e2m1_e8m0_g32     / Mxfp4
```

`QuantizedEncoding`（`manifest/quantized_manifest.h`）が唯一の on-disk encoding enum である。
その番号体系（BF16 = 0, FP8 = 5, MXFP4 = 7）は既存 artifact と対応する。

既存 manifest / tensor reference を拡張する場合も、`scales` という 1 本の opaque stream に
複数の意味を埋め込む設計には戻さない。

---

## 9. Mandatory GPU native preshuffle

GPU backend では preshuffle を **必須処理** とする。

```text
Canonical file SoA
        |
        v
Canonical host view
        |
        v
MANDATORY native-layout builder / preshuffle
        |
        v
GPU Native SoA
        |
        v
Kernel
```

以下の経路は禁止。

```text
Canonical SoA -> GPU kernel
```

preshuffle は optional optimization ではなく GPU representation を成立させる変換である。

format 別の builder:

- FP8 : `quantization::fp8::preshuffle_fp8_native`
- BF16 : preshuffle なし（row-major BF16 のまま）
- MXFP4 : `quantization::mxfp4::preshuffle_mxfp4_native`。
  canonical の row-major 4-bit codes を WMMA fragment 順へ組み替える（§11）。
  E8M0 scale は byte 順を変えない。

Kernel API は canonical view を受け取れない構造にすることを推奨する。

---

## 10. Preshuffle の目的: bank-aware physical mapping

preshuffle の目的は単なる転置ではない。

kernel の 1 load / gather 命令で必要になる codes / metadata が、対象 RDNA4 kernel の
cache / memory bank mapping 上で不要に多数の bank へ分散しないように物理配置を作る。

特に codes stream は必ず kernel access pattern に合わせて preshuffle する。
metadata も同じ native-layout builder で kernel-native order に変換する。

各 kernel について最低限以下の対応関係を定義できなければならない。

```text
lane
logical quant index
canonical byte offset
native byte offset
metadata index
bank mapping
```

「correctness が通る」だけでは native layout の完了条件にならない。
期待する bank mapping をテストまたは profiler / ISA 解析で確認する。

canonical 順を直接 gather する fallback を production path に残してはならない。

---

## 11. GPU Native SoA

native representation も SoA を維持する。
codes と metadata を Block 単位で interleave した AoS resident layout は禁止。

stream 内部の順序は kernel / RDNA4 bank mapping に合わせて変更してよい。
canonical file の順序と native の順序が同一である必要はない。

native preshuffle は 16 出力 row を 1 tile とする。
`rows` は 16 の倍数へ pad し、pad 分は 0 で埋める。

### FP8 native

```text
codes     : 16 outputs x 64 K の fragment block 順
            block 内は lane slot 32 byte 每 (lane が 4 つの 16 K fragment を
            連続 byte で読める。wave 1 本で 1 KiB 連続要求になる)
            codes_row_stride_bytes = nblk * 1024 = padded_k * 16  (nblk = padded_k / 64)
metadata1 : row-major [ceil(rows / 128)][padded_k / 128] の FP32
```

canonical の codes は row-major だが、native では WMMA consumer の fragment 順へ並べ替える。
byte 順の定義は `include/shelf/quantization/fp8/fp8_frag.h` に置き、host builder と
kernel が同じ式を使う。pad row は 0 埋めする。

activation codes (FP8 W8A8 の入力) も同じ fragment block 順を使う。
`padded_k` は 64 の倍数でなければならない（`Int8ActivationWorkspaceLayout::k_padded_for`
が 64 へ align する）。

### MXFP4 native

WMMA consumer が E2M1 を直接食うための layout。16 output x 32 K を 1 block とし、
block 内で output row が 16 byte を連続して持つ。byte b は code[k = b] を下位
nibble、code[k = 16 + b] を上位 nibble に格納する。この (k, k + 16) 対は
FP8 fragment の 1 lane が 16 K 命令で消費する 8 byte と一致するため、
E2M1 -> E4M3 展開は byte 再配置なしに operand register へ書ける。

```text
block            = 16 outputs x 32 K = 256 bytes
codes            : [tiles][blocks][256]，tiles = rows / 16
                   codes_row_stride_bytes = (padded_k / 32) * 256   (1 tile あたり)
                   同一 tile 内の行 ol の 16 byte は offset ol * 16
                   byte b = code[k = b] | (code[k = 16 + b] << 4)
metadata1        : [rows][padded_k / 32] row-major E8M0（byte 順は canonical と同じ）
                   scale_row_stride_bytes = padded_k / 32
row_ref          : [rows] 行単位の E8M0 最大値 byte（1 byte / output row）と
                   row_ref_exact flag（全行の E8M0 幅 <= kMxfp4FoldExactMaxDistance = 8）
padded_k         : 64 の倍数（activation の 64 K fragment block と揃える）
rows             : 16 の倍数（pad は作らない。満たさない tensor は load 時に unsupported）
resident size    : 0.5 byte / weight + 1 byte / 32 weight（canonical と同一）
```

`preshuffled` flag は native layout の成立条件であり、native の codes を canonical
stride で提示しても、canonical の codes を native stride で提示しても load / dispatch
側の contract 検証で失敗する。

codes を E4M3 へ展開した layout を常駐させてはならない。resident size が
0.5 byte / weight から 1 byte / weight へ倍増し、§12 の VRAM 条項に反する。
E2M1 code から E4M3 operand への展開、および E8M0 exponent の適用は
register 内で行い、metadata1（E8M0）は常に保持する。

E8M0 の operand 側 fold（`d = row_ref[row] - e8m0[row][blk]` を E4M3 仮数へ畳み、
`2^(row_ref-127)` を epilogue で 1 回適用する）は per-block rescale 命令を消す。
`d <= 8` の間は E2M1 仮数（有意 2 bit）は E4M3 で厳密なので**量子化値も加算順序も
変わらない**（correctness path と bf16 bit 一致）。`d >= 9` では E4M3 subnormal の
1.5 刻みを丸めるため値が変わるので、loader は行単位の E8M0 幅を検査して
`row_ref_exact` を立て、dispatch はその flag が true の payload でのみ fold を使う。
fold を使う variant は Prefill2D のみ（rows >= 512）。実測と判断は
[../../research/20261103/mxfp4_gate_a_prime.md](../../research/20261103/mxfp4_gate_a_prime.md)。

row_ref は 1 byte / output row であり、1 byte / weight でも 1 byte / block でもない。
例: out=17408, k=5120 の MXFP4 weight で codes + E8M0 が約 45 MB に対し row_ref は
17 KB（+0.04%）。per-weight でないため §12 の resident size 条項に反しない。

---

## 12. VRAM contract

同じ logical tensor shape / quantization type に対して、本仕様への移行によって
**resident VRAM を増加させてはならない**。

特に禁止:

- unused metadata stream の確保
- per-Block reserved byte
- per-Block / per-tile alignment padding
- codes / metadata の AoS 化に伴う padding
- canonical GPU copy と native GPU copy の二重常駐
- 高速化のための metadata duplication

推奨ロード経路:

```text
Disk canonical SoA
       |
       v
CPU canonical buffers
       |
       v
CPU native preshuffle / packing
       |
       v
H2D
       |
       v
GPU native SoA only
```

GPU 上で preshuffle する場合は temporary canonical buffer を変換後に必ず解放する。
peak VRAM が問題になる場合は chunked transform または in-place transform を使用する。

native layout は情報の permutation を基本とし、raw quantization payload の情報量を
増やさない。native preshuffle の 16-row tile padding は kernel が要求する tile 境界で
あり許容するが、今回の format 統一を理由に新たな常駐 padding を増やしてはならない。

---

## 13. Serialization / quantization / native layout の責務分離

責務は以下の 3 層に分ける。

```text
Quantization math
  - code selection
  - scale search
  - E4M3 encode
  - scale 最適化

Canonical serialization
  - sequential codes packing
  - metadata stream 生成
  - SoA file write/read
  - shape / size validation

GPU native layout
  - mandatory preshuffle
  - WMMA / lane mapping
  - bank-aware physical ordering
```

format 統一で quantization math を変更してはならない。

canonical disk format は **TP-independent** である。tensor parallel を使う場合も
disk 上の canonical payload は global tensor のままで、rank-local partition は
load 時に行われ、その rank-local canonical の後に native preshuffle が実行される。
native payload を disk contract にしない現在の方針を維持する。
順序 contract は [tensor_partition.md](tensor_partition.md)、runtime execution は
[tensor_parallel_execution.md](tensor_parallel_execution.md) を参照する。

canonical disk format は **TP-independent** である。tensor parallel を使う場合も
disk 上の canonical payload は global tensor のままで、rank-local partition は
load 時に行われ、その rank-local canonical の後に native preshuffle が実行される。
native payload を disk contract にしない現在の方針を維持する。
順序 contract は [tensor_partition.md](tensor_partition.md) を参照する。

---

## 14. Loader contract

この contract の実装は generic weights layer（`shelf::weights::load_quantized_matrix` /
`load_bf16_matrix`、`include/shelf/weights/weight_loader.h`、target `shelf_weights`）。
model 側（例: Qwen35）は tensor name → model field の binding のみを行い、
encoding の分岐を持たない。

loader は以下を順番に行う。

1. format / version を検証する
2. logical shape / padded K を検証する
3. codes / metadata stream size を検証する
4. canonical SoA を読む
5. mandatory native preshuffle を行う（`WeightLoadOptions.preshuffle = false` の場合を除く）
6. native SoA（canonical の場合は canonical SoA）を GPU へ配置する
7. kernel には native view のみ渡す

preshuffle のスイッチは generic layer（`shelf::weights::WeightLoadOptions`、
`load_bf16_matrix` / `load_quantized_matrix`）に属する。model 側（Qwen35 等）は
これを自 LoadOptions に組み込むだけで、独自に preshuffle フラグを持たない。

`preshuffle = false` の意味:

- FP8 : canonical SoA をそのまま配置する。optimized GEMM は
  未 preshuffle の weight を拒否する。
- BF16 : preshuffle の対象外（常に row-major BF16）。

preshuffle を呼び忘れても動作する API を作らない。
kernel dispatch 時に native layout であることを validate する。

この loader contract は HF 由来の real-quantized safetensors（fp8）
にもそのまま適用される。`hf_manifest.{h,cpp}` が HF の tensor naming から合成した
manifest は canonical と同一の stream size formula / scale dtype（FP8 は F32）を持ち、
loader は self-contained dir と同じ
shape / size / CRC32 validation と native preshuffle を合成 manifest に対して実行する。
差異は shard の `__metadata__`（`shelf.format` 等）を検証しないことと、
FP8 の BF16 scale を F32 へ widen した staging を metadata1 のソースにすることのみである。

---

## 15. Versioning

canonical packing または stream 構成を変更した場合は layout / version を更新する。
on-disk manifest の `format_version` は現在 4 である（`kQuantizedSafetensorsFormatVersion`）。
V3 manifest（`format_version = 3`）は互換 parser（`parse_quantized_manifest_v3`）で
読み込める。新規保存は V4 のみ。V4 の manifest 構造は
[quantization.md](quantization.md) の self-contained quantized safetensors を参照。
canonical payload の stream 構成自体は V3 から変更していない。

旧形式を新形式として silent に読み込んではならない。

旧形式をサポートする場合は:

```text
legacy file
    |
    v
legacy reader
    |
    v
canonical SoA
    |
    v
mandatory native preshuffle
    |
    v
GPU native SoA
```

とする。

writer は原則として最新 canonical format のみを書き出す。

---

## 16. Validation requirements

reader / verifier は最低限以下を検証する。

```text
logical_k > 0
padded_k >= logical_k
padded_k % block_elements == 0
known encoding
supported format version
codes size == expected
metadata1 size == expected
```

format 別の追加条件:

```text
FP8: k_padded % 128 == 0, metadata1 は FP32 で ceil(rows / 128) * (padded_k / 128) 個
MXFP4 : k_padded % 32 == 0, metadata1 は U8 で rows * (padded_k / 32) 個、
        E8M0 byte 255 を含まない
BF16        : k_padded == logical_k, codes / metadata を持たない
```

---

## 17. Required tests

### 17.1 Quantization round-trip

各 format について:

```text
FP32/BF16
  -> quantize
  -> canonical SoA
  -> CPU dequantize
```

を検証する。少なくとも以下を含む。

```text
all zero
positive / negative
mixed sign
small / large values
K = 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129
```

### 17.2 FP8 block128 tile boundary

N が 128 の倍数でない multi-row tensor を用意し、128 x 128 scale tile が
matrix 境界をまたがないことを検証する。

### 17.3 Native preshuffle correctness

```text
canonical
  -> native preshuffle
  -> test-only inverse transform
  -> canonical
```

を bit-exact で確認する。

### 17.4 Bank mapping

kernel の主要 load / gather について、logical index -> native offset -> bank mapping を検証する。
少なくとも codes stream は必須。metadata も kernel access pattern に応じて検証する。

### 17.5 VRAM regression

同一 tensor shape について resident allocation bytes を変更前後で比較し、
増加していないことを確認する。

raw payload の基準値:

```text
FP8 = 1 byte / weight + FP32 scale per 128 x 128 tile
MXFP4 = 0.5 byte / weight + E8M0 scale per 32 weight block
BF16      = 2 bytes / weight
```

---

## 18. Final format summary

```text
FP8
-----------------------------------------------
codes     : 1 byte / weight, E4M3
metadata1 : 4 bytes / 128 x 128 tile, FP32 scale
Raw       : 1 byte / weight + scale tile
```

```text
MXFP4
-----------------------------------------------
codes     : 0.5 byte / weight, E2M1, 2 codes / byte along K
metadata1 : 1 byte / 32 weight, E8M0 scale, per row
Raw       : 0.5 byte / weight + scale byte per 32 weight
```

```text
BF16
-----------------------------------------------
data      : 2 bytes / weight, row-major
Raw       : 2 bytes / weight
```

GPU execution:

```text
Canonical SoA
    |
    v
MANDATORY RDNA4 bank-aware preshuffle
    |
    v
Native SoA
    |
    v
Kernel
```

**canonical payload を production GPU kernel が直接参照することは禁止する。**
