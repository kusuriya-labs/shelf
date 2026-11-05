# Tensor Partition

logical tensor の tensor parallel (TP) partition と、それを weight へ適用する
materialization contract の source of truth である。
該当コードは `include/shelf/weights/tensor_partition.h` /
`src/shelf/weights/tensor_partition.cpp` /
`include/shelf/weights/canonical_partition.h` /
`src/shelf/weights/canonical_partition.cpp`。

## 1. 正式な materialization 順序

```text
Safetensors canonical global tensor
    ↓
canonical validation
    ↓
logical TensorPartitionDesc
    ↓
rank-local canonical materialization
    ↓
rank-local canonical validation
    ↓
backend-specific preshuffle
    ↓
GPU upload
```

絶対にしてはいけない順序:

```text
canonical global → global preshuffle → native-layout partition   (禁止)
preshuffled GPU payload を rank ごとに切り出す機構              (禁止)
```

partition は必ず preshuffle より前に行う。preshuffle の入力は
「呼び出し側が所有する logical tensor 全体」であり、TP 使用時は
その logical tensor が rank-local tensor である
（`preshuffle_fp8_native` の header contract）。
preshuffle 関数は `tp_rank` / `tp_size` を一切受け取らない。

## 2. 概念の分離

| 概念 | 意味 |
| --- | --- |
| storage / file shard | `model-00001-of-000xx.safetensors` 形式の safetensors ファイル分割。tensor partition とは無関係 |
| tensor partition | logical tensor の geometry 分割。`TensorPartitionDesc` |
| device placement | どの partition をどの GPU に置くか。tensor partition には含まれない |

- tensor partition != device placement。`TensorPartitionDesc` は
  `device_id` / `gpu_id` / RCCL rank / PCIe topology / GPU address を持たない。
- partition index は tensor storage geometry であり、どの partition をどの GPU に
  割り当てるかは将来の multi-GPU runtime / execution plan の責務である。
- manifest の `format_version` は 4 である（V3 の読み込み互換を維持）、`QuantizedTensorMetadata` は
  partition を持たない。on-disk artifact は global canonical のままである。
- そのため **同一の global canonical artifact から runtime で tp_size を選べる**
  （TP=1 / TP=2 / TP=4 …）。TP degree / rank-local geometry を disk artifact へ
  焼き込まない。

## 3. TensorPartitionDesc

```cpp
struct TensorPartitionRange {
    uint64_t global_offset;   // partition axis 上の logical element 単位の開始位置
    uint64_t extent;          // 同じく logical element 単位の長さ
};

struct TensorPartitionDesc {
    int32_t  axis;            // global tensor を分割した dimension
    uint32_t index;           // この partition の通し番号
    uint32_t count;           // partition 総数
    std::vector<int64_t> global_shape;      // 分割前の global tensor shape
    std::vector<TensorPartitionRange> ranges;
};
```

- 座標系は **canonical logical tensor coordinate** である。
  native byte offset / WMMA tile index / lane index / native codes stride /
  preshuffled offset を持たない。
- `ranges` は記述順に local tensor へ**密に連結**する。順序は sort しない。
  順序自体が local layout contract である。local offset は metadata に保存せず、
  ranges の prefix sum として導出する。
- local shape は metadata に重複保存しない。

```text
local_shape          = global_shape
local_shape[axis]    = sum(ranges.extent)
```

### ranges の例

```text
global axis:  0 .............................................. 767

ranges:       {offset=0,   extent=128}
              {offset=256, extent=128}
              {offset=512, extent=256}

local axis:   global[0:128] + global[256:384] + global[512:768]
```

1 つの rank が global axis 上の複数 range を所有できる（segmented partition）。
これは Qwen3.8 Dense の GDN fused QKV で必須である
（`single global_offset` だけの descriptor では表現できない）。

### validation

`validate_tensor_partition`（pure CPU）:

```text
count >= 2
index < count
global_shape は非空、全 dim > 0
0 <= axis < global_shape.size()
ranges は非空、各 extent > 0
global_offset < global_shape[axis]
global_offset + extent <= global_shape[axis]
同一 partition 内の global ranges は互いに重複しない（順序に関係なく）
ranges の順序は保持する（validation は sort しない）
```

全 partition を合わせて global_shape を完全被覆するかは、rank-local bundle
単体では判定できないため validation しない。

`validate_tensor_partition_plan` はさらに `tp_size >= 1` /
`tp_rank < tp_size` と、各 entry の `count == tp_size` / `index == tp_rank` を
要求する。`tp_size == 1` の plan は entry を持たない（完全 replicated）。

## 4. N partition

2D weight `W[N][K]` の axis=0 partition は canonical payload の row range を
切り出す。BF16 の canonical payload は row-major なので、行をそのまま
ranges に沿って切り出す。codes / metadata を持つ format（FP8）は scale tile の
境界制約を伴うため、下記「FP8」節の boundary 制約に従う。

- local N が 16 の倍数である必要はない。
  native preshuffle が `rows_padded = ceil(local_N / 16) * 16` で padding する。
- したがって **N partition の都合で preshuffle 側を変更しない**。

N-D tensor（例 `[E, N, K]`）では最後の dimension が K、それ以前は
logical outer dimension として扱う。axis が K 以外の場合、
complete K-row を単位に ranges に対応する row を copy する
（outer dimension の row-major 座標を global 座標へ写す）。

## 5. K partition

axis が最後の dimension の partition は canonical の **block 単位**で切り出す。
local payload は block 粒度の slice を ranges 順に連結する。

```text
W[N][2560]、tp_size=2、rank0 K=[0,1280)、rank1 K=[1280,2560)

rank0: 各 row について block 0..39 を copy
rank1: 各 row について block 40..79 を copy

local logical K = 1280、local k_padded = 1280
その後 native preshuffle（FP8 は preshuffle_fp8_native、BF16 は対象外）
```

### quantization block alignment

`validate_tensor_partition_block_alignment(partition, block_elements)` が要求する:

```text
global_offset % block_elements == 0
extent        % block_elements == 0
```

block 途中からの partition や、block を rank 間で共有する実装は行わない。
`block_elements == 0`（BF16）では alignment 検査を省略する。
block 値は encoding 側（`QuantFormatDesc::block_elements`）が source of truth であり、
partition コードは format 専用定数を hard-code しない
（FP8 block128 = 128、BF16 = なし）。
K 以外の axis にはこの制約を課さない。
`MetaCountMode::MatrixBlock` の format はさらに scale tile
（`meta.block_n` / `meta.block_k`）へ partition axis を整列させる
（FP8 節）。

### FP8

FP8 の scale は 128 N × 128 K の shared tile であるため、
BF16 と同じ per-row 切り出しは適用できない。
partition boundary を N / K とも scale tile の 128 境界に制限し、
`MetaCountMode::MatrixBlock` の grid を N 分割では 128 行単位、
K 分割では `block_k` 単位で slice する。分割→結合は codes / scales 双方
byte 完全一致（`test_canonical_partition`）。128 に不整列の boundary は
`invalid_argument` で拒否する。対応は 2D tensor のみ
（N-D は `unsupported`。`preshuffle_fp8_native` も batch=1 のみ許可）。

## 6. rank-local canonical の owning buffer

partition を適用する場合だけ、global canonical view から
rank-local canonical payload を持つ owning host object を作る。

```cpp
struct OwnedCanonicalTensor {
    std::vector<int64_t> logical_shape;
    uint64_t k_padded;
    std::vector<uint8_t> data;         // BF16
    std::vector<uint8_t> codes;        // 量子化 codes
    std::vector<uint8_t> metadata1..4; // scale / metadata
};
```

`materialize_rank_local_canonical(partition, global_view, owned, local_view)` が
local view（local `logical_shape` / `k_padded` / stream span）を生成する。
`QuantizedModelReader` の mmap 領域は一切書き換えない。
`partition_plan == nullptr` の経路では buffer を一切作らず、
追加の host copy も発生しない。

## 7. weight loader への統合

```cpp
struct WeightLoadOptions {
    bool preshuffle = true;
    const TensorPartitionPlan* partition_plan = nullptr;   // nullptr = 現状どおり
};
```

- lookup は `QuantizedModelReader::resolve()` が alias 解決後に返す
  canonical tensor name（`QuantizedTensorView::name`）で行う。
  tie_word_embeddings で `lm_head → embed_tokens` となる場合、
  plan entry は canonical 名でしか解決されないため、
  alias 元と alias 先へ矛盾する geometry を与えることはできない。
- plan に entry のない tensor は global（replicated）としてロードされる。
- `preshuffle` flag は「partition するか」を制御しない。
  `preshuffle == false` でも partition は適用され、
  `global canonical → partition → local canonical → direct GPU upload` になる。
- `MatrixWeight.rows` / `cols` / `k_padded` は partition 使用時には local geometry。
  `MatrixWeight::partition` が small metadata として partition を保持する。
  GPU kernel へ partition descriptor は渡さない。
- `QuantizedModelReader` は global canonical reader のままであり、
  `tp_rank` / `tp_size` / GPU id / partition / preshuffle を一切持たない。

## 8. encoding との直交

partition は quantization encoding から独立である。

- BF16 で共通の `TensorPartitionDesc` を使う。encoding 専用の partition
  descriptor は作らない。
- `code_bytes_per_block` / metadata bytes は `QuantFormatDesc` から取得し、
  partition コードは format 専用定数を hard-code しない。
- canonical payload / preshuffle / GEMM kernel / quantization math /
  scale layout / code packing は変更しない。

## 9. Qwen3.8 Dense の例（GDN fused QKV）

GDN の `linear_attn.in_proj_qkv.weight` は global output order が
`[Q | K | V]` である。TP=2 では rank ごとに local Q / local K / local V を
segmented ranges で所有し、local tensor `[local Q | local K | local V]` へ
密に連結する（global preshuffle 後から 3 領域を拾わない）。

```text
qk_features   = linear_num_key_heads   * linear_key_head_dim
value_features= linear_num_value_heads * linear_value_head_dim
local_qk      = qk_features    / tp_size
local_value   = value_features / tp_size

rank r の ranges:
  {offset = tp_rank * local_qk,              extent = local_qk}
  {offset = qk_features + tp_rank * local_qk, extent = local_qk}
  {offset = 2*qk_features + tp_rank * local_value, extent = local_value}
```

Qwen3.8-27B（hidden 5120 / qk 2048 / value 6144 / conv 10240）で TP=2 の場合:

```text
rank0: {0,1024}   {2048,1024}   {4096,3072}
rank1: {1024,1024} {3072,1024}  {7168,3072}
```

model-specific な plan は `build_qwen35_tensor_partition_plan`
（`include/shelf/models/qwen35/weights/tensor_parallel_plan.h`）が生成する。
`TensorPartitionDesc` へ `head_dim` / `num_heads` / GDN / unit / model family
などの model 情報は入れない。

## 10. rank-3 tensor の例

`[E, N, K]` の 3D tensor は axis=0 の single-range partition で表現できる
（`TensorPartitionDesc` の基本 case）:

```text
rank0: axis=0, index=0, count=2, ranges={{0,256}}
rank1: axis=0, index=1, count=2, ranges={{256,256}}

local_shape = [256, 1280, 2560]
```

descriptor と N-D slicing は 3D tensor を扱える
（`tests/unit/test_canonical_partition.cpp` の 3D case）。
ただし **routed expert 平面の分割はここを通さない**。MXFP4 の N-D canonical partition は
今も `Status::unsupported` で、Qwen3.5 MoE は expert 軸を `MoeExpertGroupPlan` の
`expert_begin` / `experts` で切る。matrix axis の partition descriptor は backbone と
shared expert、そして dense FFN が使う（正本は
[qwen35moe.md](qwen35moe.md) §3「expert 軸の分割」）。

## 11. 現在の runtime で未対応のもの

partition metadata の導入は weight materialization 基盤であり、それ自体は分散実行の
対応完了を意味しない。dense の TP=2 execution は
[tensor_parallel_execution.md](tensor_parallel_execution.md) に実装済みで、その局所 geometry は
plan から来る。現在も未対応なものは以下。

- 単一 GPU の weight load path（`load_qwen35_weights_from_*`）は
  `WeightLoadOptions.partition_plan` が非 nullptr なら
  `Status::unsupported` で停止する。partition を受け付けるのは tensor parallel の rank load
  （`load_qwen35_weights_tensor_parallel_rank`）と MoE の rank load のみ
- MXFP4 の N-D canonical partition（`materialize_rank_local_canonical`）は `unsupported`
- `QuantizedModelReader` 側の rank-3 payload（fused expert 平面）は partition descriptor を
  作らない。expert 範囲で絞るのは plan の責任
- MTP block は partition しない（plan に entry を作らない）
- `tp_size > 2` の execution は `TpCoordinator` が `unsupported` で拒否する
- embedding と norm、router（`mlp.gate.weight`）、`mlp.shared_expert_gate.weight` は
  replicated（partition plan に entry を作らない）

## 12. transport から独立

partition format / plan は `RCCL` / AllReduce backend / communicator /
rank handle を一切持たない。

row-parallel layer に `partial outputs must be combined` という execution
requirement は存在するが、実装方法（RCCL / HIP peer-to-peer / custom reduction /
host mediated / topology-specific transport）は将来の Gate で選ぶ。

## 13. required tests

| test | 内容 |
| --- | --- |
| `test_tensor_partition` | descriptor validation / local shape / block alignment / plan（pure CPU） |
| `test_canonical_partition` | BF16 1D/2D/3D・segmented、segmented GDN / 3D unit / N-K partition、FP8、reject（pure CPU） |
| `test_qwen35_tensor_parallel_plan` | Qwen35 Dense plan builder（mini + Qwen3.8-27B geometry、q_proj head pair、GDN qkv coverage、replicated tensor）（pure CPU） |
| `test_weight_load` | BF16 / FP8 の quantized GPU load、preshuffle on/off、Qwen35 guard（GPU） |
