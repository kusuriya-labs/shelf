# EXL3 mul24 GEMM の重み pointer hoist

## 背景

decode 1 row の graph replay 30.7 ms のうち、GEMM は約 25.9 ms（実 graph で GEMM node を同じ grid の空 kernel に置換した差）を占める。FFN の GEMM（gate/up と down）だけで 16.75 ms で、重み読み出し 8.565 GB を実効帯域 635 GB/s で換算した 13.5 ms に対して約 3.2 ms 遅い。

FFN の GEMM は grid を 25〜100% に縮めると時間がほぼ比例して減る（固定分は kernel あたり 4〜7 µs）。つまり遅い主因は定常状態のスループットである。

## 単体 microbench（gate/up 形状 n=17408, k=5120, split 4、24 buffer を巡回して cache を外す）

| 条件 | 時間 µs | 見かけ帯域 GB/s |
|---|---:|---:|
| 現行 kernel | 83.2 | 536 |
| decode 演算を外す（raw word を b に使用） | 77.8 | 573 |
| 重み load のみ（dword） | 76.3 | 584 |
| 重み load のみ（16 byte） | 75.6 | 589 |
| 連続読み（grid-stride、16 byte） | 75.0 | 594 |
| 重み pointer を hoist（採用） | 79.8 | 558 |

現行 kernel の ISA は、load ごとに 64 bit の address 計算（v_add_co_u32 と v_add_co_ci_u32）と s_wait_alu を持ち、1 group あたり約 90 個の VALU が address 計算に使われていた。per-lane の基準 pointer を 1 つ持ち、group ごとに加算して進めると、load は即値 offset になる。

## 変更

`decode.hip` の `exl3_mul24_gemm_kernel` と `projection.hip` の `group_body` で、`tile_base + (g * 8 + s) * kt_stride` の再計算をやめ、lane の 3 本の基準 pointer（i0、i1、i2）を group ごとに `8 * kt_stride` ずつ進める。演算順序は変えない。

## 結果

decode（1 row、256 token、生成 token 列は完全一致）と DFlash2 verify（8 row）の graph replay を、新旧 build を交互に実行して比較した。

| 条件 | 旧 ms | 新 ms | 差 |
|---|---:|---:|---:|
| decode 1 row（6 run） | 30.814 ± 0.247 | 29.899 ± 0.137 | −0.915（−3.0%） |
| DFlash2 verify 8 row（4 run） | 34.297 ± 0.056 | 33.476 ± 0.040 | −0.821（−2.4%） |

単体 microbench は split 4 の gate/up で 83.5 → 80.3 µs、down 形状（n=5120, k=17408）の split 12 で 84.0 → 79.4 µs。split 数による差は小さい。

検証: 全回帰 43 pass / 1 skip / 0 fail、ctest の exl3 関連 7 件 pass。

## 残りの余地（仮説を含む）

- 重み load のみの上限 75.6 µs に対して、変更後は 79.8 µs。差約 4 µs は decode 演算が約 2 µs、WMMA と activation・scale の load が約 2 µs。
- 上限の 75 µs は 636 GB/s の連続読み（70 µs）に対して約 5 µs 遅い。これは kernel ごとの起動と立ち上がり・終端の固定分に相当し、kernel 内の最適化では取れない。
