# EXL3 W16A16 kernel の ISA 調査

## 対象

- `src/shelf/kernels/optimized/linear/exl3.hip`
- `gemm_exl3_w16a16_kernel`（8 warp block + mul1 affine epilogue / packed pair 復号）
- build: `397e763` 相当、`-O3 --offload-arch=gfx1201`

ISA は `clang++ --save-temps` で device 側の `.s` を出して読んだ。

## kernel metadata

| 項目 | 値 |
| --- | ---: |
| wavefront | wave32 |
| threads / block | 256（8 warp） |
| VGPR | 92（割り当て粒度 24 で 96） |
| SGPR | 32 |
| LDS / block | 14,400 B |

## occupancy の再計算

`rocminfo` の gfx1201 は **2 SIMD/CU、Max Waves/CU = 32、wave32**。
RDNA4 の SIMD あたり 1,536 VGPR、割り当て粒度 24 とすると、96 VGPR は

```text
floor(1536 / 96) = 16 waves/SIMD
2 SIMD/CU -> 32 waves/CU
```

で **ハードウェア上限**に一致する。VGPR は occupancy を制限していない。
LDS も 14,400 B × 4 block = 57.6 KiB で、64 KiB/CU に 4 block 収まる。
つまり **資源上限は 4 block/CU（32 waves/CU）** であり、
「VGPR を 85 以下にしないと 3 block/CU にならない」という以前の見積は誤りだった。

実際の occupancy は **grid 数**で決まる。decode の rows=1 では
grid = (n/128) × 1 block、1 block = 8 warp。

| shape | blocks | waves | waves / 2048 slots |
| --- | ---: | ---: | ---: |
| n=1024 | 8 | 64 | 3.1 % |
| n=5120 | 40 | 320 | 15.6 % |
| n=17408 | 136 | 1,088 | 53.1 % |
| n=248320 (lm_head) | 1,940 | 15,520 | 100 %（資源上限で多重 wave） |

n=5120 / n=1024 は **CU あたり 0.6 / 0.125 block** しかなく、資源上限の 4 block/CU に
まったく届いていない。n=17408 でも 2.1 block/CU で、上限の半分程度。
VGPR 削減ではなく、**独立した warp を増やす**（K 分割や出力 tile の細分化）か、
1 warp 内の ILP を上げるのが筋になる。

`rocprofv3` の PMC は gfx1201 で 0 が返り、PC sampling も出力が得られなかった。
この環境では resident/active waves の実測ができないため、上記は resource と grid からの
計算値である。`wait` 命令数も待機サイクル数ではないので、stall の内訳は
段階別マイクロベンチで確認する必要がある。

## 命令 mix

全 2,168 命令（`state_from_ring` の modulo / 64-bit shift を残した状態）。

| 分類 | 数 | 割合 |
| --- | ---: | ---: |
| VALU | 1,022 | 47.1 % |
| wait / delay | 580 | **26.8 %** |
| LDS | 284 | 13.1 % |
| SALU / control | 262 | 12.1 % |
| VMEM | 20 | 0.9 % |

主な命令:

| 命令 | 数 |
| --- | ---: |
| `s_wait_dscnt` | 198 |
| `s_wait_alu` | 191 |
| `s_delay_alu` | 174 |
| `ds_load_b32` | 130 |
| `ds_bpermute_b32` | 88 |
| `v_mul_lo_u32` | 86 |
| `v_and_b32_e32` | 84 |
| `v_lshrrev_b64` | 64 |
| `v_add_co_u32` / `v_add_co_ci_u32_e64` | 45 / 46 |
| `v_cndmask_b32_e64` | 47 |
| `v_mul_hi_u32` | 16 |
| `v_sad_u8` / `v_sad_hi_u8` | 32 / 32 |
| `v_wmma_f32_16x16x16_f16` | 8 |

## 読み取れること

1. **MMA は主コストではない。**
   1 k chunk あたり `v_wmma` は 8 個（8 ks）。全 2,168 命令の 0.4 %。
   16×16×16 の WMMA 自体はボトルネックではない。

2. **wait / delay が 26.8 %。**
   `s_wait_dscnt`（LDS load 待ち）198、`s_wait_alu`（VALU 依存）191、
   `s_delay_alu` 174。命令 issue ではなく **依存 stall** が支配している。

3. **`state_from_ring` の runtime 除算と 64-bit shift が ISA に出ている。**
   `lo % word_count` / `hi % word_count` が `v_mul_hi_u32`、`v_add_co_u32`、
   `v_add_co_ci_u32_e64`、`v_cndmask_b32_e64` の列になり、
   `(pair >> shift)` が `v_lshrrev_b64` 64 個になっている。

4. **LDS の往復が多い。**
   `ds_load_b32` 130（ring の読み出しが主）、`ds_load_b128` 18（A fragment）、
   `ds_store_b32` 42（tile の u32 store）。tile を LDS に置いて A fragment を
   読み直す transpose が、decode と WMMA の間に `__syncwarp()` を挟んでいる。

5. **`ds_bpermute_b32` 88。**
   FWHT の 4 値 × 5 step × 2 row と row sum の reduction。

## 実験: 除算と 64-bit shift を消す

`state_from_ring` を次のように変えた。

- `% word_count` を条件減算へ（`lo`/`hi` は `2*word_count` 未満）。
- `(pair >> shift)` を `__funnelshift_r` へ。

ISA の変化:

| 項目 | 変更前 | 変更後 |
| --- | ---: | ---: |
| 全命令 | 2,168 | **1,980**（-8.7 %） |
| VALU | 1,022 | 893 |
| wait / delay | 580 | 528 |
| `v_mul_hi_u32` | 16 | 0 |
| `v_lshrrev_b64` | 64 | 0 |
| `v_alignbit_b32` | 0 | 64 |

性能（同じ GPU 1、`tg --context 128 --tokens 16 --warmup 1`、3 run）:

| 版 | tok/s |
| --- | ---: |
| 変更前 | 9.04 / 9.05 / 9.06 |
| 変更後 | 9.08 / 9.10 / 9.10 |

**命令数を 8.7 % 減らしても、decode は 0.5 % しか改善しなかった。**
この結果から、律速は命令スループットではなく、decode の依存 chain と
LDS / barrier / shuffle の stall であると判断した。変更は revert した。

## 段階別マイクロベンチ

production とは別に `/tmp` の kernel コピーへ `clock64()` を入れ、block 0 / warp 0 の
累積 cycles を shape 別に測った。5 sample 平均。

| shape | total | rot | decode | A load | B load | WMMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=1024, k=5120 | 0.178 ms | 10.6 % | **70.2 %** | 2.3 % | 2.6 % | 2.5 % |
| n=5120, k=5120 | 0.193 ms | 9.6 % | **72.2 %** | 2.1 % | 2.5 % | 2.3 % |
| n=5120, k=6144 | 0.230 ms | 9.6 % | **72.3 %** | 2.1 % | 2.5 % | 2.3 % |
| n=5120, k=17408 | 0.641 ms | 9.6 % | **73.1 %** | 2.1 % | 2.5 % | 2.3 % |
| n=17408, k=5120 | 0.302 ms | 9.4 % | **73.4 %** | 3.2 % | 4.1 % | 1.6 % |
| lm_head | 3.208 ms | 9.1 % | **73.9 %** | 4.4 % | 4.8 % | 1.2 % |

- phase の合計は約 88〜90 %。残りは `__syncthreads()`、出力 epilogue、その他。
- clock probe では decode が 70〜74 %、入力回転が 9〜11 %、A/B load + WMMA が 8〜10 %。
- WMMA の issue 時間は 1〜2.5 % と出たが、後述の ablation では WMMA が 43 % を占める。
  `clock64()` を tile ごとに挟むと依存 chain が切れ、decode の割合が過大に出る。
  以下の判断には ablation の値を使う。

## decode / WMMA の ablation

production と同じ kernel から一部だけを無効化し、shape と split を固定して
GEMM + reduce の時間を比較した。n=5120 / k=5120 / split=8、10 sample 平均。

| 版 | 時間 | 差分 |
| --- | ---: | --- |
| 全体 | 0.0864 ms | — |
| decode を空にする（ring stage も含めて削除） | 0.0556 ms | decode + stage = 36 % |
| decode を空にし、WMMA を 1 個の half 加算へ置換 | 0.0184 ms | WMMA = 43 % |
| decode を空にし、A/B load と WMMA を削除 | 0.0190 ms | A/B load は無視できる |
| decode の LDS store だけ削除 | 0.0689 ms | store = 20 % |
| decode の state 抽出 + hash だけ削除 | 0.0694 ms | 抽出 + hash = 20 % |

**M=1 では WMMA が最大のコスト（43 %）**。16 行のうち 1 行しか使わないため、
演算の 15/16 が捨てられる。decode は 36 % で、hash の置換や modulo の除去だけでは
全体を動かせない。残り約 21 % は入力回転、epilogue、launch、reduce である。

## hash / modulo の A/B（棄却）

参照実装 `Luke458/qwen38-27b-exl3-rdna4` は mul1 の hash を full-rate の 24-bit 乗算
2 回へ置き換えている。gfx1201 では `v_mul_lo_u32` が 1/4 rate のためである。
同じ変更と、`% word_count` を条件付き減算へ変える変更を 4 通りで比較した。

- A: 変更なし
- B: modulo だけ条件付き減算
- C: hash だけ 24-bit 乗算 2 回
- D: 両方

n=5120 / k=5120 / split=8 の GEMM 時間は A 0.0859 / B 0.0856 / C 0.0860 /
D 0.0856 ms（2 round）。**差は 1 % 未満**。`v_mul_lo_u32` は 188 → 20 命令まで
減ったが、時間は変わらなかった。この kernel は ALU スループットではなく
WMMA と tile 間の依存で律速している。両変更は revert した。

## batch-aware split-K（採用）

grid 数が足りない shape で K を分割する。`rows == 1` 固定ではなく、行 tile と
n tile の両方から決める。

- 1 block は従来どおり 16 行 × 128 列を担当する。
- `splits > 1` のとき、grid を `(n_tiles * splits, row_tiles)` にし、
  block は `split` ごとに K chunk を飛び飛びに処理する。
- 部分和 `acc` と rowsum を scratch に書き、reduce kernel が split 方向に加算して
  affine 変換、`H128`、`svh` を適用する。
- 自動 split は `grid_x <= 64 ? 8 : grid_x <= 160 ? 4 : 1`。scratch 容量と
  K chunk 数で上限を下げる。scratch は 524,288 float（2 MiB）で、
  rows=8 / n=5120 / split=8 まで収まる。
- 部分和の順序が変わるため、CPU affine 参照との差は split あり / なしの両方で
  確認する。

### 見つけた race

最初の実装は E2E の生成が崩れた。`rows=5, n=5120` の数値テストも実行ごとに
PASS / FAIL が変わった。原因は K ループ先頭で、前の chunk の `xrot` をまだ読んでいる
warp がある間に、別の warp が次の chunk の `xrot` を上書きしていたことである。
K ループ先頭に `__syncthreads()` を追加して解消した。3 回連続で PASS、
E2E は `11751,13,198`（` Paris.`）に戻った。

### 結果

同じ GPU 1、`tg --context 128 --tokens 16 --warmup 1`、3 run。

| 版 | tok/s |
| --- | ---: |
| split なし（`SHELF_EXL3_SPLIT=1`） | 8.85 |
| auto split | **12.16 / 12.20 / 12.23** |
| E2E decode | 12.67 |
| MTP-2 | 14.30 |

shape 別の GEMM + reduce（10 sample 平均、split=1 → 最良）:

| shape | split=1 | 最良 split | 時間 |
| --- | ---: | ---: | ---: |
| n=1024, k=5120 | 0.1526 ms | 8 | 0.0333 ms |
| n=5120, k=5120 | 0.1521 ms | 8 | 0.0861 ms |
| n=5120, k=17408 | 0.4960 ms | 8 | 0.2484 ms |
| n=17408, k=5120 | 0.2986 ms | 3（scratch 上限） | 0.2537 ms |

reduce kernel は平均 3.5 µs。split を増やすと block 数は増えるが、reduce と
launch の追加は小さい。prefill 128 は 199.67 tok/s で split 前と同じ。

batch>1 の GEMM 時間（n=5120, k=5120, rows=8）:

| split | 時間 |
| --- | ---: |
| 1 | 0.1697 ms |
| 2 | 0.1222 ms |
| 4 | 0.1301 ms |
| 8 | **0.0944 ms** |

rows=8 でも split 8 が効くため、scratch を 2 MiB まで広げた。rows=16 では
split 4 / 8 が 0.1024 / 0.1017 ms でほぼ同じ、rows=32 以上では split の効果は
1 % 程度になる。

## 2 系列インターリーブ（棄却）

2 本の ring と 2 枚の tile を用意し、ring 2 本を stage → sync → 2 tile を復号 →
sync → A/B/WMMA を 2 回、という形にした。barrier は 1 tile あたり 2 回から
2 tile あたり 2 回へ半減する。

| 版 | tok/s |
| --- | ---: |
| 1 系列（現行） | 9.04 / 9.05 / 9.06 |
| 2 系列 | 8.96 / 8.96 / 9.04 |

**改善せず**、LDS が 14.4 KiB → 16.4 KiB に増えて 4 block/CU から 3 block/CU へ
落ちた可能性がある。decode の依存 chain はすでに compiler が 4 pair 分
インターリーブしており、明示的な 2 系列化の追加コストが上回った。revert した。

## ring window の 4-word 一括ロード（採用）

1 lane が担当する 8 状態は 8 連続 position なので、16-bit window の開始位置は
`ka`（half では 1 bit 交互）ずつしか進まない。8 状態を覆う範囲は最大でも
約 102 bit であり、4 個の u32 に収まる。`state_from_ring` を状態ごとに呼ぶと
1 tile あたり 16 回の LDS load になるが、先頭 word を 4 個だけ load し、
`pick4` で相対 word を選ぶ形にすると 4 回で済む。

- 先頭 word `w0` は最大 `2 * word_count - 1` になる。4 個の index は
  条件付き減算を 2 回行って `word_count` 未満へ落とす。1 回では
  `w0 + 3 >= 2 * word_count` のときに範囲外になる（試作で踏んだ）。
- window の抽出式は `state_from_ring` と同じで、相対 index が 0..3 に収まる。
- 余分な選択命令は増えるが、この kernel は ALU 律速ではないため、
  LDS load の削減がそのまま効いた。

同じ shape・split での GEMM + reduce（10 sample 平均）:

| shape | 変更前 | 変更後 |
| --- | ---: | ---: |
| n=1024, k=5120 | 0.0338 ms | 0.0327 ms |
| n=5120, k=5120 | 0.0864 ms | **0.0777 ms** |
| n=5120, k=17408 | 0.2506 ms | **0.2175 ms** |

全体（同じ GPU 1、`tg --context 128 --tokens 16 --warmup 1`、3 run）:

| 条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| decode（auto split） | 12.16〜12.23 tok/s | **13.63〜13.66 tok/s** |
| decode（split なし） | 8.85 tok/s | 9.51 tok/s |
| E2E decode | 12.67〜12.82 tok/s | **14.50 tok/s** |
| MTP-2 | 14.30 tok/s | **15.48 tok/s** |
| prefill 128 | 199.67 tok/s | **226.06 tok/s** |

prefill も同じ decode_tile を使うため、行数が多い場合にも効果がある。
`test_exl3_linear` の ka=1 / 2 / 3+half / 4 / 5 / 6 / 7 / 8、half 1.5 / 2.5、
rows 1〜17、split 1〜32 がすべて PASS し、E2E は `11751,13,198`（` Paris.`）を
維持した。

## 参照実装との比較

`Luke458/qwen38-27b-exl3-rdna4` は同じ gfx1201 で Qwen3.8-27B EXL3 を動かし、
plain decode 41.3 tok/s、MTP-3 で 73.9 tok/s を記録している。dot kernel は
GEMV（`fdot2`）で、1.1〜1.25 T weights/s、ALU-bound と報告している。
multi-row GEMV は m=4 で m=1 の 1.07〜1.31 倍のコストである。

この差は主に M=1 での WMMA の無駄に由来する。現在の W16A16 は 16 行を同時に
計算するため、M=1 では演算の 15/16 を捨てる。ablation でも WMMA が 43 % を占めた。
一方、batch を増やすと WMMA の無駄は減る。n=5120 / k=5120 / split=8 の
1 step は rows=1 で 0.0861 ms、rows=8 で 0.0944 ms、rows=16 で 0.1017 ms であり、
batch 8 / 16 では token あたりコストがそれぞれ 1/7.3 / 1/13.5 になる。

GEMV は batch 1〜16 のすべてで W16A16 より 2〜3 倍速い見込みだが、
multi-request / mixed batch を前提とする現在の設計では decode 経路を 2 つ持つ
コストが大きい。**今回は W16A16 を維持し、GEMV 経路は実装しない**。
この判断は性能の上限を 1.55 倍程度に限定するが、multi-request の amortize と
prefill の WMMA をそのまま使えることを優先する。

## 次の候補

1. **rows 別の split / scratch チューニング**。batch 8 / 16 では token あたり
   1/7.3 / 1/13.5 まで amortize している。rows から split 数と scratch 容量を
   決め、multi-request の実測で確認する。
2. **decode の LDS store 削減**。store は全体の約 20 %。256 half を A fragment の
   配置で直接作れれば LDS を経由しないが、lane 間 shuffle が増える。2 系列
   インターリーブが効果なかったため、優先度は低い。
3. **WMMA の直列依存**。M=1 では WMMA が 43 % を占めるが、16 行のうち 1 行しか
   使わない構造上のコストである。W16A16 を維持する限り削れない。
4. **hash / modulo の削減**は効果なし。gfx1201 の `v_mul_lo_u32` が 1/4 rate でも、
   この kernel では律速にならない。
5. GEMV 経路は batch 1〜16 で 2〜3 倍の余地があるが、今回は実装しない判断とした。
