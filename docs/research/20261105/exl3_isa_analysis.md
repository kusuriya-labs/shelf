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

## prefill の row group 再利用（採用）

prefill は同じ重み tile を 16 行ごとに復号し直していた。rows=128 では
decode + stage が 31.5 %、うち state 抽出 + store が 14.9 %、global stage が 16.6 %。
16 行 tile を複数まとめて 1 block で処理すれば、復号と stage をその分だけ共有できる。

- kernel を `kGroups` でテンプレート化し、`kRows = 16 * kGroups` 行を 1 block が担当する。
- `xrot` と `rowsum` は `kRows` 行ぶん確保し、`acc[kGroups]` に row group ごとの
  累算を持つ。WMMA は復号した A fragment を group 間で再利用する。
- epilogue は 16 行ぶんの `area.accum` を使い回し、group ごとに affine / H128 / svh を
  適用する。
- split-K の reduce は 16 行 / block のままなので、reduce の grid は
  `ceil(rows / 16)` で別に計算する。ここを row group の grid と共有して
  rows 16 以降が書かれないバグを一度作った。

`kGroups` の比較（n=5120, k=5120, rows=128, split=1）:

| 版 | 時間 | pp128 |
| --- | ---: | ---: |
| 1 | 0.528 ms | 226.06 tok/s |
| 2 | **0.441 ms** | **265.51 tok/s** |
| 4 | — | 188.84 tok/s |

groups=4 は LDS が約 26 KiB になり 2 block/CU へ落ちるため、block 数が増える
rows=128 でも逆効果だった。groups=2 は LDS 約 18 KiB で 3 block/CU を保てる。
rows=32 で groups=2 を強制すると block 数が 40 に減り、0.202 → 0.247 ms へ悪化する。
そのため自動選択は次を条件にする。

```text
rows >= 32 かつ grid_x * ceil(rows / 32) >= 128 のとき groups=2
```

結果（同じ GPU 1、3 run）:

| 条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| prefill 128 | 226.06 tok/s | **265.51 tok/s** |
| prefill 512 | 230.85 tok/s | **273.57 tok/s** |

decode（rows=1）は groups=1 のまま。n=5120 / k=5120 / split=8 の GEMM は
変更前 0.0778 ms、変更後 0.0777 ms で差がない。`test_exl3_linear` の
rows=64 / n=8192 で groups=2 の経路を検証している。

## 入力回転の事前計算（採用）

現行 kernel の定数は `kOutBlock = 128`、`kRowTile = 16`、`kThreads = 256`
（8 warp、1 warp = 16 列）、`kGroups = 1`（decode）/ `2`（prefill 条件付き）。
入力回転は CTA ごとに、同じ Activation を n tile 数と split 数だけ繰り返していた。

まず回転ブロック（`suh` load、BF16 load、FWHT、FP16 変換、`xrot` LDS 書き込み）を
丸ごと外した ablation を、auto split で測った。

| rows | n=5120 k=5120 | n=5120 k=17408 | n=17408 k=5120 |
| ---: | ---: | ---: | ---: |
| 1 | -15 % | -17 % | -16 % |
| 4 | -33 % | -22 % | -21 % |
| 16 | -25 % | -28 % | -31 % |
| 128 | -41 % | — | — |

回転は decode でも 15 %、prefill では 41 % を占める。`fwht128` だけを外した
以前の ablation が効果なしだったのは、重いのは FWHT ではなく、
BF16 load / suh / LDS 書き込みだったためである。

実装:

- `rotate_exl3_kernel` が 1 linear につき 1 回だけ `xrot[rows][k]`（FP16）を作る。
  row ごとに 1 block、8 warp で K/128 block を分担する。
- GEMM は `kPre` template で分岐し、`xrot` を global から LDS へコピーして使う。
  行 sum はコピー後の LDS から計算するので split でも整合する。
- `xrot` は execution workspace に 80 MiB 確保する。`rows * k * 2 <= 80 MiB` かつ
  `rows >= 4` のときだけ使い、それ以外は従来の fused 経路に落とす。

結果（同じ GPU 1、3 run、`SHELF_EXL3_PRE=0` が fused）:

| prompt | fused | pre | 改善 |
| ---: | ---: | ---: | ---: |
| 128 | 265.51 | **315.25** | +18.7 % |
| 512 | 273.57 | **328.92** | +20.2 % |
| 1024 | 274.37 | **326.17** | +18.9 % |
| 2048 | 274.06 | **330.00** | +20.4 % |

32 MiB では 2048×17408 の `down_proj` が fused に落ち、pp2048 は 314.37 だった。
80 MiB に広げると 330.00 まで上がる。decode（rows=1）は fused のまま。
`n=5120 / k=5120 / split=8` の GEMM は 0.0785 → 0.0775 ms で、fused 側の回帰はない。

### decode prefetch（棄却）

次の tile の ring を現在の decode/WMMA より先に global load する形も試した。
単一 tile + 2 ring にしたが、rows=1 0.0792（base 0.0793）、rows=16 0.0997（0.0978）、
rows=128 0.4959（0.4931）で改善しなかった。compiler がすでに unroll で
load を先行させており、明示的な prefetch の追加コストが上回る。revert した。

### rowsum のガード

非 split 経路の `rowsum` 書き込みは index が `kRows` 未満で範囲内、非アクティブ行の
`row_acc` は 0、epilogue も `r >= active_rows` を弾くため誤出力はなかった。
防御的に `r < active_rows` のガードを追加した。

## ka=4 の 8 状態一括抽出（採用）

`codebook_pair_h_bits()` は既に `v_sad_u8` + `v_sad_hi_u8` を使っており、
hash は `v_mul_lo_u32` である。24-bit split hash は ka=4 抽出の後でも **±0.5 %**
で効果がなかった。`v_mul_lo_u32` は 1/4 rate だが、この kernel では律速ではない。

generic 経路は state ごとに 4 word から 2 word を `pick4` で選び、64-bit 結合と
可変 shift を行う。ka=4 / half=0 では 1 lane の 8 状態が 2 word に収まり、
オフセットが lane に依存しない。

- `w0 = lane + 31`、`w1 = lane + 32`（word_count=32、1 回だけ条件付き減算）
- `pair01 = (w0 << 32) | w1` → `v_alignbit` shifts 28, 24, 20
- `pair11 = (w1 << 32) | w1` → shifts 16, 12, 8, 4, 0
- 2 LDS load + 8 `v_alignbit` + 8 mask で 8 状態

GEMM 単体（auto split、導入前 / 導入後）:

| shape | rows | 導入前 | 導入後 |
| --- | ---: | ---: | ---: |
| n=5120 k=5120 | 1 | 0.0792 | **0.0754** |
| n=5120 k=5120 | 4 | 0.0848 | **0.0792** |
| n=5120 k=5120 | 32 | 0.1759 | **0.1673** |
| n=5120 k=5120 | 128 | 0.4961 | **0.4805** |
| n=5120 k=17408 | 1 | 0.2282 | **0.2117** |
| n=17408 k=5120 | 1 | 0.2217 | **0.2083** |

n=5120 / k=5120 の rows=16 は導入前の測定が 0.1246 ms と外れていたため表から外した。
導入後は 0.0956 ms。全体として **5〜7 %** の短縮。

production（ka=4 後の precompute on/off）:

| prompt | fused | pre | 改善 |
| ---: | ---: | ---: | ---: |
| 128 | 278.95 | **336.44** | +20.6 % |
| 2048 | 289.60 | **353.68** | +22.1 % |

decode は 13.36〜13.39 → **14.80〜14.91 tok/s**。E2E は 15.93 tok/s、
MTP-2 は 15.68 tok/s。ka=4 以外（lm_head の ka=6、half）は generic 経路のまま。

## 棄却した試行（依存チェーンと LDS）

- **Accumulator の 2 本 / 4 本分割**: `acc[g]` を偶数・奇数 K tile で分け、最後に
  FP32 加算した。2 本は rows=1 0.0746 → 0.0742 ms、rows=128 0.4835 → 0.4784 ms で
  誤差範囲。4 本は rows=128 が 0.4835 → 0.5892 ms へ悪化した（VGPR 増で occupancy 低下）。
  WMMA の累積依存チェーンは主因ではない。
- **xrot の row pitch 128 → 130 halfs**: B fragment の bank を分散させる狙い。
  rows=1 0.0746 → 0.0743 ms、rows=128 0.4835 → 0.4837 ms で変化なし。
- **tile の column pitch 16 → 18 halfs**: A fragment の bank を分散させる狙い。
  rows=1 0.0746 → 0.0983 ms（+32%）、rows=128 0.4835 → 0.5896 ms へ悪化。
  16B vector load の alignment が崩れるためで、pitch 変更は逆効果。
- 以上から、decode → LDS → WMMA の逐次依存や bank conflict を主因とする仮説は
  支持されない。次の大きな候補は M=1〜8 の Small-M カーネル（fdot2/dot2 系）で、
  WMMA の M 方向 15/16 の浪費を避ける経路である。

## INT8 1-GEMM の品質ゲート（棄却）

gfx1201 の INT8 WMMA は FP16 の **1.96 倍**（93.5 → 183.1 T-MAC/s）で、
mul1 の 11-bit `d` を 8-bit に丸めれば 1 回の INT8 GEMM で MMA を半分にできる。
既存の FP16 カーネルに「重みだけ int8 相当（`d` を 4 刻みに丸める）」シミュレーションを
入れ、prose 32 position で品質を比較した。

| 指標 | FP16 | 重み int8 |
| --- | ---: | ---: |
| PPL | 3.7558 | 3.8093（**+1.43 %**） |
| mean KL | — | 0.000615 |
| top-1 一致 | — | 100 % |
| top-5 overlap | — | 99.4 % |
| logit max abs delta | — | 0.413 |

プロジェクトの既定ゲートは PPL 回帰 1.0 % 以下である。**重みだけの int8 で既に
+1.43 %** となり、超過する。活性も int8 にすると量子化誤差は約 1.1 % RMS まで増え、
さらに悪化する。

- `d` を 2 個の int8（lo/hi）に分ける 2-GEMM は重み精度を保てるが、
  INT8 1.96 倍 × 2 回で MMA 時間は FP16 と同等になり、活性 int8 の誤差だけ残る。
- したがって、INT8 1-GEMM は品質・性能の両面で採用しない。

## Trellis ring の lane shuffle 化（採用）

ka=4 の ISA では、`global_load_b32` → `ds_store_b32`（ring）→ wave barrier →
`ds_load_b32`×2 → `v_alignbit_b32` の順だった。ka=4 で必要なのは自分の word と
1 つ前の lane の word だけなので、ring を LDS に作らず、
`source[lane]` と `__shfl_sync(0xFFFFFFFFull, b, (lane + 31) & 31, 32)` に置き換えた。
gfx1201 ではこれは `ds_bpermute_b32` 1 命令になる。

差分:

- 削除: `ds_store_b32`、wave barrier、`ds_load_b32`×2、ring のアドレス計算
- 追加: `ds_bpermute_b32`
- 変更なし: `v_alignbit_b32` / `v_bfe_u32`、`v_mul_lo_u32`、SAD、
  `ds_store_2addr_b32`、`ds_load_b128`、`v_wmma_f32_16x16x16_f16`

GEMM 単体（auto split、ring / shuffle）:

| rows | n=5120 k=5120 | n=17408 k=5120 |
| ---: | ---: | ---: |
| 1 | 0.0757 → **0.0746** | 0.2080 → **0.2051** |
| 4 | 0.0810 → **0.0797** | 0.2211 → **0.2187** |
| 16 | 0.0958 → **0.0947** | 0.2924 → **0.2852** |
| 32 | 0.1672 → **0.1652** | 0.4824 → **0.4682** |
| 128 | 0.4799 → **0.4670** | — |

**1〜3 %** の短縮。production の decode は横ばい（14.79〜14.85 tok/s）、
prefill 128 は 336.44 → 339.82、2048 は 353.68 → 356.55 tok/s。
Trellis ring の LDS 往復は主因ではなく、寄与は 1 % 程度だった。

## groups / split-K の分離スイープ（採用）

M=16→32 で総 throughput が伸びない原因を、`groups` と split-K を固定して分離した。
n=5120 / k=5120、時間は ms、括弧内は `M / time` の相対 throughput。

| M | g1 s1 | g2 s1 | g1 s4 | g2 s4 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.1513 | 0.1521 | 0.0799 | **0.0792** |
| 16 | 0.1533 | 0.1554 | 0.0863 | **0.0858** |
| 32 | 0.1856 | 0.2240 | 0.1540 | **0.1342** |
| 64 | 0.2858 | 0.2963 | 0.2874 | **0.2482** |
| 128 | 0.5529 | 0.4470 | 0.4894 | **0.4217** |

- **M=16→32 の頭打ちは groups=2 ではなく split 容量**だった。production は
  M=16 で split 6、M=32 で split 3 に容量制限され、block 数が 240 → 120 に減る。
  split=1 固定では M=16→32 の throughput は 104→172（+65%）まで伸びる。
- split=4 固定では **M≥32 で groups=2 が 15〜16% 速い**。M=8/16 はほぼ同等。
- M=1/4 は split=4 が最適（M=1: s4 0.0713 / s8 0.0771、M=4: 0.0767 / 0.0784）。
- n=1024 は block 数が少ないため **split=8** が最適（M=1: s4 0.0404 / s8 0.0299）。
- n=17408 も M≥32 で groups=2 が 8〜14% 速い（M=32: 0.4656 / 0.5138、
  M=128: 1.5492 / 1.7640）。

### 新しい dispatch

```text
split: grid_x <= 16 ? 8 : (grid_x <= 160 ? 4 : 1)
groups=2: rows >= 32 かつ grid_x * ceil(rows/32) * splits >= 128
```

- split scratch を 2 MiB → **12 MiB** に広げ、M=32/64/128 の split=4 を可能にした。
  M=32 (0.66M float)、M=64 (1.31M)、M=128 (2.62M) が収まる。
- M=8/16 は groups=1、n=1024 は split=8 のまま。

production（同じ GPU 1、3 run）:

| 条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| decode | 14.79〜14.85 | **15.00〜15.10** |
| prefill 128 | 339.8 | 340.4（横ばい） |
| prefill 2048 | 356.6 | 355.9（横ばい） |
| E2E decode | 15.89 | 15.86 |
| MTP-2 | 15.60 | 15.77 |

単一 stream の伸びは 1.5 % 程度だが、GEMM 単体では M=32 で約 28%（g1 s3 →
g2 s4）、M=64 で約 13%、M=128 で約 6% 改善する。batch decode の主対象は
この範囲なので、Continuous Batching の改善として採用する。

## pp2048 のカーネル内訳と FP16 展開の診断

pp2048（2048 token、GPU 1）を `rocprofv3 --kernel-trace` で測った。総カーネル時間
5664 ms の内訳:

| カーネル | 時間 | 割合 |
| --- | ---: | ---: |
| EXL3 GEMM `<2,false,true>` | 5423.5 ms | **95.8 %** |
| rotate_exl3 | 31.5 ms | 0.6 % |
| GDN recurrence | 51.0 ms | 0.9 % |
| PagedAttention prefill | 25.4 ms | **0.4 %** |
| SwiGLU / mul / residual / norm 等 | 約 100 ms | 1.8 % |

**PagedAttention は遅くない**。pp2048 のボトルネックは EXL3 GEMM である。
EXL3 の 400 launch の内訳は次のとおり。

| n | k | launches |
| ---: | ---: | ---: |
| 17408 | 5120 | 128 |
| 5120 | 6144 | 64 |
| 5120 | 17408 | 64 |
| 6144 | 5120 | 48 |
| 10240 | 5120 | 48 |
| 1024 | 5120 | 32 |
| 12288 | 5120 | 16 |

### 重みを FP16 に展開した場合（診断）

同じ `launch_gemm_exl3_w16a16` 構造（pre 経路、groups=2、split=1、M=2048）で、
trellis の代わりに事前復号済み FP16 tile を global から読む診断カーネルを 1 shape
ずつ実行した。FP16 tile は 0 で埋め、メモリ流量だけを再現している。

| n | k | 現行 | FP16 tile | 比 |
| ---: | ---: | ---: | ---: | ---: |
| 17408 | 5120 | 17.43 ms | 18.21 ms | **0.96** |
| 5120 | 6144 | 6.43 ms | 5.69 ms | 1.13 |
| 5120 | 17408 | 18.12 ms | 17.32 ms | 1.05 |
| 6144 | 5120 | 6.47 ms | 5.71 ms | 1.13 |
| 10240 | 5120 | 10.79 ms | 10.24 ms | 1.05 |
| 1024 | 5120 | 1.18 ms | 0.97 ms | 1.21 |
| 12288 | 5120 | 12.93 ms | 12.39 ms | 1.04 |

- 最大の `n=17408, k=5120`（gate/up、128 launch、EXL3 時間の 41 %）は
  **FP16 展開の方が遅い**。FP16 は 4 bpw trellis の 4 倍の重み流量で、
  M=2048 では同じ重みを 64 row block が読むため bandwidth-bound になる。
- 現在の pp2048 は 361.0 tok/s。診断結果を trace の shape 別時間へ適用すると、
  FP16 展開の pp2048 は **約 366〜381 tok/s**（+2〜6 %）に留まる。
- モデル全体の FP16 重みは 27B × 2 B ≈ **54 GB** で、32 GB VRAM に収まらない。
  レイヤーごとに on-demand 展開しても、最大 178 MB の書き込み・読み出しが加わる。
- したがって「先に FP16 に展開して同じカーネルで回す」案は採用しない。
  より大きな row tile（128 行）で重みの再読み出しを 4 分の 1 にする dense GEMM なら
  別だが、それは現在の `launch_gemm_exl3_w16a16` ではない。

### A0/B0/B3 分離（1 layer、M=2048、K=N=5120、g=1、s=1）

`tests/benchmarks/exl3_isolation/` に独立ベンチを作り、rocBLAS FP16 GEMM（A0）、
現行 WMMA に事前復号 FP16 fragment を直接渡す経路（B0）、現行 EXL3（B3）を
同一 shape で測った。10 warmup + 50 samples。

| Stage | kernel ms | tok/s（単一 Linear） | TFLOPS | A0 比 |
| --- | ---: | ---: | ---: | ---: |
| A0 rocBLAS FP16 | 0.7139 | 2,868,713 | **150.40** | 1.00× |
| B0 現行 WMMA + FP16 fragment | 4.7511 | 431,055 | 22.60 | 6.65× 遅い |
| B3 現行 EXL3 mul1 | 6.6431 | 308,291 | 16.16 | 9.31× 遅い |

- B0 は bit 抽出、mul1 hash、SAD、LDS tile 往復をすべて省き、事前復号した
  raw FP16 tile を WMMA の A fragment として直接読む。
- **B0 と B3 の全出力は bit 一致**（`max_abs=0`）。B0 は忠実な decode-free 基準。
- 現行 WMMA 構造は、**decode を一切含まなくても rocBLAS の 6.65 倍遅い**。
  EXL3 decode 全体の追加は B0 比 1.40 倍に過ぎない。
- したがって主因は hash / bit抽出 / ring / accumulator ではなく、
  **WMMA のタイル構造・データ供給・スケジューリング**である。decode をさらに
  削っても、このカーネルの上限は約 1.4 倍。次は 128×128 以上の large-tile WMMA
  （register blocking、K pipeline）か、128 行単位で重みを再利用する dense GEMM。
- 成果物: `tests/benchmarks/exl3_isolation/result_summary.md`、`result.csv`、
  `isa/stage_b0.s`、`isa/stage_b3.s`、`profiles/pp2048_breakdown.txt`。

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

1. **M=1〜2 の Small-M dot2 カーネル**。現在の M=1 は 16 行 WMMA の 15/16 を捨てる。
   `v_dot2_f32_f16` で復号済み tile を直接畳めば、WMMA の M 浪費を避けられる。
   M=4 以上は WMMA を維持する。品質リスクはない。
2. **ka=6 の windows8 特殊化**。lm_head はまだ generic 経路。decode 全体への寄与は
   小さいが、同じ 2-word / alignbit の考え方を適用できる。
3. **prefill の dense FP16 GEMM**。再構築 0.17 ms（5120²）は小さく、M=512 以上では
   パイプライン化した dense GEMM が fused を上回る可能性がある。素朴な 128×128 tile は
   10 T-MAC/s で、double buffering が必要。
4. **row group の拡大**。groups=2 が LDS と block 数のバランス点。xrot を global から
   直接読むようにして LDS を削れば groups=4 を再評価できる。
5. **INT8 1-GEMM は品質ゲートで棄却**。重みだけの int8 で PPL +1.43 %、既定ゲートは
   +1.0 % 以下。2-GEMM も MMA が FP16 と同等で利得がない。
