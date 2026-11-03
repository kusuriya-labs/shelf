# Gate A' — MXFP4 dense linear の optimized kernel 実装記録

Gate A（format / load / correctness path）は [mxfp4_gate_a.md](mxfp4_gate_a.md)、
現在形の contract は `docs/developer/canonical_soa_payload.md` §9 / §11、
`docs/developer/quantization.md`、`docs/developer/kernels.md` に移動済み。
ここには経緯・実測・採否だけ残す。

設計の大半は `../references/phaseshift-mxfp4` の MXFP4 W4A8 kernel 群
（`src/phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.hip`、
`include/phaseshift/quantization/mxfp4/mxfp4_native.h`）を確認した上で、
同系統の構造を shelf の contract（activation group 128、64 K fragment block、
`WeightSlot` stride 検証）へ合わせ込んだ実装である。

## 1. 決定: native layout を導入し、canonical と分離した

Gate A の「MXFP4 native == canonical（byte 変換なし）」は**撤回**し、
WMMA fragment 順の native layout を導入した。

```text
block = 16 output x 32 K = 256 B
row ol の 16 B: byte b = code[k = b] | (code[k = 16 + b] << 4)
codes_row_stride_bytes = (kp / 32) * 256（1 tile あたり）
scale は row-major [rows][kp/32] のまま（byte 順を変えない）
```

(nibble を (k, k+1) ではなく (k, k+16) で対にする) のが要点で、これで
1 lane が 16 K 命令で消費する 8 byte が FP8 fragment と一致し、
E2M1 → E4M3 展開が byte 再配置なしに operand register へ書ける。

- correctness path も native を読むように変更（FP8 と同じ扱い）。
  canonical payload を読む参照実装を別に持たない。
- loader は `WeightLoadOptions::preshuffle`（既定 true）で native を作り、
  `preshuffle=false` のときだけ canonical を残す。
  dispatch / correctness 双方が native stride を検証するので、
  canonical bytes に native stride を付ける誤りは即座に失敗する。

## 2. 展開命令の採否

E2M1 → E4M3 は `v_perm_b32` 6 本（magnitude table 2 本 + negative table 2 本 +
sign を selector bit3 として選ぶ最終 perm 2 本）で、shift / or を含まない。
sign を `((n & 8) << 4)` で OR する 6 op 版は試作段階で棄て、
permutexa 形式（`0x03020100 | ((n & 0x08080808) >> 1)`）へ寄せた。

副次的に確認した事実: **gfx1201 には `v_dot4_f32_fp8_fp8` がある**
（`__builtin_amdgcn_dot4_f32_fp8_fp8` が ISA へ出る）。
最初に scalar FMA + dot4 の GEMV 経路を試作したが、参照実装が WMMA で
DRAM 天井付近まで到達しているため、WMMA 系統へ切り替えた
（dot4 経路は測定前に削除、コードは残さない）。

## 3. micro 実測（`shelf-bench gemm --dtype mxfp4`、R9700 dev0、p50）

`--warmup 40-50 --samples 20-30 --launches 20`。
`--check` は同一 payload の correctness path との比較で、下表の全 shape / 全 variant
**mismatch = 0**（bf16 bit 一致）。

### rows=1（decode1 unroll=2）

| shape | decode1 | rowblock1 | splitk | fp8 |
| --- | ---: | ---: | ---: | ---: |
| 256:5120 | 10.33 | 12.31 (u4) | 17.93 (s4) | — |
| 1024:5120 | 10.57 | — | 18.30 (s4) | — |
| 5120:5120 | 15.79 | 21.17 | 23.97 (s4) | 18.51 |
| 10240:5120 | 23.36 | — | — | — |
| 17408:5120 | 35.74 | 48.27 | — | — |
| 34816:5120 | **146.58（608 GB/s）** | — | 163.08 (s4) | **294.08（614 GB/s）** |

DRAM を超える shape で **MXFP4 = FP8 の 2.00 倍**（weight bytes 半分どおり）。
L3 に収まる 5120:5120 では 1.17 倍（命令律速）。

### decode1 unroll（rows=1、p50 µs）

| unroll | 5120:5120 | 17408:5120 | 34816:5120 |
| --- | ---: | ---: | ---: |
| 1 | 17.79 | 38.73 | 155.07 |
| **2** | **15.79** | **35.74** | **146.58** |
| 3 | 16.30 | 37.81 | 148.83 |
| 8 | 17.33 | 40.76 | — |
| 16 | 18.00 | 42.10 | — |

### split-K の有効域（rows=2-64、p50 µs、splitk=8 対 rowblock1）

| shape | rows | splitk8 | rowblock1 |
| --- | ---: | ---: | ---: |
| 256:5120 | 2 | 11.06 | 17.64 |
| 256:5120 | 32 | 15.74 | 18.06 |
| 256:5120 | 64 | 22.43 | 18.07 |
| 1024:5120 | 2 | 11.31 | 17.83 |
| 1024:5120 | 32 | 16.51 | 18.15 |
| 1024:5120 | 64 | 23.65 | 19.70 |
| 2560:5120 | 8 | 14.98 | 18.48 |
| 5120:5120 | 2 | 21.73 | 21.28 |

→ split-K は **rows ≤ 32 かつ out_features ≤ 2560** で採用（rows=1 は decode1 が常に勝つ）。
境界 2560 と 5120 の間で勝敗が反転する（16 output block 数が device を埋めるため）。

### RowBlock MB（p50 µs）

| shape | rows | MB1 | MB2 | MB4 | MB8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5120:5120 | 2 | 21.26 | 28.81 | — | — |
| 5120:5120 | 16 | 20.83 | 28.21 | — | — |
| 5120:5120 | 32 | 31.65 | 30.30 | 53.55 | 92.20 |
| 5120:5120 | 128 | 95.18 | **76.17** | 78.59 | 100.94 |
| 5120:5120 | 512 | 465.26 | 351.07 | **340.46** | 357.45 |
| 17408:5120 | 32 | 84.75 | **70.01** | 132.64 | 240.43 |
| 17408:5120 | 128 | 385.05 | **273.27** | 285.37 | 299.07 |
| 17408:5120 | 512 | 1647.28 | **1295.13** | 1302.16 | 1340.93 |

→ selector: rows ≤ 16 → MB1、rows ≤ 192 → MB2、それ以外 → MB4。
MB8 は全条件で劣るので **variant 自体を削除**した（到達不能な instantiation を残さない）。

## 4. E2E 実測（`shelf-bench tg`、ctx 128、tokens 32、prefill-chunk 128、R9700 dev0）

`models/Qwen3.8-27B-Quark-AWQ-MXFP4`、同一ビルドの A/B。

| 条件 | tok/s | ms/token | 備考 |
| --- | ---: | ---: | --- |
| `SHELF_QWEN35_KERNEL_MODE=correctness` | 0.71 | 1,400 | Gate A のまま |
| optimized（allowlist 7 shape） | 21.12 | 47.4 | mxfp4 fallback 16 本/token |
| optimized（+ (12288,5120)） | **28.38** | 35.2 | correctness fallback **0**、optimized 52,896 / dispatch 52,896 |

- 1 shape の allowlist 漏れ（GDN 系 12288:5120）で 35% 損をしていた。
  `SHELF_LINEAR_DEBUG=1` の MISS 行と `SHELF_QWEN35_PERF_STATS=1` の
  `linear.mxfp4.fallback` で特定した（両方とも既存の計測機構）。
  perf stats は元々 mxfp4 の hit/fallback を bf16 カウンタへ混入させていたため、
  `mxfp4_gemm_hits` / `mxfp4_fallbacks` を追加した。
- 35.2 ms/token は per-token weight traffic（MXFP4 化された weight + bf16 部）に対し
  実効 420-430 GB/s ≒ DRAM 天井 636 GB/s の 67%。
  prefill は 18 token / chunk 128 条件で correctness 7.09 s → 3.45 s。
- `shelf-compute`（18 token prompt、48 token greedy）の GENERATED_IDS は
  correctness path 実行と**全 48 token 一致**（kernel が bit 一致なので当然だが、
  E2E で確認した最初のケースとして記録）。

## 5. 採否と残課題

- ACCEPT: native layout + WMMA 系統（decode1 / splitk / rowblock）、
  scale は fold しない。全 variant が correctness path と bf16 bit 一致。
- REJECT: E8M0 を E4M3 operand へ fold する prefill 高速化
  （参照実装は採用しているが、加算再結合が変わり bit 一致が失われる。
  採用する場合は品質ゲートを別途明示する）。
- 未実装: **prefill 2D staging variant**。rows ≥ 256 では MB4 でも
  17408:5120 rows=512 が 1.30 ms（FP8 との比較尚未計測）であり、
  weight 再利用がタイル幅 16 output のまま。参照実装の prefill2d
  （LDS へ weight fragment を stage して 16 行で再利用、BN/BK 128）が
  pp2048 で -16.9% を出す場所であり、fold を除いた形でなら shelf でも成立する。
- 未実装: MXFP4 の TP partition（Gate A 同様 unsupported）。
- bench 側: `--shapes` は bf16 runner でしか効いておらず、fp8 runner は常に
  `--n/--k` の 1 shape で計測していた。fp8 / mxfp4 で解消したので、
  fp8 の shape 別数値は `--shapes` 付きで取り直した（34816:5120 rows=1 = 294.08 µs）。

---

## 6. 更新（同日）: Decode1 は削除した

上記 §3 の decode1 採用は**取り消し**。`d209b1b カーネルの Decode1 経路を削除` で
FP8 / PSQ5 の rows==1 GEMV 経路（`decode1_supported` / `decode1_unroll` /
`launch_*_decode1`）を落としており、「decode / prefill の linear は wmma 経路のみ」が
現行方針である。参照実装（phaseshift-mxfp4）に decode1 が残っていたのでそのまま
移植したのは誤りだった。

削除後の実測（同一条件の再計測、p50 µs）:

| shape | rows=1 rowblock1 | rows=1 splitk8 | 削除前 decode1 |
| --- | ---: | ---: | ---: |
| 256:5120 | 17.02 | **11.01** | 10.33 |
| 1024:5120 | 17.10 | **11.25** | 10.57 |
| 5120:5120 | 21.32 | 21.55 | 15.79 |
| 17408:5120 | **46.34** | 59.00 | 35.74 |
| 34816:5120 | **150.91** | 157.26 | 146.58 |

- rows==1 は SplitK（out <= 2560）または RowBlock1 が担う。SplitK は rows>=1 を
  受理するよう gate を広げた（rows=1 で 17.0 → 11.0 µs の実測利得）。
- **E2E への影響は 3.8% のみ**（tg ctx128 tokens32: 28.38 → 27.29 tok/s）。
  27B クラスの decode は DRAM 律速で、rowblock1 が 34816:5120 rows=1 で
  590 GB/s（天井 636 の 93%）を出しているため、L3 内 shape の差は打ち消される。
- RowBlock の activation scale を 128 K chunk で LDS から register へ hoist した
  （4 block 分使い回し）。bit 一致は維持、µs 単位の差は上記表に織り込み済み。

## 7. 更新（同日）: prefill 2D staging variant を移植した

`gemm_mxfp4_w4a8_prefill_2d_kernel<64>`: 1 block = 256 行 x 64 output、8 wave。
128 K chunk ごとに weight fragment を展開して LDS へ stage（`w0f_lds/w1f_lds`）し、
4 行 tile で再利用する。E8M0 は byte のまま `ws_lds` に stage し、32 K block ごとに
`fma(c, ws * as, acc)` で適用する（fold しないので **correctness path と bit 一致**、
`test_mxfp4_linear` の `prefill2d` / `2d-guard` ケースで固定）。

micro 実測（p50 µs、`--check` で全 shape mismatch=0）:

| shape | rows | prefill2d(BN64) | rowblock2 | rowblock4 |
| --- | ---: | ---: | ---: | ---: |
| 5120:5120 | 256 | 175.8 | **149.9** | 187.1 |
| 5120:5120 | 512 | 331.0 | **298.3** | 358.5 |
| 5120:5120 | 1024 | **539.0** | 698.5 | — |
| 5120:5120 | 2048 | **1357.7** | 1569.5 | 1856.1 |
| 10240:5120 | 512 | **549.9** | 603.5 | — |
| 10240:5120 | 2048 | **2511.0** | 3011.3 | — |
| 17408:5120 | 256 | 496.3 | 498.8 | 623.7 |
| 17408:5120 | 512 | **927.2** | 1244.6 | 1605.8 |
| 17408:5120 | 2048 | **4132.1** | 5022.1 | 5853.1 |
| 5120:17408 | 512 | **1310.2** | 1389.0 | — |
| 5120:17408 | 2048 | **4214.0** | 5258.1 | — |

selector の規則（実測の crossover）: `rows >= 1024`、または
`rows >= 512 かつ out_features * k_padded >= 4.0e7` で Prefill2D。
**BN=128 は計測却下**（BN=64 に対して 1.8〜2.3 倍遅い。fold しない形では
per-block rescale の accumulator が 16 本に膨らみ VGPR が溢れる）。
参照実装が 128x128 を採用できているのは fold で rescale を消しているから、
fold なしの本実装では 64 が正解だった。

E2E prefill A/B（`shelf-bench pp --prompt-tokens 2048 --prefill-chunk 2048`、
同一ビルドで selector 規則のみ切替、4 runs）:

| 条件 | tok/s | ms |
| --- | ---: | ---: |
| Prefill2D 無効（RowBlock のみ） | 1131.0 | 1810 |
| Prefill2D 有効 | **1520.4** | 1343 |

**+34.7%（時間 -25.8%）**。decode は影響なし（chunk 128 は rows < 512 で非該当）。

## 8. 参照: `../references/radiance`（libr4d）から取れる手

`libr4d/r4d_mxfp4.h` / `r4d_gemm_mxfp4a8_decode.hip` / `r4d_gemm_mxfp4a8_gemv.hip`。
phaseshift の fold（E8M0 の絶対値を E4M3 へ畳む、下位桁こぼれ 10.75%）と違い、
**行基準の fold** を使っている。

```text
ref[n]  = 行 n の E8M0 byte の最大値
d       = ref[n] - e8m0[n][blk]        （d >= 0、16 行の定数テーブル kMxfp4Mag[d][2]）
operand = e2m1 仮数 × 2^-d を E4M3 byte として v_perm で展開（d <= 8 で厳密）
epilogue: acc * 2^(ref[n] - 127) を 1 回
```

実 checkpoint `models/Qwen3.8-27B-Quark-AWQ-MXFP4` の全 496 tensor
（760,954,880 scale block）で d を実測した:

| 帯域 | block 数 | 比率 |
| --- | ---: | ---: |
| d <= 8（fold が厳密） | 760,954,598 | **99.99996%** |
| d 9..12（RNE で丸める） | 282 | 0.00004% |
| d >= 13（ゼロ） | 0 | 0% |

d の最頻値は 1、最大 10。つまり radiance の fold はこの checkpoint では
**量子化誤差を変えない**（丸め直数が変わるだけ）。phaseshift の fold が
10.75% の block で下位桁を落として E2E のトークン列を変えたのと同じ土俵ではない。

inner loop から消せるもの: 32 K block ごとの `mul + fma` x8（RowBlock では
128 K あたり 64 VALU、Prefill2D では 1024 VALU / wave）。activation scale は
128 K chunk ごとの適用のままなので、rescale 命令は 1/4 に減る。

他に確認した設計:
- rows 1〜8 の GEMV を **LDS に weight を載せない形**（lane が自分の B fragment word を
  fragment 順の weight から直接読み、8 wave で K を分割、末尾で 1 回 LDS reduce）に分ける。
  動機は「decode は純粋な weight stream。WMMA タイルの LDS staging + barrier 2 回で
  narrow projection が DRAM の 1/3〜2/3 しか出ない」— shelf の `attn_kv 1024:5120`
  rows=1 = 237 GB/s、`gdn_ba 48:5120` = 11 µs 床と符合する。
- form 選択を **K だけで決める**（M に依存しない）。verify step の 1 行目が
  素の decode と bit 一致するための契約になっている。
- K % 128 == 0 は WMMA form、K % 64 は dot form（`v_dot4_f32_fp8_fp8`、16 B = 4 列 x 8 codes）。
  shelf でも dot4 命令は gfx1201 で確認済み（§2）。

次の一手（未実装）: 行基準 fold を Prefill2D と RowBlock へ入れる。
per-row の `ref` を native payload の付随平面（N byte もしくは N f32、
5120 行で 20 KB 程度）として preshuffle 時に作り、
`d` をテーブル index にする。量子化誤差は不変で、変わる加算の再結合順序だけなので、
判断ゲートは tolerance ではなく **E2E の GENERATED_IDS 一致**で置く。

---

## 9. 行基準 E8M0 fold を Prefill2D へ入れた（採否: 採用 / 対象 variant 限定）

`d = row_ref[out] - e8m0[out][blk]` を E2M1 仮数の fold 距離にし、
`kMxfp4FoldMag[d]`（E2M1 の 8 仮数 × 2^-d を RNE で E4M3 byte 化した 16 行テーブル、
LDS へコピーして lane 別に引く）で operand を展開する。
`2^(row_ref-127)` は epilogue で 1 回だけ掛ける。

### 9.1 厳密性（先に結論: d<=8 なら bit 一致）

E2M1 の仮数は有意 2 bit（0, .5, 1, 1.5, 2, 3, 4, 6）なので、2^-d を掛けても
E4M3（subnormal 刻み 2^-9）で **d <= 8 は厳密**。d >= 9 では 1.5×2^-d が
E4M3 subnormal の中間に落ちるため丸める。operand が厳密なら、WMMA 内の積も
f32 累加もすべて 2^-d 倍（2 冪）の関係に移るので **加算のラウンド結果は変わらず**、
epilogue の 2^ref も 2 冪乗算なので bf16 出力は unfold と一致する。

実測で裏取った（`test_mxfp4_linear`、E8M0 幅を意図的に広げた fixture、
rows=256・out=128 の全要素）:

| 条件 | prefill2d | 2d-guard | rowblock1 | splitk4 |
| --- | ---: | ---: | ---: | ---: |
| 行内 E8M0 幅 <= 8 | 32768/32768 一致 | 25600/25600 一致 | 32768/32768 一致 | 32766/32768 |
| 同 left（fold 有） | rel_l2=0 | rel_l2=0 | rel_l2=0 | rel_l2=7.1e-7 |

splitk の 2 要素だけ違うが、これは fold ではなく split-K の LDS reduce 順序による
（fold なしの splitk4 も同じ 32766/32768）。

### 9.2 実 checkpoint の E8M0 幅（Qwen3.8-27B-Quark-AWQ-MXFP4、496 tensor）

| 単位 | 総数 | 幅 > 8 | 比率 |
| --- | ---: | ---: | ---: |
| scale block | 760,954,880 | 282 | 0.000037% |
| weight 行 | 3,904,000 | **5** | 0.000128% |
| 16 行 tile | 244,000 | 5 | 0.002% |

行内の E8M0 幅ヒストグラム: 1 が最多（870,796 行）、2 が 2,793,475、
3 以降急減し、9 が 4 行、10 が 1 行。幅 > 8 は
`layers.15.self_attn.q_proj`（4 行）と `layers.58.mlp.down_proj`（1 行）に集中している。

したがって **payload 単位で厳密性フラグを持って不足分の tensor だけ fold を
切ればよい**。loader は行単位の最大値平面 `row_ref` と
`row_ref_exact = (全行の max - min <= 8)` を作り、dispatch は
`mxfp4_gemm_folds_row_ref(config.id) && e.row_ref_exact && e.row_ref != nullptr`
のときだけ kernel へ ref を渡す。

### 9.3 速度（`shelf-bench gemm --dtype mxfp4 --impl prefill2d`、p50 µs、A/B を 2 往復）

`--mxfp4-fold` で同一ビルドで切替。合成 E8M0 幅 8（実測範囲）で `--check` の
mismatch は全 shape **0**。

| shape | rows | unfold | fold | 差 |
| --- | ---: | ---: | ---: | ---: |
| 17408:5120 | 512 | 914.3 / 918.0 | **681.9 / 690.5** | -25% |
| 17408:5120 | 2048 | 4045.3 / 4057.7 | **3098.2 / 3113.4** | -23.5% |
| 5120:17408 | 512 | 1307.9 / 1261.2 | **823.2 / 828.4** | -37% |
| 5120:17408 | 2048 | 4127.9 / 4143.9 | **3176.2 / 3183.0** | -23.2% |
| 10240:5120 | 512 | 543.4 / 545.5 | **407.9 / 413.3** | -24.6% |
| 10240:5120 | 2048 | 2462.0 / 2472.0 | **1903.9 / 1920.7** | -22.7% |

E2E prefill（`shelf-bench pp --prompt-tokens 2048 --prefill-chunk 2048`）:

| 条件 | tok/s |
| --- | ---: |
| Prefill2D + unfold | 1520.4 |
| Prefill2D + fold | **1886.4 / 1896.6** |

**+24%**。decode は影響なし（rows < 512 で Prefill2D 非選択。tg ctx128 tokens32 は
27.29 → 27.37/27.38 tok/s で測定誤差内）。

### 9.4 rows == 1 では fold は損（variant 限定の理由）

同ビルド A/B、rows=1・p50 µs:

| shape | unfold | fold |
| --- | ---: | ---: |
| mlp_gate_up 17408:5120 | 47.71 | 49.54 |
| mlp_down 5120:17408 | 89.32 | 99.13 |
| attn_q 6144:5120 | 28.96 | 31.98 |
| attn_o 5120:6144 | 34.23 | 37.85 |
| attn_kv 1024:5120 (splitk) | 12.16 | 11.80 |
| gdn_ba 48:5120 (splitk) | 11.75 | 11.46 |

MB=1 では 1 K block あたり 8 本の mul が消えるだけなのに、block ごとの
`d` 計算と LDS テーブル参照（lane 別に divergent）が効く。prefill の MB=4 /
2D（nt 分行再利用）では消える命令のほうが多いため得をする。
よって fold は Prefill2D に限定し、RowBlock / SplitK は展開形のままにする
（`mxfp4_gemm_folds_row_ref`）。

### 9.5 実装で踏んだ落とし穴

`if constexpr (kGuarded)` で守っていた staging を `stage_valid` のような
**実行時フラグに落とすと rows=1 の decode が 9% 下がった**
（tg 27.29 → 24.80 tok/s）。fold/non-fold どちらでも命令が実行時分岐にならない
よう constexpr の二重化へ戻した（fold はテンプレート引数）。

## 10. A 側 scale を K ループから外す余白（ablation 実測）

Prefill2D（fold 入り）の 1 chunk あたり命令内訳は WMMA 64 本に対し
activation group scale の適用が **256 fma**（+ 元は 256 mul）で、VALU の大半を
占める。そこで A scale を K ループ内で適用しない形（WMMA を accumulator へ
直接 chain する）を bench 専用の ablation 変種 `kActRescale=false`
（`launch_gemm_mxfp4_w4a8_prefill_2d_ablated_noactrescale`）で測った。
**数値は意図的に誤り**で、速度上限のみを見る。

| shape | rows | fold 付き（A rescale あり） | ablation（A rescale なし） | 差 |
| --- | ---: | ---: | ---: | ---: |
| 17408:5120 | 512 | 789.7 / 774.7 | **525.3 / 463.6** | -34% / -40% |
| 17408:5120 | 2048 | 3181.2 / 3243.7 | **2327.2 / 2265.6** | -27% / -30% |
| 5120:17408 | 512 | 917.4 / 902.6 | **539.6 / 545.7** | -41% |
| 5120:17408 | 2048 | 3252.3 / 3334.5 | **2279.2 / 2246.4** | -30% |

参考: FP8 W8A8 の同一 shape（17408:5120 rows=2048）は 2725.4 µs。
ablation の 2266 µs はこれを 20% 下回る。

つまり A 側量子化を「group 128 F32 scale」から
**loop の外へ出せる形**（per-token F32、または E8M0 を operand へ fold できる
MXFP8 / E8M0 scale）へ変えると、Prefill2D で 27〜41% の余白がある。
4 案の精度比較は §11（実 activation によるオフライン計測）にまとめる。

## 11. A 側量子化 4 案のオフライン精度実測

実 activation を使った。`shelf-bench pp --prompt-tokens 1024 --prefill-chunk 1024` を
`SHELF_LINEAR_INPUT_DUMP=<dir>`（`inference/execution/linear_input_dump.cpp`、
`kv_calib_dump` と同じ流儀）で走らせ、ACTIVATION_QUANTIZE の**量子化前 BF16 入力**と
それを消費する linear の weight shape を captures した
（3 分布: k=5120 hidden state、k=6144 attention 出力、k=17408 MLP 中間。各 384 行）。
解析は `tools/study_act_quant.py`。weight は checkpoint の MXFP4 payload を厳密に
dequant（e2m1 × 2^(e8-127)）し、基準出力は**実 activation をそのまま**使った
f64 積算。したがって下表の誤差は A 側量子化の寄与だけを切り出した値である。

案:

| 案 | scale | 粒度 |
| --- | --- | --- |
| A | F32 | group 128 K（現行） |
| B | F32 | 行全体（per-token） |
| C | E8M0 | group 32 K（MXFP8） |
| D | E8M0 | group 128 K |

E8M0 は OCP MX に従い `2^ceil(log2(amax/448))`（amax/scale <= 448）。

### 11.1 精度（192 行 × 3 分布）

| 案 | A 再構成 rel_l2（mean / max） | 出力 rel_l2（mean / max） | bf16 反転率（mean） | subnormal code | ゼロ潰れ |
| --- | ---: | ---: | ---: | ---: | ---: |
| A g128-F32 | 1.44e-2 / 1.96e-2 | 5.55e-3 / 8.82e-3 | 85.1% | 0.24% | 0.04% |
| B tok-F32  | 2.21e-2 / 2.67e-2 | 9.45e-3 / 1.30e-2 | 90.1% | 2.26% | 0.21% |
| C g32-E8M0 | 2.18e-2 / 2.62e-2 | 7.69e-3 / 1.19e-2 | 88.7% | 0.15% | 0.02% |
| D g128-E8M0| 2.18e-2 / 2.62e-2 | 7.69e-3 / 1.19e-2 | 88.7% | 0.34% | 0.05% |

- 現行（A）だけでも出力は rel_l2 5.5e-3 動いており、bf16 出力の 85% は least bit が変わる
  （bf16 の分解能 2^-8 に対して量子化誤差が 10 倍以上大きいため）。
  **A 側を替える変更は bf16 bit 一致ゲートでは評価できない**（§12）。
- E8M0 の 2 冪丸め損は group 32 / 128 でほぼ同じ（C と D が同値）。粒度を細かくしても
  2 冪スケールの損は消えない。
- per-token（B）は行内の octave 幅（p99 = 41 octave、max はもっと広い）のせいで
  subnormal code が 0.24% → 2.26%、ゼロ潰れ 0.04% → 0.21% に増え、誤差は 1.7 倍。

### 11.2 A 側 E8M0 を operand へ fold できるか（weight 側と同じ d<=8 判定）

行内の group scale 幅（octave）を実測:

| 分布 | group 32: p50 / max | 幅 > 8 の行 | group 128: p50 / max | 幅 > 8 の行 |
| --- | --- | ---: | --- | ---: |
| k=5120（hidden） | 5 / 5 | 0/192 | 4 / 4 | 0/192 |
| k=17408（MLP 中間） | 6 / 8 | 0/192 | 5 / 6 | 0/192 |
| k=6144（attn 出力） | 12 / 22 | **192/192** | 10 / 19 | **192/192** |

hidden state と MLP 中間は fold 帯域（d<=8）に**全行が入る**。attention 出力
（o_proj / linear_attn.out_proj の入力）だけ 10〜22 octave 開くため fold 不可。
つまり C（group 32 E8M0）は operand fold して K ループから rescale を消せるが、
attn 出力の入力だけ per-block rescale に戻す必要がある（layer につき 1 tensor、
全体の 13%）。判定は weight 側と同じく payload 側のフラグで行う。

### 11.3 速度（Prefill2D、p50 µs、`--mxfp4-ablate-act` で A/B を 2 往復）

`kActRescale=false` 変種は A scale を K ループから完全に外し、epilogue で行 1 個の
F32 scale を掛ける形（= 案 B の kernel 形）。scale 平面は既存 workspace の
column 0 を読むので**数値は案 B そのものではない**（速度評価専用）。

| shape | rows | 案 A（現行 + weight fold） | 案 B 形（loop に A scale なし） | 差 |
| --- | ---: | ---: | ---: | ---: |
| 17408:5120 | 512 | 686.2 / 693.3 | **444.1 / 446.2** | -35.4% |
| 17408:5120 | 2048 | 3115.6 / 3114.7 | **2218.5 / 2219.5** | -28.8% |
| 5120:17408 | 512 | 825.0 / 825.2 | **453.7 / 454.4** | -45.0% |
| 5120:17408 | 2048 | 3188.3 / 3185.4 | **2123.8 / 2124.1** | -33.4% |

参考: FP8 W8A8 の 17408:5120 rows=2048 は 2725.4 µs。案 B 形はこれを **18.6% 下回る**
（MXFP4 が FP8 より遅い理由は rescale 命令だった、という筋が速度面でも確認できた）。

案 C / D は scale が group 32 / 128 で付き、group 128 でも 32 K block が 4 本あるため
**per-block rescale を消せない**（§10 の ablation が消せる 256 fma のまま）。
消せるのは (a) 行 1 個の scale（案 B）、(b) operand へ fold できる E8M0 scale
（案 C、ただし §11.2 の帯域条件を満たす入力だけ）の 2 通り。

### 11.4 結論と次の一手

1. 精度だけで見ると **C（MXFP8 group 32 E8M0）が最良の代替**（出力 rel_l2 7.69e-3、
   B より 1.23 倍小さい。subnormal / ゼロ潰れも現行以下）。
2. 速度は (a) 案 B 形と (b) 案 C + operand fold の両方で −29% 級。
   案 B は contract 変更が最小（scale を行 1 個にする）だが精度が 1.7 倍悪い。
   案 C は scale 平面が rows × kp/32 byte（f32 平面より小さい）で、A operand の
   e4m3 に 2^-d を畳む命令（exp -field 加算、約 2-4 op / byte）が必要。
3. どちらへ進むかもゲート次第。**A 側変更は bf16 bit 一致を必ず壊す**
   （表の反転率 85〜90%）。したがって採用判断は generated ids 一致ではなく
   logit 変動と生成文の妥当性で行うしかない（§12）。

## 12. 精度ゲートの校正（fold のゲート実験から分かった感度）

同じ prompt（1024 token）で generated ids を 512 まで伸ばしたときの一致長:

| 比較 | 一致長 | prefill logit 差 |
| --- | ---: | ---: |
| Prefill2D（unfold） vs RowBlock（unfold） | **512/512** | 0（bit 一致） |
| fold（gating 前有: 5 行が d>8） vs 同 | 54/512 | max_abs 0.208、rel_l2 1.05e-2 |
| fold（gating 後） vs 同 | **512/512** | 0（bit 一致） |

- 加算の再結合だけ（fold、d<=8）は**完全一致**で通り、`row_ref_exact` gating で
  実模型でも 512/512 一致になった。
- 逆に **weight 5 行・282 block の丸め損だけで logit rel_l2 1.05e-2、54 token で分岐**した。
  つまり generated ids 一致ゲートは logit rel_l2 ~1e-3 級の変化でも落ちる感度がある。
  案 B/C の出力 rel_l2 は 8〜13e-3（layer 1 回分）なので、E2E では即分岐する。
  A 側変更の受け入れ基準は「logit 変動の上限 + 生成文の妥当性確認」に切り替えるべきで、
  それは本 Gate の判断材料として別途測る（例: 同一 prompt で top-1 一致率と
  logprob 差、または FP8 版との相互一致）。
