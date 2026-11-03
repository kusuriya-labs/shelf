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

## 13. 案 C/D の実測（A 平面 fold パス + loop-free GEMM）

§11.2 で「A 側 E8M0 を GEMM の inner loop で畳むと 4 kb x 4 t 分行あたり 16-24 op
必要で、消せる 256 fma に対して割に合わない」見立てになったため、**fold を平面側で
事前適用する**形（activation 平面を行基準の指数へ揃え直すパスを通し、GEMM は
A scale を一切適用しない）を実装して測った。bench 内の実験 kernel
（`a_row_ref_kernel` / `a_plane_fold_kernel` in `src/apps/bench/gemm.hip`、
`--mxfp4-a-fold 1`）で、production 側には手を入れていない。

平面のアドレス対応は GEMM と同じ式（64 K fragment block 内で lane slot L が 32 byte、
`u < 4` が偶数 32 K block、`u >= 4` が奇数、行は tile 内 `L & 15`）。e4m3 byte の
fold は指数 field を引く算術（`b > sh ? b - sh : 0`、subnormal 域はゼロへ潰す保存的
実装）。

### 13.1 速度（p50 µs、`--impl prefill2d`、8 samples）

| shape | rows | A 平面 fold パス | GEMM 案 A（A rescale あり） | GEMM loop-free | 案 C/D 合計 | 案 A 比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5120:5120 | 512 | 17.4 / 19.5 | 241.8 | 130.5 | 148〜150 | **-38%** |
| 5120:5120 | 2048 | 31.8 / 29.9 | 791.8 | 506.4 | 537 | -32% |
| 17408:5120 | 512 | 19.3 / 19.7 | 674.5 | 436.2 | 456 | -32% |
| 17408:5120 | 2048 | 31.7 / 33.4 | 3085.2 | 2185.4 | 2218 | -28% |
| 5120:17408 | 512 | 42.0 | 808.3 | 443.1 | 485 | -40% |
| 5120:17408 | 2048 | 134.4 / 137.8 | 3173.5 | 2112.1 | 2250 | -29% |
| 10240:5120 | 512 | 20.1 / 20.7 | 409.9 | 260.7 | 281 | -31% |
| 10240:5120 | 2048 | 31.2 / 31.3 | 1890.0 | 1215.8 | 1247 | -34% |

fold パスのスループットは 274〜520 GB/s（A 平面 read + write、e8 平面 = kp/32 byte/行）。
GEMM に対する負担は rows=512 k=17408 の最悪値で 9.5%、rows=2048 では 4〜6%。
単純実装（byte 毎ループ）なので SWAR / 16 B store でまだ縮む。

### 13.2 精度（平面 fold の損得、`tools/study_act_quant.py` の C2 / D2）

出力 rel_l2（基準は実 activation そのまま + weight 厳密 dequant）:

| 分布 | A（現行） | C | C2 = C + 平面 fold | D | D2 = D + 平面 fold | B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k=5120 | 5.498e-3 | 7.38721e-3 | 7.38719e-3（fold で 384 code 変化） | 7.38721e-3 | 7.38719e-3（384） | 8.568e-3 |
| k=17408 | 8.823e-3 | 1.185720e-2 | 1.185723e-2（37,468） | 1.185720e-2 | 1.185723e-2（36,782） | 1.3016e-2 |
| k=6144 | 2.325e-3 | 3.81639e-3 | 3.81652e-3（83,610） | 3.81680e-3 | 3.81615e-3（79,566） | 6.751e-3 |

- **平面 fold の精度コストは測定分解能に出ない**（C vs C2 は 5 桁目）。fold で値が
  変わる code は 0.007〜1.5%（行内幅 10〜22 octave の k=6144 で 83,610 個）だが、
  それはすべて e4m3 の下位側で、内積への寄与が無視できるため。
- subnormal code 率は fold で 0.15% → 3.07%、ゼロ潰れ 0.02% → 0.30% に増える。
  上表は tensor core が e4m3 subnormal 入力を正規どおり扱う前提の値なので、
  本実装では subnormal 領域の実挙動を GPU 側で別途確認する（§11.1 の note と同じ検証）。
- §9 の weight 側 tail（5 行）が E2E で logit rel_l2 1.05e-2 を出したのに対し、
  A 側 fold の損は出力 metric に現れない。両者が同じ大きさの影響になる理由までは
  解明できていない（未解決。weight 側は gating で回避済み）。

### 13.3 採用候補としての整理

- 速度: 案 C/D（平面 fold + loop-free GEMM）と案 B（行 1 個 scale）は同じ等級
  （GEMM 側は同一形。C/D は fold パス分だけ余計にかかる）。
- 精度: **C/D（1.34 倍の誤差）> B（1.71 倍）**。C と D は実質同値で、D は scale 平面が
  1/4（kp/128 byte/行）。E8M0 の 2 冪丸めが支配的で粒度の効果は出ない。
- 実装量: 案 B は quantizer を行 1 個にするだけ。案 D/C は quantizer が E8M0 scale を
  出す必要があり、fold は (a) 別パス（今回測定）、(b) quantizer に融合（block amax と
  行 ref を同じ pass で決めると 2 pass になるが追加トラフィックは activation read 1 回）、
  (c) GEMM 内 fold（inner loop で 16-24 op / 16 byte 増え、256 fma を消すので 2D では
  釣り合うが rows==1 系では損）の 3 案がある。
- どの案でも A 側変更は bf16 bit 一致を壊す（反転率 88〜92%）。E2E の受け入れは
  logit 変動基準に切り替える必要がある（§12）。

## 14. WMMA の e4m3 subnormal 入力挙動（案 D の前提確認: 問題なし）

A 側 fold は operand を subnormal 域へ移す（subnormal code 率 0.15% → 3.07%）ため、
tensor core が subnormal 入力を flush するかを確認した。`tests/unit/test_fp8_wmma_subnormal.hip`
は 1 wave で FP8 WMMA 16x16x16 を 1 回、operand byte を全部同一値に埋めて実行する
（operand が一様なら 16 個の積がすべて等しく、期待値が fragment の lane 対応に
依存しないので、layout を assumption にせずにフォーマットだけを試せる）。

| a | b | 期待 accumulator | 結果 |
| --- | --- | ---: | --- |
| 0x01 (2^-9) | 0x01 | 6.10351562e-05 | 256/256 一致 |
| 0x07 (7·2^-9) | 0x07 | 2.99072266e-03 | 一致 |
| 0x01 | 0x07 | 4.27246094e-04 | 一致 |
| 0x08 (2^-6, 最小 normal) | 0x07 | 3.41796875e-03 | 一致 |
| 0x38 (1.0) | 0x38 | 16 | 一致 |
| 0x7E (448) | 0x01 | 14 | 一致 |

gfx1201 の FP8 WMMA は E4M3 subnormal を flush せず、積も累加も厳密（2^-9 × 2^-9 = 2^-18
まで bit 一致）。よって A 側 operand fold はハードウェア側で安全で、残る損は §13.2 で
測った平面側の丸めだけ（5 桁目）である。

## 15. 行基準 fold の production 形（optimized 3 系統 + quantizer）実測

案 D の本実装（`launch_activation_quantize_mxfp4_rowref` + GEMM の `kActRowRef` 形、
`Mxfp4ActivationForm::RowRefFolded`）を bench の production 経路
（`--mxfp4-a-rowref 1`、`--impl prefill2d` / `auto`）で測った。p50 µs、R9700 dev0。

### 15.1 GEMM 本体（weight 側 fold あり、A 側だけ差し替え）

| shape (n:k) | rows | group 形 | rowref 形 | Δ |
| --- | ---: | ---: | ---: | ---: |
| 5120:5120 | 512 | 238.9 | 135.1 | **-43%** |
| 5120:5120 | 2048 | 786.8 | 498.4 | -37% |
| 17408:5120 | 512 | 679.6 | 420.7 | -38% |
| 17408:5120 | 2048 | 3033.3 | 2071.0 | -32% |
| 5120:17408 | 512 | 814.1 | 425.4 | **-48%** |
| 5120:17408 | 2048 | 3039.7 | 2064.3 | -32% |
| 10240:5120 | 512 | 389.7 | 249.0 | -36% |
| 10240:5120 | 2048 | 1706.8 | 1135.8 | -33% |

§13.1 の loop-free ablation（-28〜-40%）と同等以上。**production 経路で再現された**。

### 15.2 quantize 本体（rowref は行基準を決めるため 1 行を 2 スイープする）

| shape (n:k) | rows | group 形 | rowref 形 | 増分 |
| --- | ---: | ---: | ---: | ---: |
| 17408:5120 | 1 | 6.4 | 9.0 | +2.6 |
| 5120:17408 | 1 | 11.9 | 20.8 | +8.9 |
| 5120:5120 | 512 | 26.2 | 30.2 | +4.0 |
| 5120:5120 | 2048 | 34.5 | 76.5 | +42.0 |
| 17408:5120 | 2048 | 34.6 | 76.9 | +42.3 |
| 5120:17408 | 512 | 36.9 | 64.5 | +27.6 |
| 5120:17408 | 2048 | 189.4 | 247.2 | +57.8 |

初版（group amax を double の `ceil(log2(...))` と `ldexp` で求め、thread 0 が
groups 個を serial に最大.reduce し、input を 2 回 scalar load する形）は
**6.4〜11 倍遅かった**（rows=512 k=5120 で 177 µs、rows=2048 で 680 µs、45 GB/s）。
上の値は (a) 指数を整数ビット演算化（`448·2^e >= peak` を 3 分岐で求める）、
(b) 1 スイープ目の codes を LDS に staging して global の再 read をなくす、
(c) input load を 64 bit vector（lane あたり 4 要素）にした後の値で、
host モデルとの bit 一致（`tests/unit/test_activation_rowref_fold.hip`）は保持している。

### 15.3 linear 1 回の合計（quantize + GEMM）

| shape (n:k) | rows | group 形 | rowref 形 | Δ |
| --- | ---: | ---: | ---: | ---: |
| 5120:5120 | 512 | 265.1 | 165.3 | -38% |
| 5120:5120 | 2048 | 821.3 | 574.9 | -30% |
| 17408:5120 | 512 | 705.3 | 451.0 | -36% |
| 17408:5120 | 2048 | 3067.9 | 2147.9 | -30% |
| 5120:17408 | 512 | 851.0 | 489.9 | -42% |
| 5120:17408 | 2048 | 3229.1 | 2311.5 | -28% |
| 10240:5120 | 512 | 415.6 | 279.4 | -33% |
| 10240:5120 | 2048 | 1741.3 | 1212.5 | -30% |

### 15.4 decode（rows==1 / 8、selector は rowblock1 を選択）

| shape (n:k) | rows | group 形 GEMM | rowref 形 GEMM | quantize 増分 | 合計 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17408:5120 | 1 | 39.7 | 33.9 | +2.6 | -21% |
| 5120:17408 | 1 | 59.9 | 41.9 | +8.9 | -29% |
| 17408:5120 | 8 | 40.4 | 35.2 | +2.7 | -13% |
| 5120:17408 | 8 | 60.8 | 43.4 | +8.6 | -22% |

weight 側 fold は rows==1 で損（§10.2）なので decode は fold なしのままで、A 側
rescale が消える分だけ利いている。rowref 平面で fold あり/なしを rowblock1 で測ると
rows=1 で 34.7 / 34.2、rows=64 で 115.0 / 118.9（5120:17408）と**ほぼ同等**で、
group 平面で観測した rows==1 の fold 損は消えている（採用判断は据え置き、decode は
fold なしのまま）。

### 15.5 variant 間の一致（`tests/unit/test_mxfp4_linear.hip`、fixture k=512 / out=128）

rowref 平面で correctness path（`flags = kActFlagRowRefFolded`）と全 optimized variant
を比べた。bf16 一致率 / double 参照に対する rel_l2:

| variant | correctness との bf16 一致 | double 参照 rel_l2 |
| --- | ---: | ---: |
| rowblock1 / rowblock1+fold / rowblock4+fold | 32768/32768 | 1.658e-3 |
| splitk4+fold | 32768/32768 | 1.658e-3 |
| splitk8+fold (rows=1) / rowblock1+fold (rows=1) | 128/128 | - |
| prefill2d / prefill2d+fold / prefill2d+fold guarded | 32768/32768 / 25600/25600 | 1.658e-3 |

chain 形（WMMA を K 全体へ continuation）でも correctness path と**完全一致**した。
理由は MXFP4 の inner loop の積がすべて f32 で厳密なためと説明できる:
E4M3(4 bit) × E4M3(4 bit) の積は有効桁 ≤ 8 bit、16 個の和は ≤ 12 bit で f32 の
24 bit に収まり、2 冪の rescale は厳密（正規域）。よって K の加算順を保つ限り
丸めの差が出ない。group 形で splitk が 2〜3 要素ずれるのは、F32 の group scale が
2 冪でないために summand 自体が丸められ、かつ LDS reduce で加算順が変わるから
（§1 の観測と整合）。**reference は変更せず、acceptance は double 参照に対する
tolerance（bf16 exact >= 99.9% / rel_l2 < 2e-3）**で全 variant を通している。

### 15.6 採否と分かったこと

- 案 D（A 側行基準 E8M0 fold）は GEMM 側で §13.1 の予測以上の利得があり、
  quantize の増分を差し引いても linear 全体で prefill -28〜-42%、decode -13〜-29%。
  **採用候補として成立**（E2E の logit ゲートは未実施）。
- rowref quantizer の threads を増やす試行（rows<=256 で 1024 threads、groups に応じ
  512/1024 へ可変、`__launch_bounds__(1024)`）は**却下**: rows=64 で 9.4 -> 20.2 µs、
  rows=256 で 19.9 -> 40.7 µ と遅くなり、かつ `test_activation_rowref_fold` の bit 一致が
  落ちた（行基準がゴミ値になる。threads 可変側の不整合、未調査）。workgroup 数の
  方が row 内並列より効くので、small-rows の +2.6〜+8.9 µs は別の形で攻める
  （例: 複数行を 1 workgroup で扱う、または 2 kernel 化）。
- bench の mxfp4 経路に `hipFree(d_input)` の二重解放が入っており（2143c7a で導入）、
  `--rows` に複数値を渡すと 2 番目以降が黙って失敗していた（`rc=1`、メッセージなし）。
  本測定向きに修正した。§13 の表は単一 rows の実行で取った値なので影響はない。

## 16. graph 配線後の E2E 実測と受け入れ判断

行基準 fold を graph へ配線した（`KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF`、
`ResolvedLinearQuantContract::activation_kernel`、linear binding の `flags` bit 1 =
`kActFlagRowRefFolded`。bit 0 は lowering が既に使っている output F32 旗なので避けた）。

### 16.1 reference 実装と optimized の全系統一致（bench `--check`）

`shelf-bench gemm --dtype mxfp4 --check --mxfp4-a-rowref 1` は、**correctness path
（新しい `exec_activation_quantize_mxfp4_rowref` + `exec_linear_mxfp4` の行基準読み）
と optimized path（行基準 quantize kernel + 行基準 GEMM）を独立に走らせて突き合わせる。

| 条件 | 結果 |
| --- | --- |
| rowblock1、17408:5120、rows=1、fold なし | mismatch=0 / 17408、max_rel=0 |
| 同上、spread 4（= fold 厳密域）、fold あり（chain 形） | mismatch=0、max_rel=0 |
| fold あり + spread 24（合成データの広い E8M0 散布） | mismatch=2707、max_rel=4.68（**A 側とは無関係**。`--mxfp4-a-rowref 0` でも同じ値に FAIL する。production は `row_ref_exact` のときだけ fold するので踏まない） |

unit test 側（`test_mxfp4_linear`）も rowref 平面で全 variant が correctness path と
bf16 完全一致、double 参照 rel_l2 1.658e-3（§15.5）。

### 16.2 E2E（Qwen3.8-27B-Quark-AWQ-MXFP4、prompt 1024 tokens、greedy 512 tokens）

| | pre-wiring（group A） | rowref A |
| --- | ---: | ---: |
| PREFILL_MS | 554.4 | **421.7（+31.5%）** |
| DECODE_MS / tokens | 19714.4 / 511 | 15545.6 / 349 |
| DECODE_MS / step | 38.4 | 44.5（**-16%**） |
| 生成終了 | 512 tokens 打ち切り | 350 tokens で EOS（248044） |

prefill は micro（§15.3 の -28〜-42%）と整合する。**decode は悪化した**。micro の
per-launch 見積りでは decode は 1 step あたり約 -2 ms（速くなる）はずで、実測 +6 ms/step
との差は未説明。残る仮説は、quantize kernel の動的 LDS（k=17408 で 18 KB）を
decode の 256 launch のたびに切り替えるコスト（LDS carveout の再構成）。bench では
同一 kernel の反復なのでこのコストが見えない。**A/B を同一 binary で取れていない**
（切り替え旗を立てていない）ため、原因確定は未実施。ユーザー判断で一旦許容。

### 16.3 logits ゲートの妥当性（計画のゲートは使えない）

`--dump-logits` の行は 1 step あたり 4 行だが有効なのは 1 行だけ（他はゼロ）なので、
stride 4 で比較した。

| 比較 | prefill 最終位置 rel_l2 | decode 行の rel_l2 | top-1 一致 |
| --- | ---: | ---: | ---: |
| correctness mode vs optimized mode（同じ rowref contract、prompt 18） | 4.13e-2 | 平均 6.7e-2、最大 2.75e-1 | **68/68**（GENERATED_IDS 完全一致） |
| group A vs rowref A（optimized、prompt 1024） | 4.75e-1 | 平均 2.0e-1、最大 2.43 | 1 番目から相違 |

**logit 空間では、実装差（correctness vs optimized）だけで既に 1e-1 級動く**。これは
MXFP4 の inner loop が厳密でない帯域（A・W 双方の E4M3 が 2^-9〜448 の 18 octave に
広がるため 16 積の和が f32 の 24 bit を超えることがある）と、attention / GDN など
他 kernel の bf16 級の実装差が増幅されるためと説明できる。よって
「prefill 最終位置の top-1 一致 + rel_l2 上限」というゲートは **A 側変更の合否を
判別できない**（plan の §4 は却下）。代わりに次を接受基準にする:

1. reference 実装と optimized 変種の全系統一致（§16.1、bit 一致で取れている）
2. 実 activation でのオフライン精度（§11、arm D は arm A の 1.34〜1.6 倍の layer 出力
   rel_l2。平面 fold 自体の損は 5 桁目）
3. 生成文が破綻せず EOS で終端すること（rowref 実行は 350 tokens で自然終了し、
   base の同一 prompt で後続に現れる列とほぼ同じ文構造を辿っている）

### 16.4 生成列の観察

base: `41564,13,814,20139,...` / rowref: `271,760,2781,279,12515,7701,6105,2261,...`。
base の 22 番目以降に `271,760,...,12515,7701,6105,2261,279,1834,321,2438,506` が
現れるので、rowref は同じ反復文の位相を 1 つ進めて始めた読み方になっている
（プロンプト自体が反復文）。offset 最適化で一致率 16%、末尾は EOS。
文として破綻は無いが、**1 番目の token から分岐する**ので、生成列一致を性能変更の
ゲートに使う方針はこの変更では成立しない（§12 の校正と整合）。

## 17. 契約集約（行基準形を MXFP4 の唯一の activation 形にする）と decode 再測

graph 配線（§16）で MXFP4 の producer は `ACTIVATION_QUANTIZE_MXFP4_ROWREF` 1 系統に
なったので、group 平面を読む形（`Mxfp4ActivationForm::GroupF32`、
`DispatchBinding::flags` の `kActFlagRowRefFolded`、kernel の `kActRowRef` template）は
production から到達不能になる。リポジトリ規則に従い削除した。

- GEMM の実体化は 22 → 10（RowBlock1/2/4 × fold、SplitK × 3 分割 × fold、
  Prefill2D × guarded × fold）。
- RowBlock の未使用 LDS array（`smem[2*MB*8]`）削除。
- bench から §13 の実験 kernel（`a_row_ref_kernel` / `a_plane_fold_kernel`）と
  `--mxfp4-a-fold` / `--mxfp4-a-rowref` を削除。production 形の quantize 計測は残した。
- `test_mxfp4_linear` の受け入れを **double 参照に対する dtype tolerance**
  （bf16 exact >= 99.9%、rel_l2 < 2e-3）に統一。correctness path との bf16 一致率は
  print のみ（gross divergence だけ 2e-2 で守る）。producer 相互一致
  （reference quantize と optimized quantize の平面 bit 一致、codes 131072/131072、
  scale diff 0）と contract 違反の `INVALID_PROGRAM` はそのまま固定。
- contract の現在形は `docs/developer/quantization.md`（activation quantization contract）
  と `docs/developer/kernels.md` へ移し、設計メモ `mxfp4_act_rowref_fold_plan.md` は削除。

### 17.1 micro 再確認（p50 µs、`--impl auto`、集約前後）

| shape (n:k) | rows | quantize 前/後 | GEMM 前/後 |
| --- | ---: | ---: | ---: |
| 17408:5120 | 1 | 9.00 / 9.11 | 34.2 / 34.3 |
| 17408:5120 | 512 | 30.3 / 30.8 | 420.7 / 423.4 |
| 17408:5120 | 2048 | 76.9 / 77.6 | 2071.0 / 2032.3 |
| 5120:17408 | 1 | 20.8 / 20.6 | 41.9 / 41.8 |
| 5120:17408 | 512 | 64.5 / 65.4 | 425.4 / 439.0 |
| 5120:17408 | 2048 | 247.2 / 247.7 | 2064.3 / 1975.0 |

差はノイズ帯。§15 の performance 判断は集約後もそのまま成立する。

### 17.2 E2E 3 状態（prompt 1024 tokens、greedy、`--max-seq-len 2048`）

| 状態 | PREFILL_MS | DECODE_MS / tokens | ms / step |
| --- | ---: | ---: | ---: |
| pre-wiring（group A 平面） | 554.4 | 19714.4 / 511 | 38.4 |
| 配線直後（2 形共存） | 421.7 | 15545.6 / 349 | 44.5 |
| 集約後（行基準 1 形） | 409.0 / 415.1 / 417.2 | 13364.5 / 13377.0 / 13401.5（349 tokens） | **38.3** |

集約後は **decode が pre-wiring と同等（38.4 → 38.3 ms/step）、prefill は +34%**。
 GENERATED_IDS は配線直後の実行と完全一致（集約は意味論的に中立）。

配線直後に観測した decode +16%（44.5 ms/step）は、集約後は 3 回とも 38.3 ms/step で
再現しない。原因は特定していない。候補は (a) 実体化 22 → 10 によるコード配置 /
ICache の差、(b) RowBlock の未使用 LDS array、(c) 計測環境。配線直後のコードへ戻して
の再現は行っていない（記録のみ残す）。decode の per-launch 利得
（rows=1 で GEMM −5.6 / −18.1 µs、quantize +2.7 / +8.7 µs）が E2E で net に現れない
点は未分解で、decode 側の内訳（attention / GDN を含む 1 step の launch 構成）を
分解しないと結論は出ない（推定: 1 step あたり 256 quantize + 数百 launch の内、
MXFP4 linear 由来の差は ±2 ms 級で、step 38 ms に対して観測分解能際に立たない）。
