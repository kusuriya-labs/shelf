# Gate A' — MXFP4 dense linear の optimized kernel 実装記録

Gate A（format / load / correctness path）は [mxfp4_gate_a.md](mxfp4_gate_a.md)、
現在形の contract は `docs/developer/canonical_soa_payload.md` §9 / §11、
`docs/developer/quantization.md`、`docs/developer/kernels.md` に移動済み。
ここには経緯・実測・採否だけ残す。

設計の大半は `../references/phaseshift-mxfp4` の MXFP4 W4A8 kernel 群
（`src/phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.hip`、
`include/phaseshift/quantization/mxfp4/mxfp4_native.h`）を確認した上で、
同系統の構造を shelf の contract（activation group 128、64 K fragment block、
`WeightSlot` stride 検証）へ合わせ込みた実装である。

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
sign を `((n & 8) << 4)` で OR する 6 op 版は試作段階で弃て、
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
