# workgroup barrier のスコープと、prefill activation の LDS staging

日付: 2026-11-02
対象: `models/Qwen3.8-27B-FP8` / gfx1201 (R9700) / ROCm 7.15.26333
比較先: `../references/radiance`（`libr4d/`）。前日の FP8 突き合わせ記録は
[fp8_kernel_alignment.md](fp8_kernel_alignment.md)。
適用コミット: `22bf6ac`（B-1 のみ適用。B-2 は棄却して revert 済み）

前提方針（ユーザー指定）: Prefill の tok/s が下がっても spill は 0 を保つ。
本記録の実験はいずれも spill 0・private segment 0 のまま実施した。

---

## B-1 workgroup barrier を LDS スコープの fence に絞る（ACCEPT）

### 機序

`__syncthreads()` は workgroup scope の全 address space に対する acquire-release fence で、
gfx1201 では `s_barrier_signal` / `s_barrier_wait` に加えて両側へ
`global_inv scope:SCOPE_SE` が出る。LDS のやり取りだけを order したい barrier でも、
次に読む global（activation codes、KV stream、重み）の L0/L1 行を捨てていた。

radiance は `libr4d/r4d_common.h:246-254` で同じ機序を明記し、`lds_barrier()`
（fence を `local` address space に限定）を使っている。彼らの実測は prefill GEMM で +1%。

### 実装

`src/shelf/kernels/optimized/detail/lds_barrier_device.h` を追加し、次を差し替えた。

| ファイル | 差し替えサイト数 |
| --- | ---: |
| `optimized/linear/fp8.hip` | 4 |
| `optimized/linear/bf16.hip` | 1 |
| `optimized/attention/paged_prefill.hip` | 13 |
| `optimized/attention/paged_decode.hip` | 16 |
| `optimized/gdn/recurrence.hip` | 12 |

選択基準は「barrier の後に、同一 block の他 thread がこの kernel 内で書いた global を読む
経路が無いか」。5 ファイルに atomic・`__threadfence`・block 間 global 通信は無く、
barrier はすべて LDS（scale ステージング、KV/V tile、split-K の LDS reduce、LUT、
wave 間 reduce）のため全サイト対象とした。atomic を扱う `radix_topn.hip` は
`__syncthreads()` のまま残した。契約は `docs/developer/kernels.md` の
「workgroup barrier のスコープ」に記録した。

### ISA

`global_inv` の出現本数が、対象 5 TU で 0 になった（変更前の fp8.hip は 3、
paged_prefill.hip は 17）。`s_barrier` 本数は不変（fp8 6、paged_prefill 34、
paged_decode 196、recurrence 134、bf16 4）。

FP8 kernel の資源は不変（spill 0・private 0 も維持）:

| kernel | VGPR | LDS |
| --- | ---: | ---: |
| `prefill_2d_wide_kernel<true>` | 144 | 1024 |
| `prefill_2d_wide_kernel<false>` | 159 | 1024 |
| `prefill_2d_kernel` | 164 | 0 |
| `wmma_kernel<4>` | 181 | 0 |
| `wmma_kernel<2>` | 98 | 0 |
| `wmma_kernel<1>` | 80 | 0 |

また fp8 GEMM について、barrier 変更前后の assembly で命令構成が同一であることを確認した
（wide kernel: fma 39 / mul 8 / wmma 64 と fma 66 / mul 4 / wmma 64、MB4: fma 86 / mul 47 /
wmma 64、両ビルドで一致。差分は `global_inv` の有無のみ）。

### 実測

`shelf-bench` device 1、変更前 / 後を交互、p50。単位 µs（pp は ms）。

fp8 GEMM（`gemm --dtype fp8 --impl auto --actual-qwen-shapes`）:

| shape (n x k) | rows | 変更前 | 変更後 | 差分 |
| --- | ---: | ---: | ---: | ---: |
| 17408 x 5120 (gate/up) | 1 | 146.0 / 147.0 | 145.9 / 144.6 | ±0 |
| 17408 x 5120 | 512 | 464.1 / 466.8 | 461.9 / 460.0 | −0.8% |
| 17408 x 5120 | 2048 | 2538.2 / 2578.8 | 2515.8 / 2402.4 | −0.9% / −6.8% |
| 5120 x 17408 (down) | 1 | 149.0 / 149.5 | 149.1 / 149.3 | ±0 |
| 5120 x 17408 | 512 | 500.3 / 503.8 | 489.6 / 489.4 | −2.4% |
| 5120 x 17408 | 2048 | 2441.5 / 2434.3 | 2413.7 / 2380.2 | −1.7% |
| 6144 x 5120 (q_proj) | 512 | 182.0 / 182.5 | 181.3 / 181.9 | −0.4% |
| 6144 x 5120 | 2048 | 668.6 / 669.1 | 651.6 / 655.1 | −2.3% |

pp2048（`pp --mode forward --prompt-tokens 2048`、runs 3 + warmup 1、ms）:

| 条件 | 変更前 | 変更後 | 差分 |
| --- | ---: | ---: | ---: |
| TP=1、device 1 | 891.0 / 916.6 | 879.1 / 899.1 | −1.3% / −1.9% |
| TP=2、device 0,1 | 544.3 / 540.7 | 534.2 / 532.9 | −1.9% / −1.4% |

paged attention（`paged-attention --variant optimized`、µs）:

| 条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| bf16 prefill rows=512 visible=1024 | 7451 / 6964 | 7013 / 6995 |
| bf16 prefill rows=512 visible=4096 | 27149 / 27140 | 27176 / 27199 |
| fp8 decode rows=1 visible=1024 | 77.8 / 77.0 | 77.6 / 77.4 |
| fp8 decode rows=1 visible=4096 | 289.7 / 285.7 | 286.2 / 285.9 |

bf16 prefill の visible=1024 は run 間ばらつきが大きく、差は判定できない。
それ以外は ±0.5% 以内。decode E2E（`shelf-compute` prompt 128 / 24 生成）の
`DECODE_TOKENS_PER_SEC` は 18.20 / 18.18 → 18.29 / 18.14 で変化なし。

### 正しさ

- required acceptance 129/129 PASS（GPU 実行、skip 0）
- `shelf-compute` Qwen3.8-27B-FP8 TP=1、24 生成の `GENERATED_IDS` sha1 が
  変更前后で一致（`5afc7d0e199de43328ae11bfb7eeb32df02b3c8e`）

---

## B-2 prefill activation codes を LDS へ staging + pitch pad（REJECT）

### radiance の設計

`libr4d/r4d_gemm_fp8a8.hip:702-784` は activation codes を LDS にダブルバッファで載せる。

- staging は 2 thread で 1 行、1 thread 32 B（`stage8_load` = global 16 B × 2、
  `stage8_store` = LDS 8 B × 4、`r4d_fp8_wmma.h:180-218`）
- LDS 行 pitch は `lds_pitch<FK>() = FK + 8`（`r4d_fp8_wmma.h:145-166`）。
  理由は bank conflict の解析: 「LDS has 32 banks of 4 B; a row pitch of P bytes puts row r
  at bank (r * P/4) % 32. With the tile width as pitch (64 or 256 B) rows collide」。
  XOR swizzle では 8 行しか分離できず row r と r+8 が衝突するため、pad は 8 の倍数
  （`ds_read_b64` の granule 整列を保つ）でなければならない
- 代价も明記済み: 「the prefill tile is 36 KiB per block instead of 32, allowing
  **3 resident blocks per WGP instead of 4**, which the saved bank cycle outweighs」
- LDS 帯域が layout を決める: 「a CU has 256 bytes of LDS per WMMA at the WMMA rate。
  両 operand を LDS 経由の 1x8 tile は read 288 B + store 64 B per WMMA で ~73% 上限。
  この layout は read 128 B + store 32 B per WMMA、weight の 64 B は vector cache から」
- weight は LDS を通さない（lane 32 B = 4 fragment、wave 1 本で 1 KiB 連続）

Shelf の `prefill_2d_wide_kernel` は activation codes を global から直接読いていた。
block の幾何（8 wave = ww 4 × wa 2、wave あたり weight 2 tile × activation 4 tile）は
radiance と同一で、違うのは activation の通り道だけだった:

| operand | Shelf（変更前） | radiance |
| --- | --- | --- |
| activation codes | global を wave ごとに読む（同じ 64 行を 4 wave → **4 回/block**） | global 1 回 + LDS |
| weight codes | global を 2 wave が読む（**2 回/block**、L1 .hit） | global 1 回（register 直読み） |

### 実装した 2 変形

いずれも LDS の並びは global と同じ **fragment order**（16 行 × 64 K = 1 KiB 連続）に保ち、
staging は wave ごとに 1 tile（32 lane × 32 B = 1 KiB 連続）、read も lane 32 B 連続のまま
とした。radiance の `FK + 8` pad は行-major LDS layout を前提にしたもので、
fragment order では 1 phase（8 lane × 16 B）が 128 B 連続を覆い、全 32 bank を 1 回ずつ使う。
したがって pad は不要であり、実際に生成命令は `ds_load_b128` 56 本 / `ds_store_b128` 8 本で
幅が落ちている（gfx12 の LDS 命令は `ds_load_*` / `ds_store_*`）。

- **stage-a**: 64 K block 単位で LDS へ stage する（ダブルバッファ 2 × 8 KiB）。
  weight load は変更前と同じく barrier の後で発行
- **stage-b**: stage-a に加え、activation / weight の次 block 先読みを barrier の前に上げる
  （radiance の `tile()` と同じ順序。barrier が weight 先読みを跨げないことが
  stage-a の劣化要因という仮説の検証）

いずれも staging で barrier が group あたり 2 本（64 K 単位で 1 本）になる。
128 K 単位で 1 本にまとめる形は、ダブルバッファが 2 × 16 KiB となって 2 block/CU
（4 waves/SIMD）に落ちるため採用しなかった。

資源:

| 変形 | VGPR `<true>` / `<false>` | LDS | waves/SIMD |
| --- | --- | ---: | ---: |
| production（B-1 のみ） | 144 / 159 | 1024 B | 10 |
| stage-a | 172 / 145 | 17408 B | 6（LDS 律速） |
| stage-b | 192 / 144 | 17408 B | 6（LDS 律速） |

いずれも spill 0・private segment 0。LDS 17408 B で 3 block/CU が上限となり、
occupancy が 10 → 6 waves/SIMD へ下がる（radiance が引き受けた値と同じ）。

### 実測（device 1、交互、p50、µs）

| shape (n x k) | rows | B-1 のみ | stage-a | stage-b |
| --- | ---: | ---: | ---: | ---: |
| 17408 x 5120 | 512 | 453.6 / 456.7 | 529.9 / 530.4 | 465.1 / 465.7 |
| 17408 x 5120 | 2048 | 2470.1 / 2487.3 | 2642.5 / 2642.7 | 2468.0 / 2473.5 |
| 17408 x 5120 | 4096 | 5064.4 / 5087.1 | — | 5030.3 / 5034.8 |
| 5120 x 17408 | 512 | 489.0 / 489.3 | 557.2 / 555.3 | 480.5 / 483.4 |
| 5120 x 17408 | 2048 | 2348.8 / 2381.7 | 2602.0 / 2619.7 | 2408.3 / 2398.5 |
| 6144 x 5120 | 512 | 181.0 / 181.1 | 189.9 / 189.3 | 168.6 / 168.6 |
| 6144 x 5120 | 1024 | 323.9 / 324.0 | — | 327.8 / 327.0 |
| 6144 x 5120 | 4096 | 1441.8 / 1440.3 | — | 1471.2 / 1466.3 |

pp2048（ms、B-1 のみ / stage-b）:

| 条件 | B-1 のみ | stage-b | 差分 |
| --- | ---: | ---: | ---: |
| unchunked、TP=1 | 860.5 / 866.6 / 867.2 | 860.2 / 868.2 / 870.3 | +0.2% |
| `--prefill-chunk 512` | 877.4 / 895.2 | 881.7 / 889.9 | −0.1% |
| `--prefill-chunk 1024` | 870.4 / 878.4 | 859.5 / 865.8 | −1.3% |

`test_gemm_fp8_optimized` は stage-b でも 283 case すべて PASS（実装自体は正しい）。

### 棄却理由

1. **performance target で利得が無い。** pp2048（chunk なし）は +0.2%、chunk 512 は ±0。
   chunk 1024 の −1.3% のみが一貫した改善で、shape 間の符号が揃わない
   （6144 x 5120 は rows=512 で −6.6%、rows=1024 / 4096 で +1〜2%）。
2. **代価が確実で、利得がshape依存。** LDS +16 KiB と staging register（+48 VGPR の live）で
   occupancy が 10 → 6 waves/SIMD に落ちる。radiance は同じ 6 waves を引き受けて
   +12%（別の baseline 比）を出しているが、Shelf の prefill wide は既に
   rows=2048 で 143 TFLOP/s（radiance 記録の gate_up 純 GEMM 160 TFLOP/s の 89%）にいて、
   activation の global 重複は L1 / L2 に収まるための代価になっていない。
3. **barrier 構造が主因ではないと分かった。** stage-a の −7〜−17% は weight 先読みが
   barrier に跨げないこと由来（stage-b で recovering）だが、stage-b の上限は ±数 % であった。

再挑戦する場合の前提は、LDS を 17 KiB 以下の設計（例: activation の一部だけを staging する、
あるいは block 形状を 128 × 64 に変える）で occupancy 10 waves/SIMD を保つこと、
および 6144 x 5120 の rows=1024 / 4096 で劣化しないことである。
occupancy pinning（`amdgpu_waves_per_eu`）を入れるなら、その contract と同時に扱う。

---

## 数値ドリフトの記録（未解決）

`test_gemm_fp8_optimized` の exact host 比較 worst rel は現在の build で
**6.167e-03**（abs 1.831e-04 at |y|=3.490e+02、同一条件 3 run で決定論的、283 case PASS）。
[fp8_kernel_alignment.md](fp8_kernel_alignment.md) に 5.7e-3 と書いた値は再現しない。

- B-1（barrier）は原因ではない: 変更前后の fp8 GEMM assembly で fma / mul / wmma 本数が
  同一（差分は `global_inv` のみ）
- B-2（staging）も原因ではない: stage-b は 6.167e-03 で、revert 後も同じ
- E2E（prompt 128、`--dump-logits`）の logits は B-1 前と B-2 適用時で bit 一致
- 5.7e-3 を記録した時点のビルドは block scale の fold 周期（32 K → 64 K）変更と同時期で、
  fold 周期変更前の値である可能性が高い（累算順が変わる）。未特定のまま残す
