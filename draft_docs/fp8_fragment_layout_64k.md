# FP8 重みの断片順を「1 レーン 32 バイト」へ変える（実装計画・未着手）

状態: **実装済み（2026-10-30）**。旧 16 行 × 32 K（1 レーン 8 B）から移行した。
実測: 9B-FP8 prefill 2,283 → **2,356 tok/s**、27B-FP8 1,145 → **1,267 tok/s**。
`unit.quant_fp8` / `unit.hip_gemm_fp8`（128 行の広いタイルを含むビット一致）/ 9B の
トークン列一致を確認。16 バイト（4 断片）単位のロードにまとめた効果は小さく
（+0.1〜0.8%）、主な利得は 32 バイト連続になったことによる coalescing 側だった。

## 目標（参照: `references/radiance/libr4d/r4d_fp8_frag.h`）

単位は **16 行 × 64 K = 1024 B**。レーン `l` の 32 B は 4 断片を連結し、バイト `8t` が
断片 `t`（k = `t*16 + (l>>4)*8`、行 = `l & 15`）に当たる。1 波の 1 リクエストは 1 KiB。
参照実装の計測: 8 B/レーン（256 B リクエスト）は LDS staging にすら負け、
32 B/レーン（1 KiB/波）が全 N・K で勝つ（decode 形で約 13%）。

## 変更点（4 か所を同時に）

1. `src/quant/include/shelf/quant/fp8_w8a8.hpp`
   - `tiled_offset(row, col, k_padded)` を新式へ:
     `tile = col/64`, `lane = row & 15`, `t = (col % 64)/16`, `byte = ((col%64)%16)`
     → `offset = (row>>4)*(k_padded*16) + tile*1024 + lane*32 + t*8 + byte`
   - `tiled_bytes(rows, k)` は不変（置換なので同サイズ）。
2. `src/quant/fp8_w8a8.cpp` の `preshuffle_codes`: 上記の順で書き出す。
3. `src/backend/hip/include/.../gemm/detail/linear_fp8_device.h`
   - `fp8_code_off()` を同式へ。
   - 断片ロードを **32 B（`int32x8`）1 命令**に: `t=0..3` を `wp + t*8` ではなく
     レーン先頭から 32 B まとめて読み、`t` ごとに 8 B を切り出す。
   - 対象: `fp8_wmma_body`（RowBlock）, `fp8_decode1_body`, `fp8_prefill2d_body`,
     `fp8_prefill2d_wide_body`。K は 64 の倍数が必要（現在の前提と一致。
     `fp8_gemm_shape_supported` に `k_padded % 64 == 0` を追加するか、
     端数は旧経路に落とす）。
4. 活性化の量子化カーネル（`kernels/elementwise/activation_quantize_fp8.hip`）も同じ
   `code_offset` を持つので同時に直す（コメント通り `quant` と二重定義）。

## 検証ゲート

- `unit.quant_fp8`: `tiled_offset` / `preshuffle` / 逆写像の整合。
- `unit.hip_gemm_fp8`: 全ケースで f64 golden 一致 + `Prefill2D`/`Wide` と `RowBlock4` の
  **ビット一致**（累積順は変えないので一致するはず）。
- `shelf-run`（9B-FP8, 24 step）: トークン列が `/tmp/ref_seq.txt` と一致。
- `shelf-bench pp`: 9B 2,285 tok/s / 27B 1,145 tok/s を下回らないこと。

## 期待

- リクエスト数が 4 分の 1（1 KiB/波）。wide タイルは 32 K 単位で回しているので、
  64 K 単位に揃えるとロード命令が 1/4 になり、参照実装の計測では 13% 程度。
- 27B の律速（活性化の再読 = L2 帯域 1.26 TB/s）はこの変更では消えないため、
  2000 tok/s には「1 ブロックが複数 N タイルを掃く」も併用が必要。

## 実装後の内訳（2026-10-30・27B-FP8 / chunk 512 / 396 ms）

| 項目 | 時間 | 実効 | 所見 |
| --- | --- | --- | --- |
| linear N=17408 ×128 | 102 ms | 114 GFLOP/s | ほぼ microbench の上限（129 GFLOP/s） |
| linear N=5120 ×128 | 68.5 ms | 50 GFLOP/s | ブロック数 160 で並列不足（重み 388 GB/s） |
| lm_head（bf16 N=248320） | 51.2 ms | 25 GFLOP/s | 重み 5.1 GB × 4 パス = 20.6 GB（DRAM 律速） |
| linear N=10240 / 6144 | 34.6 ms | 119 GFLOP/s | 上限付近 |
| linear N=48 ×96 | 13.3 ms | 2 GFLOP/s | 1 本 0.14 ms（0.25 MB）で起動律速 |
| gdn_step | 34.3 ms | — | チャンク版 |
| attention_gqa_step | 24.3 ms | — | |
| elem_mul / rmsnorm / slice / rmsnorm_gated | 35.6 ms | — | 小カーネル（20 MB/本で 90 GB/s 程度） |

27B を 2,000 tok/s にするには 54 GFLOP/token × 2,000 = **108 TFLOP/s** の持続が必要。
microbench の上限が 129 TFLOP/s なので理論上は届くが、現状は 68 TFLOP/s。
優先順は (1) N=5120 の並列性、(2) lm_head の行タイル 256 化（bf16）、
(3) N=48 級の起動律速、(4) 小カーネル 35 ms の fusion / graph 化。

## 参照実装との同一形対決（2026-10-30）

`phaseshift-fp8` を `-DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201` で
ビルドし、カーネル単体を同じ形で比較した（`phaseshift-bench gemm` と
`shelf-bench kernel gemm`、1 枚、fp8・K=5120）。

| 形 | phaseshift | shelf（対策前） | shelf（対策後） |
| --- | --- | --- | --- |
| rows=128, N=17408 | 349.0 µs | 223.8 µs | 224.5 µs |
| rows=512, N=17408 | 515.6 µs | 706.1 µs | **561.0 µs** |
| rows=512, N=5120 | 156.8 µs | 183.5 µs | **180.5 µs** |
| rows=512, N=248320（bf16） | 73,243 µs | 51,200 µs | **35,819 µs** |

* 原因は **行タイルごとの重みの読み直し**だった。旧グリッドは x = 出力タイル、
  y = 行タイルで、ブロックが x 優先で投入されるため「同じ行タイルの別出力」が
  並び、重み 89 MB を行タイル数ぶん読み直していた（rows=512 で 4 パス）。
  x = 行タイル / y = 出力タイルに替えると、同じ出力タイルの行タイルが同時に走り
  重みが L2 に残る（706 → 561 µs）。bf16 の wide タイルも同じ理由で 51.2 → 35.8 ms。
* ISA（`llvm-objdump` + `clang-offload-bundler` で device bundle を展開）では、
  命令クラス構成は参照実装と同程度（wmma 64 本に対し全 2173–2876 命令、
  `v_wmma` 比率 2.2–2.9%）。**ISA の差ではなくデータの流し方の差**だった。
* 余波: 9B 2,358 → **2,477 tok/s**、27B 1,260 → **1,342 tok/s**。

## グリッド順の修正と「折り畳み」の採否（2026-10-30）

* **グリッド順（採用）**: fp8 wide / bf16 wide の blockIdx を x=行タイル・y=出力タイルへ。
  同じ出力タイルの行タイルが同時に走り、重みが L2 に残る。
  rows=512 N=17408: 706 → 561 µs、bf16 lm_head 形: 51.2 → 35.8 ms。
  9B 2,358 → 2,477、27B 1,260 → 1,342 tok/s。
* **重みの LDS 共有（不採用）**: 2 つの波が同じ重みタイルを二重に読むのを LDS で
  共有したが、LDS 16 KiB とバリア 2 回/群のコストが勝ち、in-model で
  27B 1,342 → 1,258、9B 2,477 → 2,397 と悪化。撤回。
* **128 K 折り畳み（不採用・要再検討）**: 参照実装と同じく「スケールを外して 128 K
  積み、末尾で 1 回折る」にすると、カーネル単体は rows=512 N=17408 で
  561 → **512.6 µs**（参照 515.6 と同等）まで速くなり、9B も 2,477 → 2,543 に上がった。
  ただし 9B のトークン列が末尾（269 トークン目以降）で EOS を跨ぎ、旧ベースラインと
  不一致になった。同一ビルド内では chunk 64 と 256 が完全一致（差 0）することは
  確認したが、旧ベースラインとの差の由来を説明できなかったため**安全側で撤回**。
  再導入するなら T2 の基準取り直しとセットで行う。

## モデルロードの内訳（2026-10-30）

`shelf-run` で 27B-FP8（29.5 GB）を 1 トークンだけ流して計測。

| 段階 | 時間 |
| --- | --- |
| 全体（ロード＋1 トークン） | **58.4 s**（preshuffle をバイト単位から 8 バイト単位へ直す前は 125.0 s） |
| materialize（read + preshuffle） | **52.1 s** |
| upload（H2D、29.5 GB） | 2.0 s（約 15 GB/s） |

切り分け:

* raw 読み（`cat *.safetensors > /dev/null`、page cache 済み）: 29 GB を **1.86 s**（15.8 GB/s）
* こちらの `Reader::read_tensor`（1 シャード 0.38 GB / 20 テンソル）: **2.3 GB/s**（0.16 s）
  → 29.5 GB で 13 s。生の読みより 7 倍遅い（テンソルごとに `std::ifstream` を開き直し、
  シークして読むため）。
* preshuffle は 8 バイト単位へ変更済み（`quant/fp8_w8a8.cpp`）。ただし
  `memset` で 29.5 GB を 0 埋めしてから書くため、書き込みが 2 倍になる。
* 30 GB 級の `std::vector` をテンソルごとに確保/解放しており、ページフォルトも効く。

①〜④はいずれも実施済み（2026-10-30）。同じ 27B（29.5 GB）で

| 段階 | ①〜④ 前 | 実施後 |
| --- | --- | --- |
| preshuffle（1 バイトずつ） | 125.0 s | — |
| preshuffle（8 バイト） | 58.4 s | — |
| + AVX2 de-interleave・詰め領域のみ memset・シャード 1 回 open・再利用バッファ | — | **7.3 s**（H2D 1.9 s を含む） |

切り分けの手がかり: 30 GB 級 `std::vector` をテンソルごとに確保していたとき、読みは
**2.4 GB/s** だったが確保をやめると **18.3 GB/s**。I/O ではなくページフォルトが支配していた。
参照実装の約 3 s との差は preshuffle と H2D の分（さらに縮めるには mmap）。

## 2048 行ステップと device 予算（2026-10-30）

### 「27B の chunk 2048 が動かない」の正体は OOM でなく grid.y 上限

`--prefill-chunk 2048` は `gdn post: invalid configuration argument` で落ちていた。
GDN のゲート付き正規化は head 方向を畳んだ行で発射するため

```
行数 = step 行数 × linear_num_value_heads = 2048 × 48 = 98,304  >  grid.y 上限 65,535
```

となり、**発射自体が失敗**していた（9B は 2048 × 32 = 65,536 で 1 だけ超過）。

### 対処: 行方向の分割発射（`kernels/detail/row_grid.h`）

行ごとに独立で、行は stride 経由のオフセットでしか参照されない性質を使い、
基底ポインタを行分だけ進めて 65,535 行区間で複数発射する（1 発射と同一の結果）。
対象は `blockIdx.y` を行に写している 11 本（elementwise 9 本 + l2 normalize +
`gdn_qk_prepare`）。`rmsnorm` は 1 次元 grid なので影響なし。
検証は `test_hip_elementwise_kernels` の rows=66,000 ケース（2 チャンク目に境界が含まれる）。

### device 予算（D2 の入口）

`HipStepExecutorOptions::vram_budget_bytes`（apps 側は `--vram-budget-mib`）。
実行器が確保台帳（`dev_allocs_` / `dev_bytes_`）を持ち、**hipMalloc を試みる前に**
「確保済み X MiB + 要求 Y MiB > 予算 Z MiB」で止める。重み・状態・アリーナ・scratch を
すべて含む（モデル込みの総量として扱う）。HIP コンテキストとカーネルコード
（実測 約 600 MiB）は計上されないため、実 device 使用量は計上 + 600 MiB になる。

最適化中の計測は **31.5 GiB（32,256 MiB）** を予算として渡して行っている。
`--verbose` 相当ではステージ毎に `実 device 空き / 総 / 予算計上 / 予算` も出す。

| 実行 | chunk 1024 | chunk 2048 |
| --- | --- | --- |
| 9B-FP8（ctx 2048） | 786.6 ms = 2,603 tok/s | **772.0 ms = 2,655 tok/s**（計上 15,049 MiB・実空き 17,306 MiB） |
| 27B-FP8（ctx 2048） | 1,513.7 ms = 1,354 tok/s | **1,501.6 ms = 1,364 tok/s**（計上 31,952 MiB・実空き **96 MiB**）※ prefill logits を最終行に絞った後は 1,361.8 / 1,341.0 ms = 1,504 / 1,527 tok/s に改善（`draft_docs/perf_selection_2026-10-30.md`） |
| 27B-FP8（ctx 4096） | 3,644 ms = 1,124 tok/s（計上 30,296 MiB・実空き 1,880 MiB） | 不可（計上 28,512 + 3,568 = 32,080 MiB > 使える量） |

27B の 2048 行は 1 枚で載るが、 logits を全行で作っていた当时は実空き 96 MiB と
余裕がなかった（`draft_docs/perf_selection_2026-10-30.md` の重複計算除去で実空き 2.0 GiB まで改善）。
`out of memory` になったとき、上の表の数字がそのまま理由になる。
