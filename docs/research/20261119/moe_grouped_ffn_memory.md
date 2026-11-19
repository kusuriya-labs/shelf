# GroupedExpertFFN のメモリ効率: 参照実装の調査

実験 ID: `MOE-GROUPED-MEM-1`

prefill chunk を 2048 から 4096 / 8192 へ広げるときの
`GroupedExpertFFN`（shelf では `MoeGroupedLinearNode` + `MoeCombine` + MoE plan scratch）
のメモリ増加に対し、`~/work/references` の各実装がどう対応しているかを調べた記録である。
新しい GPU 測定は行っておらず、shelf の数値は `moe_executor_grouped_bytes` の式を
`Qwen3.8-Flash-Next` の geometry（experts 512、top_k 10、moe_intermediate 640、hidden 2560）で
計算したものである。

## 1. 結論

- **radiance**: MoE の 1 pass を `min(max_tok, 2048)` token に固定し、それより広い step は
  pass に分割する。expert の中間バッファは pass 高さで宣言し、chunk 長に比例させない。
  さらに grouped GEMM には token 高さの活性と `sorted` / `eoff`（row table）を渡し、
  gather 済みの活性 plane を持たない。中間は fp8 で持つ。
- **hipfire**: こちらは `slots = rows * top_k`、`routed_rows = slots * intermediate`、
  `expanded_rows = slots * hidden` と同じ次数のバッファを chunk 高さで持つ。
  代わりに prefill scratch を **expert の VRAM から借り**、prefill 後に戻す（decode-share）。
  FFN のバッファ構造ではなく、VRAM の時間分割で解く。
- **llama.cpp**: `[n_embd, top_k, n_tokens]` に入力を繰り返し、
  gate_up `[2*n_ff, top_k, n_tokens]`、down `[n_embd, top_k, n_tokens]` を持つ。
  メモリ削減の構造は無い。
- **vLLM (sly image)**: fused MoE の tile 設定と量子化の patch が中心で、
  grouped expert のバッファ構造を縮める変更は無い。
- **Strata**: decode / verify window（T <= 4）向けに expert を shared memory へ staging する
  `EXPECD` を持つが、prefill chunk のメモリ構造は対象外。

shelf の host expert offload 構成では、**pass 分割は「chunk を広げて expert の読み出し回数を
減らす」という目的そのものを打ち消す**。1 pass を維持したまま plane を減らすか、
expert-major に並べ替えるか、hipfire 型に VRAM を借りる必要がある。

## 2. shelf の現状

### 2.1 バッファ

`moe_executor_grouped_bytes`（`src/shelf/models/qwen35moe/runtime/executor_footprint.cpp:36`）は
1 layer の peak を次で見積もる。

| バッファ | 高さ | Flash-Next の 1 group row あたり |
|---|---|---:|
| gate_up 出力 `y13`（bf16） | group rows | 2 × 640 × 2 = 2,560 B |
| gated 出力 `act`（bf16） | group rows | 640 × 2 = 1,280 B |
| down 出力 `expert_out`（bf16） | group rows | 2,560 × 2 = 5,120 B |
| gate_up 入力の MXFP4 codes + scales | group rows | 2,560 + 80 = 2,640 B |
| down 入力の MXFP4 codes + scales | group rows | 640 + 32 = 672 B |
| plan scratch | tokens / slots | 数十 B/group row |

- `group rows` は `grouped_row_capacity(tokens, {experts, top_k})`
  （`include/shelf/runtime/graph/grouped_row_shape.h:28`）。
  group tile が expert をまたがない制約のため、`slots = tokens * top_k` に
  `experts * 15 / 16` tile 分の余白が乗る。
- gate_up の入力は token 高さの値から **gather した group row の活性 plane** を作る
  （`src/shelf/runtime/program/program.cpp:918` の `activation_ws_key` と `row_index`）。
  gate と up は同じ入力・同じ K なので 1 plane を共有する。
- `act` は SwiGLU の bf16 出力で、直後の量子化が読んで down 入力の codes を作る。
- `expert_out` は `slot_row_of` を介して `MoeCombine` が読み、token 行へ重み付き和する
  （`src/shelf/kernels/fused/moe/combine.hip:37`）。

### 2.2 chunk 長による増加（Flash-Next、計算値）

| chunk | group rows | bf16 3 plane | 入力 codes | down codes | 合計 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 28,160 | 252 MB | 74 MB | 19 MB | **347 MB** |
| 4096 | 48,640 | 436 MB | 128 MB | 33 MB | **599 MB** |
| 8192 | 89,600 | 803 MB | 237 MB | 60 MB | **1,104 MB** |

- TP=2 では group rows が 85,760（8192）になり約 1,056 MB、TP=4 では 83,840 で約 1,033 MB。
  expert 軸分割をしても 1 GB 前後である。
- 2048 -> 4096 の増分が約 252 MB、4096 -> 8192 が約 505 MB。
- 行の上限は row bucket の 8192（`include/shelf/runtime/execution/row_bucket.h:72`）。

## 3. radiance

### 3.1 pass 分割

`arch/qwen4exp_fp8/qwen4exp_fp8.cpp:518`:

```
mc.rows = g.max_tok < 2048 ? g.max_tok : 2048;
```

MoE の pass は chunk に関係なく 2048 token に固定される。
`MoeFP8::step_rows`（`arch/common/rad_block_moe_fp8.h:2121`）は

```
for (r0 = 0; r0 < T; r0 += c.rows) pass(c, T, r0, min(T - r0, c.rows));
```

と step を pass に切る。pass の各 plane は pass 高さで宣言される。
`Wire` のコメント（`:298`）:

> Only the four planes with one row per (token, slot) are a pass tall; everything else has one
> row per token, is sized at max_tok and sliced by the pass offset.

pass 高さの 4 plane は `sorted` / `inv`（row table）、`egu`（gate_up 出力 bf16）、
`eff`（量子化済みの中間）、`edn`（down 出力 bf16、slot 順）である。
`max_tok` 高さなのは `x` / `h` / `logits` / `ids` / `ew` / 共有 expert 系。

既定の Flash-Next は hybrid なので、scheduler が chunk を checkpoint interval（2048）に
clamp する（`core/sched/geometry.cpp:205`、`docs/GUIDE.md:1264`）。
したがって通常運用では 1 pass で、pass 分割のコストは払わない。
chunk を 2048 より広げる配備だけが複数 pass になる。

### 3.2 gather 済み入力を持たない

`pass()` の gate_up（`:1938`）は token 高さの `gin_q` / `gin_s` と
`sorted` / `eoff` を渡す。down（`:1968`）も `eff` と `sorted` / `eoff` を渡す。
活性を group row に複製しないため、shelf の「入力 codes plane」に相当するものが無い。

### 3.3 中間の幅

`eff` は fp8（1 B/element + 128 ごとの f32 scale）で、bf16 の `act` を持たない。
`gq_exp` が `egu` から `eff` を作る。

### 3.4 pass 高さでの plane 量（Flash-Next、計算値）

| chunk | pass | slot 数 | egu | eff | edn | 合計 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 20,480 | 52 MB | 14 MB | 105 MB | **171 MB** |
| 4096 | 2048 x 2 | 20,480/pass | 52 MB | 14 MB | 105 MB | **171 MB** |
| 8192 | 2048 x 4 | 20,480/pass | 52 MB | 14 MB | 105 MB | **171 MB** |

chunk を広げても MoE の活性は増えない。代わりに expert weight を pass ごとに読み直す。
`docs/MOE-W4.md:233` は disk tier で「non-resident experts を pass ごとに読む」と明記している。

## 4. hipfire

`crates/hipfire-arch-qwen4/src/program.rs:150` の `Qwen4ScratchLayout` は

- `slots = rows * experts_per_token`
- `grouped_rows = grouped_m_total_bound(slots, num_experts)`
- `routed_rows = slots * intermediate`
- `expanded_rows = slots * hidden`

を持つ。`grouped_m_total_bound`（`:391`）は

```
ceil((slots + min(slots, num_experts) * 15) / 16) * 16
```

で、shelf の `grouped_row_capacity` と同じ 16 行 tile の余白を入れる。
つまり **FFN のバッファ次数は shelf と同じ**で、chunk に比例する。

hipfire が違うのは VRAM の使い方である。decode-share
（`crates/hipfire-arch-qwen4/src/decode_share.rs:45`）は、prefill scratch を
resident expert の donor tail から借り、prefill の間だけ expert を host mirror に退避し、
decode 前に戻す。scratch は 1024 / 2048 / ... / prompt chunk の capacity class ごとに
view を持つ（`docs/CONFIG.md` の decode-share 節）。
また 48〜511 行の append は prefill tile として別経路で処理する（`docs/ARCHITECTURE.md:163`）。

## 5. その他の参照

- **llama.cpp**（`Qwen-QSA-llama.cpp` は upstream の graph をそのまま持つ）:
  `src/llama-graph.cpp:2313` で入力を `[n_embd, top_k, n_tokens]` に repeat し、
  gate_up `[2*n_ff, top_k, n_tokens]`（`:2323`）、down `[n_embd, ...]`（`:2456`）を持つ。
  shelf と同じ「gather + slot 高さ」の構造で、メモリ削減は無い。
- **vLLM**（`vllm-sly-radiance`）: R9700 向けの int4 fused-MoE config
  （`CHANGELOG.md:75`、`DOCKERHUB.md:23`）と量子化 patch が中心。
  grouped expert のバッファを縮める変更は無い。
- **Strata**: `docs/EXPERTDEC.md` の expert kernel は decode / verify window（T <= 4）で、
  group の entry を shared memory に staging する。prefill chunk は対象外。
- **DSpine / EAGLE-4 / Qwen-QSA-llama.cpp / qwen35-f64-oracle / phaseshift 系**:
  MoE prefill のバッファ構造に関する記述は無い（phaseshift は本プロジェクトの前身）。

## 6. shelf への設計候補

### A. MoE だけ pass 分割（radiance 型）

`moe_pass_rows = 2048` を executor の内部 sub-plan として持ち、chunk を pass に切る。
メモリは 8192 でも約 347 MB。routing は pass ごとに計算する（token の routing は
token 単独で決まるので厳密）。実装は比較的小さい。

ただし **host expert offload では expert weight の読み出しが pass 数倍**になる。
2048 token の chunk を 4 回呼ぶのと同じ PCIe traffic になり、
chunk を広げる目的（sweep 回数の削減）を打ち消す。
VRAM resident の expert 構成でのみ有効な選択肢である。

### B. 1 pass を維持したまま plane を削る

| 変更 | 削減（8192） | 累積 | 難度 |
|---|---:|---:|---|
| B1. grouped SwiGLU + 量子化の融合（`act` を廃止） | 115 MB | 989 MB | 小（既存の fused producer を grouped row に拡張） |
| B2. gate_up の epilogue で gated product を作り `y13` も廃止 | 229 MB | 760 MB | 中（row-ref 量子化は row 全体の max が要るため、group scale 化か 2 段構えが要る） |
| B3. gather 済み入力 plane を廃止（row-major A + row index） | 237 MB | 523 MB | 大（MXFP4 の 16 行 fragment layout を row-major に変える） |
| B4. slot ちょうどに詰める（89,600 -> 81,920 行） | 45 MB | 478 MB | 大（16 行 tile が expert をまたぐため、tile 分割か per-row 重み選択が要る） |

B1 は shelf の既存 `SWIGLU` + `ACTIVATION_QUANTIZE_MXFP4_ROWREF` の融合を
`row_index` 付き（grouped）に広げるだけで済む。B2 は radiance の
`gemm_nt_q_gated` / `gated_had_quant` に相当する。

### C. expert-major slab + token 高さ accumulator

expert を slab（例 32〜64）に分け、slab を保持したまま token tile を回す。
各 expert の重みは chunk につき 1 回だけ読み、活性は tile 高さに収まる。
down の寄与は token 高さの f32 accumulator に加算する。
メモリは tile バッファ（2048 で約 171 MB）と accumulator（8192 x 2560 x 4 = 84 MB）程度。
weight sweep 1 回を保ったまま 1 GB を 250 MB 前後にできる。
参照実装には同じ形が見当たらないため、shelf 固有の設計になる。
層あたり slab 数 x tile 数の launch と、combine の加算順（決定性）の設計が要る。

### D. hipfire 型の VRAM loan

workspace を 2048 のまま確保し、prefill 中だけ expert の VRAM を借りて
chunk 高さの scratch を置き、decode 前に戻す。
shelf は host expert plane に pinned copy を既に持つため、resident expert を
host へ退避する経路は作りやすい。arena が固定長なので、arena の可変領域か
別確保が要る。prefill と decode の境界でコピーが 1 回入る。

## 7. 推奨

- **4096 まで**: B1（+ 必要なら B2）で 599 -> 537 / 412 MB。
  host offload を維持したまま 1 sweep を保てる。
- **8192 まで**: B1 + B2 + B3 で約 523 MB（B4 まで含めると 478 MB）、または C で 260 MB 前後。
  A は resident expert の TP 構成でのみ使う。
- まず B1 を実装して 4096 の実測（workspace と prefill 時間）を取る。
  `moe_executor_grouped_bytes` の計算値と `execution_workspace_bytes` を突き合わせ、
  そのうえで B2 / C のどちらへ進むかを決める。

## 8. 未検証

- Flash-Next の実 workspace（計算値のみ。`shelf-bench pp` の
  `execution_workspace_bytes` で要確認）。
- host expert offload 時の pass 分割コスト（PCIe traffic の実測）。
- B2 の row-ref 量子化と epilogue 融合の両立可否。
- B3 の row-major 化による grouped GEMM の効率低下。
- backbone（hyper connection 幅 10240、attention、PLE）も chunk に比例して増えるため、
  MoE の削減だけでは 8192 の予算が足りない可能性がある。
