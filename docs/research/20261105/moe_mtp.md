# MoE の MTP speculative decode（weight / graph / drafter 接続と block 判定）

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層、`experts 256`、`top_k 8`、`hidden 2048`、
`moe_inter 512`、`shared_inter 512`、`(q_heads, kv_heads) = (16, 2)`、`head_dim 256`、
`full_attention_interval 4`、`mtp_num_hidden_layers 1`、`mtp_use_dedicated_embeddings false`）。
実験 ID `MOE-MTP`、結論 **REJECT（現状の MoE decode 精度では成立しない。理由は §5）**。

## 1. 着手時点の状態

dense（`Qwen3.8-27B` 系）側には MTP の full stack が有った: `mtp.*` weight load、
`lower_qwen35_mtp_to_primitives`、`create_mtp_drafter`、`shelf-compute --mtp`。
MoE 側は weight load 時点で factory が MTP を拒否し、`Qwen35MoeModelWeights` に MTP フィールドが無く、
`single_request.hip` に "MTP requires a dense model" gate が有った。

checkpoint の MTP block の layout（safetensors header 実測）:

| tensor | dtype | shape |
| --- | --- | --- |
| `mtp.layers.0.mlp.experts.gate_up_proj` | BF16 | `[256, 1024, 2048]` |
| `mtp.layers.0.mlp.experts.down_proj` | BF16 | `[256, 2048, 512]` |
| `mtp.fc.weight` | BF16 | `[2048, 4096]` |
| `mtp.layers.0.self_attn.q_proj.weight` | BF16 | `[8192, 2048]` |
| `mtp.layers.0.self_attn.{k,v}_proj.weight` | BF16 | `[512, 2048]` |
| `mtp.layers.0.self_attn.o_proj.weight` | BF16 | `[2048, 4096]` |
| `mtp.layers.0.mlp.gate.weight` | BF16 | `[256, 2048]` |
| `mtp.layers.0.mlp.shared_expert.*` | BF16 | gate/up `[512,2048]`、down `[2048,512]` |
| `mtp.norm.weight` / `mtp.pre_fc_norm_{embedding,hidden}.weight` | BF16 | `[2048]` |

transformer layer の expert が per-expert の U8 codes + `_scale`（Quark MXFP4）で保存されている一方、
MTP block は **expert も attention も router も shared expert も BF16** で保存されている。
HF の `Qwen3_5Moe` 側は `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` で MTP block を forward せず、
`gate_up_proj` は `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` → gate rows が前半、up rows が後半。

## 2. 実装した contract（採用分。これは有効）

- **fused rank-3 expert plane の load 時再量子化**。grouped GEMM に BF16 経路を足す代わりに、
  `gather_fused_plane` が checkpoint の BF16 payload を `quantize_mxfp4_matrix` + `preshuffle_mxfp4_native`
  で native SoA plane に落とす。`MoeExpertGroupPlan` は `mlp_prefix` / `fused_rank3` / 平面名
  （`build_moe_mtp_expert_group_plan`）を持つ。
  実 checkpoint で抜き取った expert 0, 1, 128, 255 の codes / scales / row_ref が、
  checkpoint payload を独立に再量子化して preshuffle した結果と **byte 一致**（`test_qwen35moe_model_load_real`）。
  dense MXFP4 checkpoint の MTP block も BF16 保存であり、dense 側はそれをそのまま BF16 linear で読む
  （`docs/perf/current.md` の dense MTP は BF16 FFN の結果）。
- **MTP graph**。`lower_qwen35_moe_mtp_to_primitives` は
  embedding → pre_fc norms → concat → fc → `emit_hybrid_block`（full attention 1 層）→
  `emit_moe_ffn`（MTP plan）→ head、という graph を出す。外部入力は token id と前 hidden の 2 本、
  state は KV cache 1 本のみ（GDN state なし）、`grouped_row_shape` は target と同じ。
  program build まで fixture で確認（`test_qwen35moe_lowering`）。
- **drafter の family 中立化**。`create_mtp_drafter` は `MtpDrafterInput`
  （target config へのポインタ、共有 embedding、family 側が lower 済み graph、executor の追加 workspace 項）
  を受け取るようになり、`MtpDrafter::config` は `QwenHybridConfig`、
  `mtp_drafter_reserve_bytes` は dense 項 + `moe_executor_grouped_bytes` の合計になった。
  family 固有の lower と workspace 見積もりは `LoadedModel`
  （`has_mtp_block` / `mtp_drafter_input` / `mtp_drafter_reserve_bytes`）に集約した。
  shelf-compute の "MTP requires a dense model" gate は外れ、MTP block を load していない checkpoint は
  drafter input で失敗する。
- target graph に `emit_normed_token_hidden`（MoE 側）を追加し、`ExecutorConfig` 経由で渡すようにした。

上記は required acceptance に常設（`test_qwen35moe_model_load`、`test_moe_expert_load`、
`test_qwen35moe_lowering`、`test_qwen35moe_mtp_executor`、`test_qwen35_mtp_lowering`）。

## 3. 実測: MoE MTP は動くが遅い・accept しない

`shelf-compute --model-dir <MoE> --input-ids-file <prose 1600 tokens> --max-seq-len 4096 --device 0`。
target-only は 61.23 t/s（同条件、MTP 無し）。

| 条件 | t/s | rounds | accepted drafts | mean accepted |
| ---: | ---: | ---: | ---: | ---: |
| MTP drafts=1 | 22.09 | 7 | 3 | 0.429 |
| MTP drafts=4 | 17.20 | 6 | 2 | 0.333 |
| MTP drafts=7 | 15.78 | 5 | 3 | 0.600 |

比較 anchor として dense `Qwen3.8-27B-Quark-AWQ-MXFP4`（BF16 FFN の MTP block）は
`docs/perf/current.md` §2026-11-04 で accepted/7 = 6.88（1024 token prompt）。
MoE の mean accepted は 0.3〜0.6 で、draft がほぼ弃えられるため MTP は target-only より 2.8〜3.9 倍遅い。

さらに、MTP run の `GENERATED_IDS` が target-only の greedy 列と**一致しない**
（1600 token prompt で target-only: `264,13759,506,3603,13,...`、MTP: `264,13759,220,16,17,...`。
3 番目から分岐）。greedy verify を使う限り、これは §4 の不一致で説明できる。

## 4. 根本原因の実測: MoE の forward は batch 行数に依存する

`shelf-bench quality` の teacher-forced logits dump を、同じ token・同じ position で
**スコア行的な batch 行数だけ変えて**比較した（MoE、bf16 KV、device 0/1）。
行は position で突き合わせ、logits ベクトル（vocab 248320）の差を見た。

| 比較 | position | max\|Δ\| | rms Δ | argmax一致 | top20 重複 |
| --- | ---: | ---: | ---: | :---: | ---: |
| 8 行 batch vs 1 行 batch | 136 | 8.39 | 1.50 | する | 10/20 |
| 8 行 batch vs 1 行 batch | 130 | 7.59 | 1.56 | する | 10/20 |
| 8 行 batch vs 1 行 batch | 132 | 8.09 | 1.45 | **しない** | 9/20 |
| 8 行 batch vs 4 行 batch | 136 | 6.35 | 1.23 | する | 11/20 |
| 8 行 batch vs 16 行 batch | 136 | **0.00** | 0.00 | する | 20/20 |
| 16 行 batch vs 12 行 batch | 136 | 4.92 | 0.90 | **しない** | 13/20 |
| context 128+8 行 vs context 4chunk+8 行 | 136 | 15.90 | 2.15 | **しない** | 7/20 |

同じ条件の dense（`Qwen3.8-27B-Quark-AWQ-MXFP4`、hybrid の対照）:

| 比較 | position | max\|Δ\| | rms Δ | argmax一致 | top20 重複 |
| --- | ---: | ---: | ---: | :---: | ---: |
| 8 行 batch vs 1 行 batch | 132 | 0.45 | 0.076 | する | 20/20 |
| chunk 2048 vs chunk 4 | 130..136 | 0.56〜2.36 | 0.10〜0.12 | 136 だけしない | 19〜20/20 |

つまり dense は行数・chunk 分割に対して logit rms 0.1 級で安定しているのに対し、
MoE は rms 0.9〜2.1、top20 重複 6〜13、argmax 反転が起きる。
8 行と 16 行が **bit 一致** なのに 12 行と 1 行は大きく違う、という単調でない形をしている。

MoE FFN の kernel 自体は HF oracle に対して正しい（`test_moe_layer_e2e`、16 tokens、
experts 8 / top_k 2、rel_l2 1.7e-3。`test_moe_program_execute` は program 経路を reference chain と比較）。
したがって怪しいのは program / executor 側の plumbing（row bucket と plan capacity、
grouped 行の int8 activation plane の stride、pad 行が plan に混入ときの扱い）であると**推定**されるが、
本記録では切り分けは未実施（仮説として残す）。

### 4-1. 2 行から壊れている

同じ position を、スコア行 2 本の batch と 1 本の batch で比べた
（context 134 + positions 2 と、context 135 + positions 1）。

| 比較 | position | max\|Δ\| | rms Δ | argmax一致 | top20 重複 |
| --- | ---: | ---: | ---: | :---: | ---: |
| 2 行 batch vs 1 行 batch | 136 | 6.63 | 1.04 | する | 13/20 |

logits 自身の rms は 2.59 なので**相対で 40 % 級の差**で、行数 2 から既に成立している。

### 4-2. 合成 fixture での walk 比較（形を確認する診断）

`tests/unit/moe_expert_fixture.h` の 2 層 MoE（experts 4 / top_k 2 / hidden 256）を executor で実行し、
token 行ごとの hidden を (a) 8 行 1 batch、(b) 1 行ずつ 8 步（decode class）、(c) 4 行 × 2 chunk、
(d) 1 行ずつ 8 步（prefill class）で取った。最終行の先頭 4 つ:

| walk | 最終行の値 |
| --- | --- |
| 8 行 1 batch | -1.271e+07, 1.540e+06, 5.636e+06, 1.769e+06 |
| 4 行 × 2 chunk | 3.539e+06, -9.011e+05, -1.049e+06, -4.391e+06 |
| 1 行ずつ（prefill） | 6.016e+04, -1.536e+04, -2.381e+04, -9.165e+04 |
| 1 行ずつ（decode） | 6.016e+04, -1.536e+04, -2.381e+04, -9.165e+04 |

**1 行ずつの walk は execution class を問わず一致**し、multi-row walk はそこへ一致しない。
合成 model は未学習なので hidden の絶対値自体が 1e4〜1e7 へ発散する（GDN の recurrence が
ランダム係数で増幅する）ため、この数値差を実モデルの誤差として読むことはできない。
読めるのは「single-row walk 同士は一致し、multi-row walk はそこへ一致しない」という
**形**が §4 の実測と同じ、ということだけである。
発散しない geometry で同じ比較をやり直す（fixture の expert 側にもっと小さな値を入れる等）ことは
本記録では実施していない。

再現手順（新規 code は要らない。`shelf-bench quality` だけで再現する）:

```bash
for spec in "128 8" "128 16" "124 12" "134 2" "135 1" "129 1" "131 1" "133 1"; do
  set -- $spec
  ./build/shelf-bench quality --model-dir <MoE> --tokens-file <corpus> \
    --context $1 --positions $2 --device 0 --dump /tmp/moe_q_$1_$2.psq
done
```

dump は `PSQUAL01` header 32 byte に続けて行ごと `int32 target`、`uint32 position`、
`float logits[vocab]`。position で突き合わせて logits ベクトルの差を見る。

MTP の verify は「複数行を同時に forward した結果」と「1 行ずつ forward した結果」が一致することを
前提にしている。MoE ではそれが §4 の大きさで崩れるため、
draft の良し悪し以前に **verify 自体が target-only の greedy と別の列を作る**。
この状態で MTP の速度を論じることはできない。

## 5. 採否判断

- weight load・graph・drafter 接続（§2）は **ACCEPT**。MoE に MTP block が無い状態（拒否・未 load）より
  明示的に load / validate / lower できる状態のほうがよく、contract は acceptance test で固定した。
- MoE の MTP decode 有効化（性能主張）は **REJECT**。§3 の実測で target-only より遅く、
  §4 の MoE forward の行数依存が verify の前提を壊しているため。
- 次の前提作業は §4 の解消（MoE の multi-row forward と single-row forward の一致）である。
  これは MTP 固有ではなく、`shelf-bench quality` の teacher forcing と prefill chunk 分割にも
  直接効く問題として扱う。

**（2026-11-05 更新）§4 の原因は特定・修正された**。combine の shared expert gate 平面
stride 欠落であり、[./moe_combine_gate_stride.md](./moe_combine_gate_stride.md) が正本。
§3 の数値と §5 の「verify の前提が壊れている」判定はその bug が入った forward での実測で、
§6 で再評価した。

## 6. combine 修正後の再評価

§3 と同一条件（1600 token prompt、`--max-new-tokens 32`、`--max-seq-len 4096`、TP=1、
bf16 KV、device 0、修正後 build）で再実測した。

| 条件 | t/s | rounds | accepted drafts | mean accepted | verify rows | draft ms | verify ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only | **61.74** | - | - | - | - | - | - |
| MTP drafts=1 | 28.09 | 16 | 14 | 0.875 | 32 | 46.40 | 995.88 |
| MTP drafts=4 | 45.92 | 9 | 22 | 2.444 | 43 | 95.86 | 578.93 |
| MTP drafts=7 | 41.23 | 9 | 22 | 2.444 | 63 | 150.73 | 600.91 |

- **MTP run の `GENERATED_IDS` は三条件とも target-only の greedy 列と完全一致**した
  （§3 で分岐していた 3 番目以降も含む）。verify の前提（multi-row forward の一致）は
  修正で成立した
- 同じ条件の長い生成で `test_mtp_parity`（`SHELF_MODEL_DIR_MTP=<MoE>`）が PASS:
  2048 token prompt / 64 token 生成 / drafts=7 で **rounds=15、gdn_restores=11**、
  `--mtp` ありとなしの `GENERATED_IDS` が完全一致。partial accept の compact log replay を
  含む経路が MoE で target-only greedy と一致することの gate になった
- mean accepted は 0.429 / 0.333 / 0.600 → 0.875 / 2.444 / 2.444。draft 受理率も修正で上がった。
  これは §4 の不一致が reject を量大で生んでいたことの裏返しでもある。
  ただし dense 対照（accepted/7 = 6.88 = 98 %）に対し MoE は 2.444 / 7 = 35 % のまま
- 速度は依然として target-only に劣る（最良でも 45.92 / 61.74 = 0.74 倍）。**性能主張の
  REJECT は据え置き**だが、理由は §4 の bug ではなくなった

内訳（1 round あたり）。verify 時間は drafts=1 で 62.2 ms（2 行）、drafts=4 で 64.3 ms
（平均 4.8 行）、drafts=7 で 66.8 ms（平均 7 行）で、**行数に対してほぼ一定**。
同じ model の multi-row batch forward 単体は `shelf-bench pp --mode forward --prompt-tokens 1600`
の chunk 別実測で、chunk 2 = 11.24 ms/token（2 行で 22.5 ms）、chunk 4 = 5.48 ms/token
（21.9 ms）、chunk 8 = 23.5 ms、chunk 16 = 26.2 ms と行数に対して緩やかに増える
（decode 1 行 = 16.2 ms、chunk 2048 = 3807 tok/s）。つまり 2 行 forward の既知の上限が
約 22 ms で、**verify round の 62 ms はそれを 40 ms 上回る**。この 40 ms が
forward 以外の費用（descriptor 準備・staging・sampling・state save）なのか、
DECODE class の multi-row batch が prefill class より遅いことなのかは未切り分けである
（`SHELF_OP_DUMP` は 1 dispatch ごとに fprintf するため 1355 dispatch / pass の MoE で
host 費用が支配的になり、この分離には使えなかった）。**row bucket は疑いを外した**。
`--max-num-batched-tokens 16 / 64 / 256 / 2048` で verify 合計は drafts=1 が
743.20 / 742.20 / 736.30 / 739.53 ms（12 round）、drafts=7 が 463.78 / 465.54 / 462.70 /
460.84 ms（7 round）で差が無く、生成列も同一だった。bucket の大きさで walk する行数が変わる
仮説は否定される。MoE MTP を speedup に载せるには、受理率を dense 並みに上げる前に、
この round 費用を落とすほうが先。受理率側の論点は §3 と同じく MTP expert の MXFP4 再量子化
（checkpoint では BF16 の expert を phase-2 codec で再量子化している）に残る。

break-even の目安（K=4）: round 675.03 / 9 = 75.0 ms、1 token あたり 16.2 ms なので、
round あたり 75.0 / 16.2 = 4.63 token 必要。bonus token を引くと **accepted ≥ 3.63 / 4** で
あり、実測 2.444 では足りない。

## 7. 副次で見つかった不具合（修正済み）

`create_executor` 直後の最初の `execute_batch` で、`prepare_batch_descriptor` の launch が
`invalid argument` を返し、batch が失敗していた（2 回目の同一 launch は成功する）。
kernel module の初回 launch と競合する形に見える。`create_executor` の program upload 末尾で
0 request の batch descriptor を 1 回 launch する warm を置き、初回 batch が初回 launch に
ならないようにした（`test_qwen35moe_mtp_executor` がその経路を通る）。

## 8. 再現条件

- build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHELF_ROCM_ROOT=$ROCM_PATH
  -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH
  -DCMAKE_HIP_ARCHITECTURES=gfx1201`（`ROCM_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`）、
  追加で `-DSHELF_BUILD_OPTIONAL_TESTS=ON`（`test_qwen35moe_model_load_real` は optional 側）。
- MoE MTP run: `./build/shelf-compute --model-dir <MoE> --input-ids-file <1600 ids>
  --mtp --mtp-drafts 7 --max-new-tokens 64 --max-seq-len 4096 --device 0`
- §4 の行列表現: `./build/shelf-bench quality --model-dir <MoE> --tokens-file <corpus>
  --context N --positions K --device D --dump <psq>` を (N,K) = (128,8), (128,16), (124,12), (135,1),
  (129,1), (131,1), (133,1) で取り、position で突き合わせた（dump 形式は
  `PSQUAL01` header 32 byte + 行ごと `int32 target, uint32 position, float logits[vocab]`）。
- corpus: `/tmp/moe_corpus.psktok`（2191 tokens、sha256 prefix `9136d38b`）、
  dense 対照は `/tmp/bb27b.psktok`。
