# shelf-compute

単一request推論アプリ。greedy / temperature / top-k / top-p / min-p デコード。BF16 modelと量子化model
（`shelf_quantization.json` による自動検出）の両方に対応。

## options

<!-- cli:compute -->
| option | 説明 |
| --- | --- |
| `--model-dir PATH` | BF16 or quantized model directory (auto-detected via shelf_quantization.json) (required) |
| `--input-ids-file PATH` | whitespace separated prompt token ids |
| `--max-new-tokens N` | max tokens to generate (default 32) |
| `--max-seq-len N` | max sequence length (default 512) |
| `--arena-gib N\|auto\|-N` | arena capacity in GiB; auto = free VRAM minus 0.5 GiB, -N = total VRAM minus N GiB (default auto) |
| `--page-tokens N` | paged KV page size in tokens (default 16) |
| `--max-concurrent-requests N` | max in-flight requests (default 4) |
| `--max-num-batched-tokens N` | token budget per scheduled step; also sets the executor row bucket (default 2048) |
| `--device N\|N,M` | GPU device(s); count must match --tensor-parallel or --pipeline-parallel (default 0) |
| `--tensor-parallel N` | tensor parallel degree 1 \| 2 \| 4 (default 1) |
| `--pipeline-parallel N` | pipeline-parallel stage count (default 1; requires --tensor-parallel 1) |
| `--kv-cache-dtype TYPE` | bf16 \| fp8 (default bf16) |
| `--rope-factor MODE` | auto \| 1 \| 2 \| 4; static YaRN rope factor (default auto = model config: native unless the checkpoint requests yarn) |
| `--decode-backend TYPE` | hip (default) |
| `--verify-weights 0\|1` | verify quantized payload CRC32 on load (default 0) |
| `--dump-logits PATH` | append per-step sampled logits rows (raw f32) to PATH row k aligns with GENERATED_IDS[k] (prefill row first) |
| `--hidden-taps LIST` | comma separated layer ids (max 8) for intermediate hidden capture; dump to SHELF_DEBUG_HIDDEN_FILE |
| `--tensor-parallel-combine-f32 0\|1` | keep row-parallel combine output in f32 (default 0 = bf16 round as today; tp only) |
| `--temperature F` | sampling temperature (default 0 = greedy) |
| `--top-p F` | nucleus sampling threshold in (0, 1] (default 1) |
| `--min-p F` | min-p threshold in [0, 1] (default 0 = disabled) |
| `--top-k N` | keep the N highest logits (default 0 = disabled) |
| `--seed N` | sampling seed (default 0) |
| `--dflash2-model-dir PATH` | DFlash2 draft model directory; enables speculative decoding (greedy / stochastic) |
| `--dflash2-drafts N` | speculative draft tokens per round (default 7) |
| `--dflash2-max-sessions N` | serve only: requests that may speculate at once (default min(`--max-concurrent-requests`, 8)); the rest decode without speculation |
| `--dflash2-stats 0\|1` | print speculative decode statistics (default 1) |
| `--mtp` | enable the checkpoint's MTP block as the drafter |
| `--mtp-drafts N` | speculative draft tokens per round (default 7, max 31) |
| `--mtp-stats 0\|1` | print MTP speculative decode statistics (default 1) |
| `--prefix-cache 0\|1` | enable prefix cache (default 1) |
| `--working-set-reserve-gib N` | arena headroom kept free for the working set (executor workspace allocated after the memory plan; 0 = size it from the row bucket, default 0) |
| `--prefix-cache-concurrency N` | max concurrent prefix captures (default 4) |
| `--serve-stdio` | serve JSON Lines generation requests on stdin/stdout |
| `--help` | show this help |
<!-- /cli:compute -->

`--arena-gib auto` は起動時の空き VRAM から 0.5 GiB を引いた容量を arena に使う。
`--arena-gib -0.3` は VRAM の総容量から 0.3 GiB を引いた容量を使う。
KV の物理容量は個別に指定できない。model weight、`--working-set-reserve-gib`、
DFlash2 の drafter（weight と session pool 等の runtime 領域）、または MTP の drafter
（専用 KV pool、executor、GDN snapshot）、prefix cache の GDN state / ring を確保した後に
残る arena の領域をすべて KV に割り当てる。
KV を増やすには `--arena-gib` を増やすか、`--working-set-reserve-gib` を減らす。
prefix cache は KV を予約しない（active KV pool の page を共有する）。
既定は `0`（auto）で、executor が plan 後に確保する量を row bucket から推定して予約する。
実測値を大きくすれば KV、小さくすれば KV が増える。auto の推定と実測は起動 log の
`MEMORY_PLAN auto: working set reserve` と `MEMORY_PLAN executor footprint` で比較できる。
起動時には固定 / auto のどちらでも次の memory plan を stderr へ出す。
`auto` で `required` が `available` を超える場合は workspace を縮小し、
それでも足りなければ明示エラーで終了する。

```text
MEMORY_PLAN mode=auto
  target weights    xx.xx GiB
  draft weights     xx.xx GiB
  drafter runtime   xx.xx GiB
  runtime workspace xx.xx GiB
  KV budget         xx.xx GiB (NNNN tokens)
  prefix cache      xx.xx GiB
  ------------------------------
  required          xx.xx GiB
  available         xx.xx GiB
  arena capacity    xx.xx GiB
  left              xx.xx GiB
```

`target weights` は実際に arena へ load した後の使用量、`draft weights` は
`--dflash2-model-dir` の safetensors サイズ、`drafter runtime` は drafter の
scratch と session pool の領域（DFlash2 は `--max-concurrent-requests` 分、MTP は
専用 KV pool・executor・GDN snapshot）、`runtime workspace` は
`--working-set-reserve-gib` の予約量である。`KV budget` は arena 残量から導出した
physical KV capacity、`prefix cache` は GDN state と ring の実サイズである。
`left` は arena の未使用容量であり、workspace はこの後に executor が実際に確保する。

`--arena-gib N` を明示した場合は arena 容量を N GiB に固定する（measurement の
再現用）。

`--max-num-batched-tokens N`（既定 2048）は 1 scheduled step で扱う token 数で、
executor の row bucket も決める。prefill は chunk 分割で進むため、prompt が N を
超えても生成は止まらない。N を小さくすると program workspace（executor の arena 確保の
大半）が減って KV が増え、大きくすると prefill の step 数が減る代わりに arena を
多く使う。grouped expert checkpoint（MoE）では 1 step の group 行数が token 数から決まる上限
（experts 256・top_k 8 で、`token 数 × 8` slot を expert ごとに 16 行 tile へ切った厳密最大値）に
増える。**MoE の `--max-num-batched-tokens` の上限は 4096**（group 36608 行）で、それを超える
chunk は allowlist の geometry 外なので実行時点で
`unsupported: grouped-row dispatch has no correctness path` を返し、誤った値で生成を続けることはない。
prefill throughput の実測は chunk 2048 が最良で（`shelf-bench pp --prompt-tokens 8192` の
chunk 1024 / 2048 / 4096 で 3662 / 3807 / 3631 tok/s）、上限を上げる理由にはならない（経緯は
`docs/research/20261104/moe_group_row_elementwise_envelope.md`）。実測（27B FP8 / TP2 / requests 4 / drafter 込み / `--max-seq-len 262144`）:

| `--max-num-batched-tokens` | rank の executor 実測 | 結果 |
| --- | --- | --- |
| 8192（`--max-seq-len` に跟随していた旧動作） | 1.84 GiB | KV 上限 207,695 tokens で起動失敗 |
| 2048（既定） | 0.61 GiB | `SHELF_SERVE_READY` |


`--model-dir` は必須。`--input-ids-file` なしでは既定prompt
（`248041, 77091`）を使用。

`--model-dir` には BF16 model dir、Shelf 量子化 model dir
（`shelf_quantization.json` で自動検出）、および HF 由来の
real-quantized fp8 model dir（`quantization_config` の `quant_method: fp8` から
自動検出）のいずれも指定できる。

> **BF16 (plain safetensors) checkpoint について**: 生成結果が参照実装と一致することを
> 実測で確認済みです（`Qwen3.5-9B` の greedy 12 token が HF transformers の出力と全一致）。
> GDN 層の数値は required test `test_qwen35_gdn_gate_reference` が合成 checkpoint の
> f64 参照で固定しています。実モデルの f64 oracle fixture は `Qwen3.8-27B-FP8` の分が
> commit 済みです。

### decode backend

`--decode-backend` の既定値は `hip` で、現時点で受理する値は `hip` のみである。
gfx1201 の HIP kernel で decode を実行する既定構成である。選択結果は
`DECODE_BACKEND=hip` として stdout に出力する
（`--serve-stdio` の JSON Lines protocol には混ぜない）。

### MTP speculative decoding

`--mtp` を指定すると、checkpoint の MTP block（`mtp.*`）を drafter として使う単一
request の speculative decode 経路に入る。DFlash2 とは排他的で、
`--dflash2-model-dir` と同時に指定できない。

```bash
./build/shelf-compute \
  --model-dir models/Qwen3.8-27B-FP8 \
  --mtp --mtp-drafts 4 \
  --max-new-tokens 32 --max-seq-len 512 --device 0
```

dense checkpoint を TP=2 で動かす場合:

```bash
./build/shelf-compute \
  --model-dir models/Qwen3.8-27B-exl3 \
  --mtp --mtp-drafts 4 \
  --max-new-tokens 32 --max-seq-len 128 \
  --device 0,1 --tensor-parallel 2 \
  --prefix-cache 0 --arena-gib 30
```

制約:

- checkpoint に MTP block（`mtp.*`）が必要。無い checkpoint で `--mtp` を指定すると起動時に失敗する。
- dense checkpoint は `--tensor-parallel 2` に対応する。grouped expert（MoE）の TP では
  MTP block を rank に分割しないため、現行の MoE TP load は MTP を拒否する。
- `--pipeline-parallel 2`、`--dump-logits` は未対応。
- `--serve-stdio` は MTP を 1 request ずつ直列に処理する。`--max-concurrent-requests`
  を 2 以上にしても、MTP の生成中は次の request を開始しない。
- partial accept は GDN の compact log replay で復元する。target の再 forward は行わない。
- prefix cache は MTP 経路では使わない。
- grouped expert（MoE）checkpoint でも MTP block は load / 実行され、`--mtp` の生成列は
  target-only の greedy 列と一致する。MoE でも speedup がある（`--mtp-drafts 4` で 1.88 倍）。
  実測値は [../perf/current.md](../perf/current.md)、経緯は
  [../research/20261105/moe_mtp.md](../research/20261105/moe_mtp.md)。

詳細な round 契約は [../developer/mtp.md](../developer/mtp.md) を参照。

### DFlash2 speculative decoding

`--dflash2-model-dir` 指定時のみ DFlash 経路に入る。未指定時は既存の
`ContinuousBatcher` 経路を変更しない。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

併用できない option（いずれも model load 前に exit 2）:

- `--dump-logits`
- `--dflash2-drafts` が `block_size - 1` を超える

`--serve-stdio` と `--kv-cache-dtype fp8|octave:FORMAT` は併用できる。DFlash2 の
drafter ring は target の KV dtype と独立して BF16 のままである。serve mode では
`prefix_cache_checkpoint_position`（`0` のみ受理）と `temperature > 0` を
DFlash2 経路でも受け付ける。
stdio serving の prefill 行数、環境変数、追加メモリは
[server の DFlash2 設定](server.md#dflash2-speculative-decoding)を参照。
単発推論にはこの設定を適用しない。

### static YaRN（`--rope-factor`）

`--rope-factor auto|1|2|4` は RoPE の静的拡張倍率を選ぶ（既定 `auto`）。

- `auto` はチェックポイントの `text_config.rope_parameters.rope_type` に従う。`default` なら Native、`yarn` ならその `factor` を使う。
- `1` は Native を強制する。`2` / `4` は静的 YaRN を強制する。
- YaRN の元の長さは `rope_parameters.original_max_position_embeddings`、無ければ `text_config.max_position_embeddings`（Qwen3.8 は 262,144）。
- `--max-seq-len` の上限は Native 262,144 / 2x 524,288 / 4x 1,000,000（Qwen 公式の案内値）。超える指定は起動時に拒否する。
- 起動時に `ROPE profile:` 行を stderr へ出し、type / factor / theta / rotary_dim / 元の長さ / attention factor / 決定元（cli か config）を表示する。
- KV 容量計画は factor と独立である。必要な最小ページ数は `--max-seq-len` だけで決まり、残りのページ数は arena の余りで決まる。
- 静的 YaRN は入力長に関係なく係数が一定であり、短文の品質にも影響しうる（Qwen 公式の注意）。長文を扱うときだけ有効化する。
- speculative decoding（`--dflash2-model-dir`）と併用できる。Drafter は target の factor を継承せず、自身の checkpoint の RoPE（通常は `default`）を使う。起動時に `DFLASH2 rope:` 行を stderr へ出す。
- 2x の全長（524,288）は `--kv-cache-dtype fp8 --tensor-parallel 2` で容量が成立する（FP8 KV の品質特性は別途評価が必要）。

### prefix cache

prefix cache は page（`--page-tokens` の既定 16 token）単位の radix tree で、共有 prefix の
paged KV は active KV pool の page を refcount 付きで共有する。save は page を登録するだけで、
restore は block table に page を map するだけで、KV のコピーも別の shadow pool も作らない。
GDN state と DFlash2 ring だけを VRAM の shadow slot に保持する。
共有 prefix への extension や multi-turn の output tail で prefill を省略する。
entry が増えたときは LRU で evict する。同一プロンプトの再実行は restore せず、
DFlash2 経路でも prefix cache を使う。
詳細は [../developer/prefix_cache.md](../developer/prefix_cache.md) を参照。

### tensor parallel

`--tensor-parallel 2` で 2 GPU の tensor parallel 実行。`--device` に rank ごとの GPU を
カンマ区切りで指定する（例 `--device 0,1`、非連続な `--device 0,2` も可）。
`--device` の既定は `0` であり、`--tensor-parallel 2` のときは 2 個の指定が必須である
（例 `--device 0,1`）。個数が `--tensor-parallel` と一致しない場合・重複している場合・
可視 device を超出する場合は明示エラー。model weight は rank ごとに
順次 load するため load 時間は単 GPU より長い。

`--serve-stdio` も同じ runtime の TP 経路を使う。TP2 の serve 性能は未計測であり、
性能値は [../perf/current.md](../perf/current.md)、数値互換性・対応範囲の契約は
[../developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md)
を参照。

### pipeline parallel

`--pipeline-parallel K` で model の layer を K stage に分割し、stage ごとに GPU を
割り当てる。`--device` は stage 数と同じ個数を指定する（例 `--device 0,1`）。
`--tensor-parallel` は `1` でなければならない。

```bash
./build/shelf-compute --model-dir /opt/zen/work/models/Qwen3.8-27B-FP8 \
  --input-ids-file /tmp/opencode/ids_1024.txt \
  --device 0,1 --tensor-parallel 1 --pipeline-parallel 2 \
  --max-seq-len 2048 --max-num-batched-tokens 2048 \
  --prefix-cache 0
```

- 先頭 stage が embedding、最終 stage が final norm と lm_head を持つ。
- stage 間の hidden は bf16 のまま pinned host 経由で直列に受け渡す。
- `--pipeline-parallel` は prefix cache、DFlash2、MTP、hidden taps、
  `--dump-logits` と併用できない。grouped expert checkpoint（Qwen3.6-35B-A3B MoE）も受理する
  （layer を stage で割り、担当外 layer の expert 平面は読まない）。`--tensor-parallel` と
  `--pipeline-parallel` を同時に 2 にはできない。
- decode と prefill が同じ batch に混在する実行は未対応である。
  `--max-concurrent-requests 1` で運用するか、phase ごとに batch が分かれる条件で使う。
- KV の page 数は stage 0 の auto-fit 結果を全 stage で共有する。

契約の詳細は [../developer/pipeline_parallel.md](../developer/pipeline_parallel.md)
を参照。

## serve-stdio

`--serve-stdio` はserver backendから利用されるengine modeである。modelとruntimeを
一度だけ初期化し、stdinからrequestを読んでstdoutへeventを返す。stdoutはJSON Lines
protocol専用で、`MODEL_*`・debug・load statusは出力しない（diagnosticはstderr）。
このmodeのstdoutは1行1 JSON objectである。

request:

```json
{"op":"generate","request_id":1,"input_ids":[1,2,3],"max_new_tokens":32,"temperature":0.0,"top_p":1.0,"top_k":0,"seed":0,"prefix_cache_checkpoint_position":0}
```

未知の request field は `invalid_argument` で拒否する。
`prefix_cache_checkpoint_position` は `0` のみ受理し、それ以外は
`invalid_argument` で拒否する。prefix の再利用位置は compute 側の prefix cache が
radix lookup で決め、checkpoint による指定は受け付けない。silent ignore はしない。

thinking を生成途中で閉じる field は省略できる。

| field | 既定 | 意味 |
| --- | --- | --- |
| `reasoning_budget_tokens` | `0`（予算なし） | thinking に許す最大 token 数。thinking が未クローズでこの数に達した位置の token を close token へ置換する |
| `reasoning_close_token` | `-1` | thinking を閉じる単一 token ID。`reasoning_budget_tokens > 0` のときに必須 |

`reasoning_close_token` が不正、または `reasoning_budget_tokens` が `max_new_tokens` を
超える場合は `invalid_argument` で拒否する。置換は停止条件ではなく token value の
置換であり、commit 数・KV・position は変えない。contract は
[../developer/reasoning.md](../developer/reasoning.md) と
[../developer/stop_conditions.md](../developer/stop_conditions.md) を参照。

event:

```json
{"event":"token","request_id":1,"token_id":1234}
{"event":"done","request_id":1,"generated_ids":[10,20,30],"finish_reason":"stop","prompt_tokens":3,"prefill_tokens":3,"restored_tokens":0,"cache_checkpoint_tokens":0}
{"event":"error","request_id":1,"code":"invalid_argument","message":"..."}
```

`token` eventは生成token順に送られ、その連結は `done.generated_ids` とexact一致する。
`finish_reason` は `stop` / `length` / `cancelled` / `error` / `none` のいずれかである。
`restored_tokens` は prefix cache から復元された token 数（復元なしなら `0`）、
`prefill_tokens` は `input - restored_tokens`、`cache_checkpoint_tokens` は常に `0` である。

ping / shutdown:

```json
{"op":"ping"}      -> {"event":"pong","protocol_version":1,"capabilities":{"prefix_cache":true}}
{"op":"shutdown"}  -> {"event":"shutdown"}
```

`capabilities.prefix_cache` は prefix cache の作成が成功したとき `true`、
無効化または作成失敗時は `false` である。
`capabilities` は map であり、将来の capability はここへ追加する。未知の key は client が
無視してよい。

single-shot modeの出力契約（`GENERATED_IDS` 等）は変更されない。

## sampling

`--temperature 0`（既定）は従来どおり greedy で、結果は現行と完全に同一。
sampling は GPU 上で完結し、host は logits を読まない。logits の D2H も
vocab の host sort も行わない。

| 指定 | 挙動 |
| --- | --- |
| `--temperature 0` | deterministic greedy。`--top-p` / `--min-p` / `--top-k` は無視 |
| `--temperature 0.7 --top-p 1` | temperature softmax からの categorical sample |
| `--temperature 0.7 --top-p 0.9` | nucleus sampling |
| `--temperature 0.7 --min-p 0.05` | 最大確率の 5% 未満を除外 |
| `--temperature 0.7 --top-p 0.9 --min-p 0.05 --top-k 50` | 3 条件の積集合 |
| `--temperature 0.7 --top-p 1 --top-k 1` | 最上位 token のみ |

`--top-k` は `0` で無効、`--min-p` は `0` で無効。`--top-k` / `--top-p` /
`--min-p` を厳しくすると rejection の試行回数が増える。

同じ prompt・同じ `--seed`・同じ sampling 設定なら `GENERATED_IDS` は一致する。
batch 構成には依存しない。

```bash
./build/shelf-compute \
  --model-dir models/Qwen3.8-27B-FP8 \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --temperature 0.7 --top-p 0.9 --min-p 0.05 --top-k 50 --seed 42
```

## 出力契約

stdoutは `KEY=VALUE` 行。推論結果は以下:

```text
PROMPT_TOKENS=<prompt token数>
GENERATED_TOKENS=<生成token数>
KV_CACHE_DTYPE=bf16 | fp8
KV_POOL_RESERVED_BYTES=<KV pool予約バイト数>
GENERATED_IDS=<id1>,<id2>,...
```

`GENERATED_IDS` が生成tokenのexact契約。

`--dump-logits` 指定時は追加で:

```text
LOGITS_VOCAB=<vocab size>
LOGITS_STEP_ROWS=<stepごとのlogits行数（カンマ区切り）>
LOGITS_TOTAL_ROWS=<合計行数>
```

logitsはraw f32、1行 = 1 token step（prefill行が先頭）。
`row k` は `GENERATED_IDS[k]` と対応。

## 例

```bash
./build/shelf-compute \
  --model-dir models/Qwen3.8-27B-FP8 \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --kv-cache-dtype fp8
```

### token bitmask

`generate` に `"grammar":true` と `"bitmask":"..."` を指定すると、最終
prefill の最初の sample から mask を適用する。bitmask は token ID を bit 番号とする
little-endian 32-bit word 列を base64 化した文字列で、長さは
`ceil(vocab_size / 32) * 4` byte である。sample された token event の後、次の
mask を次のように送る。

```json
{"op":"mask","request_id":1,"bitmask":"<base64>"}
```

次の mask を受け取るまでその request はスケジュールされない。他の request は
進行できる。完了、cancel 後の mask は無視される。制約付き request は通常 decode を
使い、DFlash2 / MTP による投機は行わない。
