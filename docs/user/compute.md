# phaseshift-compute

単一request推論アプリ。greedy / temperature / top-k / top-p デコード。BF16 modelと量子化model
（`phaseshift_quantization.json` による自動検出）の両方に対応。

## options

<!-- cli:compute -->
| option | 説明 |
| --- | --- |
| `--model-dir PATH` | BF16 or quantized model directory (auto-detected via phaseshift_quantization.json) (required) |
| `--input-ids-file PATH` | whitespace separated prompt token ids |
| `--max-new-tokens N` | max tokens to generate (default 32) |
| `--max-seq-len N` | max sequence length (default 512) |
| `--arena-gib N\|auto` | arena capacity in GiB; auto uses free VRAM minus 1 GiB (default auto) |
| `--page-tokens N` | paged KV page size in tokens (default 16) |
| `--max-concurrent-requests N` | max in-flight requests (default 1) |
| `--kv-cache-capacity-tokens N` | physical KV token capacity (0 = one context) |
| `--device N\|N,M` | GPU device(s); count must match --tp (default 0) |
| `--tp N` | tensor parallel degree 1 \| 2 (default 1) |
| `--kv-cache-dtype TYPE` | bf16 \| fp8 (default bf16) |
| `--rope-factor MODE` | auto \| 1 \| 2 \| 4; static YaRN rope factor (default auto = model config: native unless the checkpoint requests yarn) |
| `--decode-backend TYPE` | host (default) |
| `--verify-weights 0\|1` | verify quantized payload CRC32 on load (default 0) |
| `--dump-logits PATH` | append per-step sampled logits rows (raw f32) to PATH row k aligns with GENERATED_IDS[k] (prefill row first) |
| `--hidden-taps LIST` | comma separated layer ids (max 8) for intermediate hidden capture; dump to PHASESHIFT_DEBUG_HIDDEN_FILE |
| `--tp-combine-f32 0\|1` | keep row-parallel combine output in f32 (default 0 = bf16 round as today; tp only) |
| `--temperature F` | sampling temperature (default 0 = greedy) |
| `--top-p F` | nucleus sampling threshold in (0, 1] (default 1) |
| `--top-k N` | keep the N highest logits (default 0 = disabled) |
| `--seed N` | sampling seed (default 0) |
| `--dflash2-model-dir PATH` | DFlash2 draft model directory; enables speculative decoding (greedy / stochastic) |
| `--dflash2-drafts N` | speculative draft tokens per round (default 7) |
| `--dflash2-stats 0\|1` | print speculative decode statistics (default 1) |
| `--eagle4-model-dir PATH` | EAGLE-4 draft model directory; enables EAGLE-4 speculative decoding |
| `--eagle4-drafts N` | EAGLE-4 speculative draft tokens per round (default 7) |
| `--eagle4-tree-width N` | EAGLE-4 tree verify width; 0 or 1 = chain (default 0) |
| `--eagle4-tree-dynamic 0\|1` | EAGLE-4 決定的 tree sizing（Cost table、default 0) |
| `--eagle4-candidate-head lowrank\|gemv` | EAGLE-4 candidate head path (default lowrank) |
| `--eagle4-lm-head draft\|target` | EAGLE-4 candidate head LM head source (default target) |
| `--eagle4-stats 0\|1` | print EAGLE-4 speculative decode statistics (default 1) |
| `--prefix-cache 0\|1` | enable prefix cache (default 1) |
| `--prefix-cache-hot-tokens N` | prefix cache hot tier capacity in tokens (default 16384) |
| `--working-set-reserve-gib N` | arena headroom kept free for the working set (executor workspace, DFlash2 session pool, default 4) |
| `--prefix-cache-cold-tokens N` | prefix cache cold tier capacity in tokens (default 65536) |
| `--prefix-cache-concurrency N` | max concurrent prefix captures (default 4) |
| `--prefix-cache-cold-state-slots N` | cold GDN state slots (default 0 = auto) |
| `--model-host PATH` | attach model weights from a resident model host (default: PHASESHIFT_MODEL_HOST_SOCKET, empty = local load) |
| `--serve-stdio` | serve JSON Lines generation requests on stdin/stdout |
| `--help` | show this help |
<!-- /cli:compute -->

`--arena-gib auto` は起動時の空き VRAM から 1 GiB を引いた容量を arena に使う。
起動時には固定 / auto のどちらでも次の memory plan を stderr へ出す。
`auto` で `required` が `available` を超える場合は、workspace と prefix cache の
順に縮小し、それでも足りなければ明示エラーで終了する。

```text
MEMORY_PLAN mode=auto
  target weights    xx.xx GiB
  draft weights     xx.xx GiB
  runtime workspace xx.xx GiB
  KV budget         xx.xx GiB (NNNN tokens)
  prefix cache      xx.xx GiB (NNNN tokens)
  ------------------------------
  required          xx.xx GiB
  available         xx.xx GiB
  arena capacity    xx.xx GiB
  left              xx.xx GiB
```

`target weights` は実際に arena へ load した後の使用量、`draft weights` は
`--dflash2-model-dir` の safetensors サイズ、`runtime workspace` は
`--working-set-reserve-gib` の予約量である。`KV budget` は arena 残量から導出した
physical KV capacity、`prefix cache` は hot tier の実サイズである。
`left` は arena の未使用容量であり、workspace はこの後に executor が実際に確保する。

`--arena-gib N` を明示した場合は arena 容量を N GiB に固定する（measurement の
再現用）。

`--model-dir` は必須。`--input-ids-file` なしでは既定prompt
（`248041, 77091`）を使用。

`--model-dir` には BF16 model dir、PhaseShift 量子化 model dir
（`phaseshift_quantization.json` で自動検出）、および HF 由来の
real-quantized fp8 model dir（`quantization_config` の `quant_method: fp8` から
自動検出）のいずれも指定できる。

### decode backend

`--decode-backend` の既定値は `host` で、現時点で受理する値は `host` のみである。
`ContinuousBatcher` と Host 実行経路を使う既定の構成である。選択結果は
`DECODE_BACKEND=host` として stdout に出力する
（`--serve-stdio` の JSON Lines protocol には混ぜない）。

GPU-MCU backend はこの build に含まれない。実装と R&D 記録は
`exp/gpu-mcu` ブランチへ退避している。

### resident model host

`--model-host`（または `PHASESHIFT_MODEL_HOST_SOCKET`）を指定すると、model weight は
`phaseshift-model-host` が GPU 上に保持しており、compute process は HIP IPC で
attach する。compute が crash / timeout / hang しても weight は残り、
次 の compute process は attach で即座に再開できる。

指定が無ければ従来どおり自分で safetensors を load する。
`PHASESHIFT_DISABLE_RESIDENT_MODEL=1` で environment 経由の指定を無効化できる。

### DFlash2 speculative decoding

`--dflash2-model-dir` 指定時のみ DFlash 経路に入る。未指定時は既存の
`ContinuousBatcher` 経路を変更しない。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

併用できない option（いずれも model load 前に exit 2）:

- `--dump-logits`
- `--dflash2-drafts` が `block_size - 1` を超える

`--serve-stdio` と `--kv-cache-dtype fp8` は併用できる。serve mode では
`prefix_cache_checkpoint_position`（`0` のみ受理）と `temperature > 0` を
DFlash2 経路でも受け付ける。

### static YaRN（`--rope-factor`）

`--rope-factor auto|1|2|4` は RoPE の静的拡張倍率を選ぶ（既定 `auto`）。

- `auto` はチェックポイントの `text_config.rope_parameters.rope_type` に従う。`default` なら Native、`yarn` ならその `factor` を使う。
- `1` は Native を強制する。`2` / `4` は静的 YaRN を強制する。
- YaRN の元の長さは `rope_parameters.original_max_position_embeddings`、無ければ `text_config.max_position_embeddings`（Qwen3.8 は 262,144）。
- `--max-seq-len` の上限は Native 262,144 / 2x 524,288 / 4x 1,000,000（Qwen 公式の案内値）。超える指定は起動時に拒否する。
- 起動時に `ROPE profile:` 行を stderr へ出し、type / factor / theta / rotary_dim / 元の長さ / attention factor / 決定元（cli か config）を表示する。
- KV 容量計画は factor と独立である。必要ページ数は `--max-seq-len` と `--kv-cache-capacity-tokens` だけで決まる。
- 静的 YaRN は入力長に関係なく係数が一定であり、短文の品質にも影響しうる（Qwen 公式の注意）。長文を扱うときだけ有効化する。
- speculative decoding（`--dflash2-model-dir` / `--eagle4-model-dir`）との併用は未対応で、2x/4x 指定時は起動時に拒否する。
- 2x の全長（524,288）は `--kv-cache-dtype fp8 --tp 2` で容量が成立する（FP8 KV の品質特性は別途評価が必要）。

### prefix cache

prefix cache は page（`--page-tokens` の既定 16 token）単位の radix tree で、共有 prefix の
paged KV と GDN state を VRAM（hot、`--prefix-cache-hot-tokens`）と pinned DRAM
（cold、`--prefix-cache-cold-tokens`）に保持する。共有 prefix への extension や
multi-turn の output tail で prefill を省略する。作成に失敗したときは無効化して
続行し、警告を 1 行 stderr に出す。同一プロンプトの再実行は restore せず、
DFlash2 経路でも prefix cache を使う（drafter ring も entry に保存される）。
詳細は [../developer/prefix_cache.md](../developer/prefix_cache.md) を参照。

### tensor parallel

`--tp 2` で 2 GPU の tensor parallel 実行。`--device` に rank ごとの GPU を
カンマ区切りで指定する（例 `--device 0,1`、非連続な `--device 0,2` も可）。
`--device` の既定は `0` であり、`--tp 2` のときは 2 個の指定が必須である
（例 `--device 0,1`）。個数が `--tp` と一致しない場合・重複している場合・
可視 device を超出する場合は明示エラー。model weight は rank ごとに
順次 load するため load 時間は単 GPU より長い。

`--tp 2` で併用できない option（いずれも model load 前にエラー）:

- prefix cache（`--prefix-cache 0` が必須）
- `--dflash2-model-dir`（per-rank 配線が未実装）

`--serve-stdio` も同じ runtime の TP 経路を使う。TP2 の serve 性能は未計測であり、
性能値は [../perf/current.md](../perf/current.md)、数値互換性・対応範囲の契約は
[../developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md)
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
| `--temperature 0` | deterministic greedy。`--top-p` / `--top-k` は無視 |
| `--temperature 0.7 --top-p 1` | temperature softmax からの categorical sample |
| `--temperature 0.7 --top-p 0.9` | nucleus sampling |
| `--temperature 0.7 --top-p 0.9 --top-k 50` | top-k と nucleus の積集合 |
| `--temperature 0.7 --top-p 1 --top-k 1` | 最上位 token のみ |

`--top-k` は `0` で無効。`--top-k` を小さくする、または `--top-p` を小さくすると
rejection の試行回数が増える。

同じ prompt・同じ `--seed`・同じ sampling 設定なら `GENERATED_IDS` は一致する。
batch 構成には依存しない。

```bash
./build/phaseshift-compute \
  --model-dir models/Qwen3.8-27B-FP8 \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --temperature 0.7 --top-p 0.9 --top-k 50 --seed 42
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
./build/phaseshift-compute \
  --model-dir models/Qwen3.8-27B-FP8 \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --kv-cache-dtype fp8
```
