# 推論予算と compaction 検知（Strata 調査）

対象: `~/work/references/Strata`（0.2.x 系、2026-11-02 時点の checkout）。
目的: 「リクエスト毎の最大推論予算 32768 / thinking 24576」の数字の由来を確認し、
compaction 時にだけ予算を 65536 へ上げる機構が有るなら同じものを shelf に入れる判断をする。

## 実験 ID: BUDGET-STRATA-COMPACTION-1

結論: **採否 = 一部採用**。thinking 予算の計算式と強制終了の位置は採用した。
「compaction 要求を検知して予算を上げる」機構は shelf では採用せず、
**prompt が既に使っている context 量で判定する**方式に置き換えた。

### 1. Strata の推論予算の構造（実測: コード読了、行番号で特定）

- `serve/server.py:217-220`
  - `ANSWER_ROOM_MIN_CAP = 1024` — この `max_tokens` 以上で thinking 予算を適用する
  - `ANSWER_RESERVE_MIN = 512` — answer に残す最小 Token 数
- `serve/server.py:5485-5510`
  - client が budget を指定せず `answer_room: true`（既定）なら
    `budget = max_new - min(max(512, max_new // 4), max_new // 2)`
  - `max_new = 32768` のとき `reserve = 8192`、`budget = 24576`
  - 明示 budget があっても `max_new - reserve` まで切り下げる
- 予算に達した thinking は `serve/server.py:5841-` で model 自身の end-of-thinking tag を
  注入して閉じ、残りで回答を書かせる。shelf の `commit_sampled_token` 置換と同じ形である。

`32768 / 24576` という数字は Strata の既定 `max_tokens` と reserve 式から決まっていた。
shelf でも同じ式を使う（`openai_protocol.answer_reserve_tokens` /
`thinking_budget_tokens`）。

### 2. compaction の検知方法

Strata が検知に使っているのは **client の自己申告 2 種類**だけだった。

| 経路 | 申告 | 所在 |
| --- | --- | --- |
| Responses API（Codex CLI 0.140+） | `client_metadata["x-codex-turn-metadata"]` の JSON `request_kind == "compaction"` と `session_id` / `thread_id` | `serve/responses.py:370-410` |
| Harness（dsh / pi） | `llm.stream(options)` の `options.purpose === "compaction"`。wire には出ない | `apps/host/server/agent/plugins/strata-context.mjs:72` |

- どちらも opt-in（`codex_compaction_cache` は既定 off、`serve/server.py:4183`）。
- 目的は**予算ではなく prefix cache**である。compaction 要求は `tools: []` で送られるため、
  会話の tools を復元して同一 prefix を再利用するのが狙い（`docs/DETAILS.md:873-`）。
- Strata は prompt 文面による判定を明示的に棄却している。
  `serve/test_responses.py:216-221` 「a compact prompt in a message is not a compaction」
  （引用された compaction 指示文を持つ通常要求を誤検知する）。

### 3. 「65536」の実体

リクエスト単位の出力予算を 65536 へ引き上げる実装は見つからなかった。
`65536` が出るのは次のもので、いずれも request 予算ではない。

- `apps/host/server/agent/profile.mjs:151-153` `compactionHeadroom()` の上限。
  compaction を開始する占有率の headroom であり、`localOutputBudget()` は
  `max(4096, min(32768, window * 0.3))` で **32768 が上限**（`profile.mjs:137-143`）
- `strata_cli/launch/harness.py:151` `strata connect hermes` が書く
  `auxiliary.compression.context_length: 65536`。app が**要約専用 model に対して設定する
  window** であり、server 側の検知ではない
- `strata-context.mjs:136` の `summaryTokens * 2`。window に収めるための piece 計算でのみ
  2 倍見る係数

つまり Strata の「compaction のときだけ大きく」は、**client 側の別モデル設定**であって、
model server が要求を見て予算を変える機構ではない。

### 4. shelf で採らなかった理由と、代わりにした判定

- shelf が接する harness（OpenCode / DeepSeek Harness）は Chat Completions しか送らず、
  `request_kind` も `purpose` も wire に載らない。Strata と同じ検知を入れると
  shelf 専用 field を全 client に要求することになる。
- 仮説（未実測）: `opencode.exe` の文字列として `SessionCompaction` が組み立てる指示文
  `You MUST use this format for your response ... <template> ## Objective ...` と、
  compaction 係数の群（`0.85 / 0.7 / 0.5 / 0.35`、`16000`、`64000` 等）は確認できたが、
  wire へ出ていく要求をこの環境で実測はしていない。compaction 要求が通常の要求と区別不能な
  形（会話と同じ model・同じ出力上限で、要約指示を最後の message として送る形）だと
  すれば、検知は Strata が棄却した「文面判定」に相当する。
- 予算が必要になる実体は「会話長」であって「compaction という イベント」ではない。
  shelf は prompt の Token 数を見る。

採った規則:

```text
prompt_tokens > 262144 → 推論予算 65536（thinking 49152）
それ以外               → 推論予算 32768（thinking 24576）
```

`262144` は `--large-context-tokens`、`65536` は
`--large-context-inference-budget-tokens` で変える。`--large-context-tokens 0` で無効化できる。

#### 検証

- `tests/unit/test_thinking_budget.cpp`（CPU, required）— `commit_sampled_token` の
  置換位置、自然 close tag、予算なし、allowance 追従、EOS 優先。PASS。
  required の CPU ctest 82/82 PASS（`test_cli_docs_sync` / `test_cli_docs_sync_python` を含む）。
- `tests/server/test_inference_budget.py`（model 必須・GPU 不要）— 予算の clamp、
  閾値での予算切替、prepare の thinking 予算、close token が不明なときの予算なし。25/25 PASS。
- `tests/server/test_compute_thinking_budget.py`（GPU, reasoning group）—
  27B-FP8・TP1・`--max-seq-len 4096`。予算なしでは thinking が 128 Token を超え、
  `reasoning_budget_tokens: 128` では generated_ids の index 128 が close token で、
  その後も生成が続く。5/5 PASS。
- server regression の `reasoning` group（device 1）— `reasoning-parser` 24、
  `reasoning-request` 49、`compute-thinking-budget` 5、`server-reasoning` 27、
  `server-reasoning-stream` 21、`server-reasoning-tools` 23 が PASS。
  初回実行では `compute-thinking-budget` と `server-reasoning` が
  `hipMalloc failed: out of memory` で落ちたが、同一ホストの別 process が GPU を
  確保していたためで、単独再実行で PASS した（環境起因）。
  - 環境: `SHELF_TEST_DEVICE=1 SHELF_TEST_ARENA_GIB=31 SHELF_TEST_WORKING_SET_RESERVE_GIB=1`

#### server E2E（2026-11-02 実測）

`build-shelf/shelf-compute` を TP1 / device 1 / `--max-seq-len 4096` /
`--max-concurrent-requests 4` / `--inference-budget-tokens 2048`
（thinking 1536 / answer reserve 512）で起動。arena 31 GiB では prefix cache が確保できず
`prefix cache unavailable`（cache なしの通常 decode）。

| request | 結果 |
| --- | --- |
| `max_completion_tokens: 100000`（非 stream） | `completion_tokens 2048`・`finish_reason=length`。推論予算で clamp された |
| 同上の応答内容 | `reasoning_content` 2177 文字 + `content` 747 文字。thinking 1536 で閉じて回答が続いた |
| `stream: true` + `include_usage` | `completion_tokens 206`・`finish_reason=stop`。delta は reasoning 87 chunk → content 116 chunk。自然に閉じた場合は従来どおり |

予算がないと同じ上限（2048）では thinking が使い切られて回答が空になる系であるため、
`content` が non-empty であること自体が thinking 強制クローズの実測証拠になる。

#### DFlash2 経路の容量（当初の「容量不足」判断は誤り）

TP2 で target 重みは分割されている。`--arena-gib 20` の plan は
「tp rank arena has 0.32 GiB free after weights」を出し、rank あたり 19.68 GiB であった
（TP1 は 27.45 GiB）。内訳は transformer 層の重みが分割、`embed_tokens` と `lm_head` は rank ごとの複製
（BF16 で 2.37 GiB ずつ、`vocab_parallel_lm_head` は compute CLI に公開されていない）。

drafter が読めなかった真因は **KV pool が arena の残りを自動で確保する**ことであった。

| 構成 | 実測 |
| --- | --- |
| TP2 / `--max-seq-len 512 --prefix-cache 0 --working-set-reserve-gib 1` | `used=29.94 GiB`（arena 30.63 GiB）で drafter 3.58 GiB の空きが無い → `dflash2 weight load failed: matrix allocation failed: layers.0.mlp.up_proj.weight` |
| TP2 / 同条件で `--working-set-reserve-gib 6` | `used=25.77 GiB` → `DFLASH2_ENABLED=1`・`DFLASH2_SESSION_POOL_SLOTS=1`・`SHELF_SERVE_READY` |
| TP1 / `--max-seq-len 256 --prefix-cache 0` | weights 後 3.26 GiB 空、必要 KV 0.01 + draft 3.58 + workspace 0.12 → 0.45 GiB 不足。27B-FP8 + drafter は 1 枚では無理 |

TP1 の auto plan は draft 重み分を確保計算に入れるが、TP2 の rank arena では KV auto-fit が
それを踏まない（`tensor_parallel.cpp` の reserve 計算）ため、drafter が arena に入らなかった。

採否: **auto-fit の確保計算に drafter 分を織り込む**方を取る（順序は変えない）。
`dflash2_footprint.h` に drafter 重み + draft executor の scratch + session pool の slot
× `--max-concurrent-requests` の式を集約し、TP1 の plan と TP2 の rank auto-fit が
同じ式を使うようにした。fix 後の実測（TP2 / device 0,1 / `--max-seq-len 4096`
`--max-concurrent-requests 4` / `--prefix-cache 0` / 既定 reserve 1 GiB）:

```text
MEMORY_PLAN tp drafter: weights 3.58 GiB scratch 0.03 GiB session slot 0.34 GiB x 4
MEMORY_PLAN mode=auto tp=2 arena=30.63 GiB used=25.81 GiB available=31.78 GiB
DFLASH2_SESSION_POOL_SLOTS=4
DFLASH2_SESSION_POOL_BYTES=403874632
```

`--working-set-reserve-gib 6` を手動で与えずに `SHELF_SERVE_READY` に到達する。
session slot の見積 0.34 GiB に対し実測は 0.094 GiB / slot で、見積は約 1 GiB 過剰である
（fail するより KV が減る方向なので採った。詰める場合は `spec_gdn_conv_bytes` の
history 分の見直しが必要）。

drafter 重みの式は実測と一致した。serve 時の arena 増分を print した結果は
`DFLASH2_WEIGHT_ARENA_BYTES=3848808960`（3.584 GiB）で、`safetensors_dir_bytes` の
3.58 GiB と同じである。

| 条件（TP2 / sessions=1 / prefix cache 有） | 結果 |
| --- | --- |
| `--working-set-reserve-gib 1` | plan `used=26.82`。drafter は 3.06 GiB まで読み進めて `layers.4.mlp.gate_proj.weight` で失敗 |
| `--working-set-reserve-gib 2` | plan `used=25.82` → `SHELF_SERVE_READY` |
| `--working-set-reserve-gib 4`（既定） | plan `used=23.82`（prefix cache なし）→ `SHELF_SERVE_READY` |

reserve=1 の失敗は drafter ではなく **plan 行の後で executor が確保する作業領域**が
1 GiB を超えるためである（`MEMORY_PLAN` の `used` は executor 確保前の値）。
`--working-set-reserve-gib` は plan 後の executor / workspace 用の余白であり、
drafter 分とは別に必要である。

#### thinking 予算の DFlash2 経路（実測）

auto-fit を直した構成で `test_compute_thinking_budget.py` を実行し、**12/12 PASS** であった
（TP1 / device 1 の decode 6 件と、TP2 / device 0,1 + drafter の 6 件）。
DFlash2 経路で固定したのは次の点である。

| 検査 | 内容 |
| --- | --- |
| `dflash2-close-token-at-the-budget` | `reasoning_budget_tokens: 128` で `generated_ids[128]` が close token |
| `dflash2-thinking-stayed-open-before-the-budget` | 128 手前まで close token が出ない |
| `dflash2-generation-reaches-the-budget` | 予算に達するまで thinking が開いたまま生成が続く |
| `dflash2-close-token-appears-only-once` | close token は 1 回だけ |
| `dflash2-generation-continues-after-the-budget` | close token 後も生成が止まらない |
| `dflash2-baseline-thinking-exceeds-the-budget` | 予算無しでは thinking が 128 を超えて続く |

decode 経路と DFlash2 経路の token 列は比較しない。TP の reduction 構成で分岐しうるため、
各経路で thinking 予算の contract だけを検証する形にした。

### working set reserve を自動にできるか（実測）

`--working-set-reserve-gib` は plan の後で確保される領域の予約である。plan 後の
確保箇所を stage 別の print で調べた（TP1 / device 1 / `--max-seq-len 4096`
`--max-concurrent-requests 4` / `--prefix-cache 0` / FP8 KV / reserve 1 GiB）。

```text
ARENA auto: free=31.78 GiB capacity=30.78 GiB
MEMORY_PLAN kv pool: 2.04 GiB (arena used 29.78 GiB)
MEMORY_PLAN executor footprint: 0.04 GiB (reserve 1.00 GiB)
arena capacity exceeded: capacity 30.78 GiB used 29.82 GiB requested 1.74 GiB
```

失敗した 1.74 GiB の確保は `create_model_executor` の内部であり、
`program.workspace.total_bytes`（選択 row bucket の program workspace）が該当する。
executor の他の arena 確保（`logits {MO, V}`、`token_hidden {Npad, H}`、
sampling 系、`create_device_batch_context`）は合計 0.04 GiB 程度で支配項ではない。

採否: **plan 段階では自動計算できない**。program workspace は
`build_program_set(lowered graph, weight slots, param slots, …)` の結果であり、
その構築は `create_model_executor` の中にある。`build_program_set` 自体は host 側だけ
（GPU 確保なし）なので、

1. `create_model_executor` を plan（graph lowering + ProgramSet + footprint 計算）と
   create（arena 確保）に分ける
2. KV auto-fit は `free − executor footprint − drafter footprint − prefix cache` で頁数を出す
3. `--working-set-reserve-gib` は測定用の override に降格（既定は auto）

という順序にすれば自動化できる。split は `create_model_executor` の 1600 行台を
またぐ変更で、TP1 / TP2 / bench / single request の呼び出し側が追随するため、
別の change として扱う判断にした（`docs/homework.md`）。

### working set reserve を auto にした（実測）

上記の不确定分を、row bucket から推定して埋める方式にした。executor の arena 消費を
確保単位で print して内訳を実測した（TP1 / FP8 KV / requests 4 / prefix cache off /
bucket 4096）。

```text
EXEC_ALLOC logits                       3.79 MiB (cum 3.84 MiB)
EXEC_ALLOC stochastic_topk             40.24 MiB (cum 44.09 MiB)
EXEC_ALLOC program_workspace         1780.66 MiB (max prog 1748.62 MiB)
MEMORY_PLAN executor footprint: 1.78 GiB
```

つまり footprint の 98% は program workspace（`max program workspace + scratch 32 KiB
+ attention partials 32 MiB`）で、行あたり bf16 値数は 43.7 であった。TP2 rank では
`intermediate_size` が shard されるため 8192 bucket の実測が 22.4 値/row（合計 1.84 GiB）。
final の式は `44 × (hidden + intermediate / tp) / (hidden + intermediate)` である
（`docs/developer/runtime.md`）。

見積と実測（auto 最終形）:

| 構成 | 見積 reserve | 実測 footprint | 結果 |
| --- | --- | --- | --- |
| TP1 / bucket 512 | 0.26 GiB | 0.25 GiB | READY / KV 89,776 tokens |
| TP1 / bucket 2048 | 0.91 GiB | 0.91 GiB | READY / KV 68,496 tokens |
| TP1 / bucket 4096 | 1.79 GiB | 1.78 GiB | READY / KV 40,144 tokens |
| TP2 / bucket 8192（`--max-seq-len 262144`、drafter なし） | 2.22 GiB | 1.84 GiB | **SHELF_SERVE_READY** |
| TP2 / bucket 8192（`--max-seq-len 524288 --rope-factor 2`、drafter なし） | 2.22 GiB | 1.84 GiB | **SHELF_SERVE_READY** |
| TP2 / bucket 8192（262144 + drafter） | 2.64 GiB | 1.84 GiB | KV 上限 207,695 tokens で不足 |

### 誤った容量結論を出した原因

最初に「TP2 + drafter は 262144 が入らない（上限 114,351 tokens）」と書いたが、これは
以下の重ね書きによる見積もり过大だった。

1. KV auto-fit が drafter 分を差し引いていなかったため、drafter が arena に入らず、
   `--working-set-reserve-gib 6` で回避していた（KV からさらに 2 GiB を削っている）
2. auto reserve の定数 52 値/row を **TP1 で較正して TP2 にそのまま適用**した。
   TP2 rank の program workspace は shard されて 22.4 値/row なので、3.55 GiB を
   見積もって 1.71 GiB / rank を無駄にしていた。524288 は rank あたり 8.3 GiB の
   KV が必要で、空きは 10.95 − 3.55 = 7.40 GiB に減っていた
3. `--max-seq-len 524288` は rope limit でも別口で拒否される（model の
   `original_max_position_embeddings` が 262144、`--rope-factor auto` では 262144 が上限）。
   524288 には `--rope-factor 2` が必要である

構造的な足枷として、`main.hip` が `max_scheduled_tokens`（現 `max_num_batched_tokens`）
= `max_seq_len`
とするため、context を伸ばすと必ず R8192 bucket に張り付き program workspace が
1.75 GiB（TP1）出る。**prefill の chunk size を下げる CLI が無い**（homework）。

### run.sh の構成（TP2 + drafter + 262144）は chunk size で収まる

`--max-num-batched-tokens` を足す前は serve 経路が `max_scheduled_tokens`（現
`max_num_batched_tokens`）= `max_seq_len`
としていたため、context を伸ばすと必ず R8192 bucket に張っていた。

```text
--max-num-batched-tokens 8192相当（旧動作）: reserve 2.64 GiB / 実測 1.84 GiB
  → tp rank arena fits 207696 KV tokens but max_seq_len 262144 needs 262145
--max-num-batched-tokens 2048（既定）: reserve 0.71 GiB / 実測 0.61 GiB
  → MEMORY_PLAN mode=auto tp=2 arena=30.63 GiB used=25.56 GiB → SHELF_SERVE_READY
```

chunk 2048 で 8,192 token prompt の生成も実測済み（12 tokens、`finish_reason=length`）。

### chunk 2048 で露見した TP auto-fit の prefix cache 漏れ

`--max-num-batched-tokens` を導入して KV が増えた構成で、TP2 + drafter + prefix cache
既定が次で失敗した。

```text
matrix allocation failed: layers.4.mlp.up_proj.weight want 0.17 GiB
  arena used 30.50 GiB / capacity 30.63 GiB
```

TP の rank auto-fit は `working_set_reserve + drafter` を引いていたが、**prefix cache の
hot tier を引いていなかった**。prefix cache は KV pool の後（serve 前）に確保されるため、
drafter の残り 0.4 GiB を奪っていた。`--prefix-cache 0` では起動することを確認し、
原因を確定した。

採否: rank auto-fit で `prefix_cache_bytes()` を rank の local kv heads と GDN layout で
計算して減算する（TP1 と同じ扱い）。fix 後の実測（TP2 / drafter / prefix cache 既定 /
chunk 2048 / requests 1）:

```text
MEMORY_PLAN auto tp rank 0: working set reserve 0.68 GiB (row bucket 2048)
MEMORY_PLAN tp prefix cache: 0.70 GiB
MEMORY_PLAN tp drafter: weights 3.58 GiB scratch 0.03 GiB spec batch 0.02 GiB session slot 0.34 GiB x 1
MEMORY_PLAN mode=auto tp=2 arena=30.63 GiB used=25.86 GiB available=31.78 GiB
SHELF_SERVE_READY
```

なお drafter の spec batch context（history + compact buffer）も計算に入れていなかったが、
requests 1 では 0.02 GiB と小さく、今回の失敗の主因ではなかった。`dflash2_footprint.h`
の `spec_batch` 項として追加済みである。
scheduler は prefill を chunk 分割する（`token_budget_scheduler.cpp` の `take = min(remaining,
budget_left())` / `final_chunk`）。
