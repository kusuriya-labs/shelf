# 停止条件と zero-work の contract

request の停止（termination）と、実行単位の有無（zero-work）は**別の概念**である。
この文書が意味論の正本である。実装の所在は層ごとに異なるが、意味論は同一でなければならない。

## 概念の分離

| 概念 | 意味 | 所有者 |
| --- | --- | --- |
| request termination | EOS / budget / cancel / error による request の終了 | request lifecycle の所有者 |
| zero-work | 今回の dispatch で仕事がないこと（`rows == 0` / `grid == 0` 等） | execution 層 |
| idle | 実行すべき request が無い状態 | scheduler |

## 停止条件の評価

停止条件の **policy は一箇所**、**評価する場所は各層**。

- Host backend: `RuntimeRequest` / `ContinuousBatcher` が request lifecycle を所有する。

executor・resolver・launcher は
**request の生成意味論に基づく停止条件**
（EOS / stop token / `max_new_tokens` / `max_seq_len`）を判断しない。
これらが判断するのは execution 上の「今回仕事があるか」だけである。

execution failure は、その failure を検出した実行層から lifecycle に
`error` として通知してよい。ただし既に terminal が latch されている場合は
上書きしない（下記 one-way latch）。

GPU kernel は request lifecycle も zero-work policy も所有せず、
**計算と fault 防止用 safety guard のみ**を所有する。

### token commit 内の停止条件優先順位と terminal の競合規則

token commit 時の semantic precedence は次の3つだけである。

1. EOS / stop token
2. `max_new_tokens`
3. `max_seq_len`（defensive）

同一 token が複数条件を満たす場合は**先のものが勝つ**。
例えば「`max_new_tokens` の最後の token が EOS」なら `eos` である。

`cancel` と `error` はこの順位表に含めない。
これらは同タイミングで比較する条件ではなく、時系列で解決する。

```text
cancel:
  transaction boundary で観測
  terminal_reason == none の場合のみ cancelled

error:
  execution / commit failure の発生時
  terminal_reason == none の場合のみ error
```

最終的な競合解決は one-way latch（first terminal wins）が担う。

```text
token commit → EOS 成立 → terminal_reason = eos
  → 後から cancel を観測しても eos のまま

cancel を transaction 開始前に確定 → terminal_reason = cancelled
  → 次の transaction 自体を開始しない
```

cancel は実行中の transaction を途中破棄しない。

### one-way latch

terminal は **one-way latch（first terminal wins）** である。

```text
none → eos / max_new_tokens / max_seq_len / cancelled / error
terminal → 他の terminal     （禁止）
```

既に terminal である request への terminal 判定は拒否する。

### terminal reason

| reason | 意味 | Host 正常系 |
| --- | --- | :---: |
| `eos` | model stop token | あり |
| `stop_token` | request-specific stop | 将来用に予約 |
| `max_new_tokens` | generation budget 消費 | あり |
| `max_seq_len` | sequence limit | 原則なし（admission で防止） |
| `cancelled` | user cancel | あり |
| `error` | execution failure | あり |

### admission contract

`max_seq_len` は正常系の停止理由ではなく **admission constraint** である。
Host の `ContinuousBatcher::submit()` は

```text
input_tokens.size() + max_new_tokens <= max_seq_len
```

を入口で保証する。
比較は uint32 の加算を避け、uint64 で行う。

```cpp
static_cast<uint64_t>(prompt_length) +
    static_cast<uint64_t>(max_new_tokens) <= config_.max_seq_len
```

`max_seq_len` の terminal reason は残すが、
内部 state 不整合で sequence limit に達した場合の防御（defensive terminal）として用いる。

### `max_new_tokens == 0`

```text
max_new_tokens == 0 は「生成 token 0個で terminal」を意味し、
「一切の GPU work を禁止する」意味ではない。
```

「生成 token が 0」と「model work が 0」は同義ではない。
prefix cache 等を考慮してもこの区別を保つ。

`max_new_tokens == 0` の request は次のように扱う。

```text
prompt prefill は通常通り完了させる。

final prefill の commit に到達した時点で（safe point）:
  - sampling は行わない
  - generated token は 0
  - decode phase へ進めない
  - terminal_reason = max_new_tokens
  - terminal record を 1 個 publish
```

**partial prefill の途中では terminal にしない。**
`generated_tokens >= max_new_tokens` の判定を無条件に適用すると、
`0 >= 0` が最初の partial prefill で成立し、
prompt prefill の完了前に terminal してしまうため、
判定は **final-prefill-without-sample の safe point でのみ**行う。

## request lifecycle の所在

| 論理責務 | 所在 |
| --- | --- |
| admission | `ContinuousBatcher::submit` |
| request state | `RuntimeRequest` |
| stop-token source | `eos_token_ids` |
| token commit | `commit_sampled_token` |
| terminal latch | `RequestState::Finished` + reason |
| cancel | `ContinuousBatcher::cancel` |
| resource release | `finish_request` |
| no runnable request | `step()` が空で返る |
| backend shutdown | 通常の Host lifecycle |
| zero-work | resolver / dispatch / launcher |
| safety | kernel guard |

`finish_request()` は「停止条件を決める場所」ではなく、
**既に決まった terminal state を確定し resource を解放する場所**である。

Host backend に persistent idle は存在しない。
`has_pending()` が false なら `step()` は空の結果を返すだけである。

## zero-work（`rows == 0` / `grid == 0`）

「workgroup 0 = 仕事なし」の判断は**各 execution 層が行い、kernel は行わない**。

| 実行方式 | zero-work の判断 |
| --- | --- |
| Host backend | resolver / dispatch / launcher |
| kernel | 判断しない |

zero-work は **request terminal を意味しない**。
今回の operation を dispatch しないだけであり、request の状態遷移は起こさない。

GPU kernel に残してよいのは **GPU fault / hang / Reset を防ぐ safety guard のみ**である。

- 除算ゼロ（例: `features == 0` / `group_size == 0` による商・剰余の計算）
- 配列境界（例: tile の余剰スレッドによる `r >= rows`）
- null 参照

これらは判断ではなく防御である。「ありえない異常パス」で GPU が
ハングしたり Reset を要求したりしないようにするためのものであり、
入力の意味論（仕事の有無）の判断は kernel の職務ではない。

## stop condition の source of truth

stop token の source of truth は **model の generation config 一箇所**である。

```text
Model generation_config
        │
        └── eos_token_ids の Host 表現
```

現段階では generation config の stop token は全て `eos` reason で扱い、
`stop_token` は request-specific stop token 導入時のために予約する。

## speculative decode の commit

verify で受理した draft 数（`SpecVerifyResult::num_accepted_drafts`）と、
semantic に commit する token 数（`SpecVerifyResult::num_committed_target_tokens`）は分離する。
commit 数は受理 draft に target の bonus token 1 を加えたものから、生成 budget
（`max_new_tokens - generated_tokens`）と sequence 残り容量（`max_seq_len - position`）と
terminal token 位置で上限を切る。

**terminal token より後ろの token は、output だけでなく semantic state にも commit しない。**

`num_committed_target_tokens` と terminal reason は、
resource / KV / output への副作用**より先に**決定する。

生成 budget の残りは `max_new_tokens - generated_tokens` であり、
`generated_tokens >= max_new_tokens` のときは 0 である。
sequence の残り容量は `max_seq_len - position` であり、
prefix が既に limit 以上のときは 0 である。
`max_seq_len == 0` は limit なしを表す。

```text
sampled / accepted tokens
        ↓
stop-condition scan
        ↓
commit 数の確定（num_committed_target_tokens）
        ↓
resource / KV commit（effective 分のみ）
        ↓
token stage
        ↓
generated_tokens の更新
        ↓
terminal reason の latch
```

例えば accepted が `A B EOS C D`（`num_accepted_drafts = 5`）であれば

```text
commit 数 = 3
terminal_reason = eos
```

であり、`C D` は resource / output / semantic state のいずれにも入らない。

stop boundary と budget boundary が同率の場合は token commit 内の
semantic precedence に従い EOS が勝つ。

```text
remaining budget = 3、tokens = A B EOS
  stop boundary = 3、budget boundary = 3
  → commit A B EOS、terminal_reason = eos

remaining budget = 2、tokens = A B EOS
  → commit A B、terminal_reason = max_new_tokens
    （budget の外側に EOS があるということは EOS は発生していない）
```

`commit 数` は resource commit より**先に**確定する。
resource commit を先に行うと accepted 全数が resource へ
入り、terminal token より後ろの token が state に残ってしまう。
