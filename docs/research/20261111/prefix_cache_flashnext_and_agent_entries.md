# Prefix cache: Qwen4Exp 対応と、エージェント共有 prefix が当たらない原因

## 1. Qwen4Exp（Qwen3.8 Flash Next）の prefix cache

### 1.1 変更前

`Qwen35ComputeRuntime::initialize` が Qwen4Exp で `config_.prefix_cache = false` にしていた。
理由は PLE の n-gram history が「その process が scheduled した token」から作られており、
prefix cache が復元した prefix を history が持たないためである
（`PleInputRuntime::prepare` は `history.size() == position` を要求し、足りなければ
`invalid_state`）。

さらに `Qwen35PrefixState` は GDN state しか shadow に持たず、Qwen4Exp の QSA key state と
PLE conv state を保存していなかった。

### 1.2 変更

- `PrefixCacheRankContext` / `PrefixCache::create` に QSA pool と PLE conv pool を追加し、
  `Qwen35PrefixState` が GDN と同じく `2 × max_concurrent` slot の shadow を持つ。
  capture / hot / 出力末尾のコピーは GDN と同じ経路で行う。
- `prefix_cache_bytes` と TP の memory plan に shadow 分を計上する。
  `TpRankModel::executor_state_reserve_bytes(2 × prefix_concurrency)` を使う。
- `Executor::external_prefix_restore` を追加し、restore 後に復元 prefix の token を
  `PleInputRuntime::seed(slot, tokens)` へ渡す。TP では
  `TpBatchHook::on_prefix_restored` が全 rank へ配る。
- Qwen4Exp の `prefix_cache = false` を削除する。

### 1.3 検証（ACCEPT）

`Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、TP2、`--max-seq-len 512`、bf16 KV、arena 29 GiB。

1. X（324 token）を 1 回目に流す（entry なし、`restored_tokens=0`）。
2. X[:320]（page 揃え）を流して入力境界 entry を作る。
3. X をもう一度流す。

結果: 3 回目は `restored_tokens=320`、`prefill_tokens=4`。生成列は 1 回目と完全一致した
（`generated_ids=[248046]`）。PLE history の seed が無い、または QSA state の復元が
不正なら、restore 側の logits がずれて列が変わるため、この一致は両方の正しさを含む。

## 2. 27B で「エージェント／サブエージェントの prefix cache が効かない」原因

### 2.1 仮説

保存される境界は prefill chunk の終端（既定 `--max-num-batched-tokens 2048`）と、
入力末尾・出力末尾の page 揃え位置だけである。entry の state を保持する hot slot は
`--prefix-cache-concurrency`（既定 4）個しかなく、hot slot が埋まると LRU entry を
evict する。長い prompt の prefill は 2048 token ごとに entry を作るため、
既定では浅い境界（system prompt 付近）から先に消える。サブエージェントが共有するのは
先頭の system prompt なので、そこで一致しても entry が残っていない。

DFlash2 は加えて ring を持つ entry しか restore しない。ring slot も同じ
`max_concurrent` 個で、変更前は ring slot の確保に失敗すると entry 自体を作らなかった。

### 2.2 実測

`Qwen3.8-27B-EXL3-INT8-4bpw` + `Qwen3.8-27B-DFlash2-EXL3-INT8-4bpw`、TP2、fp8 KV、
`--max-concurrent-requests 2`、`--max-seq-len 16384`、arena 24 GiB。

- A: 12000 token の prompt（chunk 境界 2048 / 4096 / 6144 / 8192 / 10240 / 12000 と
  出力末尾 12000 の計 7 entry を保存。ログで全件 `ok=1`）。
- B: A の先頭 3000 token だけを共有する prompt。

| `--prefix-cache-concurrency` | A の lookup | B の lookup | B の `restored_tokens` |
| ---: | --- | --- | ---: |
| 4（既定） | miss | miss（`in=3000`） | 0 |
| 16 | miss | hit `state=2048` | 2048 |

既定 4 では hot slot が 4 個しかないため、7 entry のうち浅い 2048 / 4096 / 6144 が
LRU で消え、B が一致できる境界が残らない。16 では 2048 の entry が残り復元できる。

### 2.3 結論と推奨（未実装）

原因は「entry の保持数が `--prefix-cache-concurrency` に固定されていて小さすぎる」こと。
推奨する修正は次の順で、いずれも未実装である。

1. entry 用 state slot を capture 用から分離し、量を独立に指定できるようにする
   （VRAM から自動で決めるか `--prefix-cache-entries` を足す）。
2. `--prefix-cache-concurrency` の help に「capture と保存 entry の両方を決める」と書く
   （本変更で実施済み）。
3. 共有 prefix の境界（system prompt 末尾）を server から checkpoint として送り、
   そこで prefill を区切って保存する。chunk 境界に依存しなくなるが、protocol 変更になる。

## 2.4 SlotReuse（`--prefix-cache-strategy slot`）の実装と実測

2.3 の原因に対し、entry 管理だけを切り替える `slot` strategy を追加した（ACCEPT）。
KV page 共有と family state の保存・復元は `radix` と共通である。

- request ごとに固定 slot を予約し、1 slot に付き entry 1 個だけを保持する。
- 保存位置は `align_down(input_len - 1, page_tokens)` の 1 点。通常 prefill は
  `token_budget_scheduler` が、DFlash2 は `dflash2_spec_prefill` の checkpoint 引数が、
  その位置で chunk を区切る。
- 更新は新しい entry の保存成功後に旧 entry を解放する。slot 単位の LRU で置換する。
- 出力末尾の rolling capture は slot では使わない。

実測（27B EXL3 + DFlash2、TP2、fp8 KV、`--max-concurrent-requests 4`、16384 context、
`--prefix-cache-strategy slot`）:

| 操作 | `restored_tokens` | `prefill_tokens` | `prefix.entries` |
| --- | ---: | ---: | ---: |
| 12000 token の初回 | 0 | 12000 | 1 |
| 同じ 12000 token の再実行 | 11984 | 16 | 1 |
| 別の 10000 token の初回 | 0 | 10000 | 2 |
| 12008 token（延長）の初回 | 11984 | 24 | 2 |
| 同じ 12008 token の再実行 | 12000 | 8 | 3 |
| 4000〜9000 token を 6 回追加 | 各回 hit | — | 3（evictions 6） |

同じ入力の miss と restore で生成列が一致した。chunk ごとに entry が増えないこと、
slot 数（4）を超えないこと、LRU 置換が起きることを確認した。

Qwen4Exp（Flash Next、TP2、324 token）でも slot strategy で `restored_tokens=320`、
`prefill_tokens=4`、生成列が miss と一致した（PLE history の再 seed を含む）。
`tests/unit/test_prefix_cache.hip` の `slot` ケースが slot の保存・更新・LRU・
並列 writer の排他・restore を固定する。

## 3. 実装した DFlash2 ring の堅牢化（ACCEPT）

- ring slot が枯渇したとき、hot slot と同じく LRU entry を evict して回収する
  （変更前は即 `insufficient_memory`）。
- ring を保存できなくても entry 本体（KV page と family state）は保存する。
  その entry は DFlash2 request からは `ring_covers_state()` を満たさず full prefill に
  戻るが、非 DFlash2 request からは restore できる。変更前は entry ごと捨てていた。

## 4. 再現手順

`/tmp` の ad-hoc script は保存していない。手順は上記のとおりで、
`shelf-compute --serve-stdio` に `generate` を 2 回流し、`done.restored_tokens` を見る。
`SHELF_PFXDBG=1` を設定すると `save_input_boundary` / `lookup` のログが出る。
