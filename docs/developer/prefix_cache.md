# Prefix Cache

prefix cache は `shelf-compute` の runtime が所有する GPU-resident cache である。
radix tree 上の page 単位で prefix の paged KV と GDN state を保持し、
`ContinuousBatcher` が request の admission / prefill / decode / 出力末尾で
lookup・restore・save を行う。共有 prefix への extension と multi-turn の会話で
prefill を省略する。

## Radix tree

- node は KV page（`--page-tokens` の既定 16 token）単位の token edge を保持する。
  入力末尾の端数 page は tree に保持しない。
- edge は `edge_tokens` / `edge_pages` の path compression である。
  insert で既存 edge の中間まで一致した場合は node split して分岐点を作る。
- page は entry 単位の refcount で管理する。restore 中の page は refcount が
  増えるため eviction の対象外になる。
- entry は radix node ごとに 1 つずつ対応し、page 揃え後の position と
  GDN state の保持先 slot を持つ。

## 2-tier store

hot tier は VRAM（GpuArena から確保する shadow pool）、
cold tier は pinned DRAM（`hipHostMalloc`）である。

- hot KV: shadow `PagedKVPool`。既定は `--prefix-cache-hot-tokens` = 16384 token 分で、
  `--page-tokens` 未満は指定できない。
- hot state: shadow `GdnStatePool` の `2 × max_concurrent` slots
  （`max_concurrent` = `--prefix-cache-concurrency` 既定 4）。うち `max_concurrent` slots は
  capture 中 rolling slot、残り `max_concurrent` slots が entry 保持用である。
  entry 用 slot が不足すると hot state を cold へ demote する。
- cold KV: `--prefix-cache-cold-tokens` 既定 65536 token 分の pinned host buffer。
- cold state: pinned host buffer。既定は `--prefix-cache-cold-state-slots 0` = auto で
  `4 × max_concurrent` slots。
- `PrefixCache::create` は hot 分の arena 空き容量と cold 分の host memory 空きを検査し、
  不足なら作成自体を失敗させる。

### pin と eviction

- 最新 `max_concurrent` 件の entry（`last_used` 降順）の page は pin され、
  実行中の request が参照する entry を冷やさない。
- hot KV に空きが不足すると、次の順で room を確保する。

  1. 非 pin の LRU leaf node を cold tier へ migration する
  2. それでも不足すれば pin 対象の LRU leaf も cold tier へ migration する
  3. それでも不足すれば refcount 0 で entry を持たない LRU leaf を実削除する
  4. それでも確保できなければ save を `insufficient_memory` で中断する（fail-closed）。
     serving は継続し、当該 request は prefix cache を使わない。

- cold state が満杯のときは、cold LRU entry を削除して slot を回収してから
  hot state を demote する。

## hit / restore

`lookup(tokens)` は radix walk で page 単位の最長一致 M を求め、
その範囲で path が全 page 一致している entry のうち `position <= M` の最深のものを返す。

`ContinuousBatcher` は request admission 時に input 全体で lookup し、
hit かつ `state_position < input_len` のときだけ restore する。
`state_position >= input_len` は restore せず miss 扱いである
（その位置の初回サンプルの forward が存在しないため）。

restore は entry の KV page を新規 sequence の block table へ予約・複写し、
GDN state を entry slot から active state pool の slot へ複写する。
restore 後の token 数は `done.restored_tokens` に入る。

## save

save は page 揃え位置でのみ発生する。

- prefill: prefill chunk の完了時に、sequence position が page 揃えされ、
  かつ input の範囲内なら `save_input_boundary` が input prefix を保存する。
  `save_input_boundary` は `seq.position == input_len を page 揃えした値` を必須検証し、
  違反は `invalid_argument` である。`min_prefix_tokens`（64）未満は保存しない。
  DFlash2 経路では prefill を page 揃えした chunk 境界で区切り、各境界で保存する。
  最後の chunk は input 末尾より前の直近の page 境界で分割するため、input 長が
  page 揃えでなくても直近の境界までは保存される。ring は chunk 完了時点の
  drafter context から保存する。
- decode: 最終 prefill の完了時に capture を開始し、各 step 後に position が
  page 境界を新しく通過するたびに GDN state を rolling slot へ capture する。
  DFlash2 では次の page 境界までの残りトークン数に draft 数を制限し、境界を
  飛び越えずに capture できる位置で step を止める。
- 出力末尾: EOS / `max_new_tokens` で終了し、capture 位置が最終 page 境界と一致するとき、
  input と生成列から成る position 長の token 列を `save_output_tail` で entry 化する。
  cancel などで境界が一致しない場合は capture を解放して保存しない。

## options

option の一覧と既定値は [../user/compute.md](../user/compute.md) の options を正本とする。

作成に失敗したときは prefix cache を無効化して続行し、警告を 1 行 stderr に出す。
`--serve-stdio` の stdout（JSONL protocol）には混ぜない。

## protocol

`ping` の応答:

```json
{"event":"pong","protocol_version":1,"capabilities":{"prefix_cache":true}}
```

`capabilities.prefix_cache` は `PrefixCache::create` の成功時のみ `true` である。
`capabilities` は map であり、将来の capability はここへ追加する。
未知の key は client が無視してよい。

`generate` の `prefix_cache_checkpoint_position` は `0` のみ受理する。
非 `0` は `event=error` / `code=invalid_argument` で拒否し、silent ignore はしない。
prefix の再利用位置は compute 側の radix lookup に一元化されており、
server は常に `0` を送る。この validation は serve protocol 境界で行い、
Qwen runtime や GPU へこの値は渡さない。

`done` event:

```json
{
  "event": "done",
  "request_id": 1,
  "generated_ids": [...],
  "finish_reason": "stop",
  "prompt_tokens": 123,
  "prefill_tokens": 45,
  "restored_tokens": 78,
  "cache_checkpoint_tokens": 0
}
```

`restored_tokens` は restore された token 数の実値、
`prefill_tokens` は `input - restored_tokens`、
`cache_checkpoint_tokens` は常に `0` である。

## 既知の制限

- 同一プロンプトの再実行は restore しない。最長一致が input 全長に達しているため
  `state_position < input_len` を満たさない。restore が効くのは共有 prefix への
  extension と multi-turn の output tail である。
- Tensor Parallel の mirror は未対応である。

## DFlash2 経路

DFlash2 も通常 decode と同じ `ContinuousBatcher` の `prefix_*` ヘルパを通る。entry には
paged KV と GDN state に加えて drafter の `DFlash2ContextState` ring（
`SpecRingRef` / `SpecRingOut`）を保存する。restore では ring も復元し、
`dflash2_spec_prefill` は残り suffix のみを prefill する（`prefix_tokens` 引数）。
ring は hot entry slot にのみ保持し、cold tier への migration はしない。
ring を保存できないときは suffix prefill が長くなるだけで、挙動は正しい。

restore / save の入力は次のとおりである。

- admission の restore は request の `DFlash2ContextState` を先に作成し、
  `SpecRingOut` に ring buffer を渡す。restore 後の `length` / `next_position` が
  ring の状態になる。
- 出力末尾の save（`prefix_finish_capture`）は request の ring を `SpecRingRef` として
  渡す。ring は commit-only であり `sequence.position` と常に一致する。
- output tail の entry は page 揃えした position の prefix である。ring の
  `length` / `next_position` は **entry position へ clamp して** 保存する
  （final position のまま保存すると entry の KV 範囲と ring の範囲がずれる）。
- prefill 境界の `save_input_boundary` は DFlash2 では呼ばない。DFlash2 は
  prefill 完了時の capture 開始と page 境界通過時の `capture_boundary` のみを使う。
- DFlash2 request は **ring を持つ entry だけ** を restore する。`lookup` は entry の
  ring 情報（`PrefixHit::ring_length` / `ring_next_position`）を返し、
  `ring_covers_state()`（`ring_length != 0` かつ `ring_next_position == state_position`）を
  満たさない entry は restore せず full prefill する。ring は DFlash2 request の
  save だけが作るため、通常 decode が作った entry は DFlash2 では restore されない。

### spec ring pool

drafter ring の shadow は `PrefixCacheConfig::spec_ring_bytes` で指定する。

- slot 数は `max_concurrent`、1 slot のサイズは `spec_ring_bytes`（K と V の合計）。
  compute は DFlash2 有効時に `DFlash2Config::context_ring_bytes()`
  （`2 × num_hidden_layers × sliding_window × attention_kv_rows() × 2` bytes。
  27B では 40 MiB）を渡す。
- 実体は arena（VRAM）上の連続 buffer で、save / load は D2D コピーである。
- `spec_ring_bytes == 0`（既定）のときは slot を 1 つも確保しない。ring を渡した save は
  `insufficient_memory` で失敗し、entry が作られないため DFlash2 の restore も起きない。
  compute は DFlash2 有効時に必ず設定する。

## 検証

`tests/server/test_compute_prefix_contract.py` が JSONL contract を固定する。
