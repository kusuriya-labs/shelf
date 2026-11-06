# Prefix Cache

prefix cache は `shelf-compute` の runtime が所有する GPU-resident cache である。
radix tree 上の page 単位で、prefix の paged KV は active KV pool の page を共有して
保持し、GDN state は shadow slot に保持する。`ContinuousBatcher` が request の
admission / prefill / decode / 出力末尾で lookup・restore・save を行う。
共有 prefix への extension と multi-turn の会話で prefill を省略する。

## Radix tree

- node は KV page（`--page-tokens` の既定 16 token）単位の token edge を保持する。
  入力末尾の端数 page は tree に保持しない。
- edge は `edge_tokens` / `edge_pages` の path compression である。
  insert で既存 edge の中間まで一致した場合は node split して分岐点を作る。
- page は entry 単位の refcount で管理する。restore 中の page は refcount が
  増えるため eviction の対象外になる。
- entry は radix node ごとに 1 つずつ対応し、page 揃え後の position と
  GDN state の保持先 slot を持つ。

## KV page sharing

KV は model-independent な `shelf::inference::prefix::PrefixKVStore` が管理する。
store は `PagedKVPool` の物理 page を refcount 付きで共有し、shadow KV pool を持たない。

- save は active sequence の page ID を radix entry に登録するだけで、D2D copy をしない。
- restore は entry の page ID を新しい sequence の block table に map するだけで、
  D2D copy をしない。
- page は sequence 参照と cache 参照の両方を持ち、どちらも 0 になったときだけ free list に戻る。
- cache は KV pool の全 page を追跡できる。cache 専用の VRAM 予約は存在しない。
- 物理 page が不足した場合、`PagedKVPool` は cache の reclaimer を呼び、
  LRU entry を evict して page を返却する。

## state store

KV 以外のモデル状態は family 固有の adapter が管理する。

- Qwen3.5 / 3.8: GDN state と DFlash2 drafter ring。
- Qwen4Exp: QSA state、PLE conv state、必要な場合は GDN state。
- 常駐 state は family の shadow pool の `2 × max_concurrent` slots
  （capture rolling と entry 保持用）。
- entry 用 slot が不足すると、LRU entry を evict して slot を回収する。

### eviction

- prefix cache は page quota を持たない。KV pool の page を active request と共有し、
  物理 page が不足したら `PagedKVPool` の reclaimer が
  `PrefixKVStore::reclaim_pages` を呼ぶ。
- `reclaim_pages` は entry を持たない refcount 0 の leaf node を削除し、
  それでも不足すれば LRU entry を evict する。
- entry 用の GDN state slot が不足した場合も LRU entry を evict する。
- save は page を新規 allocate しない。したがって容量不足で save を fail-closed にはしない。

## hit / restore

`lookup(tokens)` は radix walk で page 単位の最長一致 M を求め、
その範囲で path が全 page 一致している entry のうち `position <= M` の最深のものを返す。

`ContinuousBatcher` は request admission 時に input 全体で lookup し、
hit かつ `state_position < input_len` のときだけ restore する。
`state_position >= input_len` は restore せず miss 扱いである
（その位置の初回サンプルの forward が存在しないため）。

restore は entry の KV page を新規 sequence の block table に map し（copy なし）、
family adapter が GDN / QSA / ring などの state を active slot へ復元する。
restore 後の token 数は `done.restored_tokens` に入る。

## save

save は page 揃え位置でのみ発生する。

- prefill: prefill chunk の完了時に、sequence position が page 揃えされ、
  かつ input の範囲内なら `save_input_boundary` が input prefix を保存する。
  `save_input_boundary` は `seq.position == input_len を page 揃えした値` を必須検証し、
  違反は `invalid_argument` である。`min_prefix_tokens`（64）未満は保存しない。
  DFlash2 経路では prefill chunk ごとの `save_input_boundary` を呼ばない。
- decode: 最終 prefill の完了時に capture を開始し、各 step 後に position が
  page 境界を新しく通過するたびに family state を rolling slot へ capture する。
  DFlash2 では次の page 境界までの残りトークン数に draft 数を制限し、境界を
  飛び越えずに capture できる位置で step を止める。
- 出力末尾: EOS / `max_new_tokens` で終了し、capture 位置が最終 page 境界と一致するとき、
  input と生成列から成る position 長の token 列を `save_output_tail` で entry 化する。
  cancel などで境界が一致しない場合は capture を解放して保存しない。

## options

option の一覧と既定値は [../user/compute.md](../user/compute.md) の options を正本とする。
`--prefix-cache` は ON/OFF のみで、KV 容量は指定しない。
`--prefix-cache-concurrency` は同時に保持する capture / GDN state entry 数を決める。

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
- KV page sharing は Tensor Parallel の rank shards に対応する。
- 物理 page が枯渇したときは LRU entry を evict する。

## DFlash2 経路

DFlash2 も通常 decode と同じ `ContinuousBatcher` の `prefix_*` ヘルパを通る。entry には
paged KV と GDN state に加えて drafter の `DFlash2ContextState` ring（
`SpecRingRef` / `SpecRingOut`）を保存する。ring は model-specific state であり、
`Qwen35PrefixState` の ring slot が管理する。generic な `PrefixKVStore` は
entry ごとの opaque payload だけを持ち、ring の意味を知らない。
restore では ring も復元し、`dflash2_spec_prefill` は残り suffix のみを prefill する
（`prefix_tokens` 引数）。ring は entry の state slot に保持する。
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
