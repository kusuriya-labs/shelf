# Prefix Cache

prefix cache は `shelf-compute` の runtime が所有する GPU-resident cache である。
radix tree 上の page 単位で、prefix の paged KV は active KV pool の page を共有して
保持し、running state（GDN / QSA / PLE / DFlash2 ring）は checkpoint として
pinned host memory に保持する。`ContinuousBatcher` が request の
admission / prefill / decode / 出力末尾で lookup・restore・save を行う。
共有 prefix への extension と multi-turn の会話で prefill を省略する。

## strategy

`--prefix-cache-strategy` が entry の管理方式を選ぶ。KV の保存形式（`PrefixKVStore` の
radix tree / page 共有）と family state の保存・復元は共通である。

- `radix`（既定）: prefix ごとに entry を作り、entry 単位で LRU する。
  save 点は 3 種類である（[save](#save)）。
- `slot`: request ごとに固定 slot を予約し、**1 slot に付き entry 1 個**だけを保持する。
  保存位置は入力末尾の直前の page 境界
  `align_down(input_len - 1, --page-tokens)` の 1 点だけである。scheduler は prefill を
  この位置で区切るため、state はその位置で確定する。出力末尾は保存しない（rolling
  capture も使わない）。slot 数は `--max-concurrent-requests`、checkpoint slot は
  `slot + 1`（更新中の旧 entry を保持するため）を確保する。
  eviction は slot 単位の LRU で、更新は「新しい entry を保存してから旧 entry を解放」の
  順に行い、保存に失敗した場合は旧 entry を残す。

slot は、エージェントが同じ会話を延長し続ける workload（自分の prefix を何度も再利用する）
に向く。radix は、複数 request が先頭の system prompt だけを共有する workload に向く。

## Radix tree

- node は KV page（`--page-tokens` の既定 16 token）単位の token edge を保持する。
  入力末尾の端数 page は tree に保持しない。
- edge は `edge_tokens` / `edge_pages` の path compression である。
  insert で既存 edge の中間まで一致した場合は node split して分岐点を作る。
- page は entry 単位の refcount で管理する。restore 中の page は refcount が
  増えるため eviction の対象外になる。
- entry は radix node ごとに 1 つずつ対応し、page 揃え後の position と
  checkpoint slot を持つ。

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

## checkpoint store

KV 以外の running state は family 固有の adapter（`Qwen35PrefixState`）が管理する。

- 対象は GDN state と DFlash2 drafter ring（Qwen3.5 / 3.8）、および GDN state、
  QSA key state、PLE conv state（Qwen4Exp）。PLE の n-gram history は state では
  なく token 列から作るため、restore 時に `Executor::external_prefix_restore` が
  復元 prefix の token を family adapter へ渡し、history を再 seed する（TP では
  `TpBatchHook::on_prefix_restored` が全 rank へ配る）。
- checkpoint slot は pinned host memory に確保する（`hipHostMalloc`）。
  slot 数は `--prefix-cache-concurrency`、1 slot は全 rank 分の
  GDN conv history / recurrent state + QSA state + PLE conv state + ring である。
  TP でも各 rank の shard をまとめて 1 slot に入る。
- save は active slot から checkpoint slot への D2H copy、restore は H2D copy である。
- VRAM に残るのは rolling capture 用の shadow state だけである。slot 数は
  `--prefix-cache-concurrency` で、decode 中に page 境界を通過した時点の state を
  D2D copy で保持する。出力末尾の save はこの shadow から checkpoint slot へ copy する。
- entry 用 checkpoint slot が不足すると、LRU entry を evict して slot を回収する。

arena の予約に入るのは capture slot 分だけであり、checkpoint 本体は host memory を
使う。したがって `--prefix-cache-concurrency` を増やしても VRAM は
capture slot 分しか増えない。

## eviction

- prefix cache は page quota を持たない。KV pool の page を active request と共有し、
  物理 page が不足したら `PagedKVPool` の reclaimer が
  `PrefixKVStore::reclaim_pages` を呼ぶ。
- `reclaim_pages` は entry を持たない refcount 0 の leaf node を削除し、
  それでも不足すれば LRU entry を evict する。
- checkpoint slot が不足した場合も LRU entry を evict する。
- save は page を新規 allocate しない。したがって容量不足で save を fail-closed にはしない。

### pin

entry は lookup で hit した回数を数え、最も hit された entry を pin する
（同数なら position が小さい方）。pin した entry は LRU の victim にしない。
`--prefix-cache-concurrency` が 2 以下のときは pin しない（pin が slot を
1 個占有すると entry が 1 個しか残らないため）。pin 以外が無い場合は
pin した entry も evict する。

## hit / restore

`lookup(tokens, max_position)` は radix walk で page 単位の最長一致 M を求め、
`position <= max_position` を満たす最深の entry を返す。walk は
`position > max_position` に達した時点で打ち切る。

`ContinuousBatcher` は request admission 時に
`max_position = input_len - 1` で lookup する。この 1 token の余白は、復元位置の
最初のサンプル forward を残すためである。したがって入力長に一致する entry は
それより長い入力の extension では使われるが、同一入力の再実行では 1 page 手前の
entry にフォールバックする。

入力境界の entry は、次の request の input が同じ token 列で延長される限り一致する。
出力末尾の entry は、client が返した assistant メッセージを再レンダリングした token 列が
生成時の raw token 列と一致する場合だけ一致する。reasoning の trim、answer の lstrip、
強制 close の挿入などでテキストが正規化されると、出力末尾の entry は使われず、
入力境界の entry まで戻る。

restore は entry の KV page を新規 sequence の block table に map し（copy なし）、
family adapter が GDN / QSA / ring などの state を active slot へ復元する。
restore 後の token 数は `done.restored_tokens` に入る。

## save

save は page 揃え位置でのみ発生する。radix では次の 3 種類である。

- 入力末尾の直前の page 境界（tail）: `align_down(input_len - 1, page_tokens)`。
  scheduler は prefill をこの位置で区切り、`save_input_boundary` が input prefix を
  保存する。同一入力の再実行を restore できるようにするための点である。
  slot ではこの点だけを保存する。
- 入力末尾（input end）: sequence position が入力長と一致し、page 揃えである場合。
  同じ prompt を延長する request の restore 点になる。
- 定期 checkpoint: sequence position が `--prefix-cache-save-interval`（既定 1024）の
  倍数であり、page 揃えである場合。分岐位置が入力末尾から遠い会話の再読みを短くする。
  この点は prefill chunk 末端でしか作らないため、間隔より粗くなることはあっても
  細かくはならない。`0` を指定すると入力末尾系の 2 種類だけになる。

decode では最終 prefill の完了時に capture を開始し、各 step 後に position が
page 境界を新しく通過するたびに family state を capture slot へ copy する。
出力末尾は EOS / `max_new_tokens` で終了し、capture 位置が最終 page 境界と一致するとき、
`sequence.position` が commit 済みの token 列を `save_output_tail` で entry 化する。
cancel などで境界が一致しない場合は capture を解放して保存しない。

save の失敗で entry を作らない場合も、既存 entry は残す。`min_prefix_tokens`（64）未満は
保存しない。

## options

option の一覧と既定値は [../user/compute.md](../user/compute.md) の options を正本とする。
`--prefix-cache` は ON/OFF のみで、KV 容量は指定しない。
`--prefix-cache-concurrency` は同時に保持する capture / checkpoint 数を決める。
`--prefix-cache-save-interval` は定期 checkpoint の間隔を決める。

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

`stats` の応答には `prefix` object が入る。`enabled`、`reuse_slots`（slot strategy の
slot 数、radix では 0）、`entries`、`saves`、`hits`、`misses`、`restores`、`evictions`。

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

- KV page sharing は Tensor Parallel の rank shards に対応する。
- 物理 page が枯渇したときは LRU entry を evict する。
- checkpoint は pinned host memory を `--prefix-cache-concurrency × 1 slot` 分だけ
  起動時に確保する。slot 1 個は GDN 48 layer の 27B で約 75 MiB である。

## DFlash2 経路

DFlash2 も通常 decode と同じ `ContinuousBatcher` の `prefix_*` ヘルパを通る。entry には
paged KV と GDN state に加えて drafter の `DFlash2ContextState` ring（
`SpecRingRef` / `SpecRingOut`）を保存する。ring は model-specific state であり、
checkpoint slot の末尾に置く。generic な `PrefixKVStore` は
entry ごとの opaque payload だけを持ち、ring の意味を知らない。
restore では ring も復元し、`dflash2_spec_prefill` は残り suffix のみを prefill する
（`prefix_tokens` 引数）。ring を保存できないときは suffix prefill が長くなるだけで、
挙動は正しい。

restore / save の入力は次のとおりである。

- admission の restore は request の `DFlash2ContextState` を先に作成し、
  `SpecRingOut` に ring buffer を渡す。restore 後の `length` / `next_position` が
  ring の状態になる。
- 出力末尾の save（`prefix_finish_capture`）は request の ring を `SpecRingRef` として
  渡す。ring は commit-only であり `sequence.position` と常に一致する。
- output tail の entry は page 揃えした position の prefix である。ring の
  `length` / `next_position` は **entry position へ clamp して** 保存する
  （final position のまま保存すると entry の KV 範囲と ring の範囲がずれる）。
- DFlash2 の prefill は `--max-num-batched-tokens` の chunk で進み、
  `prefix_save_position`（tail）でもう 1 回区切る。chunk 末端の callback が
  `save_input_boundary` を呼ぶため、prompt が 1 chunk に収まる場合も tail が保存される。
- 入力境界の entry は assistant の再レンダリングに依存しないため、強制 close や
  trim で出力末尾の token 列が変わる場合でも restore できる。
- DFlash2 request は **ring を持つ entry だけ** を restore する。`lookup` は entry の
  ring 情報（`PrefixHit::ring_length` / `ring_next_position`）を返し、
  `ring_covers_state()`（`ring_length != 0` かつ `ring_next_position == state_position`）を
  満たさない entry は restore せず full prefill する。ring は DFlash2 request の
  save だけが作るため、通常 decode が作った entry は DFlash2 では restore されない。

### spec ring pool

drafter ring の保存先は `PrefixCacheConfig::spec_ring_bytes` で指定する。

- 1 checkpoint slot の ring 領域は `spec_ring_bytes`（K と V の合計）である。
  compute は DFlash2 有効時に `DFlash2Config::context_ring_bytes()`
  （`2 × num_hidden_layers × sliding_window × attention_kv_rows() × 2` bytes。
  27B では 40 MiB）を渡す。
- ring 領域は checkpoint slot の host memory に含まれる。ring を保存できなかった
  entry も KV page と family state は保存する。その entry は DFlash2 request からは
  restore されない（`ring_covers_state()` を満たさないため full prefill に戻る）が、
  非 DFlash2 request からは restore できる。ring の失敗で entry 自体を捨てない。
- `spec_ring_bytes == 0`（既定）のときは ring 領域を確保しない。ring を渡した save は
  `invalid_argument` で失敗し、ring を持たない entry になるため DFlash2 の restore は
  起きない。compute は DFlash2 有効時に必ず設定する。

## 検証

`tests/unit/test_prefix_cache.hip` が radix tree、page 共有、checkpoint の save / restore、
eviction、pin、slot、ring を GPU 上で固定する。
`tests/unit/test_prefix_cache_batcher.hip` が 1 chunk に収まる prompt でも tail が
保存され、次の request が restore することを固定する。
`tests/server/test_compute_prefix_contract.py` が JSONL contract を固定する。
