# prefix cache: radix で 0 から再 Prefill しないための checkpoint 設計

2026-11-16。27B FP8（`models/Qwen3.8-27B-FP8`）と DFlash2
（`models/Qwen3.8-27B-DFlash2-FP8`）で実施した。

## 1. 問題

`--prefix-cache-strategy slot` は安定して hit するが、既定の `radix` は
「保存されているのに次の request が 0 から prefill する」ことが多い。
[20261111/prefix_cache_flashnext_and_agent_entries.md](../20261111/prefix_cache_flashnext_and_agent_entries.md)
の §2.3 で未実装としていた項目のうち、
1（entry 用 state slot の分離）、2（help の明確化）、
3（共有 prefix 境界の checkpoint 化）を実装した。

### 1.1 直接原因（コードから確定）

1. `prefix_save_position` は slot 経路でしか設定していなかった。
   server 既定は `--max-num-batched-tokens 2048` なので、2048 token 以下の prompt は
   1 chunk で処理され、chunk 末端が偶然 page 揃え（1/16 の確率）のときしか
   entry が作られない。非 DFlash2 経路には救済も無かった。
2. entry の state は VRAM の hot slot（`--prefix-cache-checkpoints` 個）に 1 個ずつ
   対応し、枯渇すると entry ごと LRU evict された。長い prompt の prefill は
   chunk ごとに entry を作るため、浅い境界（共有 system prompt 付近）から消えた。
3. chunk 末端にしか checkpoint が無いので、共有 prefix の境界（例: 700 token）が
   chunk 境界と一致しない場合は、その位置の state が存在せず hit できなかった。

## 2. 変更

### 2.1 入力末尾の checkpoint を必ず作る

radix でも `prefix_save_position = align_down(input_len - 1, page_tokens)` を設定し、
scheduler（`token_budget_scheduler`）と `dflash2_spec_prefill` の
`checkpoint_position` に prefill をその位置で区切らせる。
1 chunk に収まる prompt でも tail が保存される。
DFlash2 用の「先に page 境界まで別 prefill して巻き戻す」経路（shadow prefill）は
不要になったため削除した。

### 2.2 checkpoint を pinned host memory へ

- checkpoint slot は `hipHostMalloc` の連続 buffer（`--prefix-cache-checkpoints` 個）に置き、
  GDN conv / recurrent（全 rank）、QSA pooled / partial / counts、PLE conv、
  DFlash2 ring を 1 slot にまとめる。
- VRAM に残るのは rolling capture 用の shadow state だけで、slot 数は
  `--prefix-cache-checkpoints`。
- save は active → checkpoint の D2H、restore は H2D。entry 数が VRAM を消費しない。

### 2.3 checkpoint の位置

- tail: `align_down(input_len - 1, page)`（同一 prompt の再実行用）。
- input end: position が入力長と一致し page 揃え（延長 prompt 用）。
- periodic: `--prefix-cache-save-interval`（既定 1024）の倍数の chunk 末端。
- junction: lookup の最長一致 `match_tokens` が `restored_tokens < match < input_len`
  のとき、その位置。共有 prefix が分岐した境界を checkpoint にする。
  一致 prefix が分岐しただけの request は何も得ないが、次の request が
  その境界から restore できる。

### 2.4 lookup の上限

page 揃えされた input end の entry は同一 prompt では使えない（残り token が 1 個も無い）。
`lookup(tokens, max_position = input_len - 1)` として、walk をその位置で打ち切り、
より浅い entry にフォールバックする。

### 2.5 capture slot と checkpoint 数の分離（同日追記）

当初は `--prefix-cache-concurrency` が VRAM の capture slot 数と host の
checkpoint 数の両方を決めていた。capture は「同時に decode 中の request が
page 境界の state を保留する」ための作業領域なので必要数は
`--max-concurrent-requests` で決まり、ユーザーが選ぶ値ではない。

そこで option を `--prefix-cache-checkpoints`（host の checkpoint 数）へ改名し、
capture slot は `--max-concurrent-requests` から導出、slot strategy では 0 にした。
既定値（checkpoints 4 / requests 4）では確保量は変わらない。

TP2 / 27B FP8 の起動 log（実測）:

| requests | checkpoints | VRAM capture（rank あたり） | host checkpoint（全 rank 合計） |
| ---: | ---: | ---: | ---: |
| 4 | 4 | 0.15 GiB | 0.29 GiB |
| 4 | 16 | 0.15 GiB | 1.17 GiB |
| 1 | 4 | 0.04 GiB | 0.29 GiB |
| 1 | 16 | 0.04 GiB | 1.17 GiB |

slot strategy（requests 4）は VRAM 0.00 GiB、host 0.37 GiB（slot 5 個）だった。

### 2.6 pin

lookup で hit した回数が最も多い entry を pin し、LRU の victim にしない
（同数なら position が小さい方）。capacity が 2 以下では pin しない。

## 3. 実測

### 3.1 共有 prefix（非 DFlash2、27B FP8、max_seq 8192、chunk 2048）

`shared` 700 token を共通に持つ A（4000）/ B（900）/ C（1200）。

| request | 変更前 `restored_tokens` | 変更後 `restored_tokens` |
| --- | ---: | ---: |
| A 初回 | 0 | 0 |
| B 初回（A と 688 token 共有） | 0 | 0 |
| C 初回（同じ 688 token 共有） | 0 | 688 |

保存 log（変更後）: A が 2048 / 3984 / 4000、B が 688（junction）/ 896、C が 1184 / 1200。

### 3.2 1 chunk に収まる会話（非 DFlash2、700 token + 延長）

| request | 変更前 | 変更後 |
| --- | ---: | ---: |
| 初回 700 | 0 | 0 |
| 同一 700 の再実行 | 0 | 688 |
| 750 への延長 | 0 | 688 |

生成列は `--prefix-cache 0` の同時実行と完全一致した。

### 3.3 DFlash2（`tests/server/test_compute_dflash2.py`、99 checks）

`--max-num-batched-tokens 1024`（このホストの VRAM に合わせる）。
checkpoint 数 2 と 4 の両方で 99/99 PASS。
入力 80 token の extension が `restored_tokens=80`、同 prompt の再実行が 64 に
フォールバックすることを含む。

### 3.4 DFlash2 長い共有 prefix（`test_compute_dflash2_prefix_long.py`）

2 回目が 4096 token から restore し、生成列が cache 無効の DFlash2 と
非投機 greedy の両方に一致（8/8 PASS）。

### 3.5 TP2（`--tensor-parallel 2 --device 0,1`、27B FP8）

同一 harness で 6 request。A/B/C は `shared` 700 token を共通に持つ
A（4000）/ B（900）/ C（1200）。

| request | `restored_tokens` |
| --- | ---: |
| 700 token 初回 | 0 |
| 同一 700 token の再実行 | 688 |
| 750 token への延長 | 688 |
| 4000 token 初回 | 0 |
| 900 token 初回（688 token 共有） | 0 |
| 1200 token 初回（同じ 688 token 共有） | 688 |

3 種の生成列は `--prefix-cache 0` の TP2 実行と完全一致した。

DFlash2 + TP2（`test_compute_dflash2_prefix_long.py` と同じ 4300 / 4180 token）は
2 回目が `restored_tokens=4096` で、生成列は cache 無効の TP2 DFlash2 と完全一致した。

起動 log は rank あたり `MEMORY_PLAN tp prefix cache: vram 0.15 GiB host 0.15 GiB`、
実際の host checkpoint は全 rank 合計で 0.29 GiB（TP1 と同じ）である。

### 3.6 4 並列（27B FP8、`--max-concurrent-requests 4`、chunk 2048、interval 1024）

共通 root 700 token + 各会話に固有の本文を持つ 4 会話を 2 ラウンド流し、
2 ラウンド目の `restored_tokens` と `entries` を見た。

本文 3300（全長 4000）:

| checkpoints | round 1 後の entries | evictions | round 2 の restored |
| ---: | ---: | ---: | --- |
| 4 | 3 | 9 | 全 4 本 688 |
| 8 | 7 | 3 | 全 4 本 688 |
| 16 | 10 | 0 | 全 4 本 **3984** |

本文 7300（全長 8000）:

| checkpoints | round 1 後の entries | evictions | round 2 の restored |
| ---: | ---: | ---: | --- |
| 16 | 12 | 0 | 全 4 本 **7984** |
| 32 | 12 | 0 | 全 4 本 7984 |

16 未満では、共有 root（688）以外の checkpoint が 1 ラウンドの prefill 中に
LRU で落ち、各 request が本文 3300 / 7300 token を毎回 prefill し直す。
4 並列・全長 4〜8k では 1 ラウンドの save 数が 14〜16 で、それが閾値だった。

### 3.7 unit

`tests/unit/test_prefix_cache.hip` 320 checks PASS（pin、QSA/PLE の host roundtrip を追加）。
`tests/unit/test_prefix_cache_batcher.hip` は `max_batch_rows` を prompt 長より大きくして
「1 chunk でも tail を保存する」ことを固定する（この変更を戻すと 6 checks 落ちる）。

## 4. 採否

ACCEPT。radix の hit 率を決めていたのは state の保持量ではなく
「保存位置の作り方」であり、tail / input end / junction を明示的に作ることで
slot 方式に依存せず multi-turn と共有 prefix の両方が hit する。

REJECT: server から共有 prefix 境界を送る protocol 変更。junction を
compute 側の lookup 結果から作れるため不要。

未検証: Qwen4Exp（Flash Next）の host checkpoint。QSA/PLE の host copy は
unit test で stride を固定したが、実モデルの end-to-end は
`Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw` の resident 量がこのホストの 1 枚に収まらず未実施。
