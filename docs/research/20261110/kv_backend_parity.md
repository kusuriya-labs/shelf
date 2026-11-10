# BF16 / Octave の masked attention parity

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
基準は fp8 KV の masked attention（QSA block mask の prefill / decode）。

## 変更

- bf16 decode: fp8 decode と同じ page bitmap を作り、選択されていない page を skip する。
- octave prefill: 既存の per-token mask に加えて page bitmap を作り、全 row で未選択の
  page を skip する。`plan_paged_attention` の OCTAVE mask fallback を削除し、
  `rows >= 256` では bf16 / fp8 と同じ WMMA prefill を使う。
- octave decode: 32 token chunk に active token が 1 つもなければ、unpack と WMMA を
  skip する。per-token mask は従来どおり。
- octave decode の WMMA 経路を 16 q head まで 1 launch で処理するように一般化した。
  `q_per_kv=12` は 1 launch、q の load は wave ごとに 8 head ずつ進める。partial は
  `q_per_kv` 全体の layout に書く。scalar legacy 経路は 6 head group のまま。
- octave prefill に FlashNext 用の 16 query row × 192 flat row × 24 wave tile を追加した。
  `q_per_kv=12` はこの tile を 1 回 launch する。`q_per_kv > 12` だけ bf16 kMode=2
  fallback を使う。どちらの経路にも per-token mask と page filter が入る。
- kv_append selector に OCTAVE の kv_heads=2 / 1 を追加した。従来は TP2 の
  kv_heads=1 が correctness fallback に落ち、octave 非対応のため
  `ProgramStatus::INVALID_BINDING` になっていた。
- MTP+TP の GDN snapshot 予約を修正した。各 drafter が全 rank の local layout に
  snapshot を 1 面ずつ確保するため、`mtp_batch_rank_state_reserve_bytes` に
  `max_requests` 面を計上する。これがないと octave KV で
  max_seq_len 524288 の arena が数十 MiB 不足する。
- executor の Octave gate から routed-expert と TP の条件を外し、head_dim 256 /
  rotary_dim 64 / PP=1 だけを要求する。
- OCTAVE の paged attention selector に 24/2、12/2、12/1、6/1 を追加した。
- bench / compute / quality の Octave TP=1 制限を外した（PP=1 は維持）。

## 検証

- `test_kv_octave`: masked prefill を bf16 prefill と比較して max_abs=0。
  masked decode は max_abs=0.0108。
- `test_paged_attention`: 56 PASS（bf16 / fp8 の masked prefill / decode を含む）。
- `test_paged_attention_plan_splits`: mask 付きでも `rows >= 256` で prefill を選ぶことを
  bf16 / fp8 / octave で確認。
- QSA / paged attention 系 ctest: 8/8 PASS。
- `test_kv_octave`: dense TP2 (12/2) と FlashNext TP2 (12/1) の prefill / decode、
  masked prefill / decode を追加。prefill は bf16 と max_abs=0、decode は約 0.01。
- `test_qwen35_paged_attention_selector`: 52 PASS。
- `test_mtp_reserve`: snapshot を drafter 数分計上することを確認。
- `shelf-compute --serve-stdio` を run.sh と同じ条件（FlashNext、TP2、octave:c4v4、
  MTP drafts 4、rope-factor 2、max-seq-len 524288、2048 token prompt）で実行し、
  prefill 1.34 s / 1,532 tok/s、MTP 5 round で 15 token emitted を確認した。
- `shelf-server` + `run.sh` で chat completion が通り、MTP 5 round で
  accepted 10 / emitted 15 を確認した。
- dense Qwen 27B EXL3 も TP2、octave:c4v4、max-seq-len 524288 の serve で動作した。

## 参考性能

FlashNext 32K prefill、`SHELF_EXL3_MUL24_PREFILL_AB=B`:

| model / KV dtype | 8K | 32K |
|---|---:|---:|
| FlashNext fp8 | 3.46 s / 2,368 tok/s | 13.35 s / 2,454 tok/s |
| FlashNext bf16 | 3.56 s / 2,303 tok/s | 14.08 s / 2,328 tok/s |
| FlashNext octave:c4v4 TP2 | - | 12.24 s / 2,676 tok/s |
| FlashNext octave:b4n3v3 TP2 | 3.50 s / 2,338 tok/s | 12.29 s / 2,666 tok/s |
| FlashNext octave:b4n3v4 TP2 | - | 12.39 s / 2,644 tok/s |
| dense 27B EXL3 octave:b4n3v3 TP2 | 3.80 s / 2,156 tok/s | 15.92 s / 2,059 tok/s |

FlashNext の octave は TP2 で実行でき、masked prefill は q_per_kv=12 の専用 tile、
decode は 16 head の 1 launch になる。32K では fp8 より速くなった。dense Qwen 27B も
TP2 で実行できる。
