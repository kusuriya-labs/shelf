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
- octave decode の q head を 6 head 単位の group に一般化した。`q_per_kv=12` は
  2 回の launch で 6 head ずつ処理し、partial は `q_per_kv` 全体の layout に書く。
- octave prefill は `q_per_kv > 6` のとき既存の bf16 kMode=2 fallback を使う。
  この経路にも per-token mask と page filter が入る。
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

## 参考性能

FlashNext 32K prefill、`SHELF_EXL3_MUL24_PREFILL_AB=B`:

| model / KV dtype | 8K | 32K |
|---|---:|---:|
| FlashNext fp8 | 3.46 s / 2,368 tok/s | 13.35 s / 2,454 tok/s |
| FlashNext bf16 | 3.56 s / 2,303 tok/s | 14.08 s / 2,328 tok/s |
| FlashNext octave:b4n3v3 TP2 | 3.50 s / 2,338 tok/s | 13.27 s / 2,469 tok/s |
| dense 27B EXL3 octave:b4n3v3 TP2 | 3.80 s / 2,156 tok/s | 15.92 s / 2,059 tok/s |

FlashNext の octave は TP2 で実行でき、masked prefill は q_per_kv=12 の bf16
kMode=2 fallback、decode は 6 head group の 2 launch になる。32K では fp8 と同等の
prefill 速度になった。dense Qwen 27B も TP2 で実行できる。
