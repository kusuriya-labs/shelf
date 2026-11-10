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

## 検証

- `test_kv_octave`: masked prefill を bf16 prefill と比較して max_abs=0。
  masked decode は max_abs=0.0108。
- `test_paged_attention`: 56 PASS（bf16 / fp8 の masked prefill / decode を含む）。
- `test_paged_attention_plan_splits`: mask 付きでも `rows >= 256` で prefill を選ぶことを
  bf16 / fp8 / octave で確認。
- QSA / paged attention 系 ctest: 8/8 PASS。

## 参考性能

FlashNext 32K prefill、`SHELF_EXL3_MUL24_PREFILL_AB=B`:

| KV dtype | prefill | tok/s |
|---|---:|---:|
| fp8 | 13.35 s | 2,454 |
| bf16 | 14.08 s | 2,328 |

bf16 は KV が 2 倍のため fp8 より遅いが、masked prefill / decode の機能は同じ経路を使う。
octave は routed-expert モデルでは executor が拒否するため、FlashNext では比較できない。
