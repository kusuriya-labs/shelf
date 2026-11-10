# Qwen3.8-Flash-Next EXL3 prefill の二次項除去

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、`--kv-cache-dtype fp8`。

## 結論

32K prefill の二次項は 2 箇所にあった。どちらも 32K で context に比例して増えていた。

- `qsa_select_batch_kernel` が支配的だった。1 query row につき 1 CTA で全圧縮 block を走査し、
  top-k を保持するたびに `top_k` 全要素の再走査と `__syncthreads` を繰り返していた。
  dot 積も block ごとに同じ pooled key を全 row が読み直し、scalar 演算だけで実行していた。
- fp8 KV + block mask の prefill は `paged_attention_dispatch.hip` の分岐で decode kernel に
  落ちていた。decode kernel は選択済み block ではなく全 page を走査し、token ごとに mask を
  確認していた。

修正後、32K の prefill は 42.3 s / 775 tok/s から 13.93 s / 2,352 tok/s になった。
8K も 2,303 tok/s で、32K までほぼ平坦である。

## 変更

- `qsa_select_batch_kernel`
  - query を warp 単位で分担し、block ごとの dot を warp reduction で求める。
  - pooled key は shared に複製せず global から直接読む。
  - top-k を min-heap から「候補 sort + bitonic merge」へ変更する。64 block ごとの候補を
    まとめ、top-k と候補を bitonic merge して上位 k を残す。heap の依存チェーンと
    barrier を削減する。
  - top-k 配列は動的 shared とし、production の k=512 では 8 KiB に収める。
- fp8 / bf16 masked prefill
  - block mask から CTA 内の query row 群の page bitmap を構築し、選択 page だけを処理する。
  - fp8 prefill に per-token mask を追加し、plan の fp8 masked prefill を decode kernel から
    WMMA prefill kernel へ切り替える。
  - page bitmap が上限を超える場合と mask なしの場合は従来経路に戻る。
- fp8 masked decode
  - block mask から page bitmap を構築し、選択 page だけを処理する。

## 計測

条件: `shelf-bench pp --prompt-tokens 32768 --prefill-chunk 2048 --kv-cache-dtype fp8
--device 2,3 --tensor-parallel 2 --warmup 0 --runs 1`。
per-kernel 時間は `rocprofv3 --kernel-trace` の 1 run。trace は両 rank を含む。

| 状態 | wall | qsa_select (両 rank) | attention (両 rank) |
|---|---:|---:|---:|
| 変更前 | 46.87 s / 699 tok/s | 50.30 s | 18.93 s |
| qsa_select のみ変更 | 25.58 s / 1,281 tok/s | 6.43 s | 19.56 s |
| page bitmap 追加後 | 21.82 s / 1,502 tok/s | 6.55 s | 11.51 s |
| fp8 prefill + top-k merge | 14.84 s / 2,208 tok/s | 4.87 s | 5.23 s |

trace なしの最終値は `SHELF_EXL3_MUL24_PREFILL_AB=B` 指定で
8K 3.56 s / 2,303 tok/s、32K 13.93 s / 2,352 tok/s。

chunk 末尾 32K の 12 layer 合計（1 rank）:

| 状態 | qsa_select | attention |
|---|---:|---:|
| 変更前 | 2.74 s | 0.99 s |
| qsa_select のみ変更 | 0.35 s | 1.00 s |
| page bitmap 追加後 | 0.36 s | 0.37 s |
| fp8 prefill + top-k merge | 0.15 s | 0.11 s |

## 検証

- `test_qsa_select`: PASS。top-k の一致、同点時の小さい index 優先、mask の tail bit を確認。
- `test_paged_attention`: 56 PASS。fp8 masked prefill と fp8 masked decode を含む。
- `test_qsa_attention_mask`: PASS。
- `test_qsa_state_pool` / `test_paged_attention_plan_splits` /
  `test_paged_attention_split_reduce_exact`: PASS。

## 残る課題

- `qsa_select` は依然として全圧縮 block を走査する。32K で 0.15 s/chunk まで下がったが、
  query tile と tensor core を使えばさらに下げられる。現在の主コストは dot である。
- fp8 masked decode（row < 256 の decode、および短い prefill）は query head ごとに
  K/V を読む構成が残る。prefill は WMMA 経路に移したため、残るのは decode 側である。
- BF16 masked prefill は page 単位の skip のみで、page 内は密な WMMA を実行する。
