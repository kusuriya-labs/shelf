# EAGLE-4 Phase E1 Gate（TP2 prefix caching・通常経路）

- ID: EAGLE4-PHASE-E1-GATE
- 実施日: 2026-10-23
- 環境: TP2（device 2/3、model-host 2台、Qwen3.8-27B-FP8 + EAGLE4）
- ドライバ: `tools/dgate.py cache`（4プロンプト×3モードの parity）、
  `tools/e1_cache_hit.py`（同一プロセス内の hit/restore 検証）

## 判定

| 項目 | 結果 | 根拠 |
| --- | --- | --- |
| hit/miss 交互で single と完全一致 | **ACCEPT** | eagle4 / eagle4_tree / dflash2 の3モード × 4プロンプト、`--prefix-cache 1`・concurrent=4 で single と12/12一致、error なし |
| restore が実際に発生 | **ACCEPT** | 同一 serve プロセス内で `restored_tokens=128`（prefill は残り51トークンのみ）、生成列は single 参照と完全一致 |
| `--prefix-cache 0` の回帰なし | **ACCEPT** | required acceptance 全通過（cache off は従来どおり） |
| hit 時の p50 効果 | **ACCEPT** | `tools/e1_cache_hit.py --battery`（5-run、restore あり/なしを同一プロセスで比較、ms/tok 換算の p50）: restore **11.2 ms/tok** / miss **28.3 ms/tok** = **2.52×**。restore は全 round で restored=128（10.5-11.3 ms/tok） |

## 実装で確定した制約

- TP rank shard 付きの hot overflow（cold migration）と state demotion は
  **unsupported エラー**（`--prefix-cache-hot-tokens` の増設で回避。
  明示エラーで破損はしない）。現行デフォルト容量ではゲートの作業集合は
  overflow しない（保存/復元とも PFXSAVE_OK で確認）。
- ページ id は全 rank ロックステップで確保され、
  `fill_pages` 後の `num_used_pages` 一致検一致検査で発散を検出する。

## 修正した結合バグ

- main.hip の eagle4 ブロックが `config.prefix_cache = false` を
  硬コードしており、`--prefix-cache 1` でも serve の cache が無効化されていた
  （D3 時の CLI guard 解除の取りこぼし）。解除して E1 が成立。

## E2 検証（spec ring・TP2）

設計どおり ring は draft context（rank0 ローカル）依存で追加分不要。
`tools/e1_cache_hit.py` を E1_MODE=eagle4 / dflash2 の両方で実施し、
- ring の保存/復元: `ring_len=128 ring_next=128` が hit 判定を通過、
  `restored_tokens=128`・`prefill_tokens=51` を両モードで確認
- 後の spec ラウンドを含む生成列が single 参照と完全一致
  （`11,321,60445,...,248046`）
を確認。**E2 ACCEPT**。
