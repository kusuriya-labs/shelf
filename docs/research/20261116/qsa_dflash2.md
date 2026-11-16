# QSA target + DFlash2 の併用可否

実験 ID `QWEN35QSA-3`。対象は `models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`、
drafter は `models/Qwen3.8-27B-DFlash2-FP8`、R9700 1 枚。

## 1. 結論

**起動はできるが、正しくない（REJECT）。** DFlash2 の reject 時 rollback は
GDN conv / recurrent state だけで、QSA の key state（pooled block と partial block）を
snapshot / restore しない。reject された draft の token が QSA state に残り、
2048 token を超える context では block 選択が汚染される。

速度自体は出る。3k context で 47.24 tok/s（spec なしは 25.54 tok/s、1.85 倍）。

## 2. 根拠

`dflash2_spec_decoder.cpp` は verify の前に `spec_gdn_snapshot` で GDN state を保存し、
reject 時に復元する。QSA state に対応する処理は無く、`restore_qsa_spec_state` は
`mtp_drafter.cpp`（MTP drafter 経路）からしか呼ばれない。

QSA key cache は verify の全 row に対して実行され、完了した block を pooled に積み、
partial block の count を進める。reject で KV cache を巻き戻しても、この state は
巻き戻らない。

### 計測による確認

`SHELF_QSA_DUMP` の一時実装で、生成後の QSA `pool_count` / `partial_count` を読んだ。
不変条件は「4 × pool + partial = prompt token + 生成 token − 1」である
（最後に sample した token はまだ append されない）。

| prompt | 実行 | pool | partial | 合計 | 期待値 |
| --- | --- | ---: | ---: | ---: | ---: |
| 3072 | spec なし | 771 | 3 | 3,087 | 3,087 |
| 3072 | DFlash2 | 774 | 2 | 3,098 | 3,087 |
| 3071 | DFlash2 | 775 | 1 | 3,101 | 3,086 |

DFlash2 では reject された draft row の分だけ state が先に進む（3072 で +11 token、
3071 で +15 token）。pooled が期待より 3〜4 block 多いことは、reject された token の
indexer key が pooled block に混入したことを意味する。

### 表面化しない条件

- `max_context_tokens <= budget`（QSA 無効）: state 自体が無いので問題ない。
  2000 token で QSA 有効・無効の spec 出力は一致した（57.2 tok/s）。
- QSA 有効でも、リクエストが 2048 token 以下で全 block が選択される間は、
  pooled key が汚れても選択結果は変わらないため出力は保たれる。
  汚染が効くのは context が budget を超え、選択が実際に枝刈りする領域に入ってから。

## 3. 必要な対応

MTP drafter と同じ形で、DFlash2 のラウンドに QSA state の snapshot / restore を
追加する（`mtp_drafter.cpp` の `allocate_qsa_history` と `restore_qsa_spec_state`）。
それまでの間は、QSA family では DFlash2 を明示的に拒否する gate を置くのが安全である
（MTP の `LoadedModel::has_mtp_block()` が QSA family で false を返すのと同じ扱い）。

## 4. 未検証

- QSA state の snapshot / restore を DFlash2 に配線した場合の速度と正しさ。
- tree verify（`--dflash2-tree-extra`）を使った場合。今回の計測は chain verify。
