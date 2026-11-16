# Qwen35Qsa family の実装と性能判断

実験 ID `QWEN35QSA-1`。対象は `models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`
（BF16 版は `models/Qwen3.8-27B-QSA-BF16`）、R9700 1 枚、`shelf-compute`。

## 1. 目的

Qwen3.8-27B-QSA は QSA（block-sparse attention、budget 2048 / block 4 / indexer 4 head ×
128 dim / rope 64）を宣言する。それまでの shelf は root `model_type` を
`text_config.model_type`（`qwen3_5_text`）経由で受理し、**indexer を無視して dense として
実行**していた。2048 token 以下では参照実装も全 block を残すため結果は一致するが、
2048 超では dense と別関数になる。

本実験は QSA を runtime に載せ、性能と正しさを測る。

## 2. 構成の決定

- family は `qwen35qsa` として分離した。`qwen35` の config / layer weights /
  block lowering に QSA の語彙を入れない。
- 共有 block lowering を `input_norm` と `attention + residual` に分割し、
  family が indexer をその間に挟む（案 B）。`emit_hybrid_block` の呼び出し側
  （qwen35 / qwen35moe）は無変更。
- indexer の重みは backbone の weight struct と並ぶ別配列に置き、backbone の loader は
  `qwen35` のものをそのまま使う。
- indexer の `q_norm` は全 query head で共有される 1 本（`[128]`）。runtime の
  RMSNorm は parameter 要素数 == `group_size` を per-group weight として扱うため、
  複製せずそのまま渡せる。
- `qwen_hybrid` の RMSNorm block は `weight_per_feature` を parameter 要素数から決める
  （`physical_launch.hip`）。この既存契約で足りた。

## 3. 正しさ（実測）

5 token prompt（`The capital of France is`）で greedy 16 token。

```text
dense（QSA を外した同一重み）: 11751,13,198,760,6511,314,9564,369,19241,13,198,760,6511,314,14898,369
QSA（max_seq_len 512、budget 以内で dense fallback）: 同上（完全一致）
QSA（max_seq_len 8192、600 token prompt）: 220,149956,96232,14876,1510,29456,1954,14
```

16k prompt では QSA と dense の 4 token が一致（`2082,1501,24669,60690`）。
3k prompt では 5 token 目から分岐する（選択が効き始めるため。想定内）。

## 4. 性能（実測、同一重み・同一 prompt）

dense 参照は同じ checkpoint の config から `qsa` を除いたもの（hardlink + config のみ）。

| context | dense prefill | QSA prefill | dense decode | QSA decode |
| --- | ---: | ---: | ---: | ---: |
| 3k | 1,885 ms | 3,897 ms | 25.82 tok/s | 14.48 tok/s |
| 16k（保持 block 12.5 %） | 9,996 ms | 19,855 ms | 22.62 tok/s | 13.67 tok/s |

- prefill は context によらず約 2 倍遅い。保持 block を 12.5 % まで削っても逆転しない。
- decode は 0.56〜0.61 倍。selection の launch 数（16 層 × 数 launch）は無視できる大きさで、
  差は masked attention 経路のキー単価による。
- budget 以内（`--max-seq-len 512` など）は §5 の fallback により dense と同等
  （27.0 tok/s を確認）。

**判断: 現状の masked attention では QSA は速度改善にならない（REJECT）。**
契約・state・正しさは成立しており、改善はカーネル側の課題として分離する。

## 5. 採用した緩和

- **budget 以内の静的 fallback（ACCEPT）**: `max_context_tokens <= budget` のとき
  indexer を emit せず state pool も確保しない。参照実装でも全 block が残る領域なので
  意味は同じで、コストはゼロになる。
- 2048 超でも実際の列が短い場合（server の長い `--max-seq-len` で短い要求）は
  masked 経路のコストを払う。per-request に「選択が効かないなら mask を消費しない」
  判定を attention dispatch に入れる案は未実装（次段）。

## 6. 見つけた不具合

- `try_launch_qsa_select` の tiled 経路は「staged request が全 row を覆う」前提だった。
  padded 行には slice が無く、stale position で未 score の block を選び
  `qsa_topk_kernel` が mask 外へ書いてメモリフォルトした（QSA 有効 + decode で再現）。
  staged request が全 row を覆うときだけ tiled 経路を使い、それ以外は fused 経路へ
  フォールバックするよう修正した。
- identity gate が `text_config.model_type` でも通していたため、新 wrapper（QSA）が
  内側の `qwen3_5_text` だけで受理されていた。root `model_type` のみを判定に使うよう変更
  （[architecture.md](../../developer/architecture.md)）。

## 7. 残課題

1. **mask を消費しない per-request 判定**: `ceil(seq_len / block_size) <= top_k` の
   リクエストは dense 経路で走らせる（短い要求で 2 倍遅い現状を解消）。
2. **masked attention の単価**: 16k で保持 12.5 % でも dense より遅い。選択 KV を
   連続バッファへ gather して既存の dense attention を回す案と、masked kernel を
   WMMA 経路へ載せる案を比較する。
3. speculative decode（MTP drafter）との併用: draft 棄却時に QSA state
   （pooled block / partial count）を巻き戻す経路が要る。
