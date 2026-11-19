# workspace allocator の断片化と fused producer の lifetime

実験 ID: `WS-FRAG-1`

`SHELF_WS_DEBUG` で workspace の内訳を取ったところ、8192 row bucket の
`models/Qwen3.8-Flash-Next-Quark-MXFP4`（host expert 448、bf16 KV、1 カード）で
**3.63GB を確保しているのに、同時に生きているのは 0.94GB** だった。
原因は workspace allocator の断片化と、fused producer の書き込み開始が
planner の lifetime より早いことだった。

## 1. 測定

同一条件（`SHELF_MOE_HOST_EXPERTS=448`、`--device 1`、bf16 KV、8192 token）で、
workspace planner の `bucket=8192` は次のとおりだった。

| 項目 | 値 |
| --- | ---: |
| `workspace.total_bytes` | 3.479 GB |
| 同時に live な bytes の最大（peak） | 0.943 GB |
| 終了時に live な bytes | 0.343 GB |
| 新規に cursor を伸ばした回数 | 15 |

15 回の growth の内訳:

| サイズ | 回数 | 実体 | 合計 |
| ---: | ---: | --- | ---: |
| 458.75 MB | 3 | MoE down 出力（89,600 × 2,560 bf16） | 1,376 MB |
| 335.54 MB | 2 | GDN conv 出力（8,192 × 10,240 f32） | 671 MB |
| 236.54 MB | 2 | MoE gate_up 入力の活性 workspace | 473 MB |
| 201.33 MB | 1 | attention 出力（8,192 × 6,144 f32） | 201 MB |
| 167.77 MB | 4 | HC / dense stream（8,192 × 10,240 bf16） | 671 MB |
| その他 | 3 | | 86 MB |

growth のたびに free list を見ると、**合計では十分な空きがあるのに、必要なサイズの
連続 block が無い**状態だった。

| growth | 要求 | その時の free |
| --- | ---: | --- |
| 2 回目の MoE down 出力 | 458.75 MB | 37 block / 2,289 MB、最大 236.5 MB |
| 3 回目の MoE down 出力 | 458.75 MB | 40 block / 2,748 MB、最大 335.5 MB |
| 2 回目の GDN conv | 335.54 MB | 29 block / 1,549 MB、最大 291.0 MB |
| 2 回目の gate_up ws | 236.54 MB | 37 block / 2,113 MB、最大 201.3 MB |

元の allocator は「要求以上で最小の free block」を選び、余りを free list に戻すが、
隣接する free block を結合しない。458.75MB の block が 236MB の要求に切り取られ、
残りは次の 458.75MB 要求に使えない。層ごとに同じサイズが再出現するため、
この穴が 48 層分蓄積して 3.48GB になった。

2048 row bucket でも同じで、`total=0.955GB` に対して `peak_live=0.236GB` だった。

## 2. 修正

`src/shelf/runtime/program/program.cpp` の workspace planner を変えた。

- free block を **offset 順の map** に持ち、free 時に前後の隣接 block と結合する。
- 確保サイズを `kWorkspaceAlignment`（256 B）に切り上げ、分割位置も 256 B 境界にする。
- 選択は従来どおり best-fit。

これで 8192 bucket は 3.479 → 1.252GB になった。

## 3. fused producer の lifetime

coalesce した詰めた layout にすると、27B dense の 512 token prefill で生成 ID が
変わり、詰める前の layout とも一致しなかった。0 guard でも **実行ごとに同じ誤り**に
なるため、まず決定的な範囲外アクセスを疑ったが、真因は workspace の lifetime だった。

- MXFP4 の fused producer（`mxfp4_residual_norm_quantize` / `mxfp4_norm_quantize`）は、
  quantize dispatch の 1〜2 個前の RMS_NORM / RESIDUAL_ADD dispatch で
  activation workspace を書く。
- しかし planner は activation workspace の `first_command` を **quantize dispatch**
  にしていた。そのため、norm/add と quantize の間で free になった値の block を
  activation workspace が再利用でき、fused producer の書き込みがその値を壊した。
- coalesce 前の layout は隙間が大きく、この重なりが実害にならなかった。
  coalesce 後は隣接するため、壊れるようになった。
- 4KB red zone は offset をずらして重なりを偶然避けていただけで、原因ではなかった。

修正は、activation workspace の `first_command` を fused producer の開始 dispatch に
合わせることである。直前の dispatch が RMS_NORM で、その出力が quantize の入力なら
1 つ前、さらにその前が RESIDUAL_ADD で norm の入力なら 2 つ前を開始点にする。

| 状態 | 27B dense 512 token の生成 ID |
| --- | --- |
| 修正前 + 0 guard | `760,1379,25293,...`（誤り） |
| 修正前 + 4KB guard | `760,3712,314,...`（正しい） |
| **修正後 + 0 guard** | `760,3712,314,...`（3 回とも同一） |

## 4. 結果

| 条件 | 修正前 | 修正後 |
| --- | ---: | ---: |
| Flash-Next 8192、workspace reserve（WSDBG） | 3.479 GB | **1.252 GB** |
| Flash-Next 8192、`execution_workspace_bytes` | 3.626 GB | **1.399 GB** |
| Flash-Next 8192、`rank0_arena_used_bytes` | 20.97 GB | **18.74 GB** |
| Flash-Next 8192、prefill | 6.081 s / 1,347 tok/s | 5.997 s / 1,366 tok/s |
| 27B dense 2048、`execution_workspace_bytes` | 963.3 MB | **363.5 MB** |
| 27B dense 2048、prefill p50 | 919 ms | 887 ms |

lifetime を 1〜2 dispatch 延ばしたぶん、dense 2048 は red zone 時代（343 MB）より
20 MB 増えるが、断片化の修正前（963 MB）からは 62% 減っている。

## 5. 決定性の確認

- 27B dense、prose 512 token、8/128 token 生成: 修正後の build で 3 回とも
  `760,3712,314,23470,...` で一致し、修正前の正しい ID と同一。
- Flash-Next、2,056 token の chat prompt、8 token 生成:
  `--prefill-chunk 2048` で 2 回とも
  `248068,198,760,1156,682,3162,290,264` で一致。
- 8,200 token でも 2 回一致（QSA indexer budget の境界）。
- ユーザー報告の「長文で壊れる」は 8,200 token まででは再現しなかった。

## 6. レビュー指摘の検証

`mxfp4_activation_producer.hip` に対して指摘された 3 点を確認した。

- **codes の 16 行タイル**: 呼び出し側の `Int8ActivationWorkspaceLayout` は
  `rows_padded = align16(max_rows)`、`code_row_stride = align16(k_padded)` で
  16 行タイル全体を確保している。よって production の経路では不足しない。
  ただし launcher に `code_stride >= k_padded` の検証を追加した。
- **scales の capacity**: `scale_row_stride = align16(groups * 4)` なので
  `groups * 4` 以上が保証される。launcher に
  `scale_stride >= groups * sizeof(float)` の検証を追加した。
- **odd-k の LDS 整列**: `x_words = (k + 1) / 2` に直し、`sums` と `s.pack` の
  開始位置を word 単位でそろえた。`k=129` でも `sums` が 4 byte 境界に載り、
  `s.pack` と重ならない。

いずれも今回の 512 row の破壊の原因ではなかった（stride 検証を追加しても
0 guard の誤出力は変わらなかった）が、latent な不具合として修正した。
