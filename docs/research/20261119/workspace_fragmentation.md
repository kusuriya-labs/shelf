# workspace allocator の断片化と red zone

実験 ID: `WS-FRAG-1`

`SHELF_WS_DEBUG` で workspace の内訳を取ったところ、8192 row bucket の
`models/Qwen3.8-Flash-Next-Quark-MXFP4`（host expert 448、bf16 KV、1 カード）で
**3.63GB を確保しているのに、同時に生きているのは 0.94GB** だった。
原因は workspace allocator の断片化である。free block の coalesce と red zone で
1.40GB まで下げた記録である。

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
- 各 allocation の後ろに **4KB の red zone** を置く。range の `bytes` は実際のサイズのまま。
- 選択は従来どおり best-fit。

coalesce だけで 8192 bucket は 3.479 → 1.252GB になった。

red zone が必要だった理由: coalesce した詰めた layout にすると、dense の
`models/Qwen3.8-27B-Quark-AWQ-MXFP4`（prose 512 token、128 token 生成）で
**実行ごとに生成 ID が変わり**、詰める前の layout の出力とも一致しなかった。
red zone を入れると 3 回とも詰める前の ID と一致した。つまり、どこかの kernel が
自分の range の終端を越えて隣接 bytes を読み書きしており、従来の断片化した layout の
隙間がそれを隠していた。4KB の red zone はその越境を隣の値から隔離する。

red zone のコストは range 数 2,263 × 4KB = 約 9.3MB（8192 bucket）である。

## 3. 結果

| 条件 | 修正前 | 修正後 |
| --- | ---: | ---: |
| Flash-Next 8192、workspace reserve（WSDBG） | 3.479 GB | **1.252 GB** |
| Flash-Next 8192、`execution_workspace_bytes` | 3.626 GB | **1.399 GB** |
| Flash-Next 8192、`rank0_arena_used_bytes` | 20.97 GB | **18.74 GB** |
| Flash-Next 8192、prefill | 6.081 s / 1,347 tok/s | 6.031 s / 1,358 tok/s |
| 27B dense 2048、`execution_workspace_bytes` | 963.3 MB | **342.6 MB** |
| 27B dense 2048、prefill p50 | 919 ms | 899.6 ms |

prefill 時間は誤差の範囲で、workspace の削減が速度を落とすことはない。
arena 使用量が 2.23GB 減ったぶん、resident expert か KV に回せる。

## 4. 決定性の確認

- 27B dense、prose 512 token、128 token 生成: 修正後の build で 3 回とも
  `760,3712,314,23470,...` で一致し、修正前の ID と同一。
- Flash-Next、2056 token の chat prompt、16 token 生成:
  - `--prefill-chunk 512`: 旧 allocator 2 回、新 allocator 2 回とも同一 ID。
  - `--prefill-chunk 2048`: 旧 allocator 2 回、新 allocator 2 回とも同一 ID。
    旧と新の ID も一致した。
- chunk 512 と chunk 2048 では生成 ID が変わる（chunk 構成による数値差）が、
  同じ chunk 内では決定的である。

## 5. red zone が隠している越境

red zone のサイズを変えて、越境の性質を切り分けた（27B dense、prose 512 token）。

| guard | 生成 ID | 判定 |
| ---: | --- | --- |
| 0 B | `760,1379,25293,...`（3 回とも同一） | 決定的に誤り |
| 16 B | `760,3712,314,...`（3 回とも同一） | 正しい |
| 32 B / 64 B / 4 KB | 正しい | 正しい |

- 0 B でも **実行ごとに同じ誤り** になる。race ではなく、決定的な範囲外 write である。
- guard を 0x00 と 0xA5 で埋めても生成 ID は変わらない。guard の bytes は読まれておらず、
  **read の越境ではない**。write が guard に落ちている。
- 256 token の prompt では 0 B と 16 B の ID が一致する。越境は 512 row の prefill で
  だけ起きる（256 row では起きない）。
- 越境量は 16 B 以下（16 B で隔離できる）。4 KB の red zone は 256 倍以上の余裕がある。
- どの kernel かは特定できていない。no-reuse にして guard を一意にした probe は
  128 row では越境を検出せず、512 row の no-reuse workspace（約 16 GB）は
  重み（約 20 GB）と同時に 32 GB のカードへ収まらない。
  特定の producer kernel だけに guard を置く scan は、layout が変わることで
  越境先が変わり、全 kernel で正しい ID になったため切り分けにならなかった。

4 KB の red zone はコスト 9.3 MB で、越境の実体が判明するまで残す。

## 6. Flash-Next の長文

ユーザー報告の「長文で壊れる」を、chat template の 2,056 token と 8,200 token の
prompt で確認した（16 / 8 token 生成、host expert 448、bf16 KV、1 カード）。
2,056 token は chunk 512 と chunk 2048、8,200 token は chunk 2048 で、
それぞれ 2 回とも同一 ID だった。2,056 token では旧 allocator とも一致した。
8,200 token は QSA の indexer budget（2,048 block、compress 4）に相当する境界だが、
この条件では再現しなかった。

再現には別の条件（より長い context、DFlash2、複数リクエスト、別 checkpoint）が
必要と考えられる。
