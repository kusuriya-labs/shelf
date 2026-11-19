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

## 5. 未解決

- red zone が隠している **range 終端の越境** の実体（どの kernel か）は特定していない。
  4KB で足りているが、越境量が 4KB を超える入力では再発し得る。
  red zone を sentinel で埋めて実行後に検査するか、range ごとに `SHELF_OP_SYNC` で
  切り分ける必要がある。
- ユーザーが報告した「Flash-Next は長文で壊れる」は、2056 token
  （chunk 512 / 2048）では再現しなかった。より長い context（QSA の block-sparse が
  始まる 12,288 以上）や DFlash2、複数リクエストなど、別の条件が必要と考えられる。
  再現条件が特定できれば、同じ red zone/越境の枠組みで切り分ける。
