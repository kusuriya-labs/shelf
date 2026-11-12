# QSA prefill key cache の並列化と top-k radix select

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `src/shelf/kernels/optimized/qsa.hip`、`src/shelf/inference/execution/qsa_dispatch.hip`。

## 結論

`qsa_key_cache_batch_kernel` の prefill 経路をブロック並列化し、
`qsa_topk_kernel` の単発選択を radix select に置き換えた。
FlashNext 3.05 bpw、8K prompt の両 rank 合計 GPU 時間は次のとおり。

| kernel | 変更前 | 変更後 |
|---|---:|---:|
| `qsa_key_cache_batch_kernel` | 131.4 ms | 0.8 ms（blocks + counts） |
| `qsa_topk_kernel` | 35.7 ms | 19.4 ms |
| `qsa_score_wmma_kernel` | 55.1 ms | 44.5 ms |
| 合計 | 223.6 ms | 66.1 ms |

32K では key cache が 530 ms、top-k が 484 ms、score が 738 ms（両 rank 合計、
旧ビルドの trace）だった。8K の比率から、32K では QSA 全体で約 1.2 s の短縮に
相当する。

## key cache のブロック並列化

prefill の chunk は compress_ratio の倍数で、各圧縮ブロックは raw key だけで
決まる。一方、旧カーネルは 1 request を 1 CTA が全 token 分逐次処理していた。

`qsa_key_cache_blocks_kernel` は 1 CTA が 1 圧縮ブロックを担当する。

- `pending = *partial_count` を全 CTA が読み、`head`（前 chunk から繰り越した
  未完了ブロックを完成させる行）、`body`（完成ブロック数）、`tail`
  （次 chunk へ繰り越す行）を同じ手順で求める。
- `head` は保存済み partial 行と先頭の raw key を合わせて 1 ブロックにする。
- `tail` は partial を書き戻し、直後の chunk が読む `partial[tail..cr)` も
  旧実装と同じ内容にそろえる。
- カウンタは全 CTA が読むため、ブロックカーネル内では書かない。
  `qsa_key_cache_counts_kernel` を後続で起動して `pool_count` と
  `partial_count` を更新する。

dump した kernel は block あたり 1 CTA、`grid = (ceil(actual_rows / cr) + 1,
num_requests)` である。`partial_history` が必要な speculative 経路と、
`actual_rows < 256` の decode は従来カーネルのまま。

正しさは `test_qsa_state_pool` で、同じ chunk 列
（`[0,2) [2,3) [3,7) [7,11) [11,20) [20,21) [21,32) [32,40)`）を
旧カーネルと新カーネルに与え、各 chunk 終了時点の pooled / partial / count が
bit 単位で一致することを確認している。部分ブロックだけの chunk、head+tail、
head+body+tail、tail のみを含む。

## top-k の radix select

top-k の出力は最終的にブロックマスクであり、完全なソートは不要である。
単発選択（`first != 0 && last != 0 && block_begin == 0 && usable == visible`）
では、bitonic sort + merge の代わりに 8-bit radix select を使う。

1. score を単調な uint32 key に写像し、4 pass（8 bit ずつ）で k 番目の値を求める。
   各 pass で 256 バケットのヒストグラムを取り、降順に累積して境界バケットへ降りる。
2. しきい値と、それより大きい要素数 `above` から `need = k - above` を出す。
3. しきい値と等しい要素のうち `need` 個を、index の 2 pass radix で小さい順に選ぶ。
4. しきい値超の要素と、index が境界以下の同値要素を選ぶ。

同値時の「小さい index を優先」は旧実装と一致する。
`test_qsa_select` の 8192 ブロック / top-k 512 のケースを含め、
scalar 経路との一致を確認している。tiling が必要な経路（`first`/`last` の
どちらかが 0）と `selected` 出力は従来の sort/merge のまま。

## grouped EXL3 の再評価

trace の 32K では gate/up（`n_tiles=1`）が 4,666 ms / 3,008 dispatch、
down（`n_tiles=2`）が 1,750 ms / 1,504 dispatch だった。
gate と up は 2 行列なので、1 行列あたりでは 2,333 ms 対 1,750 ms、
同じ FLOPs に対して 1.33 倍である。2.7 倍ではない。

差は K 長（2560 対 640）による K chunk あたりの同期・xrot staging の
相対コストと整合する。1 chunk あたりの WMMA 数が gate/up では down の半分に
なるため、stage のオーバーヘッドが 2 倍の割合になる。
32K で gate/up を down と同じ効率にすると約 2.0 s（両 rank 合計）の短縮になる。

次の実装候補は、2 つの K chunk を 1 回の stage で処理して barrier を半減する
方式である。LDS は VGPR 律速のため余裕がある。
