# Paged KV / prefix-sharing ABI（PR-007、rev2 = PR-008c）

paged attention と prefix 共有（radix 的 KV 再利用）の契約。**カーネルが見る物理
ABI** と **共有単位の意味論** を分離して固定する。

## rev2 変更の理由

PR-007 は libr4d 準拠の単一プール interleave を正準としたが、実装実地検証で
gfx1201 の自前 kernel 族（bf16/fp8 paged decode・split-K・kv_append）は
**K/V 別プール** addressing（`[block][slot][kvh][d]`×2）であることを確認。
変換コピーを毎ステップ払うくらいなら、別プールを正準とし libr4d 型は
**併存 packing** として保持する（"移植は adapter で済ませ再設計しない" の
原則は kernel 側ではなく ABI 側が歩み寄って成立する）。

## 3 層の境界

| 層 | 置くもの | 置かないもの |
|---|---|---|
| `ir` | `StateKind::kv_cache` の**論理**意味（append/read、容量・head次元） | ブロックテーブル、ページ、物理 layout（一切知らない） |
| `backend/common` | `PagedKVLayout`（packing 2種）/ `BlockTableView` / `SeqUsedView` / `DescaleView` / `PagedKVArgs`（物理 address・検証） | 割当・共有・eviction の方針 |
| `runtime/state` | prefix 共有の同一性（`BlockKey` chained hash、token 比較契約）。pool/refcount/LRU/COW は batching 段階で同じ ABI の上に実装 | カーネル launch、デバイス管理 |

## 物理 ABI（rev2）

- **SeparatePools（正準・自前 kernel ネイティブ）**: プール内 offset =
  `b*block_stride + s*slot_stride + h*head_stride + d`
  （canonical: `head_stride=D`, `slot_stride=kvh*D`, `block_stride=bs*kvh*D`）
  K/V の区別はプールポインタ選択。
- **InterleavedKV（libr4d 移植互換）**: 単一プール
  `b*block_stride + h*head_stride + s*slot_stride + {K:0|V:D} + d`
- block table: int32 `[num_seqs, max_blocks]`（-1 未割当末尾）。uint32 表と
  **ビット互換**（-1 → 0xFFFFFFFF は num_pages 以上 sentinel として同一取扱）
- 長さの伝え方: kernel は `row_positions`（= 可視最終トークン位置 = seqused−1）
  を取る。`SeqUsedView` との対応は実行側のアダプタ（1 行）で固定
- fp8 KV: descale の粒度は kernel が **per(token, kv-head)**、中立 view は
  per(seq, kv-head)。fp8 KV 実装時にトークン側 view を追加（構造は既用意）
- **block_size はカーネル制約から取る**（現行 paged kernel は 16）

`PagedKVArgs::validate()` が table×seqused×pool 容量の整合を一箇所で検査する。

## prefix 共有（radix 相当）の契約

- 共有単位 = attention カーネルが address する**ブロックそのもの**
  （radix ノード = ページ = eviction 単位が一致する）
- ブロック ID = **全 prefix を織り込んだ 128bit chained hash**:
  `key(block) = mix(key(parent), tokens)`（`chain_block_key`、決定論的）
- 誤 hit 防止: hash 一致は**必ず token 列比較で確認**してから再利用
  （SGLang 型 trie でなく hash map の退化 radix。機能は等価で、
  eviction・fork(COW)・マルチランクが trie なしで単純に実装できる）
- GDN など hybrid 状態の prefix hit は **checkpoint interval**（attention
  block とは独立の設定）で支援する設計。実装は Phase 8（linear state の
  checkpoint 追加時に `StateKind::recurrent` 側へ対応 ABI を切る）

## 次（GPU 実装 PR）

`backend/hip/kernels` の attn_paged は `PagedKVArgs` をそのまま受け取る。
CPU 参照 executor（論理 layout）が golden のまま、同じ IR を 2 系統に
lower して Layer Parity を取る。
