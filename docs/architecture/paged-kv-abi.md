# Paged KV / prefix-sharing ABI（PR-007）

paged attention と prefix 共有（radix 的 KV 再利用）の契約。**カーネルが見る物理
ABI** と **共有単位の意味論** を分離して固定する。

## 3 層の境界

| 層 | 置くもの | 置かないもの |
|---|---|---|
| `ir` | `StateKind::kv_cache` の**論理**意味（append/read、容量・head次元） | ブロックテーブル、ページ、物理 layout（一切知らない） |
| `backend/common` | `PagedKVLayout` / `BlockTableView` / `SeqUsedView` / `DescaleView` / `PagedKVArgs`（物理 address・検証） | 割当・共有・eviction の方針 |
| `runtime/state` | prefix 共有の同一性（`BlockKey` chained hash、token 比較契約）。pool/refcount/LRU/COW は batching 段階で同じ ABI の上に実装 | カーネル launch、デバイス管理 |

## 物理 ABI（libr4d 準拠）

`~/work/references/radiance` 同梱 libr4d の `R4DArgs` と同型に保つ（移植は
adapter で済ませ、再設計をしない）:

- スロット layout（要素単位）: `block*block_stride + kvh*head_stride + slot*slot_stride + {K:0|V:head_dim} + d`
  （canonical: `slot_stride = 2*head_dim`、K→V がスロット内で隣接）
- block table: `int32 [num_seqs, max_blocks]`（row_stride 指定、-1 は未割当末尾）
- `seqused [num_seqs]`（実トークン数）、host 側で `max_ctx` 上限
- fp8 KV: `k/v_descale [num_seqs, kv_heads]`（bf16 では null。ABI は先に用意）
- **block_size はカーネル制約から取る**（radiance 則。現行 paged kernel は 16）

`PagedKVArgs::validate()` が table×seqused×pool 容量の整合を一箇所 で検査する。

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
