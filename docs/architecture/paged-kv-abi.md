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

## prefix 共有（rev2.1: ツリー共有は採用しない）

`~/work/references/Strata`・`~/work/references/XeStrata-exl3` の read-only 全監査
（2026-10-26、`docs/architecture/feature-inventory.md`）により、**Radix/TreeAttention
は不要**と判定（ユーザー結論を監査が追認）。両者共通の事実:

- ページング自体は存在（page_table + 位置 arena + pinned-RAM/VRAM 2 段）。
  vLLM 型の動的アロケータではない
- **追い出しは CLOCK ではない**（訂正）: Strata の expert tier は decayed-count
  （LFU 系・満杯時は非追い出し）、checkpoint は LRU/spread、parking は oldest-first。
  CLOCK second-chance は XeStrata の KV ストリーミング限定
- XeStrata 側も明言: "a radix cache's tree collapsed onto one branch"
  （`program/conv_cache.hpp`）= ツリーでなく **1 分岐上の checkpoint 鎖**
- 共有は「任意長のツリー」でなく **離散点**で効かせる: ターン境界 /
  `--prompt-cache-every` / system prompt root / park / 明示 pin（`pin=N` は
  長文書終端を固定し後続質問を分岐させる上位契約）。
  位置=トークン位置なので「checkpoint から再開」は KV をそのまま再利用
- 並列時はスロット単位の会話キャッシュ = 粗い共有で十分
  （"fine granularity buys nothing and costs per-token indirection"）

shelf の採用方針:

1. **ページ arena は位置インデックス**（1 文脈 = 論理ブロック列 → 物理ページ）。
   プレフィックス再利用は「位置が同じなら KV も同じ」という positional 同一性で足りる
2. 共有・再開の単位は **離散 checkpoint**（ターン境界・固定間隔・明示 pin）。
   `BlockKey`（chained 128bit hash、`runtime/state/prefix_key.hpp`）は
   **trie ではなく checkpoint→KV の同一性マップ**（prompt cache エントリ照合）
   として使う。hit 確認は token 列比較（誤 hit 不可）
3. ブロック共有 CoW / beam 共有は **当面実装しない**（必要になった時点で
   refcount+COW は同じ ABI の上に載る設計のまま温存）
4. GDN など hybrid 状態の再開は **checkpoint interval**（attention block と
   独立設定）で recurrent/conv state を保存。実装は Phase 8

## 次（GPU 実装 PR）

`backend/hip/kernels` の attn_paged は `PagedKVArgs` をそのまま受け取る。
CPU 参照 executor（論理 layout）が golden のまま、同じ IR を 2 系統に
lower して Layer Parity を取る。
