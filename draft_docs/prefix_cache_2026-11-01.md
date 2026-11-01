# Prefix Cache（T16）の実装記録 — ブロック管理の純粋部分

日付: 2026-11-01
対象: 最長共通 prefix の検索、KV ブロック参照管理、LRU、COW、世代管理。
置き場: `src/runtime/state/prefix_cache.{hpp,cpp}`（GPU・モデル非依存）。

---

## 1. 契約

* 共有の単位は **KV ブロック**（attention が block table で指す単位。既定 16 トークン）。
* ブロックの同一性は `prefix_key.hpp` の **chained key**
  `key(block) = mix(key(parent), tokens of block)`。親が違えば同じトークン列でも
  別ブロック（radix と同じ）。
* 検索は **full block 単位**。端数（block_tokens 未満）は一致に含めず、呼び手が
  その部分を再計算する。これにより「共有ブロックは読み取り専用、追記は必ず新しい
  ブロック」となり、prefill の追記で COW が不要になる。
* 参照カウント: 複数スロットが同じブロックを参照できる。`release` で 0 になっても
  **キャッシュとして残る**（次の lookup で再利用可能）。
* LRU: 参照 0 のブロックを `last_used` の古い順に再利用する。再利用時に
  **世代（generation）を進め**、古い key の entry を消す。
* COW: `fork(block)` は参照 >1 のときだけ新しいブロックを確保する（呼び手が KV の
  中身をコピーする）。参照 1 なら同じ ID を返す（`copied=false`）。
* 衝突対策: 128-bit hash の一致だけで確定せず、ブロックに保存したトークン列と
  比較する（`block_tokens()`）。

## 2. API（`PrefixCache`）

| 関数 | 内容 |
| --- | --- |
| `lookup(tokens)` | 最長一致（参照は増やさない）。`PrefixHit{blocks, matched_tokens}` |
| `acquire(tokens)` | 一致したブロックの refcount++ と touch |
| `allocate()` | 参照 0 のブロックを LRU で確保（refcount=1、世代++、未登録） |
| `insert(block, tokens, block_index)` | chained key を計算して登録（トークン列も保存） |
| `release(block[])` | 参照を手放す（0 でもキャッシュとして残る） |
| `fork(block, &copied)` | COW。参照 >1 で新ブロック |
| `free_blocks()` / `cached_blocks()` / `referenced_blocks()` | 状態 |
| `hits()` / `misses()` | 統計 |

## 3. 検証（`unit.runtime_prefix_cache`）

* 同じトークン列 → 同じブロック列、`matched_tokens` は full block 分だけ
* 途中まで一致（例: 先頭 2 トークンだけ）
* 親が違えば同じトークン列でも別ブロック
* release 後もキャッシュとして hit し、acquire で参照が戻る（2 回で refcount 2）
* LRU: 2 ブロックを交互に使い、古い方が退避され、その prefix は miss。
  再利用で世代が進む
* COW: refcount 2 の fork は新ブロック・旧は refcount 1、refcount 1 の fork は同一 ID
* 端数トークンは一致に含めない。`block_tokens()` でトークン列を確認できる
* 参照中のブロックは退避できず `insufficient_memory`

## 3.1 実行器のグローバル KV プール（2026-11-01 続き・実装済み）

* `HipStepExecutorOptions::kv_blocks`（層あたりのグローバル block 数。0 = 従来の
  `pages × max_sequences`）。`assign_kv_blocks(slot, blocks, count, offset)` で
  スロットのブロック表を任意のグローバル block へ書き換えられる。
* ブロック表は `max_sequences × pages` エントリで、`block_table_stride = pages`。
  attention / kv_append には `num_pages = kv_blocks()`（プール全体）を渡す。
* 共有プール（`kv_blocks` 明示）では `reset_slot` は **KV ページを 0 にしない**
  （他スロットが参照しうる）。スロットの表だけ無効化し、呼び手が assign する。
  従来の固定プール（`kv_blocks=0`）は今までどおりスロット範囲を 0 クリアする。
* 検証（`unit.hip_step_executor` に追加）:
  * 16 トークンの prefix を slot0 で計算（block0）
  * slot1 の表を `page0 → block0`（共有）/ `page1 → block3` にして 17 個目を流す
  * 同じ 17 トークンを別の実行器（固定プール）で計算した logits と比較 → **差 0.0**
  * 範囲外 block の assign は拒否

## 3.2 エンジンとの接続（2026-11-01 続き・実装済み）

* `shelf-compute --prefix-cache-blocks N`（N > 0 で有効）。層あたりの KV ブロック数は
  `pages × max_batch + N`。`PrefixCache` のブロック ID を全 PP ステージへ同じように
  assign する（各ステージのプールを同じ大きさにする）。
* 受付後・最初の prefill の前に `prepare_prefix`:
  1. スロットを初期化（共有プールなので KV ページは消さない）
  2. prompt の最長一致を取り、**スナップショットのある長さ**まで採用（`L < prompt_len`）
  3. GDN 状態を全ステージへ import、ブロック表を assign、スケジューラを
     `prime_prefill(slot, L)` で L から prefill 開始
* prefill 中: 書く位置の page を必要に応じて `allocate` → 表へ assign。
  `snapshot_tokens`（既定 64）の倍数に達したら全ステージの GDN 状態を退避
  （スナップショット LRU。既定 8 件）。
* prefill 完了時: full block を `insert`（key は chained key）。要求終了時に
  `release`（参照 0 はキャッシュとして残る）。
* CLI: `--prefix-snapshot-tokens` / `--prefix-snapshot-limit`。Server は
  `--prefix-cache-blocks` をエンジンへ渡す。

### 検証（9B bf16・200 トークン・max_new 2）

| 実行 | 時間 | 生成列 | ログ |
| --- | --- | --- | --- |
| 1 回目（cold） | 571 ms | `[9707, 9707]` | snapshot 64/128/192、register 12 blocks |
| 2 回目（warm） | **267 ms**（0.47x） | `[9707, 9707]` | `[prefix] hit tokens=192 prompt=200` |

* 共通 prefix 192 トークンを持つ別プロンプトでも hit（192）し、**prefix cache
  なしの基準実行と同じ生成列**（`[208, 209]`）。時間は 447 ms → 267 ms。
* `tests/scripts/check_prefix_cache.py` に E2E 検査を追加（hit ログ・生成列一致・
  時間短縮を確認）。

## 4. 実装（次ラウンド）: エンジンとの接続（ハイブリッドモデルの注意）

現状の KV は **スロットごとの固定プール**（`pages × slot` の identity 表）。
Prefix Cache を効かせるには次を接続する。

1. **グローバルな KV ブロックプール**を実行器に持たせる（スロット数に依存しない
   `num_blocks`）。各スロットの block table は「論理 page → グローバル block」を
   指す（今は identity なので、表の書き換え API を足す）。
2. エンジンが受付時に `PrefixCache::lookup(prompt)` を行い、一致したブロックを
   `acquire` してスロットの表へ割り当てる。prefill は `matched_tokens` から始める
   （`PrefillAction.begin` を進める）。
3. prefill がブロック境界まで進むたびに `allocate` → KV 書き込み → `insert`。
   共有ブロックは読み取り専用なので COW は不要。
4. 要求終了時に `release`。参照 0 のブロックはキャッシュとして残し、
   新規 `allocate` が LRU で退避する。
5. 検証:
   * 同じプロンプトを 2 回 → 2 回目は prefill が `matched_tokens` 分短くなり、
     生成列は 1 回目と一致
   * 共通 prefix を持つ 2 プロンプト → 共通部分が共有される
   * プールを小さくして LRU 退避 → miss に戻り、それでも生成列は変わらない
   * 4 並列 + prefix cache で、スロット間の共有と参照解放が壊れない
   * FP8 のバッチ不変性（T15 の残件）と組み合わせて確認する

### ハイブリッドモデルの制約（重要）

現在のモデルは full attention と GDN（linear attention）の混在。**GDN の
conv/recurrent 状態は prefix の途中で切り出せない**（全履歴の圧縮状態）。
そのため、KV ブロックを共有して prefill を `matched_tokens` から始めると、
GDN 状態が prefix 分だけ欠けて出力が変わる。エンジン接続には次のどちらかが要る。

1. **状態スナップショット**: ブロック境界（例: 64 トークンごと）で GDN の
   conv/recurrent をホストへ退避し、prefix hit 時にスロットへ戻す。
   1 スナップショットは 9B で約 6 MB（24 層 × 256 KB）。4096 文脈を 64 トークン
   刻みで取ると 384 MB になるため、件数を制限した LRU にする。
2. **full attention のみのモデルに限定**して prefix cache を有効化する
   （混在モデルでは無効化し、ログに理由を出す）。

まず 1 を実装する。実行器に `export_gdn_state(slot)` / `import_gdn_state(slot, data)`
を足し、エンジンが prefix hit 時に復元 → KV 表 assign → 残り prefill、の順に進める。

### 制約・残件（2026-11-01 時点）

* 端数ブロックの共有はしない（full block のみ）。部分ブロック共有は COW と
  組み合わせて将来対応（COW 自体は `PrefixCache` の単体テストで確認済み）。
* LRU 退避の**統合**（プールを小さくして実際に追い出す）と、同時要求での
  prefix 共有は未検証。スナップショットの LRU は実装済み。
* スナップショットは 64 トークンごと・既定 8 件（9B で約 48 MB）。長文では
  スナップショット間隔と件数の調整が要る。
* ブロックのトークン列を保持するため、プールサイズ × block_tokens × 4 B の
  ホストメモリを使う（例: 8,192 block × 16 × 4 B = 512 KiB）。
* ハッシュは 128-bit だが、衝突時はトークン比較で必ず miss にする（誤ヒットしない）。
* KV の実体は層ごとのプールなので、接続時は「層 × グローバル block」の
  アドレス計算に合わせる（現行の `block_table` は層ごとのプールを前提）。
