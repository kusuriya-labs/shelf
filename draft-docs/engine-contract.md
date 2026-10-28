# 生成エンジン抽象契約（engine-contract）

状態: 草案 v0（D1 / D8 の入口。決定済み: **sampling はエンジン側**、パラメータは
`temperature` / `top_k` / `top_p` / `min_p` / `seed` の 5 つ）
関連: `draft-docs/directory_design.md` §3.5（Server の置き場）、
`draft-docs/port_strata_cpu_gpu_coop.md`（KV 世代と協調推論）

---

## 0. 範囲

本書が決めるもの: **Server とエンジンの間の操作・イベント・生成パラメータ・キャンセル・エラー**。

本書が決めないもの（別紙）:

* HTTP / OpenAI の応答形と未知フィールド方針 → `serve-protocol.md`
* KV の予算と世代 → **本書 §8（追補）**。予算（§8.1）・世代と寿命（§8.5）・超過時の挙動
  （§8.6）は確定。残る未決は tier 移動の発動体と §8.4 の思考打ち切り方
* 数値の等価性 → `parity-policy.md`
* 予算と所在 → `memory-budget.md`

---

## 1. 操作（Server → Engine）: JSONL、1 行 1 オブジェクト

| op | フィールド | 備考 |
| --- | --- | --- |
| `generate` | `request_id`, `input_ids[]`, `max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed` | §3 の 5 パラメータは省略可（既定値を持つ） |
| `cancel` | `request_id` | 冪等（§4） |
| `ping` | – | `pong` を返す |

* **検証はエンジン側**: 未知キーの拒否、型検証、範囲検証（§3.2）。違反は **`error` イベント**で
  フィールド名つきで返す（ログに出してはいけない。ログは観測であってプロトコルではない）
* `max_new_tokens` はエンジンの語法。OpenAI の `max_tokens` は Server 側で写像する
* 1 行 = 1 オブジェクト、末尾改行。1 プロセスにつき **stdin 書込み 1、stdout 読取り 1**

---

## 2. イベント（Engine → Server）

| event | フィールド |
| --- | --- |
| `token` | `request_id`, `token_id` |
| `done` | `request_id`, `generated_ids[]`, `finish_reason`, `prompt_tokens`, `generated_tokens`, `seed` |
| `error` | `request_id`(任意), `code`, `message` |
| `pong` | `capabilities{}` |
| `shutdown` | – |

* `finish_reason` ∈ { `stop`, `length`, `cancelled`, `error` }（OpenAI 名への写像は Server 側）
* **`done` に `seed` を必ず返す**（§3.4）— 再現に必要なので最初から載せる
* `request_id` は Server 側で一意。terminal（`done` / `error`）を受けるまで mailbox を廃棄しない
* **統計フィールドは最初から持たせる**: 後付けするとプロトコルが壊れる

---

## 3. 生成パラメータ（決定: 5 つ）

| パラメータ | 型 | 既定 | 有効範囲 | 役割 |
| --- | --- | --- | --- | --- |
| `temperature` | float | **0.0** | finite, ≥ 0 | 確率分布の鋭さ。**0（または未指定）= greedy**（現状の shelf 挙動を維持） |
| `top_k` | uint32 | **0** | ≥ 0 | 確率上位 K 候補に限定。**0 = off** |
| `top_p` | float | **1.0** | (0, 1] | 累積確率が P 以上になる最小の候補集合 |
| `min_p` | float | **0.0** | [0, 1] | 最大確率に対する相対閾値 `p_i < min_p * p_max` を除外。**0 = off** |
| `seed` | uint64 | 未指定 | ≥ 0 | 乱数の種（§3.4） |

**`min_p` は OpenAI 公式には無い拡張**（vLLM / llama.cpp 系の方言）。採用する以上、
HTTP 側も `min_p` を明示的に許可フィールドとして扱う（未知フィールドの一般方針は
`serve-protocol.md`）。出所は radiance の chain 仕様。

### 3.1 実行順序（列挙順ではない — 契約）

```
top_k → top_p → min_p → temperature → 乱数抽選
```

* **temperature は最後**: `top_p` / `min_p` は**確率で**候補を切るため、先に温度を掛けると
  生存候補そのものが変わる（radiance `core/sample/host_ref.h` の規約。vLLM の
  `ChainOrder::Radiance` に追従）
* `temperature == 0` のとき乱数は一切使わず argmax（greedy）。ここが shelf の現状実装と一致する

### 3.2 検証規則（エンジン側）

* `temperature`, `top_p` は finite でない値（NaN / inf）を拒否
* `temperature >= 0`、`top_p ∈ (0, 1]`、`min_p ∈ [0, 1]`、`top_k >= 0`
* 違反は `error{code:"invalid_argument", message:"sampling ..."}`（PhaseShift の
  `validate_sampling_config` の形を踏襲。`min_p` の範囲は本契約で追加）

### 3.3 所在

**sampling はエンジン側**（PhaseShift / radiance / Strata の 3 系が一致）。
Server は値を写像するだけで、logits を受け取らない（vocab 分の転送を避ける）。

* shelf の初期実装: ホスト（現状 `run_step` → `fetch_logits` の D2H 経路を拡張）。
  後日にデバイス実装へ移しても **本書の契約は変更しない**
* 実装の置き場: `runtime/`（生成の進行層）。`backend/` は logits を出すだけ

### 3.4 seed の規約

* **乱数は `(seed, position)` の純粋関数**にする。状態付き乱数（`std::mt19937` を
  リクエストごとに持つ等）は使わない — そうすると**結果がバッチ構成に依存**する
  （radiance が明文化している失敗パターン）
* 推奨アルゴリズム: **Philox4x32-10**（ホスト実装と将来のデバイス実装で bit 一致させやすい。
  radiance が `rad_philox.h` を「ノイズの単一定義」として持つのと同型）
* `seed` 未指定時は**エンジンが採番**し、`done.seed` で返す（再現可能にする）。
  0 は正規の seed として扱う（PhaseShift は既定 0 のまま固定 — shelf はこの点だけ変える）

---

## 4. キャンセル

* `cancel` は**冪等**
* 終端は `done.finish_reason == "cancelled"`
* Server 側は「未完了なら必ず `cancel` を送り、terminal イベントを drain してから mailbox を
  解放する」（リーク防止。PhaseShift `compute_client` の `finally` と同型）
* **決定（§8.5）**: cancel で即時に解放するのは **スロット（`slot, generation`）だけ**。
  KV 正本（pinned）はターン境界まで残し、後続作業（in-flight の pass・掴み中の prefix）が
  指している間だけ解放を遅延し、commit か sweep で解放する（radiance の「即時が原則 +
  遅延例外」。Strata の「解放しない」は park 方式の前提、PhaseShift の「finish で全解放」は
  VRAM 正本の前提なので、shelf は radiance 形が噛み合う）

---

## 5. エラー

* エンジンの `shelf::Status::Code` → `error{code, message}` の対応表を本書で持つ
  （`invalid_argument` / `out_of_range` / `insufficient_memory` / `invalid_state` /
  `not_found` / `unsupported` / `device_error`）
* HTTP への写像（400 / 404 / 409 / 500）は `serve-protocol.md` の担当

---

## 6. 未決（本書の外）

**解消済み（記録）**: multi-turn / KV 世代 / `reset` op の要否 — **§8.5 で確定**
（live 保持・`reset` op 不要・世代は `BlockKey` + `(slot, generation)` の二層）。

* プロトコルのバージョン表記（`ping` の `capabilities` に含めるか、行頭に持たせるか）
* `logit_bias` / `logprobs` は受けない方針でよいか（現時点では**受けない**）
* 未知フィールドの HTTP 側方針（`serve-protocol.md`）

---

## 7. 出所

* PhaseShift: `include/phaseshift/inference/scheduling/sampling_params.h`（`SamplingConfig`:
  temperature / top_p / top_k / seed）、`.../sampling_params.cpp`（検証規則）、
  `src/apps/compute/stdio_server.hip`（JSONL、strict キー検証、`done` payload、
  `SamplingConfig` の組み立て）、`src/apps/server/serverlib/compute_client.py`
  （cancel と terminal drain、`finish_reason`）
* radiance: `core/server/iface.h`（`submit` / `submit_all` のアトミック受理 / `cancel` 幂等 +
  即時 KV 解放 / stop と detokenizer の契約）、`core/sample/host_ref.h`
  （chain 順序と「乱数は (seed, position, stream) の純粋関数」）、`core/sample/sampler.h`
  （min-p / top-k の位置づけ、top_k 0 = 候補幅）、`abi/rad_philox.h`（ノイズの単一定義）
* Strata: `serve/server.py`（`temperature=F top_p=F top_k=N seed=N` の写像、未指定 = greedy）、
  `src/program/generate.cpp`（エンジン側での適用）— 参照のみ
* shelf: `apps/shelf_run.cpp:4`（現状 greedy 専用 = `temperature` 既定 0 の根拠）

---

## 8. 追補: セッション / KV の予算と世代

状態: **予算の数値は確定**（§8.1）、multi-turn の世代と超過時の挙動は未決（§8.5 / §8.6）。

### 8.1 決定した数値

| 項目 | 決定値 | §1 / §2 への接続 |
| --- | --- | --- |
| KV の使用方式 | **PagedKV で動的使用**（ブロック単位の割当と再利用） | §4「将来: キャンセルで KV を即時解放」の前提条件 |
| **KV の正本（所在）** | **基本的に PinnedHostRAM**。VRAM は online 分だけを載せる | 予算と所在の数値は `memory-budget.md`（別紙）へ |
| 1 リクエストの decode 予算 | **32,768 tok** | `generate.max_new_tokens` ≤ 32,768。超過は `error{invalid_argument}`（§3.2 と同じ扱い） |
| うち思考（reasoning）予算 | **24,576 tok まで** | 残り 8,192 tok は非思考として常に確保 |
| ContinuousBatching | **最大 4 並列** | スロット = 4。1 decode step の出力は最大 4 tok（prefill は別） |
| **4 リクエスト時の総予算** | **単純に 4x = 131,072 tok**（予算は並列数に比例） | 正本は pinned RAM 側に確保できる。VRAM には載せなくてよい（§8.3） |
| **VRAM 上の online KV 上限** | **最大並列でも 81,920 tok = 2.50 GiB** | 「VRAM に載せてよい量」の上限。**総予算と比較してはならない**（§8.3） |

※ 曾案の「総 20,480 / 思考 14,336」は不採用。上表が正式。

### 8.2 予算式と換算（開発モデル `Qwen3.5-9B`）

`/opt/zen/work/models/Qwen3.5-9B/config.json`: 32 層のうち **full_attention は 8 層**
（残り 24 層は linear_attention = GDN でトークン比例しない）、`num_key_value_heads = 4`、
`head_dim = 256`、bf16。
（2026-10-28 に開発モデルを 9B へ選定。旧 `Qwen3.5-4B` は**同一幾何** — 32 層 / full 8 層 /
kv_heads 4 × head_dim 256 — だったため、**換算値 1 tok = 32 KiB は変わらない**）

```
KV_bytes(T) = T × L_full × 2 × kv_heads × head_dim × sizeof(bf16)
            = T × 8 × 2 × 4 × 256 × 2 = T × 32,768 B     （1 tok = 32 KiB）
```

* `L_full` は **full_attention 層のみ**を数える。GDN 状態は固定費なので式に含めず、
  ステップグラフの `StateDesc` から算出してフェーズ 0 の計測に含める
* 式はモデル非依存で固定し、モデルが変わったら `TextConfig` から再代入する
* **T の定義（重要）**: 1 リクエストの KV は **prompt と生成分の両方**を握る。
  `max_kv_tokens = prompt + (max_new_tokens − 1)`（PhaseShift
  `kv_capacity_manager.cpp:7-21` の式を採用）。
  32,768 は**生成（decode）分の予算**なので、`Σ prompt`（会話文脈）の分は
  pinned 容量に**別途 headroom として含める**。4x = 131,072 だけでは足りない

| 区分 | トークン | ブロック（block=16） | VRAM / pinned |
| --- | ---: | ---: | ---: |
| 1 リクエスト上限 | 32,768 | 2,048 | 1.00 GiB |
| ├ 思考上限 | 24,576 | 1,536 | 768 MiB |
| └ 非思考の下限 | 8,192 | 512 | 256 MiB |
| **4 リクエスト時の KV 正本（pinned RAM）** | 131,072 | 8,192 | **4.00 GiB** |
| **VRAM に載せる online 分の上限** | 81,920 | 5,120 | **2.50 GiB** |

すべて `block_size = 16` で割り切れる（2,048 / 1,536 / 512 / 5,120 / 8,192 ブロック）。
正本 4.00 GiB（pinned）と online 2.50 GiB（VRAM）は**別桁**なので大小比較はしない（§8.3）。

### 8.3 予算の桁と KV の所在（**訂正あり**）

**訂正**: 「4 リクエスト × 32,768 = 131,072 > online 81,920」という**比較は誤り**。
両者は別の桁を見ている。

* **KV の正本は PinnedHostRAM**（§8.1）。4 リクエストの総予算 131,072 tok（4.00 GiB）は
  **pinned 側に確保でき、VRAM に全部載せる必要は無い**
* **VRAM の online 上限 81,920 tok（2.50 GiB）は「その瞬間に VRAM に載せてよい量」**。
  正本より小さいのは構成として正しい（正本が pinned だから成り立つ）
* **最悪ケース全量予約は pinned 側で成立する**（正本が pinned で全量が入る）。ただし容量は
  §8.2 の式どおり `Σ (prompt_i + max_new_i − 1)` で決める。
  → **予約（admission）は pinned 容量に対して行い、VRAM 81,920 は予約対象にしない**。
  参照調査の「最悪ケース全量予約は不可（131,072 > 81,920）」は **VRAM 正本を前提とした話**で、
  shelf（pinned 正本）には該当しない
* したがって admission は **2 段**で考える。

**ただし実行側には別の制約がある**（§8.5 の caveat）。shelf は full causal attention なので、
decode は**毎トークン、active な文脈の全 block を読む**。VRAM に無い分を毎トークン pinned から
移すと、S = 81,920 tok で 1 レイヤ 320 MiB × 8 レイヤ ≈ **2.5 GiB / トークン**
（PCIe ~25 GB/s で約 100 ms/トークン）。層単位の受け持ちも同量になる。
→ **「VRAM に無い active block をどう読むか」が性能を決める**（選択肢は §8.6 の (a)(b)(c)）。

| 段 | 制約 | 尽きたとき |
| --- | --- | --- |
| 1. pinned（保存・正本） | `Σ (prompt_i + max_new_i − 1) ≤ pinned 容量`（4x 時 131,072 + prompt 分） | 受付拒否 / 保持を間引く（§8.6） |
| 2. VRAM（実行・駐在） | **active な文脈の合計 ≤ 81,920 tok** | §8.6 の (a) admission を絞る / (b) pinned 直読み / (c) 層単位受け持ち |

* 旧稿の「同時 KV 合計 ≤ 81,920」「満額で同時に走れるのは 2 件まで」は**撤回**した。
  ただし**実行側で見ると「Σ active ≤ 81,920」という形で同じ数字が制約として戻る**
  （誤りは保存側と実行側の取り違えであって、数字そのものではない）
* (a) を採ると実効並列が絞られる（満額 32,768 の同時 2 件相当）。これを避けたい場合の代償が
  (b)(c) で、どちらも上記の転送量（約 2.5 GiB / トークン）になる
* 段 2 の扱いは **§8.6 で (a) に確定**した。運用の詳細は `memory-budget.md` に書く
  （§8.5 で選んだ素材を使う）

### 8.4 思考予算（reasoning cap）の扱い — 判定主体は未決

* 構造: 総 32,768 の**内部枠** 24,576。思考が使い切っても非思考 8,192 は書ける
* 計上: 思考トークンは `done.generated_tokens` に**含める**（OpenAI の `max_tokens` の
  解釈と一致させる）。`reasoning_content` と本文の分離は `serve-protocol.md` の担当
* **未決: 打ち切り方**（2 案、推奨は B）
  * 案 A — Server が思考トークンを計上し、24,576 到達で `cancel` → 思考を閉じて再開
    （round-trip が 1 回増える）
  * **案 B（推奨）— `generate` に思考の open / close トークン ID を渡し、エンジン側で
    上限到達時に close を文脈へ注入して通常出力へ継続する**（1 リクエストで完結。
    エンジンは「語法」を持たず ID のみ受け取るため、§3.3 の分離に反しない）
  * どちらでも、思考を**強制終了せずに総 32,768 で止まる**状態は作らない

### 8.5 KV の世代と寿命 — 参照調査の取捨選択（Strata / radiance / PhaseShift）

出典: 3 系統の KV「世代と寿命」調査（2026-10-27）。決定は「骨格 / 事故対策 / 拒否・退避」に分けて採る。

#### 採用する

| # | 採用する要素 | 出所 | shelf での当てはめ |
| --- | --- | --- | --- |
| 1 | **正本 = pinned、VRAM は駐在分**（書込みは正本常常時 / reader は `page_table` 抽象 / 信頼境界の変化では**キャッシュ全破棄 → on-demand 再流入**） | Strata `include/strata/kernels/kv_stream.hpp:1-21`、`layer.cpp:1216-1227`、`conversation_snapshot.cpp:204-220` | shelf の構成そのもの。**採るのはこの 3 点だけで、miss 機構（CLOCK）は入れない**（下記 caveat） |
| 2 | **`(slot, generation)` 世代ハンドル + commit 時の stale 検証 + `poisoned`** | PhaseShift `request_handle.h`、`sequence_slot_pool.cpp:66-94`、`paged_sequence_state.cpp:240-243`、`prefix_cache.cpp:1000-1013` | **ターン毎リセット = generation のインクリメント 1 回**。VRAM の古い世代（reset / rollback / cancel 後）を読めない。radiance は世代を持たず「遅延解放 + 手動リセット」で代替しているが、shelf は世代を採る |
| 3 | **位置状態 vs 実行状態の分離** | Strata `src/program/generate.cpp:1704-1715` | KV は位置状態なので **pinned 正本からの復元は suffix 書き直しだけ**。shelf は hybrid モデルなので **GDN 再帰状態だけがチェックポイント対象**（Strata の「実行状態のみ保存」がそのまま効く） |
| 4 | **publish-then-free と cancel 契約**（即時解放・idempotent・publish は先） | radiance `core/server/iface.h:33-35`、`core/sched/scheduler.cpp:403-463` | 正本が常に書かれているので write-back は自明。**cancel はスロット（generation）だけ即時解放**し、KV 正本はターン境界まで残す（Strata の live 保持と合成） |
| 5 | **寿命の階層（live → parked → disk）と「RAM 不足でもリクエストを止めない」**（park skip / 一段解放して再試行） | Strata `conversation_buffer.hpp:18-20`、`session_save_reclaim.hpp:17-53`、`generate.cpp:9586-9646` | shelf の multi-turn の本線。park は page 揃えの**増分のみ**（`conversation_snapshot.cpp:147-181`）、restore は prefix だけ可（`:266-285`） |
| 6 | **枯渇の 3 段（evict → preempt → fail）と「キュー頭で永久待機禁止」＋ step 内チャンク切り詰め** | radiance `scheduler.cpp:1296-1323` / PhaseShift `token_budget_scheduler.cpp:85-113` | 保存側（pinned）の枯渇で使用。実行側（VRAM）の枯渇は §8.6 |
| 7 | **fail-closed なキャッシュ保存**（保存失敗は性能だけ落とし serving 継続、作成時に容量検査） | PhaseShift `prefix_cache.cpp:763-787, 129-198` | VRAM キャッシュ（81,920）に適合 |

#### 採用しない

* **PhaseShift の shadow pool（VRAM 二重複写）** — 正本 pinned では容量（81,920）と帯域の無駄
* **PhaseShift の finish / cancel での `hipStreamSynchronize`** — step 経路に同期を置かない
  （radiance 原則 `core/mem/kv.h:111-113`）。世代 + 遅延確認で代替
* **radiance の expert slab 貸付（loan / live mark）**、hybrid モデル向け checkpoint 一式
  （`GeometricBackoff` 等） — shelf に該当する plane が無く、複雑さに見合う利得がない
* **Strata の batch slot 毎 SessionState 複製**、チェックポイント連鎖運用（root pin / tail /
  spread）、VRAM 正本前提の elastic VMM（`--kv-grow`）、`page_size=4`
* **Strata の CLOCK second-chance resolve** — 下記 caveat

#### caveat: Strata の KV streaming はそのまま使えない

Strata の streaming が成立するのは **1 ステップで読むセルが疎い**（indexer top-k で
cap ≈ 2,051 セル）から。**shelf は full causal attention で、毎トークン active な文脈の全 block を読む**。

* VRAM に無い分を毎トークン pinned から移すと、S = 81,920 tok で
  1 レイヤ 320 MiB × 8 レイヤ ≈ **2.5 GiB / トークン**（PCIe ~25 GB/s で約 100 ms/トークン）
* 層単位の受け持ちも同じ量になる
* → Strata から借りるのは「正本 / 駐在の考え方・`page_table` 抽象・restore 時の全破棄」だけで、
  **miss キャッシュ機構（CLOCK / indexer 向け）は入れない**
* → 「VRAM に無い active block をどう読むか」は §8.6 の選択肢（(a) active 合計 ≤ 81,920 を
  admission 上限にする / (b) pinned 直読みを許容 / (c) 層単位の受け持ち）

#### 決定（未決だった 4 点。3 系統が揃って確定）

**① multi-turn は live 保持（Strata / radiance 型）**

前ターンの KV をターン境界で解放しない。3 系統のうち Strata・radiance が保持型（2/3）で、
shelf の「正本 = pinned」（採用 1）と「寿命の階層 live → parked → disk」（採用 5）とも整合。
PhaseShift 型（ターン毎全解放 + prefix cache 再構築）は **VRAM 正本の前提**に依存するため採用しない。

**② `reset` op は持たない（3/3 で全系統に存在しない）**

* Strata: ターン毎の reset は無く、唯一のリセットは `resume == 0` 時の `session_zero`
* PhaseShift: op は `ping` / `stats` / `generate` / `cancel` / `shutdown` の 5 つで列挙以外を拒否
* radiance: グローバルな prefix cache reset のみ（ターン毎の reset は無し）
* 続行/新規は**内容一致で自動判定**する: live に保持しているトークン列との前方一致
  （extend → 続行 / rewind → その位置まで / 不一致 → 上書き）。
  **上書き（新規）時の `reset_states()` 相当が唯一のリセット** → §6 の該当未決はこれで閉じる

**③ 世代は二層で表す（新設するのは `(slot, generation)` だけ）**

* **内容の同一性 = `BlockKey`**（`src/runtime/state/prefix_key.hpp` の既存 ABI。親キーと
  トークン列の連鎖ハッシュ + 照合時のトークン比較）— radiance の BlockHash と同型。
  prefix の一致・共有・借用はこれで表す
* **占有 = `(slot, generation)`**（採用 2）— slot 再利用時の stale 書き込みを commit 時に検出
* 両者は役割が重複しないため足し算で成立する。**session ID も会話世代番号も新設しない**
* 衛生規則として radiance の 3 点を採る: refcount 中のブロックは再配分不可 /
  adopt はブロック境界の倍数のみ / **snapshot 無しの GDN（recurrent）再利用は reuse=0 に丸める**
  （shelf は GDN 状態を持つため 3 点目が効く）

**④ 解放の単位と順序**

* 解放は `finish`（EOS / length / cancelled / error）の **1 経路**に集約する
  （PhaseShift `finish_request` 同型）。**publish（`done` または prefix への取り込み）を
  先に済ませてから**解放する（3/3 共通の順序）
* 単位は **page(16 tok) + GDN state slot + `(slot, generation)`**
* **KV ページは解放時に zeroing しない**（保護は ③ の世代とブロック表の有効化範囲で行う。
  PhaseShift / radiance 共通）
* **再利用する GDN state slot は必ず zeroing する**（radiance `StateClear`）。
  shelf は GDN 状態を持つため、これだけは必須

#### まだ未決（別紙・フェーズに接続）

* prefix 共有キャッシュ（4 並列での節約）と session save/restore（disk）は後日（フェーズ分け）
* 現状: `reset_states()` は行数のみクリアで古いバイトは残る
  （`src/backend/hip/step_executor.cpp:172-179`）— 世代の境界は上記 ①〜④ に基づき実装する
* 本契約の 32,768 は**単一リクエストの生成量**。会話全体の KV 累計は正本（pinned）が抑える

### 8.6 予算超過・枯渇時の挙動 — 確定（末尾の 1 点だけ未決）

根拠は 3 系統（Strata / radiance / PhaseShift）の共通点。**HTTP の番号は
`serve-protocol.md` §3 の表に従う**（本節は `shelf::Status::Code` とエンジンの挙動を決める）。

**方針**

1. **プロンプトは切り詰めない**（3/3）。切るのは出力予算だけ — `max_tokens` を残りへ短縮し、
   それでも入らないときのみ受付拒否（`out_of_range` → 400 `context_length_exceeded`。
   `serve-protocol.md` §5-② で `n_prompt_tokens` / `n_ctx` を添えてクライアントが履歴を詰められるようにする）
2. **外部 HTTP の生成上限は OpenAI 由来の `max_tokens` のみ**（デファクト）。
   内部 JSONL の `max_new_tokens` はエンジンの語法として維持し（§1）、写像は Server 側
3. **キュー頭で永久待機しない**（radiance `can_ever_fit`）— 最悪ケースが見込めないものは入口で拒否
4. **生成中の context 到達は `finish_reason = length` で正常終了**（3/3）
5. **全量予約（claim）+ 待機キュー**（PhaseShift `KVCapacityManager`）。予約は §8.3 どおり
   **pinned 側に対して行い、VRAM online 上限（81,920 tok）は予約対象にしない**
6. **枯渇の順序は evict → preempt → fail**（§8.5 採用 6）。**予算超過でリクエストを殺さない**

| 状況 | 決定した挙動 | `Status::Code`（→ HTTP は serve-protocol §3） |
| --- | --- | --- |
| prompt > context（受付時） | 拒否（**プロンプトは切り詰めない**） | `out_of_range` → 400 |
| prompt は入るが prompt + max_tokens が入らない | `max_tokens` を残りへ短縮して続行（既定） | エラーなし（200） |
| pinned 容量（§8.3 段 1）に収まる見込みが無い | 開始前拒否（キュー頭待機の禁止） | `invalid_state` + **`busy` 宣言** → 409 |
| 生成中の KV 枯渇（evict → preempt 後も無理） | fail。**生成の打ち切り停止はしない** | `insufficient_memory` → 503 |
| 生成中の context 到達 | `finish_reason = length` で正常終了 | エラーなし（200） |
| 正本（pinned）の確保に失敗 | 起動時エラー（または確保量を下げる。mlock の実コスト） | `insufficient_memory` → 503 |
| 状態の破損（世代不一致・テーブル汚染） | 即時 fail（再試行不可） | `invalid_state` + **`state_error` 宣言** → 500 |
| 思考枠 24,576 超過 | §8.4 の案 A / B（**未決のまま**） | – |

**(a) が確定した根拠（3/3）**: full attention で毎トークン active な全 block を読む構造では
Strata 型の miss キャッシュ（CLOCK）が成立しない（§8.5 caveat）。加えて **PhaseShift（正本 VRAM）
も radiance（publish 時 writeback + 行 run、step path で host sync 禁止）も毎トークンの
ホスト往復を免れていない**ので、毎トークン約 2.5 GiB を払う (b)(c) は選ばない。

**残る未決 1 点 — 待機キュー満杯と 429**

* 受付拒否（上表 3 行目）を 409 `busy` で吸収する現行方針（`serve-protocol.md` §3 の例示）と、
  **参照 3 系統が queue 満杯に使う 429 + Retry-After**（Strata `serve/server.py:4006-4032,
  6855-6862`、radiance `core/server/server.cpp:338-355`。PhaseShift は拒否せず待機）を
  どちらにするか。429 は `serve-protocol.md` §7（レート制限の要否）と**セットで**決める。
  決まったら本表に「待機キュー満杯」の 1 行を足す
* 事実として押さえておく: Strata の 409 は**制御系専用**（`POST /v1/vram`・`/unload`）で
  **推論には使っておらず**、radiance / PhaseShift にも推論用の 409 は無い。
  推論の受付を 409 で表すのは shelf の新規判断

**tier 移動（§8.3 段 2）の残未決**: 発動体（デバイス側 resolve カーネル方式か host 側の周期処理か）、
TTL / LRU の要否、prefetch の要否 → `memory-budget.md` と協調推論計画のフェーズに委ねる。

### 8.7 実装ギャップ（現状の shelf）

| 空白 | 根拠 | 相当フェーズ |
| --- | --- | --- |
| capacity が実行ごとの定数（`ids.size() + steps + 4`）。プール概念なし | `apps/shelf_run.cpp:130` | 協調推論 計画 |
| ブロック表が恒等写像で再利用なし | `src/backend/hip/step_executor.cpp:143` | `runtime/state` の pool / eviction（`cmake/Targets.cmake:34`「pool/eviction arrive with batching」） |
| `max_sequences = 1` 固定 | `src/backend/hip/step_executor.cpp:677,704` | 4 並列化（slots の配列化 + step 内マージ） |
| reset が世代を区別しない | 同 `:172` | §8.5 の決定後に実装 |
| GDN 固定費が予算に未計上 | 本文 §8.2 | フェーズ 0 の計測 |
| **KV の正本が device のみ**（pinned 正本・載せ替え経路が無い） | 現状 KV は upfront の device `hipMalloc`（`src/backend/hip/step_executor.cpp:132-142`）。`hipHostMalloc` はゼロ | §8.3 段 2 と `memory-budget.md`（参照調査の素材を使う） |

### 8.8 決済と接続

* 予算消費は `done.prompt_tokens` / `done.generated_tokens`（§2 の payload に既存）で確定する
* スロット毎の KV 使用量（tok）を `done` に載せるかは **未決**（§2 の「統計フィールドは
  最初から持たせる」方針に従うなら、載せる側が安全）
* `error{code:"insufficient_memory"}` は §5 の対応表に追加する
