# QSA key state の spec rollback をモデル非依存の契約にする

実験 ID `QWEN35QSA-4`。実装前の調査記録。対象は
`models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4` と DFlash2
（`models/Qwen3.8-27B-DFlash2-FP8`）。

## 1. 目的

[QWEN35QSA-3](qsa_dflash2.md) で、QSA target の DFlash2 が reject 時に QSA key state を
巻き戻さず、2048 token 超の context で block 選択が汚染されることを確認した。

本調査は、これを qwen35qsa 専用の回避策として足すのではなく、spec decode の
rolling state 全体の契約として整理できるかを確認し、実装の GO / NO-GO を決める。

## 2. 調査結果

### 2.1 QSA state の rollback は既に共有機構である

spec verify 用の per-row 履歴は既に存在し、qwen35 の MTP drafter と qwen4exp の
MTP が同じものを使っている。

- `shelf::execution::QsaSpecHistoryDeviceView`（`include/shelf/inference/execution/spec_decode_context.h`）
- `QsaKeyCacheBatchArgs::partial_history` / `count_history`（`include/shelf/kernels/optimized/qsa.h`）
- `restore_qsa_spec_state()`（`src/shelf/inference/state/qsa_state_pool.cpp`）
- `ExecuteBatchOptions::qsa_spec_history` / `ple_conv_history`（`include/shelf/models/qwen35/runtime/executor.h`）
- 配線箇所は `src/shelf/models/qwen35/runtime/mtp_drafter.cpp` のみ
  （単発 `verify_options_for` と batched の 2 経路）

意味論は「`history[row]` = その row を処理する前の state」である。MTP は partial accept で
`restore_qsa_spec_state(slot, history, row_base + accepted_drafts + 1)` を呼び、
「committed 行を処理し終えた state」に戻す。PleConvStatePool も同じ row で戻す。

qwen4exp 側の実装は
[20261108/qwen4exp_mtp.md](../20261108/qwen4exp_mtp.md) に記録されており、
`PleConvArgs::state_history_store` と `QsaKeyCacheBatchArgs::partial_history` を
qwen35 と共通の型で使う。つまり **QSA / PLE の rollback は family 固有ではなく、
target executor が任意で公開する rolling state の集合に対する操作**として既に成立している。

state pool もモデル非依存の層にある。

- `include/shelf/inference/state/qsa_state_pool.h`
- `include/shelf/inference/state/ple_conv_state_pool.h`
- prefix cache の `Qwen35PrefixState` も同じ集合（GDN / QSA / PLE / ring）を列挙している

family 名（qwen35 / qwen35qsa / qwen4exp）で分岐する箇所は無い。

### 2.2 DFlash2 側に欠けているもの

DFlash2 の spec decoder は verify の前に GDN だけを snapshot し、partial accept で
GDN を「round 開始 snapshot + compact log の再適用」で戻す
（`dflash2_spec_decoder.cpp` の `spec_gdn_snapshot` / `spec_gdn_restore` /
`commit_gdn_compact_log`）。QSA / PLE の history は確保も配線もしていない。

DFlash2 の verify は `build_gdn_options`（単発）と `build_batch_gdn_options`
（serve の複数 session）で `ExecuteBatchOptions` を作る。ここに QSA / PLE を足す
余地はあるが、現状は GDN だけを詰めている。

### 2.3 DFlash2 の既定は tree verify

`compute_cli.cpp` は drafts == 7・TP == 1・greedy（または serve）のとき
`tree_budget auto`（10 rows）を既定で有効にする。したがって「chain だけ直す」実装は
既定構成を直さない。

chain と tree で必要になる操作が違う。

- **chain**: verify の row は連続。`history[accepted + 1]` を戻せばよい（MTP と同じ）。
- **tree**: verify の row は DFS 順で、受理された path は兄弟 row を飛ばす。
  `history[walk.rows[accepted] + 1]` は兄弟の key を含むため使えない。
  GDN は compact log を path 順に再適用して解決している。QSA には同等の
  「受理 path の再適用」が無い。

### 2.4 QSA の tree 再適用に必要な情報

QSA key cache の入力は k_norm 後の raw key（`head_dim`、BF16）で、pooling は
「4 本の平均 → RMSNorm(1+w) → block 先頭位置で partial RoPE」である。受理 path を
再適用するには、この raw key と row の position だけあればよい。

- journal 量: `num_qsa_states × rows × head_dim × 2 B`
  = 16 × 10 × 128 × 2 = 40 KiB（1 rank）
- per-row 履歴量: `num_qsa_states × rows × (compress_ratio × head_dim × 2 B + 8 B)`
  = 16 × 10 × (1,024 + 8) ≈ 165 KiB

どちらも無視できる。再適用は 1 round あたり 16 layer 分で、layer を kernel の
blockIdx.y にすれば 1 launch にまとめられる。

## 3. 契約案

モデル非依存の per-row history を 1 つ定義し、MTP と DFlash2 の両方が使う。

```text
SpecStateHistory                      // target executor が公開する rolling state の集合
  rows
  qsa: partial, counts, strides       // QsaSpecHistoryDeviceView と同じ形
  qsa: key_journal, strides           // tree の path 再適用用
  ple: conv_window, stride            // qwen4exp のみ非 null

spec_state_history_bytes(pools, rows)
spec_state_history_allocate(arena, pools, rows)
spec_state_bind_options(history, ExecuteBatchOptions&)     // qsa/ple/journal を詰める
spec_state_restore(pools, slot, history, row)              // QSA + PLE
spec_state_replay_qsa(pools, slot, history, rows, positions)  // tree のみ
```

- `pools` は `QsaKeyStatePool*` / `PleConvStatePool*` の組で、null は「その state を
  持たない model」を意味する。family 判定はしない。
- 配置は `include/shelf/inference/state/`（pool と同じ層）に置き、
  `models/qwen35` に依存させない。`ExecuteBatchOptions` への詰め替えだけ runtime 側に残す。
- GDN は既存の snapshot + compact log をそのまま使う（QSA/PLE と寿命・再適用の
  形が違い、すでに動作しているため触らない）。
- chain: `restore(row = accepted + 1)`。
- tree: `restore(row = 0)` の後、`walk.rows[0..accepted]` を journal から再適用する。
  anchor row（先頭）も含めて再適用する。
- PLE + tree は今回の scope 外とし、target が PLE を持つ場合は tree を
  `Status::unsupported` で拒否する（Qwen3.8-27B は PLE を持たない）。

## 4. 実装範囲（見積り）

| 対象 | 内容 | 目安 |
| --- | --- | ---: |
| `inference/state/spec_state_history.{h,cpp}` | 新規。確保・view・restore | 150 行 |
| `fused/qsa_key_cache.hip` | journal 書き込みと raw row map（任意） | 15 行 |
| `optimized/qsa.h` / `qsa_dispatch.hip` | journal の引数追加と配線 | 20 行 |
| `spec_decode_context.h` / `executor.hip` | journal view の受け渡し | 15 行 |
| `mtp_drafter.cpp` | 既存の allocate/restore を共通 helper へ置換 | 60 行 |
| `dflash2_spec_decoder.{h,cpp}` | chain の history 配線、tree の replay、確保・解放 | 220 行 |
| tests | restore/replay の GPU unit test、state 不変条件の E2E | 200 行 |

kernel の計算自体は変えない。通常 decode（history / journal が null）の経路には
分岐が 1 つ増えるだけで、性能影響は無い。

## 5. 検証計画

1. **state 不変条件**（最も強い証拠）:
   prompt + 生成後に `4 × pool_count + partial_count == prompt + 生成 − 1` を確認する。
   QWEN35QSA-3 と同じ方法で、chain / tree / drafts を変えて確認する。
2. **selection の一致**: QSA 有効の spec 実行で、復元後の mask が「受理 path だけを
   処理した非 spec 実行」の mask と一致することを dump で確認する（任意）。
3. **MTP 回帰**: 共通 helper へ移すため、`test_qsa_state_pool` と
   qwen4exp MTP の既存テスト、`test_qwen35_spec_verify` を通す。
4. **速度**: DFlash2 の `DFLASH2_ROUND_MS` で replay の追加コストを測る。
   3k / 16k / 64k で spec なしとの比較を記録する。

注意: この build では spec と非 spec の greedy 列が一致しないことがある
（prefill の row bucket や GEMM variant が rows に依存するため）。
QWEN35QSA-3 の 2000 token でも QSA 無効で不一致を確認している。
したがって「spec と非 spec の token 一致」を合否の主判定にしない。

## 6. リスクと代替

- **tree を scope 外にする**: chain だけ直しても既定構成（auto tree）は直らない。
  その場合は QSA + DFlash2 を引き続き gate で拒否するのが正しい。中途半端な
  chain-only 対応は「動くが既定では壊れる」状態を残すため採らない。
- **journal の追加コスト**: verify 1 round あたり 40 KiB の書き込みと、1 launch の
  再適用。decode の round 時間に対して 1 % 未満と見込むが、実測で確認する。
- **multi-session（serve）**: 既存の batched 経路も同じ `row_begin + accepted + 1`
  で戻せる。history は batch 全体の row で確保する（GDN と同じ）。
- **TP**: rank ごとに QSA pool を持つため、`DFlash2GdnRankState` に保持している
  rank executor を使って rank ごとに restore する。

## 7. 判定

**GO。** 条件は次の 2 点を 1 つの変更として扱うこと。

1. per-row history をモデル非依存の `SpecStateHistory` として切り出し、
   MTP と DFlash2 の両方から使う（qwen35qsa 専用実装にしない）。
2. chain と tree の両方を実装する。tree は既定構成なので、chain だけでは
   DFlash2 + QSA を「対応」と呼ばない。

NO-GO とするのは、tree を実装しないまま QSA + DFlash2 を有効化する場合である。
その場合は現状どおり gate で拒否し、QWEN35QSA-3 の結論を維持する。

## 8. 実装結果

GO の条件どおり実装した。QSA 固有の分岐は追加せず、spec decode の rolling state の
契約として `shelf::state::SpecStateHistory` を導入した。

### 8.1 構成

- `include/shelf/inference/state/spec_state_history.h`（新規）
  - `SpecStatePools`（QSA / PLE の optional pool）
  - `SpecStateHistory`（QSA partial / counts / key journal、PLE conv window）
  - `create_spec_state_history` / `restore_spec_state`
- `QsaSpecHistoryDeviceView` に `key_journal` / `journal_state_stride` を追加。
- `QsaKeyCacheBatchArgs` に `key_journal` / `raw_row_map` / `replay_slot` / `replay_rows`
  を追加。verify は journal を書き、replay は journal を compaction なしで読む。
- `Executor::replay_qsa_spec_state()`: program の `QSA_KEY_CACHE` dispatch を
  replay mode で実行する。model 固有の norm / rope / state index は executor が持つ。
- DFlash2 は chain / tree の両方で、
  - chain partial accept: `restore_spec_state(row = accepted + 1)`
  - tree: `restore_spec_state(row = 0)` + `replay_qsa_spec_state(受理 path)`
  を行う。single と batched serve の両経路に入れた。
- PLE を持つ target の tree は `Status::unsupported` で拒否する（27B は PLE 無し）。

MTP は既存のフィールド（`qsa_partial_history` 等）をそのまま使い、同じ view 型と
restore 意味論を共有する。MTP の buffer 管理を `SpecStateHistory` に寄せる整理は
別変更とする（qwen4exp の実機検証が必要なため）。

### 8.2 状態の不変条件（実測）

`4 × pool_count + partial_count == prompt + 生成 − 1` を確認した。

| 経路 | prompt | 生成 | 期待 | 実測 |
| --- | ---: | ---: | ---: | ---: |
| single chain（drafts 4） | 3072 | 16 | 3087 | 771/3 |
| single tree（既定） | 3072 | 16 | 3087 | 771/3 |
| single tree | 16384 | 11 | 16394 | 4098/2 |
| serve batched（1 session, tree） | 3072 | 16 | 3087 | 771/3 |
| serve batched（2 session, chain） | 3072 / 3071 | 16 / 16 | 3087 / 3086 | 771/3 / 771/2 |

### 8.3 速度

R9700 1 枚、3k context、greedy、bf16 KV。

| 条件 | decode |
| --- | ---: |
| DFlash2 chain（drafts 4、修正前） | 47.24 tok/s |
| DFlash2 chain（drafts 4、修正後） | 46.96 tok/s |
| DFlash2 tree（既定 budget 10） | 46.09 tok/s |

replay / restore の追加は round あたり 1 launch と 40 KiB の journal 書き込みで、
round 時間の 1 % 未満に収まった。16k・tree では 75.5 tok/s。

### 8.4 テスト

- `test_qsa_spec_state_replay`（新規）: journal が各行の normed key を記録すること、
  replay が受理 path を直接実行した結果と pooled / partial / counts まで一致すること、
  `create_spec_state_history` の stride、`restore_spec_state` の行復元と範囲外拒否。
- `test_qsa_sparse_attention`: 変更なしで PASS。
- `test_qsa_state_pool` / `test_qsa_key_path` / `test_paged_attention` /
  `test_qwen35_*_selector` / `test_gemm_bf16_*` も PASS。
- 非 QSA model（`models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw` + DFlash2）で
  spec state が no-op のまま動作することを確認（63.79 tok/s、3 桁台の受け入れ）。
