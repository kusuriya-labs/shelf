# Shelf / Qwen4Exp PLE先読み・投機Decode改善手順書

**対象ZIP**: `shelf-develop_copen.zip`（ZIP内トップ: `shelf-develop_copen/`、アーカイブのHEAD表示 `1c5376f56e39ac682e388739c08a94eb1cc7ff65`）  
**対象モデル**: Qwen3.8-Flash-Next（Shelf内名称 Qwen4Exp）、PLE FP8 E4M3 または EXL3 trellis、TP2/TP4  
**対象GPU**: AMD gfx1201 / ROCm HIP  
**目的**: PrefillにおけるPLEのCPU計算をGPU Forwardに隠し、DFlash2投機VerifyのPLE準備を安全にバッチ化・先行化する。モデル計算の数値と受理結果を変更しない。

> 本書はコード調査を踏まえた**実装計画**であり、パッチを適用・ビルド・実測した結果ではない。コード断片は設計用の擬似コードで、そのままコンパイルできる完成差分ではない。

## 0. コード調査で確定した現在の状態

| 項目 | 現状 | 参照 |
|---|---|---|
| PLEテーブルの起動時先読み | 対応（各Rank weightロードと平行） | `src/shelf/models/qwen35/runtime/tensor_parallel.cpp`、`src/shelf/models/tp_rank_model.cpp` |
| 推論時PLEの生成 | `PleInputRuntime::prepare()` が**Host Token IDsのみ**受け入れ、sequence slotの履歴を更新しながら1 tokenずつ16 headをGatherする | `src/shelf/models/qwen4exp/runtime/ple_input.cpp` |
| 行デコード | `PleTable::gather_bf16(history, 1)` が行番号生成、16行デコード、BF16変換 | `src/shelf/models/qwen4exp/weights/ple_table.cpp` |
| Executor | `external_input_preparer(batch, host)` を同期呼出し→`hipMemcpyAsync(host→executor.input_hidden)`→`launch_program_run()` | `src/shelf/models/qwen35/runtime/executor.hip`、`include/shelf/models/qwen35/runtime/executor.h` |
| HIP Graph | `external_input_preparer` があればGraph eligibleでない | `src/shelf/models/qwen35/runtime/executor.hip` |
| 投機Decoder | Qwen4Expの**通常の製品経路はDFlash2**。`docs/developer/qwen4exp.md`ではMTP無効と記載 | `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp` |
| DFlash2のToken Bridge | 既定はGPU token IDをTarget VerifyへD2D転送（`SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE=1`） | `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp` |
| PLEとDevice Bridgeの不整合 | `PleInputRuntime::prepare()` は`TokenIdsLocation::Device`に`unsupported`を返す。**PLE有効モデルでのBridge=1は、このままでは互換性がない** | 上記2ファイル |
| DFlash2の状態復元 | DFlash2のpartial rejectはGDNを復元。QSA/PLE Conv historyと復元はDFlash2のOptionsに接続されていない | `dflash2_spec_decoder.cpp` と `mtp_drafter.cpp` の比較 |
| PLE行サイズ | 16 head × 160要素 × BF16 2 byte = **5,120 byte/token** | `docs/developer/qwen4exp.md` |

### 0.1 最重要の正しさゲート

性能最適化に先立って次を解決する。

1. **DFlash2のBridge=1 / PLE Host-onlyの不整合**を修正する。暫定的にはPLE targetの場合だけBridgeを無効化する。
2. **DFlash2のpartial rejectでQSA・PLE Convを復元**する。`mtp_drafter.cpp`の既存`restore_rank_qsa` / `restore_rank_ple`、`build_*_options`を移植の参考にする。新しい履歴バッファのサイズを`rows = K+1`（batchedなら`Σ(Ki+1)`）で予約し、`ExecuteBatchOptions::qsa_spec_history`、`ple_conv_history`に接続する。復元位置はMTP既存実装と同じ**`row_begin + accepted_drafts + 1`**を検証基準にする。
3. 機能の前後で**PLE入力BF16のビット一致**、通常Decode対DFlash2のgreedyトークン一致、full/partial reject後の次ステップ一致を通す。

**注意**: DFlash2のGDN復元コードだけを直しても、QSAとPLE Convの状態整合性は保証できない。PLEのHost-side hash historyのrollbackと、GPU上のPLE Conv stateのrollbackは**別物**である。

---

## 1. 実装フェーズと受け入れ条件

| Phase | 内容 | 終了ゲート |
|---|---|---|
| P0 | PLE profile・互換性修正・QSA/PLE Conv rollbackの正しさ | target-only vs DFlash2一致、Host/Device不整合を解消 |
| P1 | PLE計算をstateless化し、token-major bulk gather導入 | 既存`prepare()`とのPLE BF16 byte一致 |
| P2 | Prefillで次チャンクをワーカーが先行準備 | 1/2/複数request時の正しさ、GPU Forwardとの時間重なり確認 |
| P3 | Pinned Host・転送専用stream・GPU staging二重バッファ | 転送とForwardが重なり、ピークメモリ増分が予算内 |
| P4 | DFlash2提案直後のPLE一括先読みとTP共有 | full/partial/zero accept、複数requestで正常 |
| P5 | 実測に応じてGPU hash/GatherやGraphへの統合を検討 | 数値一致かつwall timeに正味の改善 |

**最初のPRはP0だけ**、続いてP1+P2、次にP3/P4と分離する。PLE改善とQSA性能改善は同じPRに混ぜない。

## 2. P0: 性能測定と正しさ基盤

### 2.1 測定点の追加

場所: `ple_input.cpp`、`ple_table.cpp`、`executor.hip`、`dflash2_spec_decoder.cpp`、TP Rank実行経路。

統計は**per batch / per request / per token**を分ける。

- `ple_hash_us`: hash→row IDs算出（CPU）
- `ple_decode_us`: 16行のデコード・BF16変換（CPU）
- `ple_prepare_wait_us`: `Future`完了待ち
- `ple_h2d_ms`: H2D転送区間（HIP event、転送streamで測る）
- `ple_staging_d2d_ms`: staging→`input_hidden`を導入した場合の追加D2D
- `ple_prefetch_hit/miss/discard`: 有効な先読み結果／再計算／キャンセル
- `ple_bytes`: `rows × hidden × 2`
- `dflash_proposal_wait_ms`: draft GPU→Host待機時間
- `dflash_ple_us`: draft確定からPLE Host準備完了まで
- `dflash_reject_count` と `accept_count`、`qsa_restore_ms`、`ple_conv_restore_ms`

`std::chrono::steady_clock`はCPU区間だけに使用し、GPU区間はHIP eventsで測る。`hipMemcpyAsync`呼出し時間を転送所要時間としない。計測時はwarmup後のp50/p95、GPUクロック・TP・コンテキスト・チャンク固定で比較する。

### 2.2 P0の暫定Bridge修正

`dflash2_spec_decoder.cpp`の単一/バッチ両方で、**PLE external-inputを持つTargetについてはBridgeをOff**。今の`bridge`判定に追加する（擬似コード）。

```cpp
const bool needs_host_ple = target_has_external_ple_preparer(...); // TPの全rankを考慮
const bool bridge = original_bridge_condition && !needs_host_ple;
```

`target_has_external_ple_preparer`は必要なら`Executor`に明示的なcapabilityを追加する。`TP leader`だけでなくrankごとの条件が同じことを起動時に確認する。暫定策として環境変数 `SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE=0` を設定した再現試験も行う。**上記修正を加えずにBridge=1を維持する場合、PLEはDevice token IDsを読めない。**

### 2.3 QSA / PLE Conv rollback修正

- DFlash2の`DFlash2GdnRankState`または別のper-rank structに`QsaSpecHistoryDeviceView`と`PleConvSpecHistoryDeviceView`の所有情報を追加。
- `create_dflash2_spec_decoder`と`create_dflash2_spec_batch_context`で履歴バッファを予約。MTPの`allocate_qsa_history`、`allocate_ple_conv_history`と対応関数が参考になる。
- `build_gdn_options` / `build_batch_gdn_options`に`options[r].qsa_spec_history`、`options[r].ple_conv_history`を設定。
- full acceptではverify済み全行を維持。partial rejectでは、GDNのrestore→commitと同じtransaction内にQSA・PLE Conv restoreを追加する。**行番号は必ずQSA/PLE実装のhistory定義と照合**し、MTP既存の`row_begin + accepted_drafts + 1`を基準にテストする。
- batched verifyで要求ごとの`slot`/`row_begin`を取り違えない。restoreイベントの完了前に同一slotへ次roundを発行しない。

**検証パターン**: K=1,2,4,6、accept=0,中間,全部、QSA 4-token圧縮境界の前後、PLE ngramのEOS境界、TP2/TP4、同一slot再利用。

---

## 3. P1: `PleInputRuntime`を非破壊・一括計算型に分割

### 3.1 なぜ必須か

現行`PleInputRuntime::prepare()`は、`history_[slot]`を伸ばしながら毎token `gather_bf16(history, 1)`を呼ぶ。これをそのまま次チャンク用に別スレッドで呼ぶと、**現在処理中のチャンクと未来チャンクで同じhistory_を書き換える**。キャンセル・rollback・要求混在でも破綻する。

さらに`PleTable::gather(history, out_len)`の複数token出力は、`ngram_rows`内部が**ngram/head groupを外側、positionを内側**として列挙するため、`out_len > 1`をそのまま`[token][hidden]`に見立てて流用してはならない。現行呼び出しは`out_len=1`なので問題が表面化していない。bulk化する際には**token-majorへ明示的に並び替える**こと。

### 3.2 新しい型とメソッド

対象:
- `include/shelf/models/qwen4exp/runtime/ple_input.h`
- `src/shelf/models/qwen4exp/runtime/ple_input.cpp`
- `include/shelf/models/qwen4exp/weights/ple_table.h`
- `src/shelf/models/qwen4exp/weights/ple_table.cpp`

提案の分離:

```cpp
struct PlePrefixSnapshot {
    RequestHandle request_handle;       // slot再利用・世代を含めて識別
    uint32_t slot;
    uint32_t absolute_position;         // このチャンクの先頭position
    std::vector<int32_t> ngram_tail;    // 過去の末尾 ngram_size-1 個
};

struct PreparedPle {
    uint32_t slot;
    uint32_t absolute_position;
    RequestHandle request_handle;
    std::vector<int32_t> input_tokens;  // 候補/チャンクと完全一致を検証
    std::vector<bf16_t> token_major;    // [rows][hidden]
};

// ワーカーで呼ぶ。共有histを変更しない。
Result<PreparedPle> prepare_stateless(
    const PlePrefixSnapshot& snapshot,
    std::span<const int32_t> tokens) const;

// Forward・accept結果で確定したtokenだけを履歴に反映。
Status commit_consumed_tokens(RequestHandle handle, uint32_t from_position,
                              std::span<const int32_t> consumed);
```

`RequestHandle`の実際の比較APIと型は`include/shelf/runtime/request_handle.h`に合わせる。識別キーに単なる`slot`だけを使わない。

**定義**: `absolute_position`は「先読み対象の先頭tokenが、Targetモデルに**入力される**位置」。DFlash2の`pending_token`は既にユーザーへemitされていても、まだTargetがconsumeしていない場合がある。これを区別する。

### 3.3 hashと行順の一致

`ngram_rows`の現行規則（EOS境界、符号変換、乗数、XOR、余りの符号補正）を変更しない。既存のCPU referenceに対して、**token-majorの`[token][head][160]`**で照合する。

- `ngram_size`に必要な末尾だけ保持。長い全履歴を毎回コピーしない。
- `row_ptr`のshard/encoding切替、FP8 scale、EXL3 trellis/biasの経路を維持。
- `float→bf16`は現行の`f32_to_bf16_rne`で**1回だけ丸める**。
- `PreparedPle`のpayloadはimmutable。
- `reserve`等で各tokenの小vector再確保を削減。1回の`[rows][hidden]`バッファを確保し、直接書き込むAPIを追加する。

Reference比較: 現行`prepare()`を**独立したruntimeインスタンス**で実行した結果と、同じPrefix+Tokensから得た`prepare_stateless()`の結果を`memcmp`で照合する。Testは、チャンク境界を1/2/16/256/2048、先頭位置0/1/2/31/32/8192、EOSを跨ぐ列で作る。

### 3.4 キャッシュキー

`{request_handle, slot, position, token_count, exact_token_sequence, model_PLE_version}`を意味論上のキーとする。実装では`slot+generation+position+token hash`で高速索引し、**照合時にtoken IDsを完全比較**する。skip/branch変更時は破棄。

現在のhistoryの寿命とは独立したprepared cacheにする。**`prepare_stateless`は絶対に`history_`を更新しない**。batchの確定行だけを`commit_consumed_tokens`で反映する。

---

## 4. P2: Prefill CPU計算を次チャンク先読み

### 4.1 先読みするタイミング

PrefillのPromptは最初からすべて既知なので、次チャンクのPrefix末尾をオリジナルPromptから直接構成できる。**現在のGPU Forwardの完了を待つ必要がない**。

初回の実装範囲は`dflash2_spec_prefill()`（`dflash2_spec_decoder.cpp`）から始める。この関数は現在`while(offset < prompt_count)`でチャンクを構築し、`run_target_batch()`する。非DFlashの通常Prefillは、続いて`ContinuousBatcher`の`scheduler`→`executor`経路に同等のprepare APIを通す。

```
CPU worker        CPU submit/main               GPU compute
-----------       ---------------------         -------------------
prepare chunk 0
                  submit chunk 0 (PLE ready) ---> Forward chunk 0
prepare chunk 1  ---------------------------> (Forward 0と並走)
                  wait chunk 1 ready
                  submit chunk 1 -------------> Forward chunk 1
prepare chunk 2  ---------------------------> (Forward 1と並走)
...               ...                           ...
```

**重要**: ワーカーに渡すのは`ScheduledBatch`への参照ではなく、token列とprefix tailの値コピー／寿命が保証されたimmutable view。schedulerが次回の行順やrequest集合を変更する場合、キャッシュキーが違えば再利用せず破棄する。

### 4.2 推奨構造

- `PlePrefetchService`: CPU worker 1本から開始。並列度は実測に応じて2/4を試す。CPUメモリ帯域・cache挙動を悪化させない。
- ahead window: **次の1チャンク**だけ（double-buffer）。たくさん先読みするとCPU RAMとNUMA bandwidthを浪費する。
- `PrefetchTicket`: `future<PreparedPle>`、request handle、position、token IDs、cancel/generation。
- `get_or_prepare`: 完成済み結果を取り出す。未完成なら待つ。キー不一致・errorなら現行同期prepareにfallback。
- `PleInputRuntime::prepare()`は互換APIとして残し、先読み成果物が渡された場合のみそれを使用。**preparer lambda内部から重ねてGatherしない**。

擬似コード:

```cpp
// chunk0を同期で作る。その後、GPU forwardをsubmitする前にchunk1の準備をenqueue。
current = ple.prepare_stateless(prefix0, tokens0);
for (each prefill chunk i) {
    auto next = (has_next) ? pool.submit([snapshot_next, copied_tokens_next] {
        return ple.prepare_stateless(snapshot_next, copied_tokens_next);
    }) : empty_ticket;

    // 既存の呼出し境界はPreparedPleを渡せるように拡張すること。
    run_target_batch_with_prepared_ple(batch_i, current);

    if (has_next) current = next.get();
    // 主経路のForwardがconsumeしたtoken列についてcommitする。
}
```

上は**処理順の説明**であり、`run_target_batch_with_prepared_ple`は未実装の拡張API。現状の`ExternalInputPreparer`は「`ScheduledBatch`だけからstd::vectorを生成する」契約なので、`ExecuteBatchOptions`のfamily-neutralな`PreparedExternalInputView`、または`ExternalInputPreparer`の拡張でprepared dataを受け渡す。

### 4.3 Executorへの接続

避けるべき設計: グローバルな「次のPLEバッファ1個」を`PleInputRuntime`に置き、各TP rankが勝手にconsumeする設計。TPの複数rankで読み出し位置がずれ、競合する。

提案:

1. scheduler/coordinatorが**論理バッチID＋対象requestの列**に対してPreparedPleを作成。
2. `TpCoordinator`の各rankに**同じimmutable prepared payload**を渡す。host PLE gatherは1回、GPU H2Dはrankごとに1回。
3. `Executor::enqueue_batch()`が既存の同期preparerよりprepared payloadを優先し、サイズ・handle・position・tokensの一致を検証。
4. 一致しなければ明示的にmissとして同期fallback。未検証のバッファを送らない。
5. rank間のready/consumptionはatomic/refcount/shared_ptr等で管理し、Hostバッファは最後のrankのH2D完了イベントまで生存。

### 4.4 バッファ容量（Qwen4Exp hidden=2560）

| chunk rows | PLE入力BF16 / 1 buffer | double bufferのHost容量 |
|---:|---:|---:|
| 256 | 1.25 MiB | 2.5 MiB |
| 512 | 2.5 MiB | 5 MiB |
| 1024 | 5 MiB | 10 MiB |
| 2048 | 10 MiB | 20 MiB |
| 4096 | 20 MiB | 40 MiB |

TP2/TP4で**Host prepared payloadは共有**。GPU stagingを後付けするなら**GPUごと**に上記と同容量が必要となる。

---

## 5. P3: Pinnedメモリ＋転送streamでH2Dも先読み

P2だけでは「CPU計算」を隠せるが、通常の`executor.input_hidden`へのH2DはForward直前のまま。P3はオプションであり、**測定でH2Dが無視できない場合のみ着手**する。

### 5.1 安全な最小構成

- Host: `hipHostMalloc`等でallocateしたPinned staging 2枚。ワーカーのvectorからPinnedへコピーするならそのcopy時間も測る。
- GPU: 次チャンク用staging `d_ple_stage[2]`をrankごとに確保（アドレス固定）。
- Stream: `transfer_stream`、`compute_stream`（既存execute stream）。
- Event: `copy_done[i]`、`consumed[i]`。同じslotを再利用する前に、前利用のGPU読み取り完了を確認。
- `transfer_stream`に `hipMemcpyAsync(d_ple_stage[i], h_pinned[i], bytes, hipMemcpyHostToDevice, ...)` → `hipEventRecord(copy_done[i])`。
- `compute_stream`で`hipStreamWaitEvent(copy_done[i])`し、その後`d_ple_stage[i]→executor.input_hidden`のD2Dコピー→`launch_program_run`。`consumed[i]`は**実際にGPUが入力を読み終えたことが確定する地点**より後に記録。初期実装はForward全体の完了イベントでもよい。

```
transfer stream :  H2D(chunk i+1) ---- record ready(i+1) ----
compute stream  :  Forward(chunk i) -- wait ready(i+1) -- D2D -- Forward(i+1)
```

**追加D2Dの分だけ遅くなる場合がある。** 最終形は`Executor`のExternal Input bindingを2枚のGPU bufferへ切り替えてD2Dを消す方式が望ましいが、programのworkspace pointer・Graph captureと衝突しないようにする。現状はPLE有効時HIP Graph無効なので、その仕様のまま先に進める。

### 5.2 バッファ寿命の落とし穴

- Page-lockedでない`std::vector`を`hipMemcpyAsync`のsourceとして投げ、関数終了後に破棄しない。既存`executor.hip`のローカル`std::vector<bf16_t> host;`は先読みの長寿命バッファに**そのまま流用しない**。
- CPU workerがGPUへDMA中のPinned領域を書き換えない。
- Rank1とRank0は異なるGPU streamを持つため、同一`hipEvent`をデバイス横断で当然に共有できると仮定せず、**rankごとのイベント**を作る。
- `hipSetDevice`/`ScopedDevice`を各GPUで適切に使用する。
- request cancelやshutdown時に全inflight copy/Forwardの完了を待ち、**stagingを先にfreeしない**。

---

## 6. P4: DFlash2投機DecodeのPLE先読み

### 6.1 計算上の限界

DFlash2 proposalがGPUで`[draft0, ..., draftK-1]`を生成するまで、それらに依存するPLE hashは確定しない。CPU RAM上のPLEテーブルの実行時先読みを行う場合、**draft生成より前に正確なK行すべてをGatherすることはできない**。

ただし、`pending_token`（anchor）は既知なので、**row0のPLE**だけはdraft中に計算できる。GPU側でdraftが完成した後、token IDsを一括D2Hし、残りK行のPLEを一括Gatherすれば、従来の1 tokenずつの手順よりallocation/分岐/TP重複を減らせる。さらにGPUの独立作業（GDN snapshotなど）と並走できる余地がある。

### 6.2 Host Bridgeに先行処理を入れる（安全な初期実装）

対象:
- `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp`
- `include/shelf/models/qwen35/runtime/dflash2_spec_decoder.h`
- `src/shelf/models/tp_rank_model.cpp`
- `src/shelf/models/qwen4exp/runtime/ple_input.cpp`

単一request:

1. `position = decoder.sequence->position`と`request_handle`をsnapshot。`pending_token`とその直前の`ngram_size-1` tokenのtailを取得。tailは**Targetがconsume済みの履歴**から取る。
2. CPU workerへ**row0（anchor）**のGatherを投げる。
3. GPUで`dflash2_propose_cached`を実行。
4. Host bridgeでproposal tokensを1回D2H（現状の`bridge=0`経路に従う）。
5. `[pending_token, drafts...]`のexact token列が確定したら、row1..Kを一括prepare。row0と連結して`PreparedPle`を完成。
6. `spec_gdn_snapshot`とPLEの後半CPU prepareを可能なら非同期に進め、Verify実行直前にjoin。
7. `make_verify_batch`に同じ`[anchor,drafts...]`を渡す。PreparedPleのrow数・順序と完全一致することを検証。
8. Target VerifyのPLE external inputとして渡す。**内部から`PleInputRuntime::prepare()`をもう一度呼ばない**。
9. acceptance完了後、**Targetがconsumeした`[anchor, accepted_drafts]`だけ**をPLE committed historyへ反映。correction/bonusは次roundのpendingとして扱う（未consume）。
10. rejectしたdraft末尾のprepared dataは破棄。次roundへ持ち越さない。

**行数例**: `K=6`なら7行×5,120byte = **35,840byte/round/request**。一括H2Dに十分小さいが、GPU proposalをD2Hするhost同期が再びcritical pathへ入る点に注意。

### 6.3 Batched DFlash2（serveで必須）

`dflash2_spec_step_batch()`は1バッチにrequestごとの`[anchor,drafts...]`を詰め、各requestの`row_begin`を持つ。PLEも**同じrow_begin/num_tokens**で作る。

```text
request A: pos=100, K=2 -> rows 0..2: [pendingA, A0, A1]
request B: pos=990, K=4 -> rows 3..7: [pendingB, B0, B1, B2, B3]

PLE prepared rows:  [A row0,row1,row2,B row0,row1,row2,row3,row4]
```

- 各requestのprefix snapshotを混ぜない。
- `RequestHandle`の世代を検証。
- `Σ(Ki+1)`が`context.max_rows`を超えないこと。
- batchの部分棄却時は各slotについて`position + accepted + 1`で履歴を確定。**batch全体のverify末尾までcommitしない**。
- 一件だけcancelされた場合も、他requestのPreparedPleは破棄しない。skipしたrequestのrow_beginを再計算する。
- TP2/TP4で同じHost prepared payloadを共有し、各rankへ転送。rankごとのPrepareを廃止する。

### 6.4 将来的にDevice Bridgeを戻す場合（選択肢を明確に）

**案A: D2H→CPU Gather→H2D（現実的な第一候補）**

- DFlash出力token IDsをHost Pinned bufferへ一括D2Hする。
- GPU eventで準備完了を知らせる。
- CPU workerがPLE hash→Gather→BF16作成。
- Host Pinned→GPUを転送streamで開始し、compute streamがevent waitしてVerifyする。
- GPU提案末尾まで候補は確定しないため、このD2H同期を完全に消せるわけではない。
- CPU Gatherと、独立したGPU処理（snapshot, descriptor準備など）は重畳し得る。依存関係のあるVerifyそのものとは重畳できない。

**案B: 完全GPU側でPLEを生成**

- token IDはGPUのまま。GPU kernelでN-gram hashを計算し、所要行をGather/デコードする。
- しかし、現在PLEテーブルは巨大な**CPU RAM上の単一共有テーブル**。全部をGPU VRAMへ持つ前提は置かない。PCIe越しrandom gather/UVA直接アクセスは高コストの恐れ。
- 本方式を採るなら、GPU側の頻出行cacheとmiss時のHost側Gatherを持つ二段方式か、モデルフォーマット別に常駐可能容量を測る実験を先に行う。
- GPU cacheは**hash row ID→embedding row**の一致・eviction race・multi-rank複製コスト・cold miss性能を測定してから採否判断。

**推奨は案A**。Device bridge有効化はP0/P4の正しさゲート通過後、ベンチの結果を見て判断する。

### 6.5 DFlash2の状態rollbackを必ず同時に検証

partial reject時、`PleInputRuntime`のHost履歴は「次回にpositionが縮んでいたら`resize(position)`」だけでも一見直るが、GPU側PLE Conv stateは別管理。**Host履歴だけを修正して完了にしない**。

`mtp_drafter.cpp`の既存実装ではpartial rejectで、GDNに加えて以下を復元する:

```cpp
restore_rank_qsa(log_ranks[ri], sequence.slot,
                 row_base + accepted_drafts + 1u);
restore_rank_ple(log_ranks[ri], sequence.slot,
                 row_base + accepted_drafts + 1u);
```

DFlash2へ相当の履歴保存・復元を追加し、境界またぎ（QSA 4-token圧縮・PLE conv window）でもTarget-onlyと結果が一致することを確認する。復元後の**次roundのlogitsと状態**を比較することが重要。

---

## 7. 実装で追加するテスト

### 7.1 CPU単体: `test_qwen4exp_ple_prefetch.cpp`（新規）

- `PleTable`の実tableを使わない小型synthetic table（APIが許す場合）と、実checkpoint双方。
- `prepare_stateless` vs 現行`PleInputRuntime::prepare`の**BF16バイナリ一致**。
- token列が同じでもdifferent prefixならcache miss。
- EOS境界、ngram不足でのpad、異なるtoken ID、negative sentinel取扱い。
- 位置0開始、分割チャンクサイズが1/2/16/256/2048。
- CPU workerの中断、例外/Status伝搬、slot再利用、同じhandleでもtoken列変化。
- `gather_bf16(history, out_len>1)`のhead-majorを誤って使わないことを直接テスト。

### 7.2 GPU連携: `test_qwen4exp_ple_async.hip`（新規）

- prepared payload→GPUの`executor.input_hidden`のbit一致。
- 2つのprefill chunkが交互に走り、前回のH2D/Forwardが終わる前にbufferを上書きしない。
- 1GPU、TP2、TP4でinputが一致。
- `request cancel`→新requestで同じslotを使っても汚染しない。

### 7.3 DFlash2連携: `test_dflash2_qwen4exp_ple_verify.hip`（新規）

- Host bridge / 非投機Target-onlyを比較。
- K=0,1,2,4,6; accept=0,1,...,K; 強制的にdraftを不一致にするテストモードを設ける。
- partial reject直後の**次round**と、QSA pooled state/PLE Conv state/GDN stateの一致。
- K+1がページ境界を跨ぐ場合・コンテキスト32K/64K位置のlong history。
- batched 2〜8requestでrow_beginが異なる場合。
- `SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE=0`経路が正常に動き、Bridge=1も明示的にHost fallbackされるか、GPU対応が完成していれば動く。
- EOS/stop token・cancel・sequence slot recycle、TP2/TP4。
- 現行`docs/developer/dflash2.md`にあるQwen4Expの`drafts=6`・64token一致アンカーは**別途既知問題を踏まえた参考**。今回の変更で新たな分岐を作らない。

### 7.4 既存の回帰テスト（リポジトリ内で名前が確認できたもの）

```bash
./build/tests/test_qwen4exp_ple_site
./build/tests/test_qwen4exp_lower
./build/tests/test_qwen4exp_layer
./build/tests/test_ple_gate
./build/tests/test_ple_conv
./build/tests/test_dflash2_mixed_batch
./build/tests/test_qwen4exp_load       # 実checkpointを使う任意テスト
```

ビルド構成により生成パスや任意テストの有無は異なるため、`cmake --build build`後に`ctest --test-dir build -N`で確認する。

---

## 8. ベンチ設計と合格基準

### 8.1 A/Bする実装構成

| Variant | Prefix Prefill | DFlash2 Verify | 目的 |
|---|---|---|---|
| B0 | 現行同期PLE | PLE Host bridge、rollback修正済 | **正しい基準線** |
| B1 | P1のbulk gatherのみ | 同じ | bulk化の単独効果 |
| B2 | B1 + CPU one-chunk ahead | 同じ | CPU準備のhide量 |
| B3 | B2 + pinned/H2D先読み | 同じ | PCIe転送のhide量 |
| S1 | B2 | draft確定後、一括PLE・TP共有 | speculative改善 |
| S2 | B3 | D2H/CPU Gather/H2Dをeventでpipeline | specでのhide量 |

### 8.2 条件

- GPUはR9700（gfx1201）、TP2を主、TP4を追加。GPU隔離、温度/クロック固定傾向。
- Prefill: 8K / 32K / 64K / 128K context。chunk 256 / 512 / 2048。warmとcoldを区別。
- Decode: 通常 vs DFlash2、drafts=1/2/4/6、accept率バケット（0%, 部分, 全部）、context 8K/32K/64K。
- concurrency: 単一requestとserve 2/4/8request。
- TP2で同一入力を5〜10run交互実行しp50,p95、tok/s・end-to-end wall・TTFT・time/round・GPU vs CPU breakdownを保存。
- **prefillのみの計測と、prompt長込みのend-to-end生成を分離**。最適化前後でactual input/output tokensが同じこと。

### 8.3 数値と性能の合格条件

**Correctness（必須）**:

- PLE external input: prefetch OFF/ONで全行`memcmp==0`（BF16 bit-level）。
- QSA/PLE Conv/GDN restored state: partial reject後もbaselineと一致。
- Greedy生成: Target-only vs投機（既存の既知不一致がある場合はそのケースを分離し、**新たな不一致を追加しない**）。
- Stochastic: 乱数消費回数・sampling indexまで揃えて比較。単純な生成テキストだけを唯一のゲートにしない。
- MSan/ASan相当のホスト側寿命チェックとHIP非同期終了、キャンセル・再利用を通す。

**Performance（提案ゲート。目標値であり現状実測ではない）**:

- B1: bulk gather CPU時間に回帰なし。
- B2: 単一Prefillのwallで**5%以上**改善、またはCPU準備待ち時間を**50%以上**削減。どちらも有意でなければP2は複雑化に見合わない。
- B3: B2から全体wallが追加**3%以上**改善しないなら無理に採用しない（追加D2D等による損失に注意）。
- S1/S2: 平均decode tok/sだけでなく低accept率/高concurrencyでのp95 latencyの悪化がない。
- 追加Pinned/GPUバッファのメモリ予算とNUMA帯域を監視。OOM/割当回数増大がない。

計算の上限: PLE準備が1チャンクで`Tple`、GPU本処理が`Tgpu`、独立部分を完全に重畳できても、改善できるのは概ね`Tple`由来の待機まで。実測で`Tple/Ttotal`が小さい場合、長文での大きな減速はIndexer/QSAやAttention側に残る。

---

## 9. 実施順のPR分割（作業チケットに転記可能）

### PR-01: Qwen4Exp DFlash2の正しさゲート

- [ ] PLE Host-only target時のDevice bridgeを明示fallback
- [ ] DFlash2 QSA/PLE Conv state history確保/Options配線/partial restore
- [ ] rollback / K=0 / K=6 / TP2 / batchedのテスト
- [ ] ログ項目を追加（Bridge chosen、Restore result）

### PR-02: PLE bulk/hash API

- [ ] Snapshot/token-major prepared buffer API追加
- [ ] EOS/multiple tokens/EXL3/FP8の参照一致
- [ ] 既存`prepare`を互換維持して遷移
- [ ] `history_`への先行書込みを排除、commit/rollbackを明文化

### PR-03: Prefill CPU one-chunk-ahead

- [ ] DFlash2 prefill loopに先読みチケット導入
- [ ] ExecutorへPreparedExternalInputを渡す経路追加
- [ ] TPでHost PLE計算を1回に集約
- [ ] 通常ContinuousBatcherのprefill経路へ拡張
- [ ] cancellation、slot recycle、`prefetch_hit/miss`ログ追加

### PR-04: Prefill transfer overlap（性能があれば）

- [ ] pinned double buffer / rankごとのtransfer stream・event
- [ ] 専用GPU staging、D2Dの追加損失を計測
- [ ] Input bindingのping-pong化を別A/B
- [ ] shutdownやerror pathにevents/drain追加

### PR-05: DFlash2 speculative PLE batch prefetch

- [ ] anchor PLEの先行準備
- [ ] Drafts D2H後、remaining PLE rows一括準備
- [ ] GDN snapshot等の独立作業とCPU prepをoverlap
- [ ] batched request/Tensor Parallelの共通prepared payload
- [ ] partial rejectのprefetch破棄・次roundの再利用条件をテスト

### PR-06: Advanced GPU path（任意）

- [ ] GPU上でのhash/row-ID計算のmicro-bench
- [ ] row cache residency/miss時Host fallback設計
- [ ] VRAM上限とPCIe帯域評価、Device bridgeとの統合
- [ ] Graph再有効化は外部Input stagingの動的pointer/shapeが安定した後に別PR

---

## 10. 失敗しやすいポイント（実装レビュー用）

1. `PleTable::gather_bf16(history, N>1)`の結果をtoken-majorと誤認する。
2. `PleInputRuntime::prepare()`を未来チャンクのworkerでそのまま呼び、`history_`を先に変更する。
3. DFlash2の**emitted token**とTargetが**consumedしたtoken**の違いを無視する。pending correction/bonusを次roundより先にhistoryへ入れる。
4. `TokenIdsLocation::Device`をHostポインタとしてdereferenceする。
5. TP各rankが同一PreparedPleを**consume時に書き換える**、またはrank0 copy完了直後にHost payloadを解放してrank1のDMAがまだ終わっていない。
6. pinned source/DMA対象の寿命、stream/event順序、GPU stagingへの上書きを取り違える。
7. partial reject時にHost hash historyだけ戻し、GPU PLE ConvとQSA partial stateを戻さない。
8. Request slotだけでcache hit判定し、世代・位置・token列を見落とす。
9. 計測をtok/sだけで行い、Prefill CPU時間・GPU実行・DFlash D2H/PLE/H2Dの内訳を見ない。
10. 改善が見えないのに、PLE用にGPU常駐table・Graph化まで一度に実装する。

## 11. 現実的な推奨着手点

**まずPR-01 → PR-02 → PR-03 → PR-05の順。** PR-04（転送stream化）は数値の分解計測でH2Dが有意なら追加する。`Qwen4Exp MTP`を最初のターゲットにせず、実際の主経路である**DFlash2単一/バッチVerify**へ着手する。QSA/PLE Conv state rollbackを修正しないまま性能だけ追わない。
