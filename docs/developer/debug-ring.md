# GPU Debug Ring

GPU 側の状態を、ホスト同期を一切入れずに観測するための診断用 ring。
`include/shelf/core/gpu/debug_ring.h` が contract の正本。

## 1. 目的と非目標

目的:

- kernel の中の値を、`hipStreamSynchronize` / `hipMemcpy` を挟まずに読む。
- 停止した実行からでも、epoch / flag / index などを事後回収できる。

非目標:

- 同期機構。順序保証や barrier には使わない。
- ホットパスの telemetry。有効化したときだけ使う。

## 2. レイアウト

- ヘッダ 64 byte（`head` / `tail` / `lost` / `capacity` / `producer_device` / `flags`）と
  64 byte 固定長の `DebugRingRecord` を 1 個の `hipHostMalloc` 領域に連続配置する。
- 確保フラグは `hipHostMallocMapped | hipHostMallocCoherent`。
  デバイスからもホストからも同じ実体を見る。
- capacity は 2 の冪。既定 1024 slot（64 KiB + 64 byte）。

`DebugRingRecord` の意味は呼び出し側が決める。`sequence` だけ transport が所有する。

## 3. 並行性

- 単一 producer（device 側の 1 スレッド）、単一 consumer（ホストの 1 スレッド）の SPSC。
- 複数 kernel が同時に push してはならない。stream 順に実行される kernel なら、
  時間的に排他なので同一 ring を使える。
- rank ごとに ring を作る。1 本の ring を複数デバイスで共有しない。

push は満杯でも待たない。

- `tail - head >= capacity` なら記録を捨て、`lost` を 1 増やして `false` を返す。
- 最も古い未読 slot は上書きしない。

## 4. memory order

- device は system scope の atomic で `head` / `tail` / `lost` を読み書きする。
  payload の通常 store の後に `tail` を release store する。
- host は `std::atomic_ref<uint64_t>` で `tail` を acquire load、`head` を release store する。
- ホスト側の native atomic 非対応は fail-closed
  （`hipDeviceAttributeHostNativeAtomicSupported`）。

## 5. ライフサイクル

1. `DebugRing::create(device, capacity)` で確保する。producer のデバイスを current にして作る。
2. `device_view()` を kernel 引数へ渡す。使わないときは nullptr の view を渡す。
3. `try_pop` / `drain` / `pending` でホストが読む。HIP API は呼ばない。
4. producer を停止してから `shutdown()` する。destructor は最終フォールバック。
5. `shutdown()` は `hipHostFree` を 1 回だけ実行する。二重呼び出しは無害。

producer が動いている間に `shutdown()` してはならない（解放済みメモリへ書く）。

## 6. 停止した実行からの読み出し

ring はホスト VA 上にあるため、プロセスがハングしていても外部から読める。
`/proc/<pid>/mem` で `head` / `tail` と slot を読む経路が原理的に使える。
device memory の ring ではこの経路が使えない。

## 7. コスト

- `device_view()` が nullptr のとき、kernel 側はポインタ 1 個の分岐だけで終わる。
- 有効時は push ごとに system scope の store とホストメモリへの 64 byte 書き込みが入る。
  hot loop に置かない。

## 8. 実戦投入: TP transport の観測

融合 collective の調査で必要になった「barrier ごとの送信を、同期を入れずに記録する」を、
実際に TP transport へ入れた例である。

### 8.1 配線

- `SHELF_TP_DEBUG_RING=1` のとき `HipPeerTpTransport::create` が **rank ごとに 1 本**
  （4096 slot）作る。
- `tp_copy_bf16_kernel` / `tp_copy_f32_kernel` の `blockIdx.x==0 && threadIdx.x==0` が、
  barrier ごとに 1 件 push する。
- 記録の意味:

| field | 内容 |
| --- | --- |
| `kind` | 1 = 送信 |
| `rank` | 送信元 rank |
| `stage` | parity（0 / 1） |
| `value0` | `rows` |
| `value1` | `row_stride` |
| `value2` | `feature_count` |
| `data[0]` | 送信要素数 |
| `sequence` | ring が採番 |

- transport の self-test（`verify()`）中の push は `verified_` で除外する。
  そうしないと起動時の probe が本番の列に混ざる。
- transport の破棄時にホストが drain して、rank ごとに 1 行を stderr へ出す。
  実行中に同期を足さないため、記録は「終わってから」読む。

### 8.2 出力例

27B EXL3 TP=2、`--context 64 --tokens 8`:

```text
[tp-debug-ring] rank=0 records=1427 lost=0 seq=0..1426 parity0=714 parity1=713 rows=64 stride=5120 features=5120 shapes=1298
[tp-debug-ring] rank=1 records=1427 lost=0 seq=0..1426 parity0=714 parity1=713 rows=64 stride=5120 features=5120 shapes=1298
```

読み方:

| 見る場所 | 意味 |
| --- | --- |
| `records` が 2 rank で一致 | 各 rank が同じ barrier 数だけ送っている |
| `parity0` / `parity1` がほぼ同数 | double buffer が交互に使われている |
| `lost=0` | ring 容量が足りている（足りなければ捨てた数が出る） |
| `shapes` | `rows` が変わった回数。prefill / decode の切替で増える |
| 片 rank だけ `records` が少ない | その rank が barrier に到達していない = 詰まりの第一証拠 |
| `lost=0` なのに値が合わない | slot 選択か可視性の問題（記録の欠落ではない） |

### 8.3 計測したコスト

27B EXL3 TP=2、ctx64 / tg8、同一 boot・同一 build、off/on を交互に 7 組:

| 条件 | tok/s（7 回） | 中央値 |
| --- | --- | ---: |
| off | 36.60 / 36.35 / 36.45 / 36.54 / 36.55 / 35.70 / 36.40 | **36.45** |
| on | 36.26 / 36.07 / 36.36 / 35.40 / 36.28 / 36.20 / 35.67 | **36.20** |

中央値で **−0.7 %**。1 run あたり 1427 件の push で、コストは barrier ごとの
64 byte ホスト書き込み 1 回である。診断専用の opt-in として許容する。

### 8.4 注意点

- `peer_write_complete` の barrier は copy 自体を省くため記録されない。
  全 barrier を記録したい場合は add kernel 側にも同じ push を足す。
- copy / add は `arrive` の外（graph capture の内側ではない）なので、
  graph replay でも記録引数は固定されない。
- 常時有効にしない。未設定なら `device_view()` は nullptr で、kernel 側は分岐 1 個。
- ring はホスト VA にあるため、プロセスがハングしても `/proc/<pid>/mem` から
  読める（本節の実測はプロセス内 drain。外部読み出しは未検証）。

### 8.5 次の対象

`SHELF_TP_FUSED_BARRIER` の device-side barrier を再評価するときは、同じ形で
fused kernel から `epoch` / `parity` / `own` / `peer_dst` を push する。
同期を入れずに、停止した状態から原因を確定できる。

