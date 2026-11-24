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

## 8. 現在の利用箇所

- `tests/unit/test_gpu_debug_ring.hip` が受入。
  - 記録の順序と全 field
  - 満杯時の drop と `lost`
  - **kernel が spin 中でもホストが記録を読めること**（同期を入れない）
  - 2 デバイスで別 ring
- production kernel への producer 追加は、対象の kernel を触るときに行う。
  診断対象は `SHELF_TP_FUSED_BARRIER` 系の device-side barrier と
  graph capture 中の dispatch を想定している。
