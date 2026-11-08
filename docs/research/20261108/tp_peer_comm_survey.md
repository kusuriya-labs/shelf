# TP 通信の現状調査

## T3-01: tag 付き peer 通信の導入前調査

結論: 未実装。複数 GPU の実機と非同期エラー伝播を検証できておらず、既存経路の置き換えは行っていない。

`TpBarrierGroup` は全 rank の CPU worker の到着を condition variable で同期し、最後に到着した worker が `sum_hidden` を呼ぶ。空の target だけの場合は通信を省く。通常の peer 経路は各 rank が他 rank ごとに copy kernel と送信 event を enqueue し、他 rank ごとに `hipStreamWaitEvent` を挟んでから add kernel を起動する。TP=2 なら各 rank で copy 1 回、event record 1 回、event wait 1 回、add 1 回である。producer が peer inbox に直接書き込んだ場合は copy を省く。BF16 と F32 を受け、BF16 は F32 で累積して最後に BF16 へ丸める。BF16 の residual add / hyper apply は条件により加算と融合する。ホスト経由は各 rank の D2H と同期、CPU の F32 加算、各 rank の H2D と同期を行う。初期化時の peer 検証やバッファ確保はこの数に含めない。

現行の `TpTransport` に all-gather の呼び出し口はなく、schedule にも `SumHidden` 以外の collective はない。TP=4 の実行は既存の developer 文書上は正式サポート外である。

次の実装では、単一 GPU の複数 stream で世代の繰り返し、遅延 rank、未到着、abort、値の可視性と再利用を検証する必要がある。世代の折り返し、in-flight slot の再利用、非同期 timeout のホストへの伝播を明示しなければ、誤った値の受理や後続 kernel の停止を防げない。GPU 上の timeout 判定を導入する場合も、呼び出し側が結果を検知して後続出力を使用しない仕組みが必要である。

複数 GPU 環境では少なくとも全方向の peer access、fine-grained device allocation の可視性と順序、IOMMU / PCIe の構成、長時間の再利用と障害時の停止、BF16 / F32 の数値、RCCL および従来経路との同条件比較を確認する。いずれも未検証である。
