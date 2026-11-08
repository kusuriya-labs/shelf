# TP=2 の peer flag 経路

`SHELF_TP_PEER_FLAG=1` を指定した場合、HIP peer transport の TP=2 で試験的な経路を選択する。既定は従来の event 経路である。生成元の fusion と peer write fusion はこの経路では無効にする。BF16 と F32 の両方を扱う。prefill の all-gather は既存の transport に口がないため実装していない。

各 rank は自分の stream で相手の fine-grained inbox に kernel store し、全 thread の system fence と block 内同期の後、64 bit の世代 tag を system scope の release store で通知する。受信 kernel はローカル flag を system scope の acquire load で待ち、到着後に inbox を加算する。データ転送に SDMA は使用しない。inbox と flag は `hipDeviceMallocFinegrained` で確保する。世代は 1 から増加し、0 は未通知を表す。最大値に達した場合は折り返さず transport を abort する。

両 rank の送信を先に enqueue し、その後に受信を enqueue する。各受信の完了を host で同期してから次の世代に進むため、2 slot の再利用時に遅い rank の読み出しと次の書き込みが重ならない。待機時間の目安は device の公称 clock rate に基づく約 5 秒で、timeout は device 側の error を host へ読み出して通知し、それ以降の transport 利用を拒否する。CPU 側 barrier でも rank の到着を最大 30 秒待ち、未到着なら abort する。GPU の停止や driver 障害そのものを kernel 内の timeout で回復することはできない。現状は 1 block の転送と host 同期を使うため性能上の利得は未検証である。

TP=4 に拡張する場合は送信元ごとに inbox と世代 flag を設け、すべての送信を投入してから受信し、全 rank が読み終わるまで slot を再利用しない必要がある。待機と通信量の増加、peer topology と system scope の可視性は別途検証が必要であり、現実装は TP=4 に適用しない。

## 複数 GPU での確認項目

- 両方向の peer access と peer store の成否。
- fine-grained inbox と flag の可視性、system fence と release/acquire の順序。
- IOMMU の設定、PCIe topology と経路。
- 世代と slot の長時間再利用、および遅い rank の影響。
- 相手の未到着、timeout 後の abort、GPU 障害時の挙動。
- BF16 と F32 の数値を従来経路と bit 単位で比較。
- 同一形状、同一同期条件で従来経路と RCCL を比較する。RCCL 比較は別途用意する。

`./build-gfx1201/shelf-bench tp-reduce --backend p2p --compare-flag --rows 1,4,16,64 --features 5120` と `--features 2560` はそれぞれ Qwen3.8-27B と Flash Next の hidden に対応する。`--dtype f32` で F32 も選べる。既存 benchmark の測定で、host 同期を含む 1 回あたりの時間を表示する。RCCL の benchmark は含まない。複数 GPU 実行は未検証。
