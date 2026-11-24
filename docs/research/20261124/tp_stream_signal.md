# TP StreamSignal barrier

対象: `HipPeerTpTransport::sum_hidden_stream_signal` / `tp_publish_block_writes` /
`TpBarrierGroup::arrive`

## 目的

TP combine の barrier から、host の rendezvous と kernel 内 spin の両方を消す。
前段の device-side fused barrier は EP2 で確率的に生成列が壊れたため不採用にした。

## 方式

各 rank の stream に次を積む。

1. 自分の部分和を peer の inbox へ copy する。copy kernel の最後の block が peer の
   signal に epoch を system-scope release で書く。
2. `hipStreamWaitValue32`（`hipStreamWaitValueGte`）で自分の signal を待つ。
3. inbox を読んで sum / residual add / hyper apply。

- host は barrier で止まらない。
- kernel は spin しない。待ちは command processor が行う。
- copy と add は full grid で走る。fused barrier の「64 block 上限」がない。
- signal と arrival counter は host coherent / device memory に事前確保する。

## 計測

### barrier 単体（`shelf-bench tp-reduce --group`、200 回連続）

| rows | Event | StreamSignal | 旧 fused |
| ---: | ---: | ---: | ---: |
| 1 | 44.0 µs | **30.3 µs** | 38.9 µs |
| 8 | 40.1 µs | **28.6 µs** | 84.4 µs |
| 16 | 47.0 µs | **36.2 µs** | 82.9 µs |
| 32 | 49.5 µs | **42.0 µs** | 82.0 µs |

旧 fused は grid-wide barrier の latency が block 数で増えるため、rows=8 以上で負ける。
StreamSignal は copy / add が full grid なので、verify の形でも勝つ。

### 27B EXL3 TP=2 decode

同一 boot の交互測定: Event 35.2 / StreamSignal 38.4 tok/s（**+9.1%**）。
別 boot では Event 36.5 / StreamSignal 40.1（+9.9%）。
`GENERATED_IDS` は 20/20 一致（sha1
`c75f3eea6473dd56b6e927ca29b4dbe18e5c4597`）。

### 27B + DFlash2（16 token）

同一 boot: Event 67.5 / StreamSignal 70.3 tok/s（**+4.2%**）。
verify は Event 138.7 ms / StreamSignal 134〜135 ms（5 round 合計）。
採択率と生成列は一致。当時「最良」としていた 78.5 tok/s（fence なし）は、
後述の切り分けで生成列が壊れていることが分かったため、性能値として採用しない。

### Flash-Next EXL3 EP=2

Event 31.7 / StreamSignal 33.8 tok/s（**+6.6%**）。
64 token 生成で 10/10 一致した。この時点では copy 側の system fence を有効にしていた。

## signal を kernel から書く

`hipStreamWriteValue32` は stream 順で signal を書くが、copy kernel の write が
peer から見えることは保証しない。fence なしでは 27B 0/5、EP2 8/10 で生成列が分岐した。

当初は copy の末尾に system fence を置いた。ただし全スレッドで実行すると combine
あたり数万回の `__threadfence_system()` になり、decode が 10% 落ちる。最後の block
だけに限定しても 2.4% 残る。

そこで signal 自体を kernel から書く。block ごとに system-scope の arrival counter を
acq_rel で進め、最後に到着した block の 1 スレッドが peer の signal に epoch を
system-scope release で書く。`__syncthreads()` が block 内の write を release より
前へ置くため、signal を見た peer は inbox を読める。`hipStreamWriteValue32` は
`peer_written` の combine だけに残す。

| 方式 | launch / combine | 27B | DFlash2 | 生成列 |
| --- | ---: | ---: | ---: | --- |
| copy + WriteValue32 + 全スレッド fence | 4 | 38.1 | 70.5 | 一致 |
| copy + WriteValue32 + 最後の block fence | 4 | 39.7 | 76.5 | 一致 |
| copy が signal も書く + WaitValue32 | **3** | **41.2** | **77.4** | 一致 |
| copy + WriteValue32 + fence なし | 4 | 40.0 | 78.5 | 不一致 |

採否は launch 数と ISA で確認した。copy kernel の ISA は HEAD の 173 命令から
222 命令へ増える。増分は末尾の 41 命令（barrier、arrival counter、release store）と
追加引数の 8 命令で、末尾は 1 スレッドあたり 1 回しか実行しない。削除した system fence
は全スレッドが毎回実行していたため、動的な命令数と待ちは減る。launch は 4 → 3。
両 family で decode が速いので ACCEPT とする。

これに伴い `TpCombinePolicy`、`fence_copy` / `fence_add`、`SHELF_TP_FENCE_COPY` /
`SHELF_TP_FENCE_ADD` を削除した。fence は family で差がなく、常に不要になったため。

## HIP graph

`hipStreamWriteValue32` / `hipStreamWaitValue32` は graph node として capture できる。
2 rank、write + wait + check kernel の 3 node graph を 200 replay して ordering 不一致 0。
現状の segment graph は compute 区間だけを capture し、signal は eager のままである。
step 全体を capture するには、epoch を device memory から読む必要がある
（`hipStreamWriteValue32` は即値を書くため、capture した epoch が固定される）。

## 78.5 からの退行の切り分け

78.5 tok/s（DFlash2、16 token）は StreamSignal の値である。同じ boot で 78.4〜78.9 が
再現したため、clock の状態ではなくコード差分が原因だった（後に、この 78.5 は
copy fence を切った壊れた経路の値だと分かる）。78.5 の作業 tree を
`git stash` して `baseline-78.5` ブランチに保存し、HEAD の変更を 1 グループずつ
戻して切り分けた。

| 戻した範囲 | decode tok/s |
| --- | ---: |
| HEAD（全変更） | 70.5 |
| auto arena margin 0.8 → 0.5 | 70.5 |
| PLE staging を baseline へ | 70.5 |
| TP グループ全体を baseline へ | **78.0** |

TP グループ内でさらに分けた。

| 条件 | decode tok/s | 生成列 |
| --- | ---: | --- |
| baseline + HEAD の template kernel、copy fence なし | 78.5 | 不一致 |
| 同上、copy fence あり（全スレッドが `__threadfence_system()`） | 70.5 | 一致 |

地雷は copy kernel の system fence だけで、kernel のテンプレート化、backend 選択、
`TpCombinePolicy`、PLE staging、arena margin は性能に寄与していなかった。
ただし fence を切った 78.5 自体が壊れた生成列であり、78.5 を目標にしてはいけない。
fence を最後の block の 1 スレッドに限定すると、正しい生成列のまま 76.5 まで戻る。

PLE の external input は通信機構とは独立した不具合だった。`hipMemcpyAsync` の送信元が
関数ローカルの `std::vector` で、enqueue 後に寿命が切れる。送信元を executor 所有の
pinned staging に移して解消した。

## 採否

ACCEPT。Dense / Qwen4Exp の既定 backend を `StreamSignal` にする。
その他は `Event`。`SHELF_TP_SIGNAL_BARRIER=0` で Event に固定できる。

## 残り

- consumer の prologue に add を融合して、combine あたりさらに 1 dispatch を削る
  （producer 側の signal 融合は完了）。
- step 全体の graph capture は epoch を device から読む方式にする。
