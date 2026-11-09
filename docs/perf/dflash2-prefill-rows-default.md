# DFlash2 prefill 行数の serving 比較

**判断: 既定化は保留を推奨する。** 測定した生成列は一致したが、入力別の平均受理数が
低下したケースがある。各条件 1 回の観測であり、再現性と実運用への影響を確認してから
既定化を判断する。

2026-11-09、`origin/main` の `c51f57f` を土台とした変更を、R9700 gfx1201、
ROCm 7.14 / LLVM 23、Release、TP=1、BF16 KV、draft 7、temperature 0 で比較した。
target は `/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24`、
drafter は `/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2`。
`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` を指定し、各 GPU 実行の開始から終了まで
個別に `flock /tmp/r9700.lock` を保持した。`rocprofv3` は使っていない。

CMake の configure は `-DCMAKE_HIP_ARCHITECTURES=gfx1201`、
`-DCMAKE_BUILD_TYPE=Release`、指定 toolchain と hipBLASLt を用い、
`nice -n 10 cmake --build build-gfx1201 -j 16` で build した。
以下の `build-gfx1201/rows_*.py` は作業中だけ使用した計測 script で、製品には含めない。
各 script は `shelf-compute --serve-stdio --device 0 --kv-cache-dtype bf16
--dflash2-model-dir "$DRAFT" --dflash2-drafts 7` を起動し、JSON Lines の `generate` と
`shutdown` を使う。実行例は次のとおり。比較では `8` と未設定を交互に実行した。
TTFT の最初の実行は `256` を明示しており、未設定と同じ設定である。

```bash
export ROCM_PATH=/opt/rocm/core-7.14
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
SHELF_DFLASH2_PREFILL_ROWS=8 flock /tmp/r9700.lock python3 build-gfx1201/rows_bench.py 8
SHELF_DFLASH2_PREFILL_ROWS=256 flock /tmp/r9700.lock python3 build-gfx1201/rows_bench.py 256
SHELF_DFLASH2_PREFILL_ROWS=8 SHELF_DFLASH2_SERVE_STATS=1 flock /tmp/r9700.lock python3 build-gfx1201/rows_generate.py 256
SHELF_DFLASH2_SERVE_STATS=1 flock /tmp/r9700.lock python3 build-gfx1201/rows_generate.py 256
```

## TTFT とメモリ

prefix cache 無効、同時 1 request、生成 1 token、各 process で 512 / 2048 / 8192 の順に
2 回実行した。512 の初回は暖機として除外し、2048 / 8192 は両回を含めた。
2 process ずつ、順序は旧・新・新・旧。TTFT は JSON 送信から最初の token 受信までの
wall time で、表は中央値である。arena 記録のため `SHELF_BATCH_TRACE=1` を指定した。
旧の 2048 は 1447.1〜1818.1 ms と変動した。
入力は `/tmp/shelf-prefill-32768.psktok` の 20 byte header 後の int32 ID、
ファイル SHA-256 は `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb`。

| 入力 token | 旧 TTFT ms | 新 TTFT ms | 変化 | 旧 / 新 prefill ms | 測定数（旧 / 新） |
|---:|---:|---:|---:|---:|---:|
| 512 | 449.508 | 356.891 | -20.60% | 449.0 / 356.35 | 2 / 2 |
| 2048 | 1666.912 | 1192.471 | -28.46% | 1665.8 / 1191.7 | 4 / 4 |
| 8192 | 6221.344 | 5073.408 | -18.45% | 6219.1 / 5071.0 | 4 / 4 |

context 構築を囲む GPU event は、一時的に計装した別 binary で計測した。target 側の
実行後と tap の投影・K/V ring 追記後に event を置き、chunk ごとに同期して時間を
合算する。event 間には host が次の処理を enqueue するまでの GPU 待ちも含まれ得る。
この同期を含むため、上の無計装 TTFT とは別測定である。

初回の 8192 token・旧設定では 4 回中 1 回が 1848.640 ms、残りは
1231.423〜1233.742 ms だった。1 chunk の 925.097 ms が外れ値で、原因は未特定。
30% を超える変動があったため、旧・新・新・旧の順で再測定した全結果を下に示す。
各 process で 512 / 2048 / 8192 を 2 回、512 の初回だけ暖機として除外した。

| 入力 token | 旧 event ms（範囲） | 新 event ms（範囲） | 中央値の変化 | 測定数（旧 / 新） |
|---:|---:|---:|---:|---:|
| 512 | 77.250（77.059〜77.442） | 10.706（10.695〜10.716） | -86.14% | 2 / 2 |
| 2048 | 308.191（308.005〜308.827） | 41.575（41.197〜42.411） | -86.51% | 4 / 4 |
| 8192 | 1233.407（1231.899〜1234.943） | 164.067（164.044〜164.535） | -86.70% | 4 / 4 |

再測定は `SHELF_DFLASH_PREFILL_TIMING=1` と一時計測 binary を使い、各 process を
個別に `flock /tmp/r9700.lock` で実行した。一時計装は commit に含めていない。

同じ auto arena、同時 1 request の実測を示す。arena 使用量には KV pool を含むため、
scratch の増分がそのまま総使用量に現れるわけではない。

| 項目 | 旧 | 新 | 差 |
|---|---:|---:|---:|
| feature scratch の増分 | 0 | 17,776,640 B | 16.953125 MiB |
| arena 使用量 | 33,227,545,928 B | 33,227,496,776 B | -49,152 B |
| KV pool | 11,993,612,288 B | 11,975,786,496 B | -17 MiB |
| KV pool の token 容量 | 182,992 | 182,720 | -272 |

scratch は `(256 - 8) × (25600 + 2 × 5120) × 2` byte。
KV は page 単位に丸められるため減少量は scratch 増分と完全には一致しない。
この KV 容量は全 request が共有する物理容量であり、単一 request も `--max-seq-len`、
同時 request の予約と利用可能な KV page に制約される。モデル、KV dtype、arena、
同時数によって減少 token 数は変わる。

## 単独 request の生成と受理

cache 無効、同時 1 request。BetterBench `corpus/v1` の 8 category の先頭 2 件ずつと
`/tmp/prompt_{code,prose,ja}.ids` を使用。BetterBench は指定 tokenizer の
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` で token 化した。
各設定を別 process とし、256 上限は旧→新、512 上限は新→旧で実行した。
停止 token による早期終了を含み、同一入力の生成 ID 全体を比較した。
tok/s は生成数の合計を各 request の送信から完了までの時間の合計で割り、prefill を含む。
受理統計は process 全体の集計で、round には draft なしの round も含み得る。

| 生成上限 | 生成列一致 | 生成 token 合計 | 旧 round / 受理 / 平均 | 新 round / 受理 / 平均 | 旧 / 新 tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 19 / 19 | 3,747 | 974 / 2,754 / 2.828 | 973 / 2,755 / 2.831 | 65.25 / 63.89 |
| 512 | 19 / 19 | 6,476 | 1,728 / 4,729 / 2.737 | 1,725 / 4,732 / 2.743 | 64.14 / 64.47 |

各条件 1 回のため速度差の安定性は未評価。短い prompt での集計結果を、長い context や
他の入力全般における受理品質の保証とみなさない。

## 入力別の受理数

受理数と round は一時計測 build で request 完了時に累積 counter を読み、直前との差から
求めた。同 build と通常 build の生成列は各条件 19/19 件一致した。`decode tok/s` は
server の完了 log に出る値で、上の prefill を含む集計 tok/s とは定義が異なる。
各条件 1 回の測定であり、速度の小さな差の再現性は未評価。平均は受理数 / 全 round で、
提案された draft 数を分母とした受理率ではない。

256 上限の `chat-clarify` は 4.091 → 3.870、`prose-continue-medium` は
1.429 → 1.406 に低下した。512 上限では前者が同じ値、後者が 1.517 → 1.505 に
低下した。合計の平均だけではこの入力別の低下を検出できない。

### 生成上限 256 token

| 入力 | 旧 受理 / round（平均） | 新 受理 / round（平均） | 旧 / 新 decode tok/s | 生成列 |
|---|---:|---:|---:|---|
| chat-followup | 152 / 103（1.476） | 152 / 103（1.476） | 41.3 / 41.5 | 一致 |
| chat-clarify | 90 / 22（4.091） | 89 / 23（3.870） | 85.8 / 82.5 | 一致 |
| code-gen-medium | 215 / 37（5.811） | 215 / 37（5.811） | 114.2 / 115.2 | 一致 |
| code-debug-medium | 192 / 51（3.765） | 193 / 50（3.860） | 79.5 / 81.6 | 一致 |
| edit-refactor-medium | 83 / 23（3.609） | 83 / 23（3.609） | 77.2 / 77.8 | 一致 |
| edit-searchreplace-short | 28 / 5（5.600） | 28 / 5（5.600） | 114.0 / 114.8 | 一致 |
| json-extract-short | 44 / 7（6.286） | 44 / 7（6.286） | 124.7 / 135.3 | 一致 |
| json-schema-medium | 211 / 44（4.795） | 211 / 44（4.795） | 97.6 / 105.6 | 一致 |
| math-word-medium | 212 / 43（4.930） | 212 / 43（4.930） | 99.1 / 107.2 | 一致 |
| math-proof-medium | 209 / 46（4.543） | 209 / 46（4.543） | 96.4 / 100.5 | 一致 |
| prose-story-long | 159 / 96（1.656） | 159 / 96（1.656） | 47.6 / 47.9 | 一致 |
| prose-continue-medium | 150 / 105（1.429） | 149 / 106（1.406） | 43.5 / 43.3 | 一致 |
| reason-plan-medium | 195 / 60（3.250） | 195 / 60（3.250） | 76.4 / 76.6 | 一致 |
| reason-causal-medium | 177 / 78（2.269） | 178 / 77（2.312） | 59.0 / 59.9 | 一致 |
| summ-short-out | 67 / 19（3.526） | 67 / 19（3.526） | 82.0 / 82.2 | 一致 |
| summ-tldr | 33 / 7（4.714） | 33 / 7（4.714） | 105.2 / 105.5 | 一致 |
| prompt_code | 210 / 45（4.667） | 210 / 45（4.667） | 102.1 / 102.3 | 一致 |
| prompt_prose | 168 / 87（1.931） | 168 / 87（1.931） | 52.7 / 52.8 | 一致 |
| prompt_ja | 159 / 96（1.656） | 160 / 95（1.684） | 47.7 / 48.4 | 一致 |

### 生成上限 512 token

| 入力 | 旧 受理 / round（平均） | 新 受理 / round（平均） | 旧 / 新 decode tok/s | 生成列 |
|---|---:|---:|---:|---|
| chat-followup | 311 / 200（1.555） | 311 / 200（1.555） | 42.5 / 42.6 | 一致 |
| chat-clarify | 90 / 22（4.091） | 89 / 23（3.870） | 85.9 / 82.2 | 一致 |
| code-gen-medium | 215 / 37（5.811） | 215 / 37（5.811） | 114.3 / 114.4 | 一致 |
| code-debug-medium | 192 / 51（3.765） | 193 / 50（3.860） | 79.6 / 80.7 | 一致 |
| edit-refactor-medium | 83 / 23（3.609） | 83 / 23（3.609） | 77.5 / 83.4 | 一致 |
| edit-searchreplace-short | 28 / 5（5.600） | 28 / 5（5.600） | 114.3 / 123.0 | 一致 |
| json-extract-short | 44 / 7（6.286） | 44 / 7（6.286） | 124.9 / 134.3 | 一致 |
| json-schema-medium | 381 / 83（4.590） | 381 / 83（4.590） | 92.9 / 100.1 | 一致 |
| math-word-medium | 429 / 82（5.232） | 429 / 82（5.232） | 105.8 / 111.7 | 一致 |
| math-proof-medium | 426 / 85（5.012） | 426 / 85（5.012） | 108.1 / 108.1 | 一致 |
| prose-story-long | 317 / 194（1.634） | 318 / 193（1.648） | 47.0 / 47.2 | 一致 |
| prose-continue-medium | 308 / 203（1.517） | 307 / 204（1.505） | 44.9 / 44.7 | 一致 |
| reason-plan-medium | 411 / 100（4.110） | 411 / 100（4.110） | 90.9 / 91.5 | 一致 |
| reason-causal-medium | 350 / 161（2.174） | 351 / 160（2.194） | 56.8 / 57.1 | 一致 |
| summ-short-out | 67 / 19（3.526） | 67 / 19（3.526） | 82.1 / 82.2 | 一致 |
| summ-tldr | 33 / 7（4.714） | 33 / 7（4.714） | 105.6 / 105.5 | 一致 |
| prompt_code | 420 / 91（4.615） | 420 / 91（4.615） | 100.7 / 100.7 | 一致 |
| prompt_prose | 306 / 165（1.855） | 306 / 165（1.855） | 51.0 / 51.0 | 一致 |
| prompt_ja | 318 / 193（1.648） | 320 / 191（1.675） | 47.0 / 47.8 | 一致 |

## 並列と prefix cache

並列は共通 512 token に code / prose / 日本語 / `chat-followup` をそれぞれ続け、
最大 256 token を生成した。同一 write で 2 または 4 request を投入、prefix cache は無効。
各条件 1 process・1 回で、TTFT は全 request 送信から各最初の token 受信まで。
process ごとの初回であり、前節の暖機後 TTFT と直接比較しない。

| 同時数 | 入力 | 旧 / 新 TTFT ms | 生成列一致 |
|---:|---|---:|---|
| 2 | code | 2361.7 / 2322.6 | 一致 |
| 2 | prose | 2814.5 / 2741.3 | 一致 |
| 4 | code | 2044.4 / 2052.8 | 一致 |
| 4 | prose | 2480.8 / 2474.9 | 一致 |
| 4 | 日本語 | 2917.4 / 2896.8 | 一致 |
| 4 | chat-followup | 3382.2 / 3345.9 | 一致 |

prefix 比較は長さ 328 token、先頭 320 token を共有し末尾 8 token が異なる seed と
extension を使用。cache 有効では seed の後に extension を投入し、無効では extension を
単独投入した。各条件で最大 64 token を生成した。

| 行数 / cache | 復元 token | 再 prefill token | 旧・cache 無効との生成列 |
|---|---:|---:|---|
| 8 / 無効 | 0 | 328 | 一致（64 token） |
| 256 / 無効 | 0 | 328 | 一致（64 token） |
| 8 / 有効 | 320 | 8 | 一致（64 token） |
| 256 / 有効 | 320 | 8 | 一致（64 token） |

これらは測定した入力に限る。target 側の行数や batch 構成に依存する出力差が
別入力で生じないことを保証するものではない。

## テスト

| 検証 | 結果・範囲 |
|---|---|
| `test_dflash2_footprint` / `test_qwen35_spec_verify` | 両方 PASS。既定、8、TP>1、短い window、block size、無効値、scratch 増分を含む |
| `flock /tmp/r9700.lock ./build-gfx1201/tests/test_dflash2_exl3_head` | PASS、relative L2 0.007154 |
| `python tests/server/test_compute_dflash2.py` | BF16 / FP8 KV、66 項目 PASS。非整列の長い prefix の復元を含む |
| `python tests/server/test_server_dflash2.py` | 21 項目 PASS |
| `SHELF_DFLASH2_PREFILL_ROWS=9` を指定した stdio 起動 | `expected 8 or 256`、exit 2 |
| TP>1 | 選択関数の単体テストのみ。複数 GPU の serving 実走は未実施 |

server テストは `SHELF_TEST_DEVICE=0`、`SHELF_TEST_ARENA_GIB=31`、指定 model path と
新規 build を使い、compute binary の wrapper で個々の GPU process を `flock -F` で囲んだ。
HTTP テストは最初に `aiohttp` 不足、次に PIL / torchvision 不足で起動前に失敗した。
build 配下に依存を用意した再実行が上表の PASS である。

CPU テストは次で実行した。

```bash
ctest --test-dir build-gfx1201 --output-on-failure -R '^(test_dflash2_footprint|test_qwen35_spec_verify)$'
```
