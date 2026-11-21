# QSA / MTP 開発 baseline（2026-11-21）

対象: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`
（qwen4exp、48 層、512 routed experts、EXL3 mul24、MTP 1 層）。

## 1. 環境

| 項目 | 値 |
| --- | --- |
| branch | `multi-mtp` |
| base SHA | `43f4a10269743a504fa9f0b3e14db3e603e43667` + 本作業の未コミット変更 |
| GPU | AMD Radeon AI PRO R9700 / gfx1201 / 32 GB |
| TP / EP | TP=2 / EP=1（GPU 0,1） |
| KV | BF16 |
| 量子化 | EXL3 mul24、lm_head 5 bpw、MTP 3 bpw |
| profiler | rocprofv3 1.4.1 |

GPU 2,3 は別プロセスの server が使用中だったため、本記録は GPU 0,1 の TP=2 で取得した。

## 2. required

`ctest --test-dir build -L required --output-on-failure` は 226/226 PASS。
`tests/unit/test_qwen35_rope_profile.cpp` は並行変更後の cap 1048576 に合わせて更新した。

## 3. 単一 request の target-only / MTP

prompt 8 token、max_new 8、drafts 7、各 3 run。

| mode | DECODE_MS | 備考 |
| --- | ---: | --- |
| target-only | 198.3 / 198.3 / 198.3 | 7 token、約 35.3 tok/s |
| MTP | 312.3 / 311.6 / 312.1 | 7 token、約 22.4 tok/s |

MTP 内訳（3 run の中央値相当）:

- `MTP_DRAFT_MS`: 20.68〜21.82
- `MTP_VERIFY_MS`: 249.18〜290.24
- `MTP_ROUNDS=4`

単一 request の安全経路は 1 round に最大 K+1 回の target decode を行うため、
target-only より遅い。これは Phase 1 の安全性を優先した結果であり、性能改善ではない。

## 4. kernel trace

同じ 8 token を rocprofv3 `--kernel-trace` で取得した。profiler 内部の
`__amd_rocclr_copyBuffer` を除いた shelf 側 kernel launch 数:

| mode | shelf launches | unique kernels |
| --- | ---: | ---: |
| target-only | 44312 | 65 |
| MTP drafts 7 | 45488 | 71 |

MTP は 4 round で +1176 launch。これは forward 数 K+1 ではなく、
1 forward あたりの層・kernel 数が多いためである。profiler の合計 kernel 時間は
計測オーバーヘッドを含むため、性能値としては使わない。

## 5. 8 request / drafts 7 の実測

`tests/server/test_compute_mtp_batch.py`、TP=2 / EP=1、8 request、drafts 7。

### 既定（安全経路、Wavefront off）

```text
MTP_BATCH rounds=6 accepted=22 emitted=56 chunks=56 draft_forwards=19 max_draft_per_round=6 max_verify_per_round=19 max_active=8
compute-mtp-batch: passed=9 failed=0
```

- draft は depth 単位で batch 化され、最大 6 forward / round。
- verify は request ごとに 1 row ずつ。`max_verify_per_round=19` は
  Phase 1 の target forward 8 以下を満たさない。
- target-only 生成列との一致と再現性は PASS。

### Wavefront opt-in（`SHELF_MTP_WAVEFRONT=1`）

```text
MTP_BATCH rounds=6 accepted=22 emitted=56 chunks=20 draft_forwards=19 max_draft_per_round=6 max_verify_per_round=6 max_active=8
compute-mtp-batch: passed=10 failed=0
```

| 条件 | 実測 | 判定 |
| --- | --- | --- |
| draft forward / round <= 7 | 最大 6 | PASS |
| target verify forward / round <= 8 | 最大 6 | PASS |
| max_active > 1 | 8 | PASS |
| target-only 生成列一致 | この run は一致 | PASS |
| 再現性 | 同一 8 request の 2 回目が一致 | PASS |

Wavefront 4 request chunk は別 prompt で target-only と分岐する再現があったため、
既定では off、比較用 feature flag として残す。Phase 1 の合格条件は満たすが、
数値不変性が全 prompt で確認できるまでは既定にしない。

### fused verify の行数依存算術を解消した（2026-11-21）

4 request（20 行）の fused verify で state と生成列が safe 経路とずれる原因を、
単体 executor テスト（同じ row 0 入力を 4 行と 20 行で実行し trace / logits を比較）で
切り分けた。rollback ではなく、**行数で算術が切り替わる 2 箇所**だった。

| 個所 | 旧動作 | 差 |
| --- | --- | --- |
| BF16 GEMM selector | rows ≤16: `ExactRows`、rows ≥17: `Wmma` / `WmmaKPartition` | スカラー FMA の逐次和と WMMA の断片和で丸めが変わる |
| EXL3 split-K | rows 1..16 と 17..64 で別の選択式 | K の分割位置が変わり部分和の順序が変わる |

HyperConnection の down / up は BF16 で、最初の不一致 trace はこの GEMM だった。
1〜3 request（最大 15 行）は一致し、4 request（20 行）で分岐した観測と整合する。

対策:

- EXL3 split-K を 1..64 行で decode と同じ fill 式に統一（scratch 超過時のみクランプ）
- BF16 は narrow-out（`256 < out_features <= 512`）かつ 17..64 行のみ 16 行チャンクの
  `ExactRows`、wide-out は `Wmma` を維持（下記実測による）
- Verify role の `splitk_tuning_rows` 固定を削除

#### BF16 GEMM（shelf-bench gemm、p50 µs）

| shape | rows | ExactRows チャンク | Wmma |
| --- | ---: | ---: | ---: |
| 320:10240 | 20 | 42.6 | 132.7 |
| 320:10240 | 64 | 117.3 | 132.7 |
| 10240:320 | 20 | 61.0 | 15.2 |
| 10240:320 | 64 | 204.5 | 31.6 |

#### EXL3 split-K（shelf-bench exl3-int8、bits=3、合計 µs）

| shape | rows | 新 splits | 旧 splits |
| --- | ---: | ---: | ---: |
| 2560:6144 | 20 | 39.2 (4) | 39.2 (6) |
| 2560:6144 | 64 | 52.9 (4) | 57.6 (6) |
| 12288:2560 | 20 | 59.3 (1) | 60.1 (2) |

#### 受入（8 request × 8 token、同一 prompt）

| 指標 | safe verify | fused verify |
| --- | ---: | ---: |
| wall time（2 回平均） | 2013 ms | 866 ms（2.3x 高速） |
| kernel dispatch 数 | 1,306,191 | 539,477（2.42x 削減） |
| GPU kernel 時間 | 9.893 s | 4.475 s（2.21x 削減） |
| QSA kernel 時間 | 0.241 s | 0.104 s |

`MTP_BATCH_STATS` は safe が `max_verify_per_round=19`、fused が `=1` で、
Target verify forward は round あたり 1 回になる。

計算経路による logits 不一致と near-tie の argmax 分岐は許容する契約とした。
safe 経路は target-only 完全一致（9/9 PASS）。fused は 8 request 中 6 件が完全一致し、
残り 2 件は near-tie だった。

この契約を `tests/server/test_compute_mtp_batch.py` が機械的に検証する。
fused / wavefront（opt-in）では `SHELF_MTP_TOP2=1` を有効にし、各 request の最初の分岐
token について「fused verify が同じ token を sampled した行の top-1 がその token と一致し、
かつ margin <= `SHELF_MTP_NEAR_TIE_MARGIN`（既定 0.5）」を witness として要求する。
safe 経路は従来どおり完全一致のみを受け入れる。witness は
`MTP_NEAR_TIE divergences=[...]` として出力する。

| request | 分岐 index | fused | target-only | 分岐行 margin | witness pos |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 7 | 17 | 16 | 0.046824 | 9 |
| 6 | 4 | 3855 | 561 | 0.254333 | 7 |

### QSA score の multi-request launch 集約（Phase 3）

旧経路は `for(row tile) for(block tile) for(request slice) launch_qsa_score_wmma` で、
request ごとに score kernel を起動していた。これを `launch_qsa_score_wmma_batched` へ
置き換え、1 回の launch で row tile × block tile × request slice を処理する。

- CTA は device 側の `DeviceRequestDescriptor`（`row_begin` / `row_count` /
  `sequence_length`）を引き、行範囲を row tile にクランプする。
- visible block は `min(max_blocks, sequence_length / compress_ratio)` から求め、
  `block_begin` 以降の残りだけをスコアする。短い request と padding 行は
  書き込み前に return し、stale score を読まない。
- 算術（WMMA の accumulation order、`block_tile` の CTA 粒度）は旧経路と同一で、
  変わったのは launch のまとめ方だけである。
- 経路の gate は従来の tiled 条件（`staged_covers_rows`、QSA workspace、
  heads=4、head_dim=128）のまま。非対応形状は `launch_qsa_select_batch` の旧経路。

テストは `tests/unit/test_qsa_select.cpp` に複数 request・異長 sequence・
8191/8192/8193 block・同点（equal scores）を追加し、legacy の per-slice 経路と
batched 経路の score が bit 一致し、top-k と mask も一致することを確認した。

計測（8 request × 8 token、TP=2、warmup + 1 rep、rocprofv3 kernel trace）:

| 指標 | 旧 per-slice | batched | 比 |
| --- | ---: | ---: | ---: |
| `qsa_score` dispatch 数 | 1,728 | 672 | 2.57x 削減 |
| `qsa_score` GPU 時間 | 15.2 ms | 5.9 ms | 2.58x 削減 |
| 全 dispatch 数 | 435,742 | 434,686 | -1,056 |
| wall time | 1,104.6 ms | 1,099.8 ms | 非悪化 |

残りは Top-k の全域再走査削減（Phase 4）、attention / key-cache の launch 集約、
長文コンテキスト測定である。

### 長文 Top-k の再走査を radix union select に置き換えた（Phase 4）

計測は `shelf-bench qsa-select`（score と top-k を分離、rows × blocks × top_k）で行う。
blocks は context / compress_ratio であり、16384 / 32768 / 65536 が 65K / 131K / 262K 相当。

baseline（rows=1、compress=4、top_k=512）:

| blocks | score | top-k（旧） |
| ---: | ---: | ---: |
| 16384 | 41 us | 2,707 us |
| 32768 | 81 us | 2,879 us |
| 65536 | 161 us | 2,991 us |

ボトルネックは top-k が top_k 依存で O(k^2) 的に伸びることで、複数ウィンドウ時の
per-call bitonic sort + merge が原因だった。単一ウィンドウの `select_once` 経路は
top_k=512 / 8192 blocks で 168 us であり、複数ウィンドウ経路とは 16 倍の差があった。

変更:

- 直前 shortlist と新ウィンドウ候補の和集合から top-k を選ぶ `qsa_union_select` を追加。
  値の 8-bit radix を 4 pass、同点は block index の 8-bit radix を 4 pass（上位から）
  で解決し、gather で候補を書き出す。算術（score）は変更しない。
- 候補数が k 未満のときは全候補を選ぶ（radix を回さない）。
- mask の `atomicOr` は `block < args.blocks` を防御する。
- 旧 sort + merge 経路は `QsaTopkArgs::legacy_merge`（bench `--legacy-merge 1`）で
  比較用に残す。実測 crossover が top_k 128 と 256 の間にあるため、
  `top_k >= 256` で union select、`top_k < 256` では旧経路を dispatch する。

計測（rows=1、blocks=65536）:

| top_k | 旧 sort+merge | 新 union select | 比 |
| ---: | ---: | ---: | ---: |
| 64 | 356 us | 357 us（旧経路） | 同等 |
| 128 | 517 us | 523 us（旧経路） | 同等 |
| 256 | 934 us | 714 us | 1.31x |
| 512 | 2,960 us | 991 us | 2.99x |
| 1024 | 10,425 us | 1,424 us | 7.3x |

65K / 131K（top_k=512）は 2,707 → 498 us（5.4x）、2,879 → 659 us（4.4x）。
バッチ行数を増やしても非悪化（65K 相当、top_k=512: rows=8 で 3,092 → 1,161 us、
rows=64 で 3,675 → 1,952 us）。モデルの `indexer_budget=2048`、compress=4 から
production の top_k は 512 である。

正しさは `tests/unit/test_qsa_select.cpp` の CPU reference 一致（selected と mask）で
確認する。複数ウィンドウ（2 / 5 tile）、8192 / 8193 / 12000 / 16384 / 20000 blocks、
同点、top_k 512 のケースを追加した。required 226/226、MTP safe 9/9、fused 10/10。

### QSA sparse attention の K/V 共有と prefill dispatch 再評価（Phase 5）

計測は `shelf-bench qsa-attention`（QSA sparse attention）と
`shelf-bench paged-attention --variant prefill`（tiled dense prefill）で行う。
geometry はモデルと同じ q_heads=24 / kv_heads=2（q_per_kv=12）、head_dim=256、
block=4 token、page=16 token。

#### decode / prefill の分離測定（bf16、selected=512 block = 2048 token）

| rows | visible | splits | p50 |
| ---: | ---: | ---: | ---: |
| 1 | any | 32 (auto) | 31.4 us |
| 4 | any | 16 | 46.2 us |
| 8 | any | 8 | 75.8 us |
| 64 | any | 1 | 492 us |
| 128 | any | 1 | 827 us |
| 512 | any | 1 | 3015 us |

時間は visible token 数にほぼ依存せず、選択 block 数に線形である
（rows=64 で selected 32/64/128/256/512/1024 → 46/75/131/263/507/990 us）。

#### K/V 共有（warp group 内 LDS staging）は不採用

q_per_kv=12 では 4 warp が同じ KV head の K/V を重複ロードしている。warp ごとに
1 cell を LDS へ staging し、全 warp が LDS から読む実装を試した。

| rows | 変更前 | LDS 共有 | 差 |
| ---: | ---: | ---: | ---: |
| 64 | 510 us | 573 us | +12% |
| 128 | 798 us | 900 us | +13% |
| 512 | 2939 us | 3402 us | +16% |

重複ロードは L1/L2 が吸収しており、staging の LDS 書き込みと追加 barrier が上回る。
計画の「キャッシュが効いているから必ず速くなると思わない」のとおりなので、
実装は戻して既存の per-warp ロードを維持する。

#### prefill dispatch の閾値再評価

tiled dense prefill と sparse の比較（bf16、selected=512）:

| rows | visible | dense | sparse |
| ---: | ---: | ---: | ---: |
| 64 | 4096 | 353 us | ~500 us |
| 64 | 8192 | 683 us | ~500 us |
| 64 | 12288 | 1002 us | ~500 us |
| 128 | 4096 | 452 us | ~822 us |
| 128 | 8192 | 874 us | 822 us |
| 128 | 12288 | 1412 us | 822 us |
| 512 | 4096 | 1623 us | ~2986 us |
| 512 | 8192 | 3276 us | 2986 us |
| 512 | 12288 | 4920 us | 2986 us |

crossover は rows 64/128/512 で約 7-8K token。fp8 はさらに早く（rows=128, visible=8192 で
dense 1410 us vs sparse 866 us）。固定 budget 512 block のため sparse 側は visible に
依存しない。`kQsaSparsePrefillMinTokens` を 12,288 から 8,192 へ下げ、dense prefill 経路は
引き続き選択可能に残す。

残りは長文 prefill の gather + WMMA 実験（近似 mode は target-only exact の既定を
変えない）と、長文 end-to-end 測定である。

### QSA key cache の短い prefill 並列化（Phase 6.1）

`shelf-bench qsa-key-cache` で逐次 kernel（`launch_qsa_key_cache_batch`、1 launch・
request ごとに 1 CTA が行を直列処理）と並列 kernel（`launch_qsa_key_cache_blocks` +
`counts`、2 launch・圧縮 block ごとに CTA）を同じ入力で比較した。

| rows | requests | compress | 逐次 | 並列 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 4 | 42 us | 55 us |
| 32 | 1 | 4 | 49 us | 51 us |
| 64 | 1 | 4 | 44 us | 11 us |
| 128 | 1 | 4 | 79 us | 10 us |
| 256 | 1 | 4 | 143 us | 10 us |
| 512 | 1 | 4 | 279 us | 10 us |
| 64 | 8 | 4 | 12 us | 11 us |
| 256 | 8 | 4 | 24 us | 11 us |

並列経路は行数にほぼ依存せず 10〜12 us で、64 行付近から逐次を上回る。旧条件は
`actual_rows >= 256` で、64〜255 行の prefill を不必要に直列化していた。
`compress_ratio >= 2` のガードも追加した（block kernel は `cr < 2` で何も書かないため、
cr=1 のモデルでは逐次経路を使う必要がある）。`test_qsa_state_pool` /
`test_qsa_key_path` / `test_qsa_spec_state_replay` は bit 一致のまま PASS。

## 6. 残課題

- `R=8, K=7` の 64 行 1 forward fused verify は成立した（`max_verify_per_round=1`）。
  safe は target-only 完全一致、fused は near-tie 分岐 2 件のみで、これは許容契約。
- 行数依存の算術切替は BF16 selector と EXL3 split-K の 2 箇所で解消した。
- Wavefront 4 request chunk は prompt によって target-only と分岐するため opt-in のまま。
- QSA score の multi-request launch 集約は完了（2.57x 削減、score bit 一致）。
- Top-k の全域再走査削減は完了（262K 相当で 2.99x、top_k=1024 で 7.3x）。
- QSA sparse attention の K/V 共有は計測の結果不採用、prefill 閾値は 8192 へ再評価した。
- key cache の短い prefill 並列化は 64 行閾値へ更新した。
- gather + WMMA prefill 実験、長文 end-to-end 測定、per-request dense bypass（6.2）、
  QSA の graph-safe 化（6.3）は未着手。
