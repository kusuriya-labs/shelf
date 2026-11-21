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

## 6. 残課題

- `R=8, K=7` の全 64 行を 1 回の `SPEC_VERIFY` にする fused verify は未達。
  GDN 上限を 8 request / 64 行へ拡張し、`SHELF_MTP_FUSED_VERIFY=1` で
  4 request fused を試したが、`paged-attention: tree committed > capacity` で停止した。
  state / mask / spec history の容量契約を先に修正する必要がある。
- 単一 request の MTP は target-only より遅い。速度評価は fused verify 後にやり直す。
- Wavefront 4 request chunk は prompt によって target-only と分岐する。数値不変経路が
  できるまで既定 off。
- QSA score / Top-k / sparse attention の launch 集約と長文測定は未着手。
