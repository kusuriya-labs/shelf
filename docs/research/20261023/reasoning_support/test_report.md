# Reasoning 対応のテスト結果

- 実施日: 2026-10-23
- モデル: `models/Qwen3.8-27B-FP8`
- GPU: AMD Radeon AI PRO R9700 1 枚（device 0）
- 環境: `SHELF_TEST_DEVICE=0 SHELF_TEST_ARENA_GIB=31
  SHELF_TEST_WORKING_SET_RESERVE_GIB=1`、`--max-seq-len 1024`（reasoning 系 E2E）

## 1. 実行コマンド

```bash
export SHELF_MODEL_DIR=models/Qwen3.8-27B-FP8
export SHELF_TEST_DEVICE=0
export SHELF_TEST_ARENA_GIB=31
export SHELF_TEST_WORKING_SET_RESERVE_GIB=1
python3 tests/server/run_server_regression.py --group reasoning --group core \
    --group tools --group unsupported --group concurrency-cancel
```

## 2. Unit / contract（GPU 不要）

| test | 結果 | 内訳 |
| --- | --- | --- |
| `test_reasoning_parser.py` | PASS（0.0s） | 24 checks。model / GPU 不要。byte-level の fake tokenizer で U06–U12, U17, U18 を検証 |
| `test_reasoning_request.py` | PASS（3.8s） | 49 checks。実 tokenizer と chat template（GPU 不要）で U01–U05, U12–U14, U16, U20 を検証 |

`test_reasoning_parser.py` が固定する挙動:

- 単一 Token / 複数 Token / chunk 分割をまたぐ close tag を 1 回だけ検出する。
- 境界 tag は reasoning / answer のどちらの文字列にも現れない。
- close 未到達は reasoning のみ（answer を捏造しない）。
- answer の先頭空白 strip、空 answer、Unicode（UTF-8 の分割を含む）。
- 1 Token ずつの投入と一括投入で完全一致する。
- 2 個の Parser を交互に進めても状態が混線しない。

## 3. E2E（実モデル・GPU）

| test | 結果 | 主な確認内容 |
| --- | --- | --- |
| `test_server_reasoning.py` | PASS（30.2s、27 checks） | 既定 thinking ON で `reasoning_content` と `content` が分離する。token log の ID 列から再構成した文字列と wire が一致。`length` 打ち切りで回答を捏造しない。履歴の reasoning 往復、`preserve_thinking=false`、`/v1/meta`、OpenAI Python SDK での受信 |
| `test_server_reasoning_stream.py` | PASS（25.8s、21 checks） | stream と非 stream の reasoning / content が完全一致。1 chunk に両 field が混ざらない。reasoning → answer の遷移が 1 回。usage chunk の `completion_tokens` が実 ID 数と一致。thinking ON/OFF の並列 stream が混線しない |
| `test_server_reasoning_tools.py` | PASS（24.5s、23 checks） | thinking ON + tool calling で `tool_calls` が正しく、回答領域の markup のみが解析される。reasoning に tool markup が無いこと、履歴の tool round-trip、streaming の `delta.tool_calls` と非 stream の一致。thinking 無効の tool calling も従来どおり |

主要な実測値:

- 既定（`reasoning_effort` 省略 = xhigh）の 1 要求で、reasoning と answer の両方が得られた。
- `max_completion_tokens=16` では `finish_reason="length"`、`content=""`、
  `reasoning_content` 非空（回答の捏造なし）。
- streaming の usage chunk の `completion_tokens` は token log の ID 数と一致した
  （reasoning token も含む）。

## 4. 既存サーバー回帰

実行した group の結果（`--tail 25`）。数値は各 test の check 数。

| group | test | 結果 |
| --- | --- | --- |
| core | `test_compute_pids_scope.py` | PASS 4 |
| core | `test_compute_service.py` | PASS 26 |
| core | `test_server_models.py` | PASS 7 |
| core | `test_server_meta.py` | PASS 19 |
| core | `test_server_chat.py` | PASS 10（下記 §5） |
| core | `test_server_stream.py` | PASS 16 |
| core | `test_server_eos_contract.py` | PASS 11 |
| tools | `test_server_tools.py` | PASS 19 |
| tools | `test_server_stream_tools.py` | PASS 14 |
| tools | `test_server_tool_regression.py` | PASS 5（173.8s） |
| unsupported | `test_server_unsupported.py` | PASS 34 |
| unsupported | `test_compute_prefix_contract.py` | PASS 19 |
| concurrency-cancel | `test_compute_cancel.py` | PASS 8 |
| concurrency-cancel | `test_compute_concurrency.py` | PASS 30 |
| concurrency-cancel | `test_server_concurrency.py` | PASS 21 |
| concurrency-cancel | `test_server_stream_cancel.py` | PASS 6 |

`core` の `test_compute_dflash2.py` / `test_server_dflash2.py` は
`SHELF_DFLASH2_MODEL_DIR` 未設定のため SKIP（従来どおり）。

## 5. 初回回帰で検出した 2 件

### 5.1 `test_server_chat.py` の CLI parity（修正済み）

初回は `cli_generated_ids()` が固定で device 1 を使うため、device 1 が別 workload に
占有されているホストで `hipMalloc failed` になった。server 側の 8 checks は全て PASS。

CLI parity は server 終了後に実行されるため、`SHELF_TEST_DEVICE`（server が使った
device）で実行すれば十分である。`cli_generated_ids()` の device 既定を
`SHELF_TEST_DEVICE` に従わせ、再実行で 10/10 PASS を確認した。

### 5.2 `test_server_context_boundary.py`（環境制約。本変更と無関係）

32K context + Qwen3.8-27B-FP8（bf16 KV）が 1 枚 32 GiB の GPU に収まらない。

```text
target weights 27.45 GiB / runtime workspace 4.00 GiB / KV 2.00 GiB (32769 tokens)
prefix cache 1.58 GiB / required 35.03 GiB / available 31.00 GiB
```

同一条件を変更前の main（`95e4c99`）のテストで再現し、同じ `required 35.03 GiB` で
失敗することを確認した。reasoning 対応による回帰ではない。

## 6. 未実施（理由付き）

| 項目 | 理由 |
| --- | --- |
| DFlash2（E13） | drafter 3.6 GB + target 27.45 GiB + prefix cache 1.58 GiB が 31 GiB arena に収まらない。`SHELF_DFLASH2_MODEL_DIR` を設定した回帰は 1 枚 32 GiB では起動できない |
| EAGLE4（E14） | drafter 5.8 GB。同上。加えて TP2 向け EAGLE4 の変更は revert 済みで、単一 GPU の EAGLE4 回帰は本作業の対象外 |
| OpenCode agent（E18） | `MODEL_CONTEXT=32768` 固定のため起動時に `fits 14640 KV tokens but max_seq_len 32768 needs 32769` で失敗する（§5.2 と同じ制約）。E17 の OpenAI Python SDK 経由で `reasoning_content` の受信は確認した |
| 品質ベンチマーク（`tests/data/quality/reasoning.json` 等） | Thinking ON の新ベースラインは未採取。旧 baseline は `enable_thinking=false` で採点しており、比較条件が異なる |
| 性能（TTFT / TPS / CPU 時間） | 本変更は host 側の分割処理のみで、計測は未実施。streaming は領域ごとの累積 decode であり、長い reasoning での decode コストは未計測 |

## 7. 備考

- E2E の実行には、リポジトリ外の `/tmp` wrapper（`shelf_server.py` を直接起動する
  シェルスクリプト）を使った。worktree には build 成果物を置かず、`shelf-compute` は
  メインの build を使用している。リポジトリへの追加物はない。
- 単体試験と E2E の PASS は別物として扱った。境界判定の正しさの正本は
  `test_reasoning_parser.py` と `test_reasoning_request.py` の固定 Token 列であり、
  E2E は wire・実 template・実計算経路の整合性を見るものである。
