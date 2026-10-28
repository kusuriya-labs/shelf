# shelf-validate 仕様

対応実装: `apps/shelf-validate`（現行フェーズ V0）。
計画・未実装フェーズの詳細は `draft-docs/shelf-validate.md`。

---

## 1. 概要

実モデルで **無損失経路（exact, bf16）** を同一入力で複数回走らせ、生成トークン列が
一致するかを検証し、結果を JSON に保存する CLI。

* 内部での量子化変換は行わない（渡されたディレクトリをそのまま読む）
* 比較の対象は decode で生成したトークン列（StopToken は含めない）
* **現行の対応範囲は T0（同一構成の決定性）のみ**

| 項目 | 状態 |
| --- | --- |
| exact 単独での実行・T0 判定・JSON 出力 | 対応 |
| `--approx-model`（T1 完全一致 / T2 誤差判定） | **未対応**（指定で起動拒否） |
| `--pipeline-parallel`（K ≥ 2） | **未対応**（指定で起動拒否） |

---

## 2. ビルドと実行形態

```bash
cmake --build --preset hip  # → build/hip/apps/shelf-validate（GPU 経路）
cmake --build --preset cpu  # → build/cpu/apps/shelf-validate（f32 参照経路）
```

```bash
./build/hip/apps/shelf-validate \
    --exact-model=models/Qwen3.5-9B \
    --runs=2 --device=hip --gpu=1 --out=/tmp/sv-result
```

実装フェーズ: `cpu` ビルドは HIP 非対応（`--device=hip` で起動拒否）、
`hip` ビルドは `hip` / `cpu` 両方を受け付ける。

---

## 3. CLI

値の形式は `--key=value`（`--key value` も可）。未知の引数・値の欠落は起動エラー。

```text
使い方:
  shelf-validate --exact-model=<dir> [--approx-model=<dir>] [オプション]

オプション:
  --exact-model=<dir>       無損失 (bf16) 側のモデルディレクトリ（必須）
  --approx-model=<dir>      量子化側のモデルディレクトリ（V1 以降で有効）
  --prefill-tokens=<N>      prefill 長（既定 256、範囲 128..384）
  --budget=<N>              prefill + decode の合計上限（既定 512）
  --stop-id=<id>            StopToken（既定: モデル設定の eos_token_id）
  --pipeline-parallel=<K>   ステージ数（既定 1、K>1 は V2 以降で有効）
  --device=<hip|cpu>        実行デバイス（既定: HIP ビルドなら hip、それ以外 cpu）
  --gpu=<n>                 GPU 番号（既定 0）
  --runs=<n>                決定性チェックの反復数（既定 2）
  --ids-file=<path>         入力 token ids（既定: 同梱 fixture。区切りは空白・カンマ）
  --out=<dir>               結果 JSON の保存先（既定 shelf-validate-out）
  --help                    このヘルプを表示

終了コード: 0 = 合格 / 1 = 条件付き合格（T2、V1 以降）/ 2 = 不合格または起動エラー
```

### 3.1 起動時検査（すべて exit 2 で拒否）

| 条件 | 制約 |
| --- | --- |
| `--exact-model` | 必須 |
| `--approx-model` | 指定すると拒否（未実装） |
| `--pipeline-parallel` | `K = 1` のみ許可 |
| `--prefill-tokens` | `128 ≤ N ≤ 384` |
| `--budget` | `budget − N ≥ 128`（decode が下限 128 に達すること） |
| `--runs` | `≥ 1` |
| `--device` | `hip` / `cpu`。`hip` は HIP ビルドのみ |
| 入力長 | `fixture 長 ≥ N`（不足はエラー、超過は末尾切り詰め） |
| 入力トークン | `0 ≤ id < vocab_size`（語彙外はエラー） |

---

## 4. 入力（fixture）

* 既定入力は同梱の **`apps/fixtures/prefill_fixture.hpp`**（**256 tok 固定**）。
  チャットテンプレート適用済みの完成形であり、**同じ ids を両経路にそのまま適用する**。
* テンプレート条件: **思考モード OFF**（`enable_thinking=false`、アシスタント側に空の思考ブロックを前置）、
  末尾はアシスタント生成開始位置。
* 長さ規則:
  * `N > fixture 長` → 起動エラー
  * `N < fixture 長` → 末尾切り詰め（`config.input.truncated = true`）。
    末尾にテンプレート末尾部があるため、実運用は **N = 256** を基本とする。
* `--ids-file` で置換可能（区切りは空白・カンマ、整数列）。
* fixture の生成は開発時のみ:

```bash
python3 apps/fixtures/make_fixture.py   # 256 tok の厳密一致を検査してヘッダを書き出す
```

  生成物が 256 tok に一致しない場合はエラーで終了するので、スクリプト内の `PROMPT` を調整する。

---

## 5. 予算

* **`prefill N + decode G ≤ budget`（既定 512、N = 256 → G ≤ 256）**
* StopToken に達せずに `G = budget − N` に達したら打ち切り、停止理由を
  **`budget_exhausted`** として記録する（これは不一致ではない）
* StopToken で停止した場合は停止理由を **`stop_token`** とし、
  **StopToken はトークン列に含めない**

---

## 6. StopToken の解決順

1. `--stop-id` 指定時 = その値（`stop.source = "cli"`）
2. `generation_config.json` の `eos_token_id` — 配列なら**全要素が停止条件**、
   **先頭の要素を primary として採用**（`stop.source = "generation_config.json"`）
3. `config.json` 由来の値（shelf の `TextConfig.eos_token_id`）（`stop.source = "config.json"`）
4. いずれも無ければ起動エラー

出力には `stop.ids`（全停止条件）と `stop.primary` を記録する。

---

## 7. 実行処理

1. **設定読み**: `config.json` / `generation_config.json` から幾何・`eos_token_id` を得る
2. **ロード**: `model.safetensors.index.json` → shard 一覧。bf16 テンソルは
   checkpoint のビットをそのまま（変換なし）、f32 以外・bf16 以外は CPU 経路で f32 に変換
3. **状態リセット**: 毎 run ごとに `reset_states()`（ステート全ゼロ + 使用行のクリア）を実行
4. **prefill**（位置 `0..N-1`）:
   * HIP: 1 ステップ上限 64 行のため **64 行チャンク**。端数は 1 行ステップで処理。
     最終ステップの logits を採用する（チャンク終端なら最終行）
   * CPU: トークン 1 行ずつ
5. **decode**（位置 `N` 以降、最大 `budget − N` 個）:
   * 各ステップで **greedy top-1**。位置規則: 直前の生成トークン `t_{s-1}` を位置 `N + s − 1` へ投入
   * StopToken なら停止（列に含めない）。予算上限なら `budget_exhausted`
   * 各ステップの **top-1 / top-2（id と値）を保持**（T2 の margin 判定用）
6. **比較**: run 0 を基準に各 run と比較（§8）

---

## 8. 比較判定（現行: T0 のみ）

比較は `comparisons[]` に 1 件以上入る（`runs = 1` では空、`t0 = "skipped"`）。
軸は `"T0"`、比較ペアは `a = run0`、`b = run i（i ≥ 1）`。

| verdict | 条件 | 追加記録 |
| --- | --- | --- |
| `identical` | トークン列と停止理由がともに一致 | `token_count` |
| `incomparable` | トークン列は同じだが停止理由が異なる | 両側 `stop_reason` |
| `diverged` | トークン列が異なる | `diverged_at`（最初の不一致位置 k）、`a_token_at_k` / `b_token_at_k`、k 位置の `a_top1{id,value}` / `a_top2_value` / **`a_margin`（top1 − top2）**、`b_top1{id,value}` / `b_top2_value`、`top1_value_gap`（top-1 値の差） |

* diverged の場合、`note` に「logits 差は top-1 値のみ（完全な vocab 差分は V1 以降）」を記録する。

---

## 9. 終了コード

| コード | 条件 |
| --- | --- |
| `0` | 全比較が `identical`（または `runs = 1` で比較なし） |
| `1` | 条件付き合格（T2 誤差判定）— **現行では発生しない**（V1 以降） |
| `2` | 比較が `diverged` / `incomparable`、または起動・実行時のエラー |

---

## 10. 出力 `result.json`

`--out` ディレクトリに `result.json` を書く（実行に失敗した場合は書かない）。

| キー | 内容 |
| --- | --- |
| `schema` | `"shelf-validate/v0"` |
| `exit_code` | 終了コード（§9） |
| `t0` | `identical` / `diverged` / `incomparable` / `skipped` |
| `config.device` / `config.gpu` | 実行デバイス / GPU 番号 |
| `config.runs` | 反復数 |
| `config.prefill_tokens` / `config.budget` / `config.decode_cap` | `N` / `budget` / `budget − N` |
| `config.pipeline_parallel` | 現行は常に `1` |
| `config.models.exact` / `config.models.approx` | モデルディレクトリ（approx は現行 `null`） |
| `config.input` | `source`（`fixture:...` または `ids-file:...`）、`length`、`truncated` |
| `config.stop` | `ids`（全停止条件）、`primary`、`source`（§6） |
| `config.device_mem_used_bytes` | VRAM 常駐量（HIP: `hipMemGetInfo` の前後差、CPU: `null`） |
| `runs[]` | 毎 run: `config`、`index`、`token_count`、`stop_reason`、`prefill_ms`、`decode_ms`、`tokens[]`（decode 生成分のみ） |
| `comparisons[]` | §8 の判定結果 |
| `timing` | `load_ms`、`elapsed_ms` |

---

## 11. リソース仕様

### 11.1 アリーナ（HIP）

* `HipStepExecutorOptions::arena_bytes` の既定 **`0` = 自動算出**。
  `declare_states(g)` に渡したグラフから
  **`64 MiB + 2 × Σ（全 primitive 出力の要素数 × sizeof(float)）`** を確保する
  （バンプ割当でステップ内再利用が無いため、必要量は行数に比例する）。
* 本アプリは **行数の大きい prefill グラフ（rows = 64）を `declare_states` に渡す**
  （rows = 1 の decode グラフを渡すと prefill で不足する）。
  両プランのステート集合が同一であることを起動時に検査する。
* 指定値（`arena_bytes > 0`）を使う場合はそれが上限となり、枯渇時は
  `insufficient_memory`（`arena exhausted`）を返す。null 出力でのカーネル発射は行わない。

### 11.2 容量の目安（開発モデル = `Qwen3.5-9B` bf16、1 GPU）

| 項目 | 目安 |
| --- | --- |
| 重み payload（mtp を含む・ディスク） | 19.31 GB（safetensors ヘッダ集計の実測） |
| アリーナ（rows = 64、自動算出） | **2,299 MiB（実測 2026-10-28、表示は "2299 MB"）**／最小必要 **777〜784 MiB**（`docs/spec/bench.md` §11.2） |
| VRAM 常駐合計（重み + KV 状態 + arena、`free` 差分） | **20.56 GB（実測）** |
| KV（既定 `budget = 512` の single sequence） | 数十 MB 程度（1 tok = 32 KiB） |
| approx `Qwen3.5-9B-exl3`（EXL3 4bit）の重み | 7.19 GB（ディスク実測・VRAM 常駐は未計測） |
| 参考: 旧 `Qwen3.5-4B`（削除済み・実測） | 重み 8.7 GB / アリーナ約 2 GB / 合計 10.1 GB |

* GPU が占有されていると重みアップロードが `out of memory` で失敗するため、
  `--gpu` で空きを選ぶ（`rocm-smi` で確認）。
* 実走の実測（prefill / decode / stop_reason）は `draft-docs/shelf-validate.md` §7
  「V0 実績（9B・2026-10-28・修正後）」を参照。

### 11.3 CPU 経路

* f32 参照経路（トークン 1 行ずつ）で、性能要件は持たない
  （大規模モデルの通し実行は現行の対象外）。

---

## 12. 現行の対応外（計画のみ）

未実装フェーズ（V1〜V3）の内容と未決事項は `draft-docs/shelf-validate.md` §7 / §8 に記す
（本書は実装済みの仕様のみを書く）。
