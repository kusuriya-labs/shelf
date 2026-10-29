# shelf-validate 操作手順

現行の仕様は `docs/spec/validate.md` を参照。本書は運用手順（ビルド・実行・対処）。

* 実モデルの **無損失経路（exact, bf16）** を同一入力で 2 回走らせ、生成トークン列が
  一致するか（**T0 決定性**）を検証し、結果を JSON に残す道具。
* `--exact-model` 単独で動作する。`--approx-model`（T1/T2 比較）と
  `--pipeline-parallel`（K>1）は**現状未実装**（起動前に exit 2 で拒否）。
  それらの計画は `draft_docs/shelf_validate.md` §7 を参照。

---

## 1. 手順（概要）

1. ビルドする（§2）
2. モデルディレクトリを用意する（`config.json` + `model.safetensors.index.json` + shard）
3. GPU の空きを確認し `--gpu` を決める（§5。既定 0 は埋まっていることがある）
4. 実行する（§3）
5. `result.json` と終了コードで判定する（§7, §8）

---

## 2. ビルド

```bash
# 計測用 = Release（prefill / decode の時間は必ずこれで取る）
cmake --preset hip-rel          # 初回のみ（設定）
cmake --build --preset hip-rel  # → build/hip_rel/apps/shelf-validate

# 動作確認・決定性確認用 = Debug
cmake --preset hip
cmake --build --preset hip      # → build/hip/apps/shelf-validate

# CPU（f32 参照）版
cmake --preset cpu
cmake --build --preset cpu      # → build/cpu/apps/shelf-validate
```

> **性能値は `hip-rel`（Release）で測定する**（2026-10-28 決定）。

---

## 3. 実行

```bash
./build/hip/apps/shelf-validate \
    --exact-model=models/Qwen3.5-9B \
    --runs=2 \
    --device=hip \
    --gpu=1 \
    --out=/tmp/sv-result
```

出力例（開発モデル `Qwen3.5-9B`・2026-10-28 実測、R9700 / gfx1201・**Release ビルド（hip-rel）**）:

```text
  arena: 2299 MB（自動算出: prefill グラフ rows=64）
  prefill チャンク 位置 0..63 (196.8 ms)
  prefill チャンク 位置 64..127 (202.4 ms)
  prefill チャンク 位置 128..191 (202.5 ms)
  prefill チャンク 位置 192..255 (202.0 ms)
run 1/2: 230 tok 生成 (stop_token), prefill 831.8 ms / decode 7121.2 ms
  prefill チャンク 位置 0..63 (199.1 ms)
  prefill チャンク 位置 64..127 (202.6 ms)
  prefill チャンク 位置 128..191 (202.8 ms)
  prefill チャンク 位置 192..255 (203.1 ms)
run 2/2: 230 tok 生成 (stop_token), prefill 836.5 ms / decode 7126.0 ms
T0 (run0 vs run1): identical
判定: 合格 (exit 0) — 結果: /tmp/sv-result/result.json
```

### 3.1 CPU 経路

`--device=cpu` は f32 参照経路（トークン 1 行ずつ）。**9B / Release 実測（2026-10-28、差分計測）**:
**prefill 4.90 s/token、decode 3.68 s/token**（ロードは約 23 s で別途。旧 4B の Debug 実測
16 s/token は履歴）。既定の入力（prefill 256 + decode 230）なら 1 run は**約 35 分**。

* CPU 経路は **小規模な経路確認にのみ使う**（27B 級は対象外。
  `draft_docs/shelf_validate.md` §8 の未決事項に同じ）。
* 全トークンでの実走が要る場合は `--runs=1` とし、放置可能な時間帯に
  バックグラウンドで回す。

---

## 4. CLI

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--exact-model=<dir>` | （必須） | 無損失 (bf16) 側のモデルディレクトリ |
| `--approx-model=<dir>` | なし | 量子化側。**現行未実装**（指定で exit 2） |
| `--prefill-tokens=<N>` | 256 | prefill 長。範囲 **128..384** |
| `--budget=<N>` | 512 | prefill + decode の合計上限。**`budget - N ≥ 128` が必須** |
| `--stop-id=<id>` | 設定由来 | StopToken の明示指定 |
| `--pipeline-parallel=<K>` | 1 | **K>1 は現行未実装**（指定で exit 2） |
| `--device=<hip\|cpu>` | hip※ | ※HIP ビルドなら `hip`、CPU ビルドなら `cpu` |
| `--gpu=<n>` | 0 | HIP 使用時の GPU 番号 |
| `--runs=<n>` | 2 | 決定性チェックの反復数（1 で T0 省略） |
| `--ids-file=<path>` | 同梱 fixture | 入力 token ids（空白・カンマ区切りの整数列） |
| `--out=<dir>` | `shelf-validate-out` | 結果 JSON の保存先 |
| `--help` | — | 使い方を表示 |

* 予算・範囲違反は **起動前に exit 2 で拒否**する（例: `--prefill-tokens=64`、
  `--budget=300`（N=256 のとき））。
* StopToken の解決順（省略時）: `generation_config.json` の `eos_token_id`
  （配列なら全要素が停止条件・先頭を primary 採用）→ `config.json`（shelf の TextConfig）
  → どちらも無ければエラー。

---

## 5. GPU の選び方

埋まった GPU を指すと重みアップロードで失敗する:

```text
shelf-validate: upload model.language_model.embed_tokens.weight: ...: malloc weight bf16: out of memory
```

```bash
rocm-smi                       # VRAM% / GPU% を見る
# GPU0 が埋まっていれば空きを指す
./build/hip/apps/shelf-validate ... --gpu=1
```

空き 1 GPU で開発モデル `Qwen3.5-9B` bf16 を動かす目安（2026-10-28 実測）:
重み payload 19.31 GB ＋ arena **2,299 MiB**（rows=64 自動算出）で、
**VRAM 常駐合計 20.56 GB**（重み + KV 状態 + arena の `free` 差分）。
approx（`Qwen3.5-9B-FP8`、FP8 W8A8）の重みはディスク 14.01 GB（`language_model` ＋ `lm_head` で 12.61 GB）。
内訳は `docs/spec/validate.md` §11.2。

---

## 6. 入力（fixture）

* 既定入力は同梱の **`apps/fixtures/prefill_fixture.hpp`（256 tok 固定）**。
  チャットテンプレート適用済みの完成形で、同じ ids を両経路にそのまま適用する。
* 長さ規則: `N ≤ fixture 長さ`（不足はエラー）。超過分は末尾切り詰めだが、
  末尾にはアシスタント開始トークンがあるため切ると意味が変わる。
  **実質 N = 256 を基本とし、それ以外は fixture を再生成する。**
* 再生成（開発時のみ・`transformers` 必要）:

```bash
python3 apps/fixtures/make_fixture.py   # 長さ 256 の厳密一致を検査してヘッダを書き出す
```

  * 思考モード **OFF**（`enable_thinking=false`）+「約 150 語で即答」の指示で、
    decode が 128..256 tok の範囲で StopToken に到達する設定にしてある
    （思考 ON だと decode が思考だけで予算を消費し StopToken に到達しない）。
  * 長さが一致しない場合はエラーになるので、`PROMPT` を調整して揃えること。

---

## 7. 判定と終了コード

| コード | 意味 |
| --- | --- |
| `0` | 合格（T0 一致） |
| `1` | 条件付き合格（T2 誤差判定。**現行では発生しない**＝未実装） |
| `2` | 不合格（T0 不一致・判定不能）**または起動エラー** |

比較結果は `result.json` の `comparisons[]` に入る:

* `identical` — トークン列と停止理由がともに一致
* `diverged` — 最初の不一致位置 `diverged_at`、その位置での exact 側
  `a_top1` / `a_top2_value` / `a_margin`（top1 − top2）、top-1 値の差 `top1_value_gap`
  （vocab 全体の logits 差は現行では保持しない）
* `incomparable` — トークン列は同じだが停止理由が異なる（片方だけ予算打ち切り等）

---

## 8. 出力 `result.json`

| キー | 内容 |
| --- | --- |
| `schema` / `exit_code` | `shelf-validate/v0` / 終了コード |
| `t0` | `identical` / `diverged` / `incomparable` / `skipped`（`--runs=1`） |
| `config` | `device`, `gpu`, `runs`, `prefill_tokens`, `budget`, `decode_cap`, `pipeline_parallel`, `models{exact,approx}`, `input{source,length,truncated}`, `stop{ids,primary,source}`, `device_mem_used_bytes` |
| `runs[]` | 実行毎: `token_count`, `stop_reason`（`stop_token` / `budget_exhausted`）, `prefill_ms`, `decode_ms`, `tokens[]`（decode 生成分のみ、StopToken は含めない） |
| `comparisons[]` | §7 の判定結果 |
| `timing` | `load_ms`, `elapsed_ms` |

---

## 9. トラブルシュート

| 症状 | 対処 |
| --- | --- |
| `malloc weight bf16: out of memory` | GPU が埋まっている。`rocm-smi` で確認し `--gpu` を変える |
| `budget_exhausted` だけで StopToken に達しない | fixture の思考モードが ON になっていないか `make_fixture.py` で再生成する |
| `入力長が --prefill-tokens より短い` | `--prefill-tokens` を fixture 長（256）以下にするか、`--ids-file` / fixture 再生成で補う |
| `このビルドは HIP 非対応です` | `cpu` プリセットのビルドで `--device=cpu` を使う |
| `approx 経路は未実装です` / `pipeline-parallel > 1 は未実装です` | 未実装フェーズの実装を待つ（`draft_docs/shelf_validate.md` §7） |
| `prefill: ... arena exhausted` | arena は executor の自動算出（`declare_states` に渡した rows=64 グラフから）で確保される。§3 出力の `arena:` 行が実際の確保量。継続する場合は issue 化 |

---

## 10. 未実装（計画のみ）

未実装フェーズ（V1〜V3）の内容・設計・未決事項は `draft_docs/shelf_validate.md` §7 / §8 に記す
（本書は実装済みの扱いのみを書く）。
