# P0/P1 実装・検証の記録（2026-11-01）

対象: 生成機能の完成（T01〜T05）、実モデルでの確定（T06〜T09）、テストと文書（T10）。
環境: R9700 (gfx1201) ×4、HIP 7.15、`build/hip_rel`（Release）、Python 3.12 / transformers 5.20.2。

---

## 1. P0（T01〜T05）

### T01 EOS / StopToken

* 置き場: `src/runtime/generate/stop_tokens.{hpp,cpp}`（純粋関数。モデル名・backend を知らない）
* 解決規則（**3 系統の和集合**）:
  1. `generation_config.json` の `eos_token_id`（整数 / 整数配列）
  2. モデル設定の `text_config.eos_token_id`（無ければトップレベル）
  3. `tokenizer_config.json` の `eos_token` を `tokenizer.json` の `added_tokens` /
     `model.vocab` で ID に解決
* 3 が必要だった理由（実測）: 9B 系の checkpoint は generation_config が
  `248044`（`<|endoftext|>`）だけを宣言し、チャットの実際の終端 `248046`
  （`<|im_end|>`）を落としている。和集合にしないと **im_end の後ろの改行まで生成し、
  `generated_tokens` が 2 多く数えられる**（修正前 5 → 修正後 3）
* 判定は `decide_step`（純粋関数）: EOS は `generated_ids` / `token` イベントに載せない。
  EOS と `max_new_tokens` が競合したら **EOS 優先**（最後の 1 トークンでも `stop`）
* `shelf-compute` に `--eos-token`（複数指定可）/ `--no-eos` を追加
* 検証:
  * `unit.runtime_generate_stop`（解析・併合・複数 EOS・競合・cancel 照合）
  * 実 9B-FP8: `--eos-token 9707` で 0 トークン `stop`、`max_new_tokens=1` でも `stop`、
    複数指定 `1234 9707` が両方登録される
  * 実 HTTP: 9B-FP8 の「1+1は?」が `stop` / `completion_tokens=3`（`2 です。`）

### T02 done の伝搬

* `EngineClient.generate()` はトークン dict を逐次返し、最後に `Done`
  （`finish_reason` / `prompt_tokens` / `generated_tokens` / `seed` / `generated_ids`）
* HTTP は推測せず `done` を写す:
  * 非ストリーム: `finish_reason` / `usage` / `seed`（トップレベル）
  * SSE: 最終 chunk に `finish_reason` / `usage` / `seed`、最後に `data: [DONE]`
* `token` 列と `done.generated_ids` が食い違えば 500（プロトコル違反）

### T03 Cancel と異常終了

* `runtime::generate::CancelToken`: `request_id` を照合。一致しない cancel は
  実行中の生成を止めず、次の要求向けに保留（上限 64）。開始前 cancel は
  その ID の開始時に `done(cancelled, 0 トークン)`
* `shelf-compute` は Prefill の**チャンクの合間と端数の 1 行ごと**、decode の各ステップで
  制御行を読む（`drain_controls`）。Prefill 中でも止まる
* Server 側:
  * HTTP 切断を検出したら消費を止め、`finally` で `cancel` → terminal まで drain
  * 子プロセスの終了は待っている要求を `EngineError(state_error)` で起こす
  * `close()` は契約に無い `shutdown` op を送らず、stdin を閉じる
* 検証（実 9B-FP8・JSONL 直）:
  * 別 ID の cancel → 10 トークン継続
  * 一致する cancel → `cancelled`（30 トークン時点）
  * 後続要求 → `done(length)` で流れる
  * 開始前 cancel → `done(cancelled, 0)`
  * 2048 トークンの Prefill 中に cancel → 0.23 秒で `cancelled`（トークン 0）
* 検証（HTTP・フェイク/実）: 切断後に次の要求が 0.30 秒（9B）/ 5.25〜5.92 秒（27B）

### T04 HTTP パラメータ検証

`validate_chat_body` がエンジン到達前に 400 / 413 / 415 を返す。主な規則:

| 対象 | 規則 | code |
| --- | --- | --- |
| `messages` | 空でない配列・最大 4096 件・各要素は object | `invalid_argument` |
| `messages[i].role` | `system` / `user` / `assistant` のみ | `unsupported_parameter`（その他） |
| `messages[i].content` | 文字列のみ | `unsupported_parameter`（マルチモーダル） |
| 未知のメッセージキー | 拒否（`name` / `tool_calls` など） | `unsupported_parameter` |
| 入力サイズ | 本文 16 MiB / 合計 4 MiB / `prompt + max_tokens <= max_context` | `request_too_large` / `context_length_exceeded` |
| `stream` | bool のみ | `invalid_argument` |
| `stop` | 形は検証し、値は未対応 | `unsupported_parameter` |
| `n` | 1 のみ | `unsupported_parameter`（>1） |
| `chat_template_kwargs` | `enable_thinking` のみ（bool） | `unsupported_parameter` |
| Content-Type | `application/json` のみ | `unsupported_media_type`（415） |

* テスト: `apps/server/tests/test_validation.py`（表駆動）と
  `test_server_http.py`（HTTP 経由・413 の本文上限を含む）
* `openai_errors` の表に `request_too_large`(413) / `unsupported_media_type`(415) を追加

### T05 SSE の文字列復元

* `IncrementalDecoder`: **受け取った全 ID** を毎回 decode し、送出済み文字列との差分を返す。
  末尾が未完成 UTF-8（`U+FFFD`）の間は次トークンまで保留し、最後に `finish()` で残りを出す
* SSE の連結が非ストリームの `decode(全 ID)` と一致しない場合は 500（自己検査）
* 検証: バイト境界がトークン境界とずれる `ByteTokenizer`（`abcdefg日!`）で
  旧実装（8 個ずつ decode）が壊れる形を再現し、新実装が一致することを固定
* 実モデル: 日本語の応答（`日本の首都は東京です。\n`）で SSE と非ストリームが一致

---

## 2. T06 Chat Template の実機確認

`tests/scripts/check_chat_template.py` を追加（モデル・GPU 不要）。
9B / 9B-FP8 / 27B-FP8 / 27B-MXFP4 の 4 つで全項目 PASS。

| 項目 | 実測 |
| --- | --- |
| 語彙 | 248077（4 モデルで一致） |
| tokenizer の eos | 248046（`<|im_end|>`） |
| 特殊トークン | `<|im_start|>`=248045 / `<|im_end|>`=248046 / `<|endoftext|>`=248044 / `<think>`=248068 / `</think>`=248069 |
| 思考 ON | 生成プロンプト末尾が `<think>\n` |
| 思考 OFF | `<think>\n\n</think>\n\n` で閉じる |
| 既定 | ON と同じ |
| role | system / user / assistant / 複数ターンが正しく描画 |
| 文字列と ID | `tokenize=False` と `tokenize=True` が同じ内容 |

* 実装の修正: transformers 5 系の `apply_chat_template(tokenize=True)` は
  `BatchEncoding` を返す。旧 `HfTokenizer.prompt_ids` は `isinstance(dict)` で分岐して
  いたため ID を取り出せず落ちていた。`_ids_from_template_output` で
  `BatchEncoding` / 入れ子 / Encoding 配列 / tensor を受ける（`test_tokenizer.py`）

---

## 3. T07 OpenAI API の E2E

`tests/scripts/serve_e2e.py` を追加。実 `server.py` を子プロセスで起動し、HTTP で検証する。

確認項目: 通常応答 / finish_reason / usage / seed / 同一要求 2 回の一致 /
SSE（role・増分連結・usage・seed）/ EOS で `stop` / 400・404・415 /
HTTP 切断後の cancel と後続要求。

| 実行 | 結果 | 通常応答 | 切断後の次要求 |
| --- | --- | --- | --- |
| 9B-FP8 PP=1 | 全 PASS | `2 です。` / stop / 3 tok | 0.30 s |
| 9B-FP8 PP=2 | 全 PASS | `2 です。` / stop / 3 tok | 0.31 s |
| 9B-FP8 PP=4 | 全 PASS | `2 です。` / stop / 3 tok | 0.32 s |
| 27B-FP8 PP=1 | 全 PASS | `2です。` / stop / 3 tok | 5.25 s |
| 27B-MXFP4（W4A4） | 全 PASS | `2です。` / stop / 3 tok | 5.92 s |

---

## 4. T08 決定性・数値回帰

`tests/scripts/check_generation_parity.py` を追加（JSONL を直接駆動）。
各実行を 1 プロセスで 2 回繰り返し、生成トークン列を比較する。

### 9B-FP8（prompt 192 tok、greedy 8 tok）

`[45484, 99162, 182807, 97551, 14876, 5205, 15015, 220]`

| 比較 | 結果 |
| --- | --- |
| 同一プロセスの連続要求（全 5 実行） | 一致 |
| Prefill 32 / 64 / 128 | 一致 |
| PP=1 / 2 / 4 | 一致 |

### 27B（prompt 192 tok、greedy 8 tok）

`[45484, 99162, 182807, 97551, 14876, 5205, 332, 12512]`

| 比較 | 結果 |
| --- | --- |
| 同一プロセスの連続要求 | 一致 |
| 27B-FP8 vs 27B-MXFP4（W4A4） | 一致 |
| PP=1 vs PP=2 | 一致 |

※ 9B と 27B は別モデルなので列が違うのが正しい（同じ prompt で比較しない）。

---

## 5. T09 MXFP4 W4A4 の実行器統合

* 選択条件: Quark の `quantization_config.global_quant_config.input_tensors`
  が `dtype=fp4` / `qscheme=per_group` / `group_size=32` / `scale_format=e8m0` のとき
  **W4A4**。`e4m3` または宣言無しは従来の **W4A8**
* `Fp8ModelFormat::mxfp4_activation` に解析結果を載せ、
  `PipelineOptions::mxfp4_activation` → `HipStepExecutorOptions` へ渡す
* 実行器: 活性化を `launch_activation_quantize_mxfp4`（断片順ニブル + E8M0）で量子化し、
  `launch_gemm_mxfp4_w4a4_decode1` / `_prefill` を呼ぶ。
  W4A4 のときは FP8 への swiglu 融合をしない（次の linear が E2M1 で量子化し直すため）
* `shelf-compute --act-quant auto|fp4|fp8` で A/B できる（既定 auto）。
  `server.py` と `serve_e2e.py` / `check_generation_parity.py` も同じ指定を通す
* 起動ログに選択結果を出す:
  `activation = MXFP4 (W4A4)（quark mxfp4 (fp4/group32/e8m0) + A=fp4 (W4A4)）`

### 見つけた欠陥と修正（W4A4 prefill の端数）

W4A4 prefill は単体テストが rows=128 / features=128 or 256 だけだったため、
**端数で HSA のメモリフォルト**を起こしていた（E2E の 24 トークンは tail 経路＝
decode1 なので通っていた。192 トークンの prefill で再現）。

| 端数 | 原因 | 修正 |
| --- | --- | --- |
| `rows` が 128 の倍数でない | 活性化の行群を 128 行タイルのまま読んでいた（存在しない行群を読む） | 実際の行数 `(rows+15)/16-1` で行群を clamp（読み値は捨てる） |
| `features` が 128 の倍数でない（例: 48） | 重みの出力行を clamp せずに読んでいた | `min(o, features-1)` で clamp（W4A8 と同じ） |

回帰テスト: `unit.hip_gemm_mxfp4` に rows=32/64/96、features=48、
rows=64×features=48 の W4A4 prefill を追加（すべて相対誤差 0）。
実測: 64 トークン入力の prefill が 210 ms（2 トークン decode 込み）で完走。

### E2E 数値・速度（27B Quark MXFP4・prompt 192 tok・greedy 8 tok）

| 実行 | 生成列 | 時間（8 tok） | ms/tok |
| --- | --- | --- | --- |
| 27B-MXFP4 **W4A4** | `[…, 332, 18]` | 855 ms | 106.8 |
| 27B-MXFP4 W4A8 | `[…, 332, 12512]` | 631 ms | 78.9 |
| 27B-FP8（参考） | `[…, 332, 12512]` | 588〜600 ms | 73.5〜75.0 |

* **数値**: 先頭 7 トークンは W4A4 / W4A8 / FP8 で一致し、8 個目で分岐
  （`18` vs `12512`）。T0（各 2 回）は全実行で一致
* **T2 判定**（`tests/scripts/compare_logits.py`、分岐位置の logits）:
  * 差: max abs 4.446 / mean abs 0.492 / 相対 L2 0.313
  * W4A4 の top1=18（17.506）、top2=12512 側は 17.235、**margin 0.271**
  * W4A8 の top1=12512（17.228）、top2 は 16.465、margin 0.763
  * margin（0.271）は量子化差の規模（max 4.446）より十分小さい
    → **T2 の「量子化・丸め由来」に該当**（バグではなく W4A4 の A 側 4 bit 化の誤差）。
    W4A4 は W4A8 より **約 1.35 倍遅い**（E2E。prefill の 0.57 倍を反映）
* 分岐を 1 トークンでも遅らせたい用途では W4A8 を既定にする選択肢がある。
  `--act-quant` で明示的に切り替えられる

---

## 6. T10 テストと文書

* `tests/CMakeLists.txt` に `serve.python` を追加:
  `tests/scripts/run_python_tests.sh` 経由で pytest を実行し、
  `pytest` / `aiohttp` が無い環境は exit 77 で skip
* `ctest`（CPU）: 26/26 PASS（`serve.python` を含む。当時の件数）
* pytest: 116 PASS（T18 の 8 件を追加。`draft_docs/ops_2026-11-01.md`）
* 更新した文書: 本ファイル、`draft_docs/engine_contract.md`、
  `draft_docs/serve_protocol.md`、`draft_docs/dependency_rules.md`、
  `draft_docs/directory_design.md`（D8）、`apps/server/README.md`、`README.md`
* `tests/scripts/serve_protocol.sh` の `shelf-serve` 表記を `shelf-compute` へ修正

---

## 7. 残り（次のラウンド以降）

* T09 の数値・速度の実測値を確定する
* T11〜T14（Prefill の固定費、HIP Graph、性能回帰、VRAM 配置）
* T18 の残り（idle timeout・複数キー・分散メトリクス。`draft_docs/ops_2026-11-01.md` §4）
* T15〜T18（Continuous Batching、Prefix Cache、CPU/GPU 協調、運用機能）
* `stop`（文字列の停止列）は未対応のまま 400。実装するなら engine_contract に
  停止列を足してから（T18 の外部公開と合わせて判断）
