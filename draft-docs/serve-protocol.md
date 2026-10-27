# serve-protocol — OpenAI API 互換 Server のプロトコル仕様（骨）

状態: 骨（**Status → HTTP の写像を本文とした**。エンジン抽象と JSONL の詳細は §7 の未決参照）
対象: `draft-docs/directory_design.md` §3.5（パス A / パス B、`src/api/` の Server 層）
関連: `draft-docs/dependency-rules.md`（依存方向）、`draft-docs/parity-policy.md`（検証の基準）

本書は Server 層（`src/api/`）と外部クライアントの契約を定める。目的はただ一つ、
**エンジン内部の `shelf::Status` を Server にも、クライアントにも漏らさないこと**。
Server はエンジンを「`code` と `message` だけ」で知り、クライアントは OpenAI の
error envelope だけを読む。

---

## 1. エラー応答の形（確定）

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

* envelope は **4 フィールド固定**（`message` / `type` / `param` / `code`）。
  OpenAI 公式の形と一致する
* **エラー型は 2 種類だけ**: 受付・引数起因（400 系）とサーバ起因（500 系）。
  例外／エラー値をこの 2 種類に正規化してから HTTP を決める
* **`type` は status から機械生成**する（`status < 500` → `invalid_request_error`、
  `500 以上` → `server_error`）。分岐を書かなくてよく、公式の分類と一致する。
  専用 type（`not_found_error` / `rate_limit_error` / `authentication_error` など）は
  クライアント要件が出たときだけ §7 に足す
* `message` は**クライアントが直す方法が分かる文言**にする。内部パス・スタック・
  デバイスハンドルは含めない

## 2. 3 段の正規化（確定）

エラーは必ず 3 段を通る。**段を飛ばして `Status::Code` を HTTP に直結させない**。

```
（パス A）JSONL の error event（code / message）
        → クライアント側例外（engine error / invalid / cancelled / protocol）
（パス B）仮想 IF の返すエラー値
        ↓
受付・サービス層で 2 種類に正規化（protocol = 400 系 / service = 500 系）
        ↓
HTTP ハンドラが type を status から生成し envelope を組む
```

規約:

1. `shelf::Status::Code` は **`code` に写してから** Server に出る（Server が shelf の
   型を知らない — パス A の利点を壊さない）
2. 受付時に分かるエラー（未知フィールド・非対応パラメータ）は **エンジンに到達する前**に
   400 で弾く（エンジン由来の `unsupported_parameter` は本末転倒）
3. 未処理例外は **ログに残し、クライアントには 500 のみ**（トレースバックを出さない）
4. クライアント切断（`ConnectionReset` / `BrokenPipe` / キャンセル）はエラー応答を作らない

## 3. `Status::Code` → `code` → HTTP の対応表（確定・本文）

| `shelf::Status::Code` | `code` | HTTP | `type` | 備考 |
| --- | --- | --- | --- | --- |
| `invalid_argument` | `invalid_argument` | 400 | invalid_request_error | 分かれば `param` を添える |
| `overflow` | `overflow` | 400 | invalid_request_error | |
| `out_of_range`（文脈超過） | `context_length_exceeded` | 400 | invalid_request_error | 追加フィールド §5-② を添えるとクライアントが履歴を詰めて再試行できる |
| `unsupported` | `unsupported_parameter` | 400 | invalid_request_error | 本来は受付時に弾く（§2-2）。エンジンから来た場合は要因を `param` に |
| `not_found` | `model_not_found` | 404 | invalid_request_error | 専用 type は §7 |
| `invalid_state` | **`busy` / `state_error` に分岐** | **409 / 500** | invalid_request_error / server_error | 再試行可能（busy）と壊れている（broken）を分ける。**ここでしか決らない** |
| `insufficient_memory` | `overloaded` | 503 | server_error | リトライ可。サーバ側ログには原因を残す |
| `device_error` | `device_error` | 500 | server_error | |
| 未知・内部 | `internal_error` | 500 | server_error | §2-3 |

`invalid_state` の 2 分岐は、**各呼び出し点がどちらかを宣言する**（例: KV 満杯の受け入れ拒否は
`busy`、状態テーブルの破損は `state_error`）。宣言を漏らすと 500 扱いになるので、
分岐表はこの文書に追記していく。

## 4. 運用系 status（確定・HTTP レイヤの一般則）

エンジンを経由しない、Server 自身の判定で返すもの。

| status | 条件 |
| --- | --- |
| 404 | 未知のルート（`/v1/*` 以外も同じ envelope で返す） |
| 413 | リクエスト本文が上限超過 |
| 415 | `Content-Type` が `application/json` でない |
| 403 | 許可されない Origin（API key 検証が要件になったときは 401 を §7 に追加） |
| 409 | Server 側で既に進行中のリクエストと衝突（エンジン由来の `busy` とは別。発生源をログで区別する） |
| 503 | エンジン未起動・停止中（`overloaded` と同じ type） |

## 5. 追加フィールド（有限に固定する）

envelope に足してよいのは次の 2 つだけ。追加はここに書き足してから実装する。

1. `supported`: 列挙値エラー（未知の値を名指ししたとき）に**許容値の一覧**。
   クライアントが即再試行できる
2. `n_prompt_tokens` / `n_ctx`: コンテキスト超過時。**クライアントが履歴を圧縮して
   再試行できる唯一の実用情報**（名前は llama.cpp 互換に合わせる）

## 6. 検証

* **表駆動テスト**: フェイクエンジン（§7 の JSONL フェイク / 仮想 IF フェイク）が
  §3 の各行の `Status::Code` を発生させ、**status / type / code / param の 4 点**を検証する
* envelope の形（4 フィールドの存在と型）を固定フィクスチャで照合する
* 「traceback が出ないこと」「受付時の 400 がエンジンに届かないこと」も表に 1 行ずつ足す
* parity と同様、**この文書の表とテストは 1 対 1**。表を変えたらテストを変える

## 7. 未決

**解消済み（記録）**: エンジン抽象と JSONL の詳細（メッセージ集合・`capabilities` 握手・
背圧・切断時挙動）は `engine-contract.md` に書いた。本書と 1 対 1（位置づけは
`draft-docs/directory_design.md` §3.5 / §6）。

* 専用 `type` を採るか（`not_found_error` / `rate_limit_error` / `authentication_error`）—
  互換クライアントが読むことを確認してから
* 認証（API key）とレート制限（429）の要否
* パス B（C++ 同プロセス）の場合、§3 の `code` をどう表すか（仮想 IF のエラー値に
  同名フィールドを持たせて表を無変更で使えるようにする想定）

## 8. 出典（写像の選定根拠）

* **PhaseShift（土台）**: `src/apps/server/serverlib/openai_protocol.py`
  （`OpenAIProtocolError` が `message/param/code/status`、`ChatServiceError`、`error_body` が
  envelope 4 フィールド）、`serverlib/chat_service.py:241-245`（エンジン例外 → 2 型への正規化）、
  `phaseshift_server.py`（`status=exc.status`、404 / 500 の扱い、traceback をクライアントに出さない）
  ※ `/opt/zen/work/references/phaseshift-fp8`
* **Strata（運用 status と type の機械生成）**: `serve/server.py` の `error(code, message)`
  （`type = "invalid_request_error" if code < 500 else "server_error"`）、
  `docs/VRAM_ELASTIC.md` の応答表（400 / 403 / 409 model_busy / 415 / 503）
  ※ `/opt/zen/work/references/Strata`
* **radiance（付加フィールド 2 つ）**: `core/server/oai.h`（`ApiError{status, type, message,
  param, code, supported, n_prompt_tokens, n_ctx}` と `err_bad` / `err_overloaded` 等のファクトリ）
  ※ `/opt/zen/work/references/radiance`
* 不採用: Strata の envelope が `param` / `code` を持たないこと、radiance の独自 status
  （426 など）— 要件が出たとき §4 に追記してから採る
