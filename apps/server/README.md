# shelf server（任意コンポーネント / パス A）

`draft_docs/directory_design.md` §3.5 のパス A（OpenAI 互換の HTTP 面を Python で持ち、
エンジンは `shelf-compute` の JSONL で駆動する）の実装。

**このディレクトリはビルド（CMake）に結線していません。** 製品のビルド・ctest は
C++ のみのまま（Python 依存は「サーバを動かすときだけ」）。移植元は
`references/phaseshift-fp8/src/apps/server/`（自作品）。

## いま入っているもの

| ファイル | 内容 | 状態 |
| --- | --- | --- |
| `engine_client.py` | `shelf-compute` の JSONL クライアント（`request_id` は文字列・`min_p` 送信・`done.seed`） | 実装済み |
| `openai_errors.py` | `serve_protocol.md` §3 の code → HTTP 表と envelope（4 フィールド） | 実装済み |
| `tests/fake_engine.py` | モデル・GPU 不要のフェイクエンジン | 実装済み |
| `tests/test_engine_client.py` | 上記のテスト（pytest） | 19 件通過 |
| `server.py` | aiohttp の HTTP 層（`/healthz`・`/v1/models`・`/v1/chat/completions` のストリーム/非ストリーム）と受付時検証 | 実装済み |
| `tests/test_server_http.py` | HTTP 面の表駆動テスト（フェイクエンジン・実ポート不要） | 10 件通過 |

## 動かし方

```bash
# テスト（モデル・GPU・ポート不要）
/opt/zen/.venv/bin/python -m pytest apps/server/tests -q

# 実エンジンに繋ぐ（要 GPU）
/opt/zen/.venv/bin/python - <<'PY'
import asyncio, sys
sys.path.insert(0, "apps/server")
from engine_client import EngineClient

async def main():
    c = EngineClient(["./build/hip/apps/shelf-compute", "--model", "models/Qwen3.5-9B-FP8", "--gpu", "0"])
    caps = await c.start()
    print("capabilities:", caps)
    ids = []
    async for ev in c.generate([248045, 846, 198], max_new_tokens=4):
        ids.append(ev["token_id"])
    print("generated:", ids)
    await c.close()

asyncio.run(main())
PY
```

## 次の段（残り）

1. `server.py`: aiohttp の `/v1/chat/completions`（ストリーム/非ストリーム）、受付時の検証（未知
   フィールド・非対応パラメータをエンジンに渡す前に 400）
2. chat template の実機確認: `HfTokenizer` は `AutoProcessor.apply_chat_template` を使う実装を
   入れたが、**Qwen3.5 での実機確認は未実施**。思考モード制御・tool 呼び出しは参照の
   `phaseshift_chat/{codec,reasoning}.py` から未移植
3. 実モデルでの `/v1/chat/completions` 通し（要 GPU・要 chat template 確認）
