# Responses API 設計メモ

2026-11-10。OpenAI の [Responses API](https://platform.openai.com/docs/api-reference/responses/create) と [ストリーミングガイド](https://developers.openai.com/api/docs/guides/streaming-responses) を参照する。

`POST /v1/responses` はステートレスなテキストと function calling の部分集合とする。`instructions` は system、文字列 input は user message に変換する。message のテキスト部分は連結し、function_call は assistant の tool_calls、function_call_output は tool message に変換する。reasoning は次の assistant message または function_call に結び付ける。履歴と tool の実行結果はクライアントが毎回送る。

Responses 形式の function 定義と生成パラメータを Chat の既存検証・生成経路に変換し、結果を reasoning、message、function_call item に戻す。stream は既存の生成イベントから Responses の SSE を構成し、切断時は生成を閉じる。キャッシュ済み入力 token は usage に引き継ぐ。

画像・音声・ファイル、hosted tool、構造化出力、状態付き継続、未対応パラメータは明示的に拒否する。`store`、`metadata`、`user` は受理するが保存・利用しない。`store: true` でも保存は行わない。
