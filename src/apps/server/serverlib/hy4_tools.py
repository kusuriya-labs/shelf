import json
import re


CALLS_OPEN = '<tool_calls:opensource>'
CALLS_CLOSE = '</tool_calls:opensource>'
CALL_OPEN = '<tool_call:opensource>'
CALL_CLOSE = '</tool_call:opensource>'
KEY_OPEN = '<arg_key:opensource>'
KEY_CLOSE = '</arg_key:opensource>'
VALUE_OPEN = '<arg_value:opensource>'
VALUE_CLOSE = '</arg_value:opensource>'


def parse_answer(text, tools):
    functions = {item['function']['name']: item['function'] for item in tools}
    start = text.find(CALLS_OPEN)
    if start < 0:
        if CALL_OPEN in text or CALLS_CLOSE in text:
            raise ValueError('不正な Hy4 tool call 形式です')
        return text, []
    end = text.find(CALLS_CLOSE, start + len(CALLS_OPEN))
    if end < 0:
        raise ValueError('Hy4 tool call が途中で終了しました')
    content = text[:start] + text[end + len(CALLS_CLOSE):]
    if CALLS_OPEN in content or CALLS_CLOSE in content:
        raise ValueError('複数の tool call コンテナは未対応です')
    body = text[start + len(CALLS_OPEN):end]
    calls = []
    while body.strip():
        body = body.lstrip()
        if not body.startswith(CALL_OPEN):
            raise ValueError('tool call の開始がありません')
        body = body[len(CALL_OPEN):]
        close = body.find(CALL_CLOSE)
        if close < 0:
            raise ValueError('tool call の終端がありません')
        call, body = body[:close], body[close + len(CALL_CLOSE):]
        name = call.split(KEY_OPEN, 1)[0].strip()
        if name not in functions or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'未定義の関数です: {name}')
        schema = functions[name].get('parameters') or {}
        properties = schema.get('properties') or {}
        args = {}
        rest = call[len(call.split(KEY_OPEN, 1)[0]):]
        while rest:
            if not rest.startswith(KEY_OPEN):
                raise ValueError('引数の開始がありません')
            rest = rest[len(KEY_OPEN):]
            key, sep, rest = rest.partition(KEY_CLOSE)
            if not sep or not key or key in args or not rest.startswith(VALUE_OPEN):
                raise ValueError('引数名または値が不正です')
            raw, sep, rest = rest[len(VALUE_OPEN):].partition(VALUE_CLOSE)
            if not sep:
                raise ValueError('引数値の終端がありません')
            prop = properties.get(key, {})
            kind = prop.get('type')
            if kind == 'string' or kind is None:
                value = raw
            else:
                try:
                    value = json.loads(raw)
                except ValueError as exc:
                    raise ValueError(f'引数 {key} は JSON ではありません') from exc
            args[key] = value
        calls.append({'type': 'function', 'function': {'name': name, 'arguments': args}})
    if not calls:
        raise ValueError('tool call がありません')
    return content, calls
