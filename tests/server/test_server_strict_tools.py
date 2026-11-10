#!/usr/bin/env python3
"""Real-model strict tool constraints through both HTTP APIs."""
import json
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ServerHarness, http_json, http_sse

SCHEMA = {"type":"object", "properties": {
    "text":{"type":"string", "enum":["  yes  ", "no"]},
    "count":{"type":"integer"}, "ratio":{"type":"number"},
    "flag":{"type":"boolean"},
    "items":{"type":"array", "items":{"type":"integer"}},
    "nested":{"type":"object", "properties":{"ok":{"type":"boolean"}},
              "required":["ok"], "additionalProperties":False},
    "optional":{"type":"string"}},
    "required":["text", "count", "ratio", "flag", "items", "nested"],
    "additionalProperties":False}

def check(args):
    v = json.loads(args)
    assert v['text'] in ('  yes  ', 'no'), repr(v)
    assert type(v['count']) is int and type(v['ratio']) in (int, float)
    assert type(v['flag']) is bool and isinstance(v['items'], list)
    assert all(type(i) is int for i in v['items'])
    assert type(v['nested']['ok']) is bool
    assert set(v) <= set(SCHEMA['properties'])

def main():
    if importlib.util.find_spec('xgrammar') is None:
        print('SKIP: xgrammar is not installed')
        return 77
    with ServerHarness(max_seq_len=2048) as server:
        f = {"name":"record", "description":"Record values", "parameters":SCHEMA, "strict":True}
        for choice in ('required', 'named', 'auto'):
            for stream in (False, True):
                body = {"model":"shelf", "messages":[{"role":"user", "content":"Call record with text yes, count 2, ratio 1.5, flag true, items [1], nested ok true."}],
                        "tools":[{"type":"function", "function":f}], "temperature":0.8, "max_tokens":256,
                        "chat_template_kwargs":{"enable_thinking":False}, "stream":stream,
                        "tool_choice": {"type":"function", "function":{"name":"record"}} if choice == 'named' else choice}
                if stream:
                    events = list(http_sse(server.base_url + '/chat/completions', body))
                    calls = [c for e in events for c in e.get('choices',[]) for c in (c.get('delta',{}).get('tool_calls') or [])]
                    assert calls, events
                    name = calls[0]['function']['name']
                    args = ''.join(c['function'].get('arguments','') for c in calls)
                else:
                    calls = http_json(server.base_url + '/chat/completions', body)['choices'][0]['message']['tool_calls']
                    name, args = calls[0]['function']['name'], calls[0]['function']['arguments']
                assert name == 'record'
                check(args)
                print('chat', choice, stream, 'OK', flush=True)
        # A named choice must exclude the other available function, and the
        # grammar must start only after the generated reasoning close token.
        other = dict(f, name='other')
        body = {"model":"shelf", "messages":[{"role":"user", "content":"Call record with suitable values."}],
                "tools":[{"type":"function", "function":f}, {"type":"function", "function":other}],
                "tool_choice":{"type":"function", "function":{"name":"record"}},
                "temperature":0.8, "max_tokens":1024, "chat_template_kwargs":{"enable_thinking":True}}
        message = http_json(server.base_url + '/chat/completions', body)['choices'][0]['message']
        assert message.get('reasoning_content')
        call = message['tool_calls'][0]
        assert call['function']['name'] == 'record'
        check(call['function']['arguments'])
        print('chat named with thinking OK', flush=True)
        for stream in (False, True):
            body = {"model":"shelf", "input":"Call record with count 2 and the other required values.",
                    "tools":[{"type":"function", **f}], "tool_choice":{"type":"function", "name":"record"},
                    "max_output_tokens":1024, "temperature":0.8, "stream":stream}
            if stream:
                events = list(http_sse(server.base_url + '/responses', body))
                args = ''.join(e['delta'] for e in events if e.get('type') == 'response.function_call_arguments.delta')
                assert any(e.get('item',{}).get('name') == 'record' for e in events)
            else:
                data = http_json(server.base_url + '/responses', body)
                call = next(i for i in data['output'] if i['type'] == 'function_call')
                assert call['name'] == 'record'
                args = call['arguments']
            check(args)
            print('responses', stream, 'OK', flush=True)
    return 0

if __name__ == '__main__': raise SystemExit(main())
