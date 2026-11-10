#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/apps/server'), str(ROOT / 'src/apps/common'), str(Path(__file__).resolve().parent)]

from aiohttp.test_utils import TestClient, TestServer
from shelf_chat import codec
from shelf_server import create_app
from serverlib.chat_service import ChatService
from support import Checker


class Compute:
    def __init__(self, ids):
        self.ids = ids
        self.prompt = None

    async def shutdown(self):
        pass

    async def generate(self, prompt, *args, **kwargs):
        self.prompt = prompt
        return {'generated_ids': self.ids, 'finish_reason': 'stop'}

    async def generate_stream(self, prompt, *args, **kwargs):
        self.prompt = prompt
        for token in self.ids:
            yield {'token_id': token}
        yield {'done': {'generated_ids': self.ids, 'finish_reason': 'stop'}}


async def run(checker, processor):
    generated = processor.encode('Hello, world.', add_special_tokens=False)
    compute = Compute(generated)
    service = ChatService(processor, compute, 'shelf', 1024)
    messages = [{'role': 'user', 'content': 'こんにちは、世界。'}]
    args = {'source_lang': 'ja', 'target_lang': 'en', 'context': '挨拶',
            'glossary': [{'source': '世界', 'target': 'world'}]}
    body = {'model': 'shelf', 'messages': messages, 'chat_template_kwargs': args}
    expected = processor.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=True, **args)
    if hasattr(expected, 'keys'):
        expected = expected['input_ids']
    async with TestClient(TestServer(create_app(service, compute))) as client:
        response = await client.post('/v1/chat/completions', json=body)
        payload = await response.json()
        checker.check('translation-response', response.status == 200 and
                      payload['choices'][0]['message']['content'] == 'Hello, world.', repr(payload))
        checker.check('template-arguments', compute.prompt == list(expected), repr(compute.prompt))
        response = await client.post('/v1/chat/completions', json={**body, 'stream': True})
        raw = await response.text()
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        streamed = ''.join(choice.get('delta', {}).get('content', '')
                           for chunk in chunks for choice in chunk.get('choices', []))
        checker.check('translation-stream', response.status == 200 and
                      streamed == 'Hello, world.', raw)
        response = await client.post('/v1/chat/completions', json={**body, 'chat_template_kwargs': {}})
        checker.check('missing-target', response.status == 400, str(await response.text()))
        response = await client.post('/v1/chat/completions', json={**body, 'chat_template_kwargs': {'target_lang': 'EN'}})
        checker.check('invalid-target', response.status == 400, str(await response.text()))
        response = await client.post('/v1/chat/completions', json={**body, 'messages': [{'role': 'system', 'content': 'x'}, *messages]})
        checker.check('system-rejected', response.status == 400, str(await response.text()))


def main():
    model = Path(os.environ.get('SHELF_PLAMO_MODEL_DIR', ''))
    if not model.is_dir() or not (model / 'tokenizer_config.json').is_file():
        print('SKIP: SHELF_PLAMO_MODEL_DIR is not a PLaMo tokenizer directory')
        return 77
    checker = Checker('plamo3-template')
    asyncio.run(run(checker, codec.load_processor(str(model))))
    return checker.done()


if __name__ == '__main__':
    sys.exit(main())
