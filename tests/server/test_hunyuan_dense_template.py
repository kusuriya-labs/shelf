#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / 'src/apps/server'),
    str(ROOT / 'src/apps/common'),
    str(Path(__file__).resolve().parent),
]

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


async def run(checker, model_dir: Path):
    processor = codec.load_processor(str(model_dir))
    generated = processor.encode('こんにちは、世界。', add_special_tokens=False)
    compute = Compute(generated)
    service = ChatService(
        processor, compute, 'shelf', 1024, reasoning_close_token=-1,
        supports_reasoning=False,
    )
    messages = [
        {'role': 'user', 'content': 'Translate the following segment into Japanese, without additional explanation.\n\nHello, world.'},
    ]
    body = {'model': 'shelf', 'messages': messages}
    expected = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(expected, 'keys'):
        expected = expected['input_ids']
    async with TestClient(TestServer(create_app(service, compute))) as client:
        response = await client.post('/v1/chat/completions', json=body)
        payload = await response.json()
        checker.check(
            f'{model_dir.name}-response',
            response.status == 200
            and payload['choices'][0]['message']['content'] == 'こんにちは、世界。',
            repr(payload),
        )
        checker.check(
            f'{model_dir.name}-prompt',
            compute.prompt == list(expected),
            f'got {compute.prompt} vs {list(expected)}',
        )
        response = await client.post('/v1/chat/completions', json={**body, 'stream': True})
        raw = await response.text()
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        streamed = ''.join(
            choice.get('delta', {}).get('content', '')
            for chunk in chunks
            for choice in chunk.get('choices', [])
        )
        checker.check(
            f'{model_dir.name}-stream',
            response.status == 200 and streamed == 'こんにちは、世界。',
            raw,
        )


def main():
    dirs = []
    for key in ('SHELF_HUNYUAN_18B_DIR', 'SHELF_HUNYUAN_7B_DIR'):
        val = os.environ.get(key, '')
        if val:
            dirs.append(Path(val))
    if not dirs:
        for candidate in (
            Path('/home/homelab1/datapool/ai_models/safetensors/Hy-MT3-1.8B-exl3-4bpw-mul24'),
            Path('/home/homelab1/datapool/ai_models/safetensors/Hy-MT3-7B-exl3-4bpw-mul24'),
        ):
            if candidate.is_dir() and (candidate / 'tokenizer_config.json').is_file():
                dirs.append(candidate)
    if not dirs:
        print('SKIP: no Hy-MT3 tokenizer directory available')
        return 77
    checker = Checker('hunyuan-dense-template')
    for model_dir in dirs:
        asyncio.run(run(checker, model_dir))
    return checker.done()


if __name__ == '__main__':
    sys.exit(main())
