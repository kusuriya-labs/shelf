#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src/apps/server"))
import shelf_server
from shelf_chat import codec
from serverlib.chat_service import ChatService
from serverlib.openai_protocol import parse_chat_request

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()
processor = codec.load_processor(args.model)
service = ChatService(processor, None, "shelf", 2048, vocab_size=107520)
body = {"model": "shelf", "messages": [{"role": "user", "content": "今日は良い天気ですね。午後は公園を散歩しましょう。"}],
        "temperature": 0, "max_tokens": 192,
        "chat_template_kwargs": {"source_lang": "ja", "target_lang": "en"}}
request = parse_chat_request(body, inference_budget_tokens=service.inference_budget_ceiling)
for i in range(3):
    start = time.monotonic()
    prepared = service.prepare(request)
    print(f"run={i + 1} seconds={time.monotonic() - start:.6f} tokens={len(prepared.prompt_ids)}", flush=True)
