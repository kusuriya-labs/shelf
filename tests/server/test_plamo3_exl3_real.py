#!/usr/bin/env python3
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plamo3_server import main
from support import Checker, ServerHarness, http_json

model = Path(os.environ.get("SHELF_PLAMO_EXL3_MODEL_DIR", ""))
if not (model / "config.json").is_file():
    print("SKIP: SHELF_PLAMO_EXL3_MODEL_DIR is not a model directory")
    sys.exit(77)
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "plamo3" or config.get("quantization_config", {}).get("quant_method") != "exl3":
    raise ValueError("expected a PLaMo EXL3 model")
os.environ["SHELF_PLAMO_MODEL_DIR"] = str(model)
os.environ["SHELF_SERVER_MODEL_DIR"] = str(model)
result = main()
check = Checker("plamo3-exl3-startup")
try:
    with tempfile.TemporaryDirectory() as temp:
        log = Path(temp) / "server.log"
        with ServerHarness(log_path=log, max_seq_len=2048, max_concurrent_requests=8) as server:
            text = log.read_text()
            check.check("warmup-before-ready",
                        "PLaMo EXL3 prefill warmup:" in text
                        and "PLaMo EXL3 chat warmup:" in text, text[-2000:])
            body = {"model": "shelf", "messages": [{"role": "user", "content": "今日は良い天気ですね。"}],
                    "temperature": 0, "max_tokens": 48,
                    "chat_template_kwargs": {"source_lang": "ja", "target_lang": "en"}}
            response = http_json(server.base_url + "/chat/completions", body)
            choice = response["choices"][0]
            check.check("translation-after-warmup", bool(choice["message"]["content"].strip())
                        and choice["finish_reason"] == "stop", repr(response))
except Exception as exc:
    check.check("exception", False, repr(exc))
sys.exit(1 if result or check.done() else 0)
