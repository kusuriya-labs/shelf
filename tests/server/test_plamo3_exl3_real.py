#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plamo3_server import main

model = Path(os.environ.get("SHELF_PLAMO_EXL3_MODEL_DIR", ""))
if not (model / "config.json").is_file():
    print("SKIP: SHELF_PLAMO_EXL3_MODEL_DIR is not a model directory")
    sys.exit(77)
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "plamo3" or config.get("quantization_config", {}).get("quant_method") != "exl3":
    raise ValueError("expected a PLaMo EXL3 model")
os.environ["SHELF_PLAMO_MODEL_DIR"] = str(model)
os.environ["SHELF_SERVER_MODEL_DIR"] = str(model)
sys.exit(main())
