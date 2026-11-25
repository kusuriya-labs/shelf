import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qwen35_gdn_tiny_fixture import bf16_bits, write_safetensors


def check_dump(bench, root, vocab, head_rows):
    hidden = 256
    config = dict(model_type="hunyuan_v1_dense", num_hidden_layers=1, hidden_size=hidden,
                  num_attention_heads=2, num_key_value_heads=2, intermediate_size=hidden,
                  vocab_size=vocab, rms_norm_eps=1e-5, max_position_embeddings=128)
    (root / "config.json").write_text(json.dumps(config))
    prefix = "model.layers.0."
    weights = {
        "model.embed_tokens.weight": np.eye(vocab, hidden),
        "lm_head.weight": np.eye(head_rows, hidden),
        "model.norm.weight": np.ones(hidden),
    }
    for name in ("input_layernorm", "post_attention_layernorm"):
        weights[prefix + name + ".weight"] = np.ones(hidden)
    for name in ("self_attn.query_layernorm", "self_attn.key_layernorm"):
        weights[prefix + name + ".weight"] = np.ones(hidden // 2)
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        weights[prefix + name + ".weight"] = np.zeros((hidden, hidden))
    write_safetensors(root / "model.safetensors",
                      {name: ("BF16", bf16_bits(value)) for name, value in weights.items()})

    ids = [0, vocab - 1, 1, vocab // 2, 2, 3]
    tokens = root / "tokens.bin"
    tokens.write_bytes(struct.pack("<8sIQ6i", b"PSKLDTOK", 1, len(ids), *ids))
    dump = root / "logits.bin"
    subprocess.run([bench, "quality", "--model-dir", str(root), "--tokens-file", str(tokens),
                    "--context", "4", "--prefill-chunk", "4", "--positions", "1",
                    "--arena-gib", "1", "--dump-prefill", "--dump", str(dump)], check=True)
    with dump.open("rb") as stream:
        header = struct.unpack("<8sIIQII", stream.read(32))
        assert header == (b"PSQUAL01", 1, vocab, len(ids) - 1, 0, 1), header
        for row, token in enumerate(ids[:-1]):
            assert struct.unpack("<iI", stream.read(8)) == (ids[row + 1], row + 1)
            logits = np.frombuffer(stream.read(vocab * 4), dtype="<f4")
            expected = np.zeros(vocab)
            expected[token] = hidden ** 0.5
            np.testing.assert_allclose(logits, expected, rtol=0.01, err_msg=f"row={row}")
        assert not stream.read(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for vocab, head_rows in ((127, 127), (126, 126), (127, 128)):
            check_dump(sys.argv[1], Path(tmp), vocab, head_rows)
