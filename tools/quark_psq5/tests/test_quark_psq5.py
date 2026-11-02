import json
import math
import os

import pytest
import torch
import torch.nn as nn

from quark_psq5 import (
    PSQ5CB28Config,
    PSQ5CB28Processor,
    dequantize_weight,
    make_scale_codec,
    quantize_weight,
    read_artifact,
    reconstruct,
    write_artifact,
)
from quark_psq5.artifact import build_manifest, stream_keys
from quark_psq5.codebook import (
    BLOCK_K,
    CB28,
    cb28_to_e4m3,
    nearest_index,
    pack_magnitude,
    pack_sign,
    unpack_magnitude,
    unpack_sign,
)
from quark_psq5.scale_codec import ue4m3_table, ue4m4_table, ue4m4_value
from quark.torch.algorithm.api import get_processor
from quark.torch.quantization.config.algo_configs import get_supported_algorithm_types
from quark.torch.quantization.config.config import _load_quant_algo_config_from_dict

SPEC_CB28 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28]
SPEC_E4M3_RAW = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16]


def e4m3_to_float(raw: int) -> float:
    sign = -1.0 if (raw & 0x80) else 1.0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    if exponent == 0:
        return sign * mantissa * (2.0 ** -9)
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def test_cb28_matches_spec() -> None:
    assert list(CB28) == SPEC_CB28


def test_cb28_to_e4m3_matches_spec_table() -> None:
    q5 = torch.arange(32, dtype=torch.int64)
    raw = cb28_to_e4m3(q5)
    for magnitude, expected in enumerate(SPEC_E4M3_RAW):
        assert int(raw[magnitude]) == expected
        assert int(raw[magnitude | 0x10]) == expected | 0x80


def test_e4m3_times_512_is_cb28_for_all_codes() -> None:
    for q5 in range(32):
        magnitude = q5 & 0x0F
        sign = (q5 & 0x10) << 3
        raw = (magnitude + ((magnitude >> 3) * (magnitude & 0x07))) | sign
        expected = -SPEC_CB28[magnitude] if (q5 & 0x10) else SPEC_CB28[magnitude]
        assert math.isclose(e4m3_to_float(raw) * 512.0, expected, rel_tol=0.0, abs_tol=1e-9)


def test_nearest_index_rounds_to_closest_code() -> None:
    values = torch.tensor([0.0, 0.4, 0.5, 0.6, 9.0, 9.4, 9.5, 9.6, 27.0, 31.0])
    assert nearest_index(values).tolist() == [0, 0, 0, 1, 8, 9, 9, 9, 15, 15]


def test_pack_unpack_roundtrip() -> None:
    generator = torch.Generator().manual_seed(7)
    index = torch.randint(0, 16, (5, 3, BLOCK_K), generator=generator)
    sign = torch.rand((5, 3, BLOCK_K), generator=generator) < 0.5
    assert torch.equal(unpack_magnitude(pack_magnitude(index)), index)
    assert torch.equal(unpack_sign(pack_sign(sign)), sign)


def test_pack_byte_layout_matches_spec() -> None:
    index = torch.arange(BLOCK_K, dtype=torch.int64) % 16
    packed = pack_magnitude(index)
    assert int(packed[0]) & 0x0F == 0
    assert int(packed[0]) >> 4 == 1
    assert int(packed[15]) & 0x0F == 14
    assert int(packed[15]) >> 4 == 15
    sign = torch.zeros(BLOCK_K, dtype=torch.bool)
    sign[0] = True
    sign[7] = True
    sign[8] = True
    sign[31] = True
    packed_sign = pack_sign(sign)
    assert int(packed_sign[0]) == 0b10000001
    assert int(packed_sign[1]) == 0b00000001
    assert int(packed_sign[2]) == 0
    assert int(packed_sign[3]) == 0b10000000


def test_ue4m3_scale_error_bound() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    generator = torch.Generator().manual_seed(11)
    scale = torch.rand((4096,), generator=generator) * 0.05 + 1e-4
    stored = codec.encode(scale)
    assert stored.dtype == torch.uint8
    assert int(stored.max()) <= 0x7E
    restored = codec.decode(stored)
    relative = ((restored - scale).abs() / scale)
    assert float(relative.max()) <= 1.0 / 16.0 + 1e-6


def test_ue4m3_zero_scale_stays_zero() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    stored = codec.encode(torch.zeros(4))
    assert stored.tolist() == [0, 0, 0, 0]
    assert codec.decode(stored).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_ue4m3_sub_min_positive_rounds_up() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    stored = codec.encode(torch.tensor([1e-9]))
    assert float(codec.decode(stored)[0]) == pytest.approx((1.0 / 1024.0) / 512.0)


def test_ue4m3_table_is_monotonic() -> None:
    table = ue4m3_table()
    assert table[0] == 0.0
    assert torch.all(table[1:] > table[:-1])


@pytest.mark.parametrize("name", ["ue4m4", "ue4m3", "fp16", "bf16"])
def test_scale_codec_relative_error(name: str) -> None:
    codec = make_scale_codec(name, 512.0)
    generator = torch.Generator().manual_seed(13)
    scale = torch.rand((8192,), generator=generator) * 0.05 + 1e-4
    restored = codec.decode(codec.encode(scale))
    relative = ((restored - scale).abs() / scale)
    limit = {"ue4m4": 1.0 / 32.0, "ue4m3": 1.0 / 16.0,
             "fp16": 1.0 / 1024.0, "bf16": 1.0 / 256.0}[name]
    assert float(relative.max()) <= limit + 1e-5


def test_quantize_reports_bits_per_weight() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn((64, 128), generator=generator)
    result = quantize_weight(weight, codec)
    rows, columns = result["shape"]
    blocks = columns // BLOCK_K
    total_bytes = (result["magnitude"].numel() + result["sign"].numel()
                   + result["scale"].numel())
    assert total_bytes == rows * blocks * 21
    assert total_bytes * 8 / (rows * columns) == pytest.approx(5.25)


def test_quantize_reconstruction_matches_dequantize() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn((33, 70), generator=generator) * 0.03
    result = quantize_weight(weight, codec)
    rebuilt = dequantize_weight(
        result["magnitude"], result["sign"], result["scale"], codec,
        result["shape"][0], result["shape"][1], result["block_k"])
    assert torch.equal(rebuilt, result["reconstruction"])


def test_quantize_error_stays_near_codebook_floor() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn((256, 512), generator=generator) * 0.02
    blocks = weight.reshape(256, 16, BLOCK_K)
    exact_scale = blocks.abs().amax(dim=-1) / 28.0
    normalized = blocks / exact_scale.unsqueeze(-1)
    index = nearest_index(normalized.abs())
    table = torch.tensor(CB28, dtype=torch.float32)
    floor = torch.where(blocks < 0, -table[index], table[index]) * exact_scale.unsqueeze(-1)
    floor_error = float((blocks - floor).pow(2).sum())
    result = quantize_weight(weight, codec)
    actual_error = float((weight - result["reconstruction"]).pow(2).sum())
    assert actual_error <= floor_error * 1.10


def test_quantize_zero_block() -> None:
    codec = make_scale_codec("ue4m3", 512.0)
    weight = torch.zeros((2, BLOCK_K))
    weight[1, :4] = torch.tensor([1e-9, -1e-9, 5e-10, 0.0])
    result = quantize_weight(weight, codec)
    assert torch.isfinite(result["reconstruction"]).all()
    assert int(result["scale"][0]) == 0
    assert float(result["reconstruction"][0].abs().sum()) == 0.0


def test_registry_dispatch() -> None:
    assert get_processor("psq5_cb28") is PSQ5CB28Processor
    assert "psq5_cb28" in get_supported_algorithm_types()


def test_config_deserializes_from_dict() -> None:
    config = _load_quant_algo_config_from_dict({
        "name": "psq5_cb28",
        "block_k": 32,
        "scale_format": "fp16",
        "exclude": ["lm_head"],
    })
    assert isinstance(config, PSQ5CB28Config)
    assert config.scale_format == "fp16"
    assert config.exclude == ["lm_head"]
    assert config.scale_fold == 512.0


def test_processor_quantizes_linears_and_writes_artifact(tmp_path) -> None:
    torch.manual_seed(29)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    original = model[0].weight.detach().clone()
    output = os.path.join(tmp_path, "psq5.safetensors")
    config = PSQ5CB28Config(output_path=output, include=["0"], exclude=["2"])
    processor = PSQ5CB28Processor(model, config, None)
    processor.apply()

    assert processor.processed == ["0"]
    assert processor.skipped == ["2"]
    streams, manifest = read_artifact(output)
    assert manifest["format"] == "PSQ5-CB28"
    assert manifest["block_k"] == 32
    assert manifest["codebook"] == SPEC_CB28
    assert manifest["bits_per_weight"] == pytest.approx(5.25)
    assert manifest["scale_format"] == "ue4m4"
    assert set(streams) == {
        stream_keys("0")["magnitude"], stream_keys("0")["sign"], stream_keys("0")["scale"],
    }
    rebuilt = reconstruct(streams, manifest, "0")
    assert torch.allclose(rebuilt, model[0].weight.detach(), atol=0, rtol=0)
    assert not torch.equal(model[0].weight.detach(), original)
    assert "psq5_magnitude" in model[0]._buffers
    assert "psq5_magnitude" not in dict(model.state_dict())
    manifest_from_disk = json.loads(open(output + ".json", encoding="utf-8").read())
    assert manifest_from_disk == manifest


def test_processor_excludes_everything_warns(tmp_path) -> None:
    model = nn.Linear(32, 8)
    config = PSQ5CB28Config(exclude=["*"])
    processor = PSQ5CB28Processor(model, config, None)
    processor.apply()
    assert processor.processed == []
    assert processor.skipped == [""]


def test_artifact_manifest_bytes_per_block(tmp_path) -> None:
    codec = make_scale_codec("fp16", 512.0)
    config = PSQ5CB28Config(scale_format="fp16")
    manifest = build_manifest({}, config, codec)
    assert manifest["bytes_per_block"] == 22
    assert manifest["bits_per_weight"] == pytest.approx(5.5)
    assert manifest["scale_storage"] == "float16"
    assert math.isclose(manifest["scale_fold"], 512.0)


def test_apply_advanced_quant_algo_dispatches_psq5() -> None:
    from quark.torch.algorithm.api import apply_advanced_quant_algo
    from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType

    from quark.torch.quantization.observer.observer import PlaceholderObserver

    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    model.device = torch.device("cpu")
    before = model[0].weight.detach().clone()
    untouched = model[2].weight.detach().clone()
    weight_spec = QTensorConfig(
        dtype=Dtype.bfloat16, qscheme=QSchemeType.per_tensor, observer_cls=PlaceholderObserver)
    config = QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec),
        algo_config=[PSQ5CB28Config(exclude=["2"])],
    )
    result = apply_advanced_quant_algo(model, config, False, None)
    assert result is model
    assert not torch.equal(model[0].weight.detach(), before)
    assert "psq5_magnitude" in model[0]._buffers
    assert "psq5_magnitude" not in model[2]._buffers
    assert torch.equal(model[2].weight.detach(), untouched)


def test_ue4m4_table_is_monotonic_and_reserved() -> None:
    table = ue4m4_table()
    assert table[0] == 0.0
    assert torch.all(table[1:] > table[:-1])
    assert len(table) == 0xF0
    assert ue4m4_value(0xF0) != ue4m4_value(0xF0)
    assert ue4m4_value(0x00) == 0.0
    assert ue4m4_value(0xEF) == pytest.approx((1.0 + 15.0 / 16.0) * (2.0 ** 7))


def test_ue4m4_scale_error_is_half_of_ue4m3() -> None:
    codec = make_scale_codec("ue4m4", 512.0)
    generator = torch.Generator().manual_seed(37)
    scale = torch.rand((8192,), generator=generator) * 0.05 + 1e-4
    stored = codec.encode(scale)
    assert stored.dtype == torch.uint8
    assert int(stored.max()) <= 0xEF
    relative = ((codec.decode(stored) - scale).abs() / scale)
    assert float(relative.max()) <= 1.0 / 32.0 + 1e-6


def test_ue4m4_quantize_reconstruction_matches_dequantize() -> None:
    codec = make_scale_codec("ue4m4", 512.0)
    generator = torch.Generator().manual_seed(41)
    weight = torch.randn((48, 96), generator=generator) * 0.01
    result = quantize_weight(weight, codec)
    rebuilt = dequantize_weight(
        result["magnitude"], result["sign"], result["scale"], codec,
        result["shape"][0], result["shape"][1], result["block_k"])
    assert torch.equal(rebuilt, result["reconstruction"])


def test_ue4m4_bits_per_weight_is_5_25() -> None:
    codec = make_scale_codec("ue4m4", 512.0)
    config = PSQ5CB28Config()
    manifest = build_manifest({}, config, codec)
    assert manifest["scale_format"] == "ue4m4"
    assert manifest["bytes_per_block"] == 21
    assert manifest["bits_per_weight"] == pytest.approx(5.25)
