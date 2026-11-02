# quark_psq5

`docs/quant/PSQ5-CB28.md` の 5bit codebook 量子化（PSQ5-CB28）を AMD Quark の
algorithm plugin として実装したもの。

- Quark の `ALGORITHM_REGISTRY` に `psq5_cb28` として登録される
  （`quark.torch.algorithm.registry`、コアファイルは無改変）。
- `PSQ5CB28Processor` は `nn.Linear` の重みを W32 block + 8bit scale の 5bit へ置換し、
  復元値を fake-quant weight として書き戻す。
- artifact は SoA 3 ストリーム（magnitude 16B / sign 4B / scale 1B 毎 W32、
  **5.25 bpw**）の safetensors と `model.safetensors.json` で出力する。

## ファイル

| ファイル | 役割 |
|---|---|
| `codebook.py` | CB28、`cb28_to_e4m3`、bitpack / unpack、量子化・復元 |
| `scale_codec.py` | 8bit / 16bit scale 符号化（`ue4m4` / `ue4m3` / `fp16` / `bf16`） |
| `config.py` | `PSQ5CB28Config`（Quark の `AlgoConfig` 派生） |
| `processor.py` | `PSQ5CB28Processor`（Quark の `BaseAlgoProcessor` 派生） |
| `artifact.py` | SoA safetensors と JSON manifest の書出・読込・復元 |
| `tests/` | 仕様・pack・scale・registry・processor の単体検証 |

## 使い方

単体検証:

```bash
python3 -m pytest tools/quark_psq5/tests
```

モデルの量子化（safetensors 単位の file-to-file。`model.safetensors.index.json` を持つ
sharded model もそのまま受け付ける。sharded は shard 単位で並列処理し、
同じ shard 構成で出力する）:

```bash
python3 tools/quantize_psq5.py \
  --model-dir models/Qwen3.8-27B-DFlash2 \
  --output-dir models/Qwen3.8-27B-DFlash2-Quark-AWQ-PSQ5 \
  --scale-format ue4m4

python3 tools/quantize_psq5.py \
  --model-dir models/Qwen3.8-27B \
  --output-dir models/Qwen3.8-27B-Quark-AWQ-PSQ \
  --jobs 8 --verify-sample 0
```

既定の対象は `*q_proj.weight` / `*k_proj.weight` / `*v_proj.weight` /
`*o_proj.weight` / `*gate_proj.weight` / `*up_proj.weight` / `*down_proj.weight` /
`*in_proj_qkv.weight` / `*in_proj_z.weight` / `*linear_attn.out_proj.weight` /
`*kernel_projection.weight` / `fc.weight` / `*hidden_projection.weight` で、
`*codebook*` / `mtp.*` / `model.visual.*` は除外する。
`--include` / `--exclude` で上書きできる。
`--dry-run` で対象数と容量、`--verify-sample 0` で全数の bit exact 検証を行う。

Quark の設定経由で使う場合は、config JSON に次の `algo_config` を記述し、
探索前に `import quark_psq5` を行う:

```json
{"name": "psq5_cb28", "scale_format": "ue4m4", "block_k": 32}
```

## 依存

**`vendor/quark` はこのリポジトリに含めない。clone していないと本プラグインは
import できず、PSQ5-CB28 などの一部のオリジナル量子化モデルを作成できない**
（既存の BF16 / FP8 E4M3 は Quark が無くても扱える）。

```bash
git clone --depth 1 --branch release/0.13 https://github.com/amd/quark vendor/quark
# 計測時の commit: 313cb0b78ef364cb937598bc3431c46e28a226f4 (tag v0.13, MIT)
```

取得後、初回の HIP 拡張 JIT ビルドでは次の環境変数が必要
（`vendor/quark` 配下にビルド済み `.so` がある環境では不要）:

```bash
export PYTORCH_ROCM_ARCH=gfx1201
export CPLUS_INCLUDE_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/include
export LIBRARY_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
```

## 制約

出力 artifact は PhaseShift の runtime がロードできない（対応 format は
BF16 / FP8 E4M3 のみ）。本体側への接続は未実施。

## scale 符号化

いずれも fold = 512（`格納値 = 512 × scale`）で 1 つの W32 に 1 つの scale を置く。

| `scale_format` | 定義 | byte | bpw |
|---|---|---:|---:|
| `ue4m4`（既定） | 符号なし 指数4bit / 尾数4bit、バイアス 7、有効 `0x00..0xEF`、subnormal `man×2^-10`、最大 248 | 1 | 5.25 |
| `ue4m3` | PhaseShift 既存 codec（バイアス 8、有効 `0x00..0x7E`、最大 224） | 1 | 5.25 |
| `fp16` | IEEE half | 2 | 5.50 |
| `bf16` | bfloat16 | 2 | 5.50 |

## オフラインリカバリ（AWQ 等）について

本プラグインは重みの 5bit 化のみを担当する。AWQ / GPTQ 等のオフラインリカバリは
Quark の `algo_config` に本プラグインより前に並べて逐次実行できるが、
calibration forward を要するため対象モデルに HF 互換クラスが必要。
可否の確認結果と手順・リスクは
`docs/research/20261023/psq5_cb28_quark_plugin.md` の「次段階メモ」に記録している
（Qwen3.8-27B-FP8 は機構上可能、Qwen3.8-27B-DFlash2 は現状不可）。
