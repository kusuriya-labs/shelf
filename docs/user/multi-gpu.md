# マルチ GPU 実行

`shelf-compute` / `shelf-server` は pipeline parallel（PP）、expert parallel（EP）、
tensor parallel（TP）の 3 軸を受け付ける。8 GPU の例では次の割当になる。

```text
GPU 0 1 2 3 4 5 6 7
PP 0 0 0 0 1 1 1 1
EP 0 0 1 1 0 0 1 1
TP 0 1 0 1 0 1 0 1
```

- PP は layer を分割する（例: PP=2 で 0〜23 / 24〜47）。
- EP は routed expert を分割する。
- TP は rank 内の並列を表す。現在の MoE では expert 分割が TP に相当するため、
  EP と TP のランタイムへの展開は未配線である（後述）。

`--device` には `pipeline_parallel * expert_parallel * tensor_parallel` 個の
GPU を、PP → EP → TP の順で指定する。

```bash
./build/shelf-server \
    --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
    --pipeline-parallel 2 --expert-parallel 2 --tensor-parallel 2 \
    --device 0,1,2,3,4,5,6,7 \
    --prefix-cache 0 --port 8123
```

K128 一括セットを要求する量子化方式（例: EXL3 INT8 3.05bpw）では、
各 TP グループの後ろの rank に重みを多く割り当てる設計である。
これは前半 GPU がデスクトップ表示やマルチモーダル / MTP 層で VRAM を
消費するケースを想定しているが、現状は重み配置の実装まで配線されていない。

## モデルと量子化方式ごとの対応

| モデル | model_type | 構造 | 量子化 | PP | EP | TP |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-Quark-AWQ-MXFP4` | `qwen3_5` | dense / 64 層 | MXFP4 | 対応 | — | 2 |
| `Qwen3.8-27B-EXL3-INT8-4bpw` | `qwen3_5` | dense / 64 層 | EXL3 INT8 4bpw | 対応 | — | 2 |
| `Qwen3.8-Flash-Next-Quark-MXFP4` | `qwen4_exp` | MoE / 48 層 / 512 expert | MXFP4 | 未対応 | CLI 可 | 2 |
| `Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw` | `qwen4_exp` | MoE / 48 層 / 512 expert | EXL3 INT8 3.05bpw | 未対応 | CLI 可 | 2 |
| `Qwen3.8-Flash-Next-DFlash2` | `qwen3` | drafter / 5 層 | — | 未対応 | — | 1 |

- `qwen4_exp` は compute 側が pipeline execution 未対応を返す（`qwen4exp.md` 参照）。
- MoE の TP は expert 軸の分割として実装されている。EP は CLI で受け付けるが、
  stage 内での EP / TP 展開は未配線で、PP > 1 のときは各 stage の先頭 GPU
  だけが使われる。
- QSA family は TP=2 までで PP は未実装（`qwen35qsa.md` 参照）。
- MTP は PP と併用できず、grouped expert（MoE）との TP 併用も拒否される
  （`mtp.md` 参照）。

## 制約

- PP > 1 では prefix cache を無効にする（`--prefix-cache 0`）。server は
  併用をエラーで拒否する。
- PP > 1 は DFlash2、MTP、hidden taps、`--dump-logits` と併用できない。
- octave KV は PP と併用できるが、PP stage 内での EP / TP 展開は未配線のため
  性能は検証されていない。

詳細は [compute.md](compute.md)、[server.md](server.md)、
[developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md)、
[developer/quantization.md](../developer/quantization.md) を参照。
