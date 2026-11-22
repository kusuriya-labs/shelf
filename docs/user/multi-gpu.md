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
- EP は routed expert の expert ID を分割する。
- TP は backbone、shared expert、routed expert の matrix 軸を分割する。
- MoE は TP reduce の後に EP reduce を行い、routed expert の部分和を集約する。
  shared expert は TP reduce にだけ参加する。
- EP=1 では全 expert を各 TP rank に複製し、TP が各 expert の行列を分割する。
- EP>1 かつ TP=1 で head 数が EP で割り切れる場合、qwen4exp は backbone も
  EP rank で分割する。Qwen3.8-Flash-Next EXL3 は EP=2 で backbone も分割し、
  EP=4 は KV head が 4 で割れないため backbone を複製する。qwen35moe の EP 単独
  構成は backbone を複製する。

`--device` には `pipeline_parallel * expert_parallel * tensor_parallel` 個の
GPU を、PP → EP → TP の順で指定する。

```bash
./build/shelf-server \
    --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
    --pipeline-parallel 2 --expert-parallel 2 --tensor-parallel 1 \
    --device 0,1,2,3 \
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
| `Qwen3.8-Flash-Next-Quark-MXFP4` | `qwen4_exp` | MoE / 48 層 / 512 expert | MXFP4 | 対応 | 2 / 4 | 2 |
| `Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw` | `qwen4_exp` | MoE / 48 層 / 512 expert | EXL3 INT8 3.05bpw | 対応 | 2 / 4 | 1 |
| `Qwen3.8-Flash-Next-DFlash2` | `qwen3` | drafter / 5 層 | — | 未対応 | — | 1 |

- `qwen4_exp` の PP は各 stage が担当地の層、QSA state、PLE state を持つ。
  stage 間では HyperConnection の stream 全体（`hc_count * hidden_size`）を
 受け渡す。PP=2 と single-stage で `GENERATED_IDS` が一致することを確認済み。
- MoE の routed expert は EP で expert ID を分割する。backbone、shared expert、
  および各 routed expert の行列は TP で分割する。EP=1 では全 expert を複製して
  TP で行列を分ける。EP=2 / 4 と EP=2 + TP=2 で生成列が TP=2 と一致することを
  実機で確認済み。
- EXL3 INT8 の routed expert は N と K が 128 の倍数でなければならない。
  `Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw` は routed expert の intermediate が 640 で、
  TP=2 では 320 になり 128 の倍数にならないため、TP>1 では起動できない。
  EP で expert ID を分ける構成（`--expert-parallel 2/4 --tensor-parallel 1`）は使える。
- dense model は EP で何も分割せず、EP グループごとに同じ計算を複製する
  （EP を増やしても高速化しない）。
- QSA family は TP=2 までで、PP にも対応する（`qwen35qsa.md` 参照）。
  ただし QSA checkpoint がリポジトリに無いため実機検証は未実施。
- MTP は PP と併用できない。Qwen3.5 MoE の MTP block は routed expert の online
  MXFP4 再量子化を無効化したため未対応である（`mtp.md` 参照）。

## 制約

- PP > 1 では prefix cache を無効にする（`--prefix-cache 0`）。server は
  併用をエラーで拒否する。
- PP > 1 は DFlash2、MTP、hidden taps、`--dump-logits` と併用できない。
- PP stage 内の EP / TP 展開は配線済みである。routed expert の TP 分割は
  量子化形式の alignment 条件を満たす checkpoint だけが起動できる。
- octave KV は PP と併用できる。PP と EP / TP を同時に使う構成の性能は未検証。

詳細は [compute.md](compute.md)、[server.md](server.md)、
[developer/multi-gpu.md](../developer/multi-gpu.md)、
[developer/quantization.md](../developer/quantization.md) を参照。
