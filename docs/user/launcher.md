# shelf-launcher

resident model host の起動と child(`shelf-compute` / `shelf-server` /
`shelf-bench`)の起動を統括する process supervisor。Python実装で、GPU演算は
行わない。

launcher は model host を tensor parallel rank ごとに起動し、すべての host が
protocol ping に応答することを確認してから child を起動する。child の終了時に
すべての host を shutdown する。

不変条件:

- 指定されたターゲットモデルは必ず model host 経由でロードされる。launcher は
  child に `--model-host` と `--tp` を注入する。
- ドラフトモデルは child(compute / server)が local load する。launcher は
  `--dflash2-model-dir` を child に透過するのみで、host には warmup させない。

`--model-dir` / `--device` / `--tp` / `--model-host` は launcher が所有する。
child の argument に書くとエラーになる。

## options

<!-- cli:launcher -->
| option | 説明 |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--model-dir MODEL_DIR` | target model directory (loaded into the hosts) |
| `--device DEVICE` | comma separated GPU devices, one per rank; count must match --tp (default 0) |
| `--tp {1,2}` | tensor parallel size; one host per rank (default 1) |
| `--host-staging-gib HOST_STAGING_GIB` | host staging arena in GiB (default 30) |
| `--host-timeout-s HOST_TIMEOUT_S` | host ready timeout in seconds (default 600) |
| `--socket-dir SOCKET_DIR` | directory for the host sockets (default /tmp) |
| `--log-dir LOG_DIR` | directory for the host logs (default /tmp) |
| `--host-bin HOST_BIN` | shelf-model-host path |
| `--allow-any-dir` | let the hosts acquire models outside --model-dir |
<!-- /cli:launcher -->

## 使い方

compute(single-shot):

```bash
./build/shelf-launcher compute \
  --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 2,3 \
  --input-ids-file build/prompt.ids --max-new-tokens 128
```

speculative decode を有効にするには、drafter を child に透過する。

```bash
./build/shelf-launcher compute \
  --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 2,3 \
  --dflash2-model-dir models/Qwen3.8-27B-DFlash2 \
  --input-ids-file build/prompt.ids --max-new-tokens 128
```

serve(OpenAI互換API):

```bash
./build/shelf-launcher server \
  --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 2,3 --port 8123
```

serve 経由の chat は `shelf-server` の OpenAI互換APIを使う。
[server.md](server.md) を参照。

bench:

```bash
./build/shelf-launcher bench pp \
  --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 2,3 \
  --prompt-tokens 2048 --runs 3
```

`bench` は `pp` / `tg` / `quality` / `batch` に `--model-host` が対応する。
`--tp` の注入は `pp` / `quality` のみ（それ以外の bench サブコマンドには
`--device` の先頭要素のみ渡す）。[bench.md](bench.md) を参照。

## host の lifetime

host は launcher の子プロセスとして起動され、child の終了とともに shutdown する。
host の stderr は `--log-dir`(既定 `/tmp`)の
`shelf-launcher-host-<pid>-<rank>.log` に出力される。ready 待ちが失敗した
場合はこのファイルを確認する。

socket は `--socket-dir`(既定 `/tmp`)の `rank<N>.sock` に配置される。再実行時は
既存の socket を削除してから起動する。
