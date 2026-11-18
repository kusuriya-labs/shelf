# Hy4-flash 専用 fork の計画

対象は Hy4-flash のテキスト生成。原本の text_config と tensor header、rocm_exl3 の組立仕様を照合する。最初は decoder 平均 4 bit、head 6 bit の EXL3-INT8、Engram はホスト側 FP8 表から必要行を集める。KV は Octave を使う。

| 段階 | 実装・検証 | 完了の判定 |
|---|---|---|
| 入力 | 原本から Engram 4 表を分割読み込みして FP8 に変換し、行単位で復元 | 各表の形状・丸め・ゼロ行・出力サイズの検査 |
| CPU | iHC の縮約と展開、sliding と疎な MLA の causal 境界、indexer の未完グループ、router と latent expert、Engram の履歴を原実装に合わせる | 短い固定列を一括・逐次で比較 |
| EP | R9700 が attention、shared expert、出力を担当。V620 は WMMA に依存しない EXL3-INT8 routed expert を担当。各 GPU に expert を分割する | 両 ISA へのオフラインコンパイル、実機で各 rank の部分和と単一入力の参照値を比較 |
| 本体 | checkpoint の全 tensor と量子化付帯情報を照合。42 層、層別 KV、4 stream の実行を既存 CLI に接続 | prefill、decode の最終 logits と greedy 列を比較 |
| DFly | 本体の量子化後に付属 5 層を追加量子化。層 25・29・33・37・41 の KV/index を渡し、候補 5 token を検証 | 全採択、部分採択、全棄却の KV・Engram 履歴・位置を通常生成と照合 |
| 受入 | 重み・各 rank の作業域を確保後、2,097,152 token 分の予約を試す。失敗時は解放して 1,048,576 で再試行 | rank 別の要求量・空き容量・結果を記録。推論上限は config の 1,048,576 |
| 最終 | 4 bit 版の正しさと性能検証後に 3.5 bpw を作成 | 同じ入力で品質、実保存量、速度を再測定 |

## 実機解放後の順序

本体の conversion_report.json と head_complete を確認し、DFly の追加量子化を完了する。その後、load average の 3 指標がすべて 8 未満のとき、/tmp/r9700.lock、/tmp/v620.lock、/tmp/v620-76a08.lock の順に排他を取り、ROCR_VISIBLE_DEVICES に 3 台の UUID を指定する。最初に各台の実空き量と 2Mi（不足時 1Mi）の予約結果を記録する。続いて peer access と転送方式を確認して短い入力の生成を照合し、pp2048 と DFly 有効時の decode を測る。

性能目標は pp2048 で 500 tok/s、decode で 40 tok/s。ウォームアップ後、同一入力、同一出力長で複数回実行し、生成済み token 数を実時間で割る。DFly の受理長分布も記録する。各層の選択頻度と rank ごとの容量から expert 配置を決める。最初は均等頻度を仮定し、実測で hot expert を高帯域 rank に寄せる。送るのは latent activation と routing 情報、戻すのは rank 内で加重和を取った出力。prefill は rank ごとにまとめた転送、decode は shared expert との重畳を比較する。計測には HIP event と壁時計を用いる。

## 実装に合わせた検証手順

現在の段階と測定値は [進捗記録](hy4_flash_progress.md) にまとめる。専用構成は SHELF_HY4_ONLY=ON。R9700 を rank 0、V620 を rank 1・2 とし、4 表はホストに置く。本体の配布形式では量子化しなかった indexer の pooling の線形だけは原本の精度を使う。

1. 本体完了前は tools/hy4_flash/check_chain.py で、完成済み層まで同じ 4 token の CPU 出力を保存する。配置表も保存して同じ条件で再開する。本体完了後、同じコマンドへの --body の指定で 6 bit head の FP32 logits も保存する。
2. 本体の head_complete と実出力の conversion_report.json を確認し、原本、組立済み本体、校正 work、DFly 出力先をこの順に quant_dfly.py に渡す。量子化側と同じ mul24 用 rocm_exl3 の Python 環境で実行する。コンテナ使用時はホストで locked.py v620 を取り、docker run に -e ROCR_VISIBLE_DEVICES を渡す。
3. ホストで locked.py ep3 を経由して qualify.py を起動する。引数は shelf-compute、本体、Engram ディレクトリ、DFly、配置 JSON、CPU 照合結果ディレクトリ、校正 safetensors、実測記録先。実予約、温度 0 の logits と生成列、正の温度で同じ seed の DFly 生成の再現性の順に検査する。
4. 同一常駐モデルで pp2048 と decode を 4 回測定し、初回を除いて中央値を取る。PCIe 転送・top-k のホスト同期、kv_b_proj の再展開、R9700 上の行列演算を分けて評価する。直列にかかる時間が支配的な部分から高速化し、各変更後に CPU 照合を繰り返す。

## このホストでのビルドと入力

この fork の既定は Hy4-flash 専用構成である。実行形式は worktree 直下で次のように作る。

```bash
cmake -S . -B build-hy4 -G Ninja \
  -DSHELF_HY4_ONLY=ON -DSHELF_BUILD_TESTS=ON \
  -DSHELF_ROCM_ROOT=/opt/rocm/core-7.14 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DSHELF_LIBCXX_LLVM_ROOT="$HOME/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23"
cmake --build build-hy4 --target shelf-compute test_hy4_flash_layer -j1
ctest --test-dir build-hy4 -R '^hy4_flash_cpu$' --output-on-failure
```

専用 linear とモデル演算の HIP ターゲットは gfx1030 と gfx1201 の両方を生成する。WMMA の translation unit は gfx1201 だけでコンパイルする。rank の割り当ては実行時に ISA を照合する。

```bash
original="$HOME/datapool/ai_models/safetensors/Hy4-flash"
quantroot="$HOME/datapool/rocm-exl3-rdna2"
pilot="$quantroot/hy4-flash-pilot"
work="$pilot/pipeline-v620-128x1024"
body="$quantroot/models/Hy4-flash-exl3-4bpw-mul24"
draft="$pilot/shelf-dfly/Hy4-flash-dfly-4bpw"
engram="$pilot/engram-e4m3fn"
```

Engram の新規生成には `python3 tools/hy4_flash/convert_engram.py "$original" "$engram"` を使う。このホストでは 4 表を作成済み。本体完了後は配置表を新しい出力名で生成する。CPU 連続照合の再開には、その照合を始めた時点と同じ配置表が必要である。

```bash
options=(--model-dir "$body" --engram-dir "$engram" --placement "$pilot/shelf-placement/owners-4bpw.json")
build-hy4/shelf-compute hy4 inspect "${options[@]}" --dfly-dir "$draft"
python3 tools/hy4_flash/locked.py ep3 build-hy4/shelf-compute hy4 reserve \
  "${options[@]}" --dfly-dir "$draft" --max-seq-len 2097152
python3 tools/hy4_flash/locked.py ep3 build-hy4/shelf-compute hy4 generate \
  "${options[@]}" --dfly-dir "$draft" --max-seq-len 4096 \
  --input-ids-file prompt-ids.txt --max-new-tokens 128
```

入力は空白区切りの token ID または既存の PSKLDTOK 形式。専用構成では shelf-compute に直接オプションを渡すと generate を実行する。本体単独との比較は --dfly-dir を省く。採用版は R9700 の WMMA、worker の dot4、短い query の吸収形 MLA、8 行以上の expert 集約を自動選択する。推論時の比較フラグは不要。測定時は --chunk 1024 を指定する。候補ごとの採否・生成列・測定条件は進捗記録の P5 に残す。

## 3.5 bpw の再現

4 bpw 版の候補撤去後の受入と 256 token 照合の後に開始する。quant_mixed.py は校正頻度をもとに各層の 266 または 267 expert を選び、合計 10,931 本の gate/up/down を BF16 原本から 3 bit に再量子化する。選択した ID と実 trellis の形状は組立時に照合する。作業中は .work/checkpoint と shard 別 marker を保持し、入力署名が同じ実行は保存済み shard の SHA-256 を確認して再開する。残りの 4 bit 重みと 6 bit head は元の値を保つ。4 枚の FP8 Engram は新しい checkpoint の index に含める。

前節の変数を使い、ホストの 2 台分の排他を変換終了まで保持する。

```bash
outdir="$pilot/shelf-mixed"
mixed="$outdir/Hy4-flash-exl3-3.5bpw-mul24"
quantrepo="$HOME/coding-local/rocm_exl3/.worktrees/hy4-flash"
hfsource=("$quantroot"/hy4-flash-env/transformers-*/src)
mkdir -p "$outdir"
python3 tools/hy4_flash/locked.py v620-pair docker run --rm --pull=never --network=none \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --user "$(id -u):$(id -g)" --shm-size=8g --entrypoint /opt/venv/bin/python -w /src \
  -e ROCR_VISIBLE_DEVICES -e ROCM_PATH=/opt/rocm/core-7.14 -e PYTORCH_ROCM_ARCH=gfx1030 \
  -e LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1 \
  -e LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -e PYTHONPATH=/hf-src:/extension:/src:/src/tools/hy4_flash -e XDG_CACHE_HOME=/tmp/cache -e OMP_NUM_THREADS=2 \
  -v /opt/rocm:/opt/rocm:ro -v "$original:$original:ro" -v "$body:$body:ro" -v "$engram:$engram:ro" \
  -v "$work:$work:ro" -v "$outdir:$outdir:rw" -v "$PWD/tools/hy4_flash:/shelf-tools:ro" \
  -v "$quantrepo:/src:ro" -v "${hfsource[0]}:/hf-src:ro" -v "$quantroot/lib-mul24-v620:/extension:ro" \
  rocm-exl3-investigation:tested -u /shelf-tools/quant_mixed.py "$original" "$body" "$engram" "$work" "$mixed"
```

組立後は --model-dir と --engram-dir に同じ mixed ディレクトリを渡す。DFly は校正署名が一致する既存 sidecar を指定する。4 bpw と同じ配置・入力・chunk 1024・seed の測定条件で比較する。

## 実施結果

4 bpw は 2Mi 予約・生成照合・速度の受入を完了した。3.5 bpw も同じ配置・入力で実機測定を終えた。2Mi 予約と 256 token の通常生成・DFly 照合は成功し、decode と CPU logits は 4 bpw と同じ基準に届かなかった。最終 checkpoint と項目別の結果は[進捗記録の M6・M7](hy4_flash_progress.md)と[性能記録](../../perf/current.md#hy4-flash-専用-fork-の-ep3)に記載した。

第 8 回レビューで検証回数と rank 別 expert 時間を実測した。4 bpw を採用し、3.5 bpw は比較用に維持して調査を完了した。[進捗記録 M8 の切り分けと最終判断](hy4_flash_progress.md)
