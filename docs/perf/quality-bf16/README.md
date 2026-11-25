# Qwen3.8-27B の BF16 基準品質測定

`cases-v1` の主 8 case、2,632 位置を teacher forcing で測る。基準は llama.cpp 0.4.0-dev、BF16 GGUF、FP16 KV、chunk 64 の保存済み全 logits。候補は shelf の BF16 KV。両者の KV 型と chunk が異なるため、量子化だけの誤差ではない。KL(softmax(基準) ‖ softmax(候補)) を有効語彙 248,077 個で FP64、温度 1、nats で計算し、top-1 一致率も報告する。English repeat は除外する。

先頭最大 256 token は prefill で全行を出力し、残りは逐次 decode する。短い case は最後の 1 行を decode とする。入力位置 i の logits は token i を入力した後、token i+1 の予測であり、最終行は便宜上同じ token を target とする。基準ファイルは末尾を含む全位置の full FP32 logits である。短い case では 256 行に達せず A の prefill 行列経路は使われない。ケース間でモデルを再ロードする。

ROCm 10.2、libstdc++ 先行でビルドし、`build-bf16-kld/shelf-bench` を用意する。例えばこの環境では次のように設定する。

```bash
llvm23=$HOME/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23
cmake -S . -B build-bf16-kld -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.2/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.2/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.2 -DSHELF_ROCM_ROOT=/opt/rocm/core-10.2 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSHELF_LIBCXX_LLVM_ROOT="$llvm23" \
  -DSHELF_LOAD_LIBSTDCXX_FIRST=ON \
  -DCMAKE_CXX_FLAGS="-I$HOME/coding-local/shelf/vendor" \
  -DCMAKE_HIP_FLAGS="-I$HOME/coding-local/shelf/vendor"
cmake --build build-bf16-kld --target shelf-bench -j 8
```

`/tmp/shelfvenv/bin/python` に numpy が必要。基準・token・語彙とモデルの既定パスは `--help` を参照。必要に応じ `--bench`、`--data`、`--model` を指定する。

```bash
export LD_LIBRARY_PATH="/opt/rocm/core-10.2/lib:$llvm23/lib"
./tools/quality/kld-vs-bf16.sh B
./tools/quality/kld-vs-bf16.sh A
```

ラッパは `/tmp/r9700.lock` を取得し UUID と HIP device を設定する。呼び出し元の `with-gpu` が既に同じ lock を保持する場合は `--already-locked` を指定する。環境変数は引き継ぐ。nt 外側ループなど別 worktree の候補は、本変更を適用して `--dump-prefill` を備えた `shelf-bench` を構築し、`--bench` で指定する。JSON の git revision はこの比較 script の worktree を示すため、別 worktree の候補 revision は構成名や別記録に残す。構成名ごとに JSON と `results.md` の表を更新し、実行ファイル・入力・基準・各 dump の SHA-256、構成と所要時間を記録する。同名結果は上書きする。候補 dump は一時領域に最大約 1 GiB を使用し、case ごとに上書き・終了時に削除する。基準は約 2.6 GiB。所要時間は JSON と表の実測値を参照。

## 過去値との照合

2026-11-20、ROCm 10.2、R9700、libstdc++ 先行、rocRoller 指定なしの B は 196 秒で平均 KLD 0.02074514、top-1 94.8708%。同じ BF16 基準と corpus に対して過去の rocm_exl3 INT8 の full logits を再集計すると 0.02166897、94.0350% で、差はそれぞれ −4.26%、+0.836 ポイント。過去の候補は FP16 KV、chunk 64 の prefill で、今回の候補は BF16 KV、最大 256 token の先頭 prefill と後続 decode である。エンジンと計算経路も異なり、差の寄与は分離していない。過去基準の chunk 1 対 64 の揺れは平均 KLD 0.000496、top-1 99.62%。

過去の候補のケース別 KLD と今回の B は、English 0.079815 → 0.079034、日本語 0.026947 → 0.023063、技術文 0.016990 → 0.017454、Rust 0.020653 → 0.023268、Python 0.021165 → 0.018292、SQL 0.011592 → 0.011374、算術 0.108550 → 0.100726、構造化文 0.014754 → 0.013136。過去の入力・候補は `/home/homelab1/datapool/rocm-exl3-rdna2/kld27b-mul24n/int8`、基準と corpus は `~/datapool/qwen38-kld-20260918/`。再集計は同ディレクトリの `compare.py` を使用した。

gfx1201 の既定は A（`1d96ff52`）で、rows<=16 の decode は融合経路のままなので、測定時の `A-fused`（`SHELF_EXL3_MUL24_FUSED_DECODE=1`）は既定の A と同じ構成になった。この環境変数は現在の main には存在しない（比較器は、環境変数名が実行ファイル中にない場合に測定前に停止する）。`A-fused.json` は、環境変数を実装していた当時の記録として残している。

B の再実行は 185 秒、全 8 case で dump の SHA-256 が初回と一致し、全 logits が bit 一致した。256 行の output 展開を含む `test_token_bitmask` は PASS。

CPU 側の式の照合として、過去の rocm_exl3 raw logits から主 8 case の先頭・中間・末尾の計 24 位置を計算し、保存済み `compare.py` の位置別値と絶対差 1e-12 未満で一致した。

同条件の A は KLD 0.02117609、top-1 94.7568%、190 秒。B より KLD は +0.00043094、top-1 は −0.1140 ポイント。過去の rocm_exl3 と比べると KLD −2.27%、top-1 +0.722 ポイント。A＋融合 decode は KLD 0.02117609、top-1 94.7568%、197 秒で、8 case すべてで融合 launch を確認し、A と全 dump の SHA-256 が一致した。各構成の JSON は case 別値を含む。A と融合版は異なる実行ファイルだが、その差分は上記 opt-in と launch 確認である。結果はこの corpus・分割・環境に限る。

位置単位の A/B 対応比較、case bootstrap、3.05bpw、長文・資源の追加評価と既定化判断は [A 経路の評価](a-path-assessment.md) を参照。位置別 KL と BF16 top-1 一致を含む JSON の比較は `tools/quality/paired_kld.py BのJSON AのJSON` で再現する。

位置別 JSON（`paired-*.json`、約 3.8 万行）は容量のためリポジトリに含めない。`kld-vs-bf16.sh` で再生成できる。
