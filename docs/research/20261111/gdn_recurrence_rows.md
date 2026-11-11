# GDN recurrence の行数と帯域（調査途中）

対象は `integ/all` の 6c4c763、R9700 gfx1201、Qwen3.8 27B の 16 key heads・48 value heads・head 次元 128・GDN 48 層。モデル全体ではなく **1 層 1 回の kernel** を hipEvent で測定した。3 回 warmup の後 20 回連続で呼び、平均した。入力・state はゼロ、slot stride は 48 層分。従って実際の値や層間のキャッシュ遷移、クロック変化、他の GPU 利用時の差は含まず、小さい構成はキャッシュに入りうる。GPU 使用は `flock /tmp/r9700.lock`、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`。デフォルトの lossy WMMA を使用。

再現用 `tests/benchmarks/gdn_recurrence_rows.hip` は cmake の HIP コンパイラ設定で `recurrence.hip` と一緒にコンパイルする。state を毎回初期値に戻す時間は計測に含まない。history のときは全行の fp16 state を書き出す。compact log と tree parent は使用していない。

## 実装の構成

- 非 verify の WMMA は request・value head・64 value 列を 1 block（256 thread）で処理し、最大 32 行ずつまとめる。fp16 state `[slot][layer][head][key][value]` を読み fp32 レジスタと LDS に置き、最後に fp16 で書く。通常は history / compact log の書き出しはない。
- verify は同じ block 配置で各行を逐次処理し、行ごとに fp16 に丸める。history が有効なら各行を書き出す。compact log が有効なら delta、key、係数を書き、deferred commit なら最後の state 書き込みを省略する。tree parent で直前以外の親を指定した場合は history から state を読み戻す。
- 汎用 F32 kernel は 32 列/block、行ごとに state を global memory 経由で更新する。head 形状により選択されるが、今回のモデルでは WMMA が選ばれる。WMMA は既定で lossy、環境変数で exact に切替可能。

ビルドと実行例（既存の `/tmp/venv-so` は変更しない）:

```sh
LLVM=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23
/opt/rocm/core-7.14/lib/llvm/bin/clang++ -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -Iinclude -Isrc -O3 -std=gnu++20 --offload-arch=gfx1201 -stdlib=libc++ \
  -stdlib++-isystem "$LLVM/include/c++/v1" -x hip tests/benchmarks/gdn_recurrence_rows.hip \
  src/shelf/kernels/optimized/gdn/recurrence.hip --hip-link -L"$LLVM/lib" -lc++ -lc++abi \
  -L/opt/rocm/core-7.14/lib -lamdhip64 -o /tmp/gdn-rows-bench
export LD_LIBRARY_PATH="$LLVM/lib:/opt/rocm/core-7.14/lib" ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock /tmp/gdn-rows-bench
```

## 基準計測

帯域下限は **state の最初の読みと最後の書きのみ**、各 request につき `48 × 128 × 128 × 2 byte × 2 / 640 GB/s` = 4.915 µs/層。history や入出力、演算、キャッシュ、実効帯域は含まない。各数値の単位は µs/層、比率は verify 時間 / この下限。行あたり値は verify 時間を全行数で割ったもの（並列実行の平均であり、単独行のレイテンシではない）。

| request × 行 | decode/prefill | verify | history あり | 最終 state 書込なし | state 下限 | verify / 下限 | verify µs/行 | verify µs/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×8 | 16.968 | 93.532 | 97.424 | 92.420 | 4.915 | 19.03 | 11.692 | 93.532 |
| 1×10 | 17.928 | 113.224 | 118.306 | 112.398 | 4.915 | 23.04 | 11.322 | 113.224 |
| 1×16 | 17.636 | 176.551 | 184.453 | 175.119 | 4.915 | 35.92 | 11.034 | 176.551 |
| 4×8 | 57.403 | 347.395 | 362.047 | 343.045 | 19.661 | 17.67 | 10.856 | 86.849 |
| 8×8 | 111.416 | 690.590 | 797.667 | 683.669 | 39.322 | 17.56 | 10.790 | 86.324 |
| 16×4 | 208.135 | 701.693 | 859.866 | 690.515 | 78.643 | 8.92 | 10.964 | 43.856 |
| 64×1 | 911.421 | 1000.638 | 1124.614 | 755.248 | 314.573 | 3.18 | 15.635 | 15.635 |

verify 1×8→1×10→1×16 は 93.5→113.2→176.6 µs と行にほぼ比例し、コード上も 1 行ずつ 32 行向けの WMMA・LDS・同期を繰り返す。8×8 の最終書き込み省略は 690.6→683.7 µs と約 1.0% なので、この条件の主な費用は最後の state 書き込みではない。history を付けると 690.6→797.7 µs（+15.5%）だが、これだけで ALU、LDS、occupancy の個別寄与は区別できない。64×1 verify では書き込み省略で 1000.6→755.2 µs（約 24.5%）で、state 書き込みの寄与が大きい。decode 64×1 は 911.4 µs/層で state 下限の 2.90 倍。48 層の単純な積は約 43.7 ms だが、実際の過去計測 37 ms と条件が異なり、step 時間として扱えない。

木の N=8 と N=10 を 1 request の chain としてのみ近似すると verify は 93.5→113.2 µs/層（+19.7 µs/層）。実際の分岐では親 state の再読込があり、この差を木の実測値と見なせない。

## 単一行 variant の試行

既存 body の template に単一行専用の実体を追加し、出力更新と state 更新の後半 16 行分の WMMA を省く。既存 variant の instantiation と通常 selector は変更していない。対象は lossy・各 request 1 行の実験呼び出しのみ。有限入力に限ると想定しており、非有限値や演算途中の overflow で従来と同じ伝播をするかは未検証である。launch 時の `actual_rows == num_requests` だけでは各 request の row_count が 1 だとは検証できず、呼び出し側がその前提を満たす必要がある。このため通常 selector には接続していない。

64×1、同一プロセス内で従来と実験 variant を順番に測った値は以下（µs/層）。

| 試行 | 従来 | 実験 | 短縮率 |
|---|---:|---:|---:|
| 1 | 924.431 | 882.580 | 4.53% |
| 2 | 939.260 | 897.182 | 4.48% |
| 3 | 941.668 | 895.144 | 4.94% |
| 4 | 942.659 | 902.554 | 4.25% |

48 層すべてに同じ差を仮定した単純な積は約 1.9〜2.2 ms だが、実モデルの step の削減量ではない。非 verify 専用であり verify や tree の改善量もない。既存の有限・非ゼロ入力の recurrence テスト（4B と 27B TP2、単一および複数 request）では既存 WMMA と出力・state が bit 一致した。64 request・27B フル geometry や非有限値の一致は未検証。

## 検証と未達

- configure と全ビルド成功。`test_gdn_recurrence`、`test_gdn_recurrence_exact_history`、`test_gdn_spec_history`、`test_gdn_compact_log_commit`、`test_qwen35_gdn_recurrence_selector` は実験追加後に全成功。実験 kernel は recurrence テスト内で直接比較した。
- `test_compute_mtp_rejects_dflash2`、`test_compute_dflash2_accepts_octave` は基準実装で成功。サーバー `--group core` は 18 pass、1 skip（`test_plamo3_template.py`）、0 fail。通常 selector は実験 kernel を選ばないので、これらは実験 kernel を通した token 一致の証明ではない。
- 実モデルの投機なし 16/32/64 および DFlash2 4/8/16/32 の before/after は未測定。通常経路で有効な改善がないため、37 ms と 1 step の実際の削減量も未測定。目標の 0.35 ms/行以下には到達を示せていない。
- 次の課題は request ごとの行数と非有限値を安全に扱う selector と数値検証、実際の model state と層間アクセスを用いた測定、64×1 の state 読み書き効率と occupancy の分離、verify の 1 行あたりの WMMA/LDS コスト削減である。compact log 有効時と分岐あり tree も個別に計測する必要がある。
