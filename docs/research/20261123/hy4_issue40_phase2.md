# Hy4-flash の prefill 最適化

基準は Phase 1 の `605d6194040f`。ROCm 10.2、gfx1201、Release、`SHELF_HY4_ONLY=OFF` でビルドした。実測には 1 台の R9700 を使い、すべての GPU コマンドで `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` と `flock /tmp/r9700.lock` を指定した。本体は Hy4-flash-exl3-4bpw-mul24、Engram は FP8、投機試験の drafter は Hy4-flash-dfly-4bpw である。

## P1　固定線形と routed expert

固定線形の trellis を射影ごとに 1 回展開し、出力タイルを先に並べた INT8 配列から WMMA の B フラグメントを連続して読む。128 要素ごとの入力スケールと K 分割の加算順を維持した。選択した経路には hipBLASLt の行単位への再量子化が不要で、展開領域は出力回転の後に再利用する。1024 行の共有 FFN で、展開を融合したまま 64 行タイルに広げる試作は AB/BA で 0.933 → 1.050 ms となり棄却した。

routed expert は行を最大 32 行にまとめ、gate と up を同じ grid で処理した。表の固定 FFN は第 1 MoE 層の共有 expert の gate、up、SwiGLU、down を含む。MoE 全体には実 router、共有 FFN、routed expert、加重和と正規化を含む。入力は固定した 1024 行の BF16 である。各プロセスで 2 回ウォームアップし、続く 5 回の中央値を取った。

| 対象 | 論理 EP | Phase 1 → 採用版（ms） |
|---|---:|---:|
| 固定 FFN | 1 | 0.935 → 0.678 |
| MoE 全体 | 1 | 13.987 → 9.935 |
| MoE 全体 | 3 | 14.213 → 10.995 |
| MoE 全体 | 4 | 14.005 → 11.086 |

各条件で全 BF16 出力のハッシュが基準版と一致した。

同じ入力の `rocprofv3 --kernel-trace` で各起動の GPU 時間を計測した。最初の 2 回を除いた 5 回について、routed expert の 3 射影に使う WMMA カーネルの合計は中央値 5.927 → 4.181 ms。基準は 3 回の起動、採用版は gate と up の同時起動および down の 2 回である。入力回転、SwiGLU、逆回転、経路計画はこのカーネル時間に含めていない。

## P2　チャンク長と VRAM

`test_hy4_graph BODY prefillbench LENGTH CHUNK 1 LOGITS` で 42 層を通し、最後の位置の全 120,832 logits を保存した。単一カードに収めるため routed expert だけを第 1 MoE 層から共有した。各層の router、固定射影、attention、Engram、head は各層固有の実重みである。入力 ID は位置 i に対して `100 + (37 * i) % 128` とし、同じ Engram 行を繰り返し使った。計測値は prefill 本体の実時間で、checkpoint のロードを含まない。A は Phase 1 の chunk 1024、B・C・D は採用版の chunk 1024・2048・4096。順序を ABCD と DCBA で交互に入れ替えた。2048 token の入力では chunk 4096 を指定しても実行行数は 2048 である。

| 入力長 | Phase 1・1024（秒） | 採用版・1024（秒） | 2048（秒） | 4096（秒） | 最終 greedy ID |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 2.553 | 2.142 | 2.149 | 2.143 | 100 |
| 8,192 | 12.494 | 10.736 | 10.836 | 11.145 | 138 |

各条件 5 回の中央値である。8k の既定チャンクでは 655.7 → 763.1 token/s となった。各入力長の全条件で logits の相対 L2 と最大絶対差はゼロだった。大きいチャンクを指定できるようにし、server の既定値は 1024 とした。

実 4 bpw の層別照合では、CPU で量子化重みを展開し、FP32 積和とモデル固有の BF16 丸めを通して基準出力を作った。短い照合は校正列の先頭 4 token で行い、chunk 1 と 4 ともに層 0・1・2 の相対 L2 は順に 0.01147662、0.00371337、0.00247323 だった。16 行で GPU 経路計画を通した層 1 と 2 は、それぞれ 0.00569287、0.00324147 だった。長チャンクでは 1024 個の校正入力と各層の入力 hidden state を 4 回繰り返し、同じ 4096 行に対して因果履歴を更新しながら CPU 基準を作った。

| 実 4 bpw 層 | chunk 1024 | chunk 2048 | chunk 4096 |
|---:|---:|---:|---:|
| 0 | 0.01157267 | 0.01157267 | 0.01157267 |
| 1 | 0.00672598 | 0.00672598 | 0.00672598 |
| 2 | 0.00415720 | 0.00415720 | 0.00415720 |

EP の作業域は rank 0 が 2 GiB、peer が 512 MiB のままである。実際の配置表のヘッダーと DFly2 を使い、32 GiB 容量、Graph 用 256 MiB、安全余裕 0.5 GiB で試算した。最大使用量には Graph 用余裕を含む。各 rank の `peak_estimate` と同じバイト数を単一 R9700 で 1 rank 分ずつ確保し、全領域をゼロ初期化し、同期して解放した。物理 3・4 台での同時起動および PCIe 帯域は未測定である。

| 台数 | スロット × 予約長 | 最大使用量（GiB） | 安全余裕控除後の最小空き（GiB） |
|---:|---:|---:|---:|
| 3 | 1 × 2,097,152 | 29.039 | 2.461 |
| 3 | 16 × 4,096 | 29.580 | 1.920 |
| 3 | 16 × 8,192 | 29.842 | 1.658 |
| 4 | 1 × 2,097,152 | 21.966 | 9.534 |
| 4 | 16 × 4,096 | 24.044 | 7.456 |
| 4 | 16 × 8,192 | 24.306 | 7.194 |


実 checkpoint と配置表を読み込む `Residency::reserve` も検証した。論理 device を同じ R9700 に割り当て、各 rank の領域確保と初期化が完了した時点でその arena だけを解放してから次の rank に進めた。6 構成すべてで予約に成功し、Graph 用余裕の判定も通った。単一 2Mi での実予約量の最大値は 3 台で 28.855 GiB、4 台で 21.785 GiB だった。

## P3　DSA Indexer

16 query 行、16 head、head 幅 128 の BF16 入力を使った。`index_select` は各位置で完了済みの 4-token グループのうち 511 個を選ぶ。これに現在グループの先頭から現在 token までを加え、2045〜2048 token を参照する。表の「スコア」は `rocprofv3` による score カーネル単体の中央値、「境界再計算」は採用版の scalar 再計算カーネル単体の中央値である。「選択全体」はスコア、誤差範囲の算出、境界再計算、最終 radix 選択を含む。各モードを 1 回ウォームアップし、12 回連続起動の平均時間を AB/BA で 5 回測定した。

| 開始位置 | スコア単体 scalar → WMMA（ms） | 境界再計算（ms） | 選択全体 scalar → 採用版（ms） |
|---:|---:|---:|---:|
| 8,192 | 0.134724 → 0.006000 | 0.012680 | 0.2018 → 0.1337 |
| 32,768 | 0.700179 → 0.013800 | 0.026280 | 0.8327 → 0.1873 |
| 131,072 | 1.491881 → 0.050441 | 0.075242 | 1.7494 → 0.4435 |

乱数入力の 3 条件とも選択集合は 8,176 / 8,176 グループで一致し、全有効スコアの最大絶対誤差は 0.00000573 以下だった。各行で選択グループの重複や未来位置の混入がなく、有効範囲に続くスコア欄がゼロのままであることも検査した。位置 2045 で選択全体は scalar 0.1049 ms、WMMA と境界再計算 0.1277 ms だった。位置 4096 では 0.1278 → 0.1201 ms となり、実行時の切り替えを位置 4096 に置いた。

ヘッドを正負の組にしてスコアを相殺する回帰入力でも、再計算後に 8,176 / 8,176 グループと全スコアが一致した。このようにほとんどの候補が境界付近に集まる入力では再計算量が増える。

WMMA の結果だけで順位を決める試作は、8k の反復プロンプトで chunk 間の logits が一致したが、基準版に対する相対 L2 が 0.2653 だった。同じバイナリで Indexer だけを scalar に戻すと差はゼロになった。そこで 511 位を含むスコア区間を radix の上位 2 バイトで絞り、区間の両端を `128 × max|q| × max|key| × Σ|head_weight| × 2^-14` だけ広げ、その中のスコアを scalar と同じ順に再計算した。

実重みの照合では、層 0 の校正入力 8 行と固定 seed 19 の合成 KV 履歴を使用した。各行について同じ query、head weight、group key から両方式を起動し、全有効スコア、グループ番号の集合、causal な末尾を比較した。開始位置 2045 では先頭 3 行が dense、残る 5 行が sparse となる。

| 開始位置 | sparse 行数 | 一致した選択 token 数 | スコア最大絶対誤差 | 層出力の CPU 相対 L2 |
|---:|---:|---:|---:|---:|
| 2,045 | 5 | 10,231 / 10,231 | 0.00010682 | 0.01006208 |
| 8,192 | 8 | 16,372 / 16,372 | 0.00009156 | 0.01218199 |
| 32,768 | 8 | 16,372 / 16,372 | 0.00010682 | 0.00941199 |
| 131,072 | 8 | 16,372 / 16,372 | 0.00012208 | 0.01072971 |

## P4　転送と受入検証

GPU で histogram、prefix sum、scatter を行い、expert 順の逆写像で BF16 の加算順を再現した。以前は 1024 入力行ごとに 196,608 バイトの routing をホストへ戻し、さらに各 rank へ並べ替えた経路、offset、逆写像、batch を送っていた。採用版では host 転送を指定しても、この CPU partition と同期は発生しない。

| 論理 EP | 転送方式 | MoE 全体 Phase 1 → 採用版（ms） |
|---:|---|---:|
| 3 | push | 14.213 → 10.995 |
| 3 | peer | 14.193 → 10.979 |
| 3 | host | 14.302 → 11.950 |
| 4 | push | 14.005 → 11.086 |
| 4 | peer | 13.645 → 11.383 |
| 4 | host | 14.711 → 12.903 |

push は 5 回の交互計測、peer と host は AB と BA の各 1 組を使った。いずれも各プロセス内ではウォームアップ後 5 回の中央値を取った。

論理 EP=3 の push の rank 間 payload は、1 層あたり 12,582,912 → 12,672,304 バイトだった。
論理 EP=4 の push の rank 間 payload は、1 層あたり 18,874,368 → 18,860,568 バイトだった。
基準版の値は全 peer が全行の latent と部分和を 1 回ずつ転送した場合のサイズである。採用版の値は実 router の所有マスクに基づく論理バイト数であり、PCIe トランザクションのカウンター値ではない。経路のない行は転送を省ける。全行に経路がある peer では、GPU 間で渡す選択情報の分だけ直接の peer バイト数が増える。旧版の経路 D2H と各 rank への経路・逆写像の H2D は合計で少なくとも 458,752 バイトだった。これらを含む論理 payload の削減量は EP=3 で少なくとも 369,360 バイト、EP=4 で少なくとも 472,552 バイトである。

BF16 の符号と仮数を保存し、32 要素の指数差を 4 ビットに収められる場合だけ送信量を減らす可逆圧縮も試した。論理 EP=3 の push では peer 1 への送信が 3,185,168 → 2,486,112 バイトになり、出力ハッシュは一致した。単一カード上で MoE 全体が 10.945 → 11.056 ms となったため棄却した。
decode・verify の `microbench` は Phase 1 → 採用版 → 採用版 → Phase 1 の順で測った。固定射影と共有 FFN の 1・6 行、routed expert の B=1・4・8・16 および各 6 行 verify で、時間差は −0.87%〜+0.34% だった。代表値は B=16 の routed expert が 1520.63 → 1521.52 µs、96 行 verify が 3165.27 → 3166.61 µs である。両ビルドの gfx1201 コードオブジェクトから各シンボルの命令バイト列を取り出し、decode 用の expert WMMA、6 通りの固定 `gemm_fused`、入力回転ペア、SwiGLU 付き回転と出力回転がそれぞれ一致することを確認した。

- 論理 EP=3・4 の層別実 expert を選択集合に従って常駐させ、単一スロットと 16 スロット予約を検証した。push / peer / host、Graph 有効・無効、DFly2 有効・無効、B=1〜16 の 408 条件、延べ 3,288 スロットで 4 token の greedy 生成列が同じ EP の Phase 1 基準と一致した。集合外 expert の参照は 0 件で、Graph の再生回数も設定と一致した。
- 既存の単一スロット Graph 回帰、バッチ・スロット再利用・2048 境界の混在、TP・EP の合算、Continuous Batching の全項目が通った。サーバー試験には同一 R9700 内で第 1 MoE 層の expert を共有する compute を使った。
- Release 全体ビルド、必須 CPU CTest 133 件、Hy4 Python 単体テスト 9 件、サーバー CLI 22 件、共通 EXL3 の再構成・WMMA 回帰が通った。
- 論理 EP=3・4 の各転送方式で 4096 行の 65,536 経路を rank 0 または末尾の peer に集中させた。全件数を処理し、両配置の出力がビット一致した。

追加したソースコメントは 0 件で、差分の空白検査を通過した。最終 GPU 試験後、排他ロック内で KFD プロセスが 0 件であることを確認した。

### 再現手順

`BODY` は 4 bpw の本体と FP8 Engram を含むディレクトリ、`DFLY` は drafter のディレクトリである。CPU fixture は原本の BF16 checkpoint、量子化時の作業ディレクトリ、FP8 Engram を順に指定して作る。

```bash
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
cmake --build build --target test_hy4_graph test_hy4_flash_layer test_hy4_reduce
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" prefillbench 1024 1024 3 moe
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" prefillbench 128 64 1 index
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" prefillbench 8192 4096 1 pp.f32
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" prefillbench 4096 4096 4 capacity
python3 tools/hy4_flash/check_layer.py "$BF16" "$WORK" "$ENGRAM" --layer 0 --tokens 4096 --fixture layer0
flock /tmp/r9700.lock build/tests/test_hy4_flash_layer layer0
python3 tools/hy4_flash/check_layer.py "$BF16" "$WORK" "$ENGRAM" --layer 0 --tokens 8 --start 131072 --fixture sparse128k
flock /tmp/r9700.lock build/tests/test_hy4_flash_layer sparse128k
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" "$DFLY" batch
ctest --test-dir build -L '^cpu$' -L '^required$' --output-on-failure
```

## 変更ファイル一覧

| ディレクトリ | ファイル |
|---|---|
| `cmake` | [targets.cmake](../../../cmake/targets.cmake) |
| `docs/developer` | [hy4_flash.md](../../../docs/developer/hy4_flash.md) |
| `docs/research/20261123` | `hy4_issue40_phase2.md` |
| `docs/user` | [hy4_flash.md](../../../docs/user/hy4_flash.md) |
| `include/shelf/kernels/fused/detail` | [radix_select.h](../../../include/shelf/kernels/fused/detail/radix_select.h) |
| `include/shelf/kernels/optimized/linear` | [exl3_mul24.h](../../../include/shelf/kernels/optimized/linear/exl3_mul24.h) |
| `include/shelf/models/hy4_flash` | [attention.h](../../../include/shelf/models/hy4_flash/attention.h)、[linear.h](../../../include/shelf/models/hy4_flash/linear.h)、[moe_graph_plan.h](../../../include/shelf/models/hy4_flash/moe_graph_plan.h)、[residency.h](../../../include/shelf/models/hy4_flash/residency.h) |
| `src/apps/server` | [shelf_server.py](../../../src/apps/server/shelf_server.py) |
| `src/shelf/kernels/optimized/linear/exl3_family/mul24` | [prefill.hip](../../../src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill.hip)、[prefill_wmma.hip](../../../src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill_wmma.hip) |
| `src/shelf/models/hy4_flash` | [attention.hip](../../../src/shelf/models/hy4_flash/attention.hip)、[generate.hip](../../../src/shelf/models/hy4_flash/generate.hip)、[linear.hip](../../../src/shelf/models/hy4_flash/linear.hip)、[moe.hip](../../../src/shelf/models/hy4_flash/moe.hip)、[ops.hip](../../../src/shelf/models/hy4_flash/ops.hip)、[residency.cpp](../../../src/shelf/models/hy4_flash/residency.cpp)、[wmma.hip](../../../src/shelf/models/hy4_flash/wmma.hip) |
| `tests/kernels` | [hy4_decode_bench.cpp](../../../tests/kernels/hy4_decode_bench.cpp)、[test_hy4_flash_layer.cpp](../../../tests/kernels/test_hy4_flash_layer.cpp)、[test_hy4_graph.cpp](../../../tests/kernels/test_hy4_graph.cpp) |
| `tests/server` | [test_server_cli.py](../../../tests/server/test_server_cli.py) |
| `tools/hy4_flash` | [check_layer.py](../../../tools/hy4_flash/check_layer.py) |
