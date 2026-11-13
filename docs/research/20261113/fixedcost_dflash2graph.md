# DFlash2 tree verify の HIP graph

## 条件

R9700 / gfx1201、Qwen3.8-27B EXL3 mul24、DFlash2、drafts 7、c=1。
比較元は `integ/fixedcost` の `integ-fc/build-gfx1201` であり、変更していない。
単発の速度は `prompt_code.ids`、greedy、128 token、max sequence 1024、
基準と候補を ABABAB の順に各3回。HTTP は tree budget 10、greedy、128 token、
同じ prompt で1回 warmup した後の1 request を測る。HTTP の速度には request 全体の時間を含む。
実行コマンドと生の値は [スクリプトと JSON](fixedcost_dflash2graph/) に残す。

## 確認された既存の greedy 不一致

これは graph の ON/OFF 比較とは別の、baseline 内の既存差である。
固定 `prompt_ja.ids`、最大256 token、`--temperature 0 --dflash2-drafts 7`、
max sequence 1024 で比較した。位置は生成列の0始まりである。
baseline の非投機出力と候補の非投機出力は一致した。
baseline の tree 出力は、各 drafter とも候補の `SHELF_GRAPH_VERIFY=0` および
初期 graph 有効の出力と一致した。
さらに baseline 非投機 decode で graph instantiate を失敗させ、通常経路に切り替えた
256 token も元の非投機出力と一致した。同じ故障注入と `SHELF_GRAPH_VERIFY=0` を
指定した baseline tree の両 drafter でも、下表の最初の差は同じ位置に残った。

| drafter | mode | 非投機との最初の差 | 非投機 ID | 投機 ID |
|---|---|---:|---:|---:|
| BF16 | tree、budget auto | 172 | 3950 | 96724 |
| BF16 | chain、budget 0 | 130 | 97965 | 96089 |
| mul24 | tree、budget auto | 255 | 18232 | 39 |
| mul24 | chain、budget 0 | 256 token 一致 | — | — |

最初の差の logit margin は測っていないため、near-tie が原因かどうかは未確定。
`baseline_ja.py` と `greedy.py`、`verify.py` が比較に使った手順であり、
GPU UUID、ライブラリ、`flock /tmp/r9700.lock`、出力先はスクリプトに記載する。
例として baseline の chain は次のように再現できる。

```bash
flock /tmp/r9700.lock env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 SHELF_GRAPH_VERIFY=0 \
  LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib \
  /home/homelab1/coding-local/shelf/.worktrees/integ-fc/build-gfx1201/shelf-compute \
  --model-dir /home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 \
  --input-ids-file docs/research/20261113/exl3_general/prompt_ja.ids \
  --max-new-tokens 256 --max-seq-len 1024 --temperature 0 \
  --dflash2-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2 \
  --dflash2-drafts 7 --dflash2-tree-budget 0
```

tree は最後の `--dflash2-tree-budget 0` を省略し、mul24 drafter は
`/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24`
を指定する。非投機は DFlash2 関連の引数をすべて省く。decode graph も通常経路に
落として比較する場合は、`cc -shared -fPIC -o /tmp/fc_dflash2graph/fail_graph.so
docs/research/20261113/fixedcost_dflash2graph/fail_graph.c` で故障注入ライブラリを作り、
上記 `env` に `LD_PRELOAD=/tmp/fc_dflash2graph/fail_graph.so` と
`SHELF_GRAPH_DEBUG=1` を加える。graph instantiate が呼ばれた場合、失敗ログを確認する。

この差は既存の graph 外の経路にあり、今回の対象には含めない。

## 検討した graph cache

初期実装では row bucket ごとに graph を1件保持した。HTTP の BF16・2 request の
診断では、verify や decode の1〜10行が同じ bucket を奪い合い、capture 63回・
replay 12回だった。10行だけでも15回 capture した。capture の host 時間は
verify で約27〜29 ms、decode 1行で約41 ms だった。

8行と10行の DFlash2 verify を別々に cache し、それ以外の verify は通常経路へ戻す。
同じ HTTP 診断では capture 3回・replay 40回で、2 request の生成 token は baseline と一致した。
cache の再利用条件には行数、tree 配列と GDN capture mask / history / compact log の
pointer、および attention plan などを含める。

HIP graph instantiate を一時的な `LD_PRELOAD` で失敗させた実行では、エラーを検出し、
通常経路への fallback 後の BF16・code 256 token が graph 無効と一致した。

## 長時間実行

`stress.py` は1 process で固定3 prompt を切り替えながら36 request を投入する。
tree shape trace を有効にした BF16 実行では、有効・無効とも2,460 tree round、
各 request 256 token を完了し、全出力 JSON が byte 一致した。有効側は capture 3回・
replay 2,408回・再 capture 0回・graph エラー 0回だった。無効側は decode graph の
capture 1回・replay 11回である。各実行は5分未満の GPU ロックで完了した。
実行ログに GPU fault はなかった。kernel journal は参照できず、kernel 側の fault の有無は確認できない。

## 第1段階の測定

| drafter | baseline tok/s | 初期 graph tok/s | 変化 |
|---|---:|---:|---:|
| BF16 | 90.61 ± 0.30 | 88.76 ± 0.14 | -2.0% |
| mul24 | 90.45 ± 0.21 | 89.22 ± 0.40 | -1.4% |

HTTP の2回目の request は、BF16 が baseline 68.47、初期 graph 39.18 tok/s、
mul24 が baseline 75.24、初期 graph 49.16 tok/s だった。これは各1回の値であり、
単発の平均・標本 SD とは比較条件が異なる。各測定で baseline と候補の生成 token は一致した。

## 第2段階の HTTP

同じ方法の再測定では、BF16 が baseline 68.61、候補 69.58 tok/s、
mul24 が baseline 75.42、候補 76.57 tok/s だった。各1回の値である。

## attention plan と context 境界

`context_boundary.py` は code prompt を繰り返して入力を2,000 token とし、末尾に
元の prompt を置いた後、256 token 生成する。BF16・tree、max sequence 2304。
graph ON/OFF の生成列は byte 一致した。52 round の中で、10行の graph は
attention key 5 で一度 capture、key 65 への変化で一度再 capture され、49回 replay した。
長時間の36 request 試験では key 5 のままで再 capture はなかった。
2,038 token 開始の別試行も256 token 一致し、key 65 で一度 capture・54回 replay だった。

## 第2段階の単発測定と一致

| drafter | baseline tok/s | 8 / 10 行 cache tok/s | 変化 |
|---|---:|---:|---:|
| BF16 | 90.93 ± 0.06 | 91.32 ± 0.08 | +0.43% |
| mul24 | 90.59 ± 0.05 | 91.18 ± 0.05 | +0.64% |

両 drafter と code・prose・ja の固定3 prompt、それぞれ256 token で、
`SHELF_GRAPH_VERIFY=0` と graph 有効の生成 ID は byte 一致した。
初期実装でも同じ6条件で一致した。

指定の `test_compute_dflash2.py`、`test_compute_dflash2_unaligned.py`、
`test_server_dflash2_tree.py`、`test_server_dflash2.py` は通過した。
cache の所有権に関わる `test_executor_move_assignment` と、
`test_dflash2_footprint`、`test_qwen35_spec_verify` も通過した。
`test_compute_dflash2_tree.py` は一度だけ実行し、各 GPU 呼び出しの間でロックを
解放して全 prompt を通過した。この既存テストは最初の不一致が一定の logit margin 以下なら
許容するため、テスト通過はすべての tree / chain 構成で greedy と完全一致する意味ではない。
なお、このテストは一時 logits dump を `TemporaryDirectory` の終了時に自動削除した。
測定終了まで dump を保持する指定に反しており、個別の logits は再確認できない。
テストログとその他の測定出力は保持している。再実行用の `run_tree_test_split.py` は
一時 logits を削除せず保持するよう修正した。

## drafter forward の見込み

第2段階の単発3回の平均では、BF16 の draft host 投入時間は round 当たり
0.362 ms、GPU event 区間は 8.263 ms、round 全体は 53.430 ms だった。
mul24 はそれぞれ 0.487 ms、5.165 ms、49.721 ms だった。
host 投入時間と GPU event 区間は並行し得るため、その差や host 時間をそのまま
GPU の launch 待ち時間とはみなせない。

`rocprofv3` を使わずに `rocprof-sys-run --rocm=kernel,hip --trace` で
mul24 の GPU timeline を取得しようとしたが、終了コード132で停止し、trace は得られなかった。
したがって launch 間の空き時間と +0.5% 以上の改善見込みは確認できていない。
条件を満たす見込みが得られなかったため、drafter forward の graph 化は実装しない。

## 複数 request の decode graph

調査のみ。現行 gate は単一 request・単一 token の decode を維持する。
過去の複数 request replay による server 500 の直接原因は今回再現・特定していない。
コード上の要調査点として、graph key には request 数がなく、GDN convolution と
recurrence は host の request descriptor から request 数・最大行数を求め、
これを launch 引数や grid に使う。一方、device の descriptor は replay ごとに更新する。
同じ総行数でも request 構成が変わるケースでは、この不一致が危険な候補になる。
これが過去の500の原因であるとは確認していない。複数 request の graph を有効化していない。
