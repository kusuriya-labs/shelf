# 構造化出力と投機 decode の接続案

この文書は設計案であり、server、stdio protocol、DFlash2、MTP による
構造化出力は未実装である。まず通常 decode の各 token で matcher を進め、
初回 prefill を含む sampling の前に bitmask を渡す経路を完成させる。
文法の状態は prefix cache と共有せず、request ごとに生成開始位置から初期化する。

DFlash2 では drafter の K 個の token を host に取得した時点で compute から
server に送る。server は request の matcher から先頭行の mask を作り、
draft を順に accept して各次行の mask を作る。不正な draft の最初の位置で
打ち切り、その位置の target sample には対応する mask を使用する。
仮に進めた matcher は rollback し、compute が verify の結果として返す採用 token
だけを改めて accept する。mask 群は round ごとに 1 往復で送る。mask と token
の順序、打ち切り位置、途中終了、cancel、複数 request の識別を protocol で
検証する。verify は K+1 行の行別 mask を必要とするため、現在の単一行用
executor contract の拡張が必要である。

MTP は別の serve 実行経路を持つため、同様の分岐を別途実装し検証する。
将来 Draft forest を扱う場合は `GrammarMatcher.traverse_draft_tree` で各 node の
mask をまとめて作れる。

Python 側 matcher の利点は既存 wheel を使い compute の clang/libc++ ABI と
分離できることである。一方、round ごとの IPC と device 転送が必要になる。
compute 内への XGrammar C++ core 組み込みはこの IPC を省けるが、clang/libc++
での依存関係と build は未検証である。両方式の実効速度は未測定である。

受け入れ時は同じ model、prompt、seed、sampling 条件で、制約あり通常 decode と
DFlash2 の greedy 出力を token 単位で比較し、temperature > 0 でも JSON Schema
への適合を確認する。制約なしの投機速度、および混合 batch の速度を別途測る。

## M3 時点の残作業

現行実装は制約付き request を `submit_unconstrained_decode` へ送り、DFlash2 / MTP を
使わない。reasoning 併用でも同じ通常 decode を使い、close token から制約を開始する。
`dflash2_spec_step_batch` は draft、target verify、採用結果の確定までを同期関数内で
行うため、単に request に mask pointer を追加しても server との往復を挟めない。

1. batcher と decoder の round を draft 完了・mask 待ち・verify に分割し、待機中の
   request のみ scheduler から除外する。request ID、round ID、draft token 列を stdio
   event として送り、応答の行数と round ID を照合する。cancel、disconnect、終了済み
   request に対する遅延応答を処理する。
2. server の matcher で各 draft 手前の mask を埋め、受理可能な token のみ一時 accept
   する。最初の不正 token で draft を切り、そこで target の correction を sample する
   行まで返す。EOS、matcher 終端、最大長も考慮する。一時 accept した個数を rollback
   してから、compute が実際に採用した token 列だけを順に accept する。
3. verify の各 logits 行に対応する mask 行を GPU へ転送して sampling 前に適用する。
   draft 打ち切り後の行数と GDN / KV の rollback、bonus token の処理を一致させる。
   reasoning 中に close token が draft に含まれる場合はその境界で round を切り、次の
   round から grammar を開始する。制約なし batch は既存の高速経路を維持する。
4. MTP は別 serve 経路のため別途対応する。通常 decode と DFlash2 の greedy token 列
   完全一致、temperature > 0 の schema 適合、混合 batch・cancel・強制 close を検証し、
   制約なし投機速度と制約あり / reasoning ありの overhead を測定する。

matcher の手動 accept / rollback と各 verify 行の mask は [XGrammar の統合ガイド](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html) に沿う。

## M5 の進捗

server の `GrammarState.draft_masks` は draft 列を一時 accept し、最初の不正
位置で打ち切った prefix の各位置と correction / bonus 用の mask を返して、最後に
matcher を rollback する。EOS で matcher が終端する draft は、その位置の mask で
通常の target sampling を行うため、終端 token の直前で打ち切る。例外時も一時状態を
rollback する。XGrammar 0.2.9 の CPU 単体テストで、各 mask、正常・不正 prefix、
EOS、空 draft、例外時の rollback を検証した。

この補助関数はまだ stdio protocol や decoder に接続されていない。`dflash2_spec_step_batch`
は proposal の直後に同期的に target verify まで実行する。接続には proposal の host
取り出しと batch 状態の保持、待機 request の識別と cancel、行別 mask の検証・転送と
verify sampler への接続、採用 token の server matcher 反映が必要である。現行の
constraint 付き request は引き続き通常 decode にフォールバックし、DFlash2 との
併用、greedy 一致、性能と往復時間は未検証・未測定である。

## M6 の行別 mask 基盤

`ScheduledRequest` は verify に device 上の mask 行列と word 単位の行間隔を
渡せる。descriptor 展開時に各 output 行の pointer を進め、既存の sampling 前の
mask kernel を使う。行間隔が語彙サイズに必要な word 数より短い場合は拒否する。
通常の単一行 mask は行間隔 0 の従来契約を保つ。mask が無い request では従来の
sampling 経路を使用する。

これは sampler 側の基盤であり、DFlash2 から mask 行列を供給する経路ではない。
draft / verify 分割、round ID と cancel を含む stdio 往復、採用 token の matcher
反映は未実装である。制約付き request は引き続き通常 decode にフォールバックする。
reasoning 中の draft が close token を含む場合は close token で打ち切り、次の
round から grammar を開始する方針だが、この境界処理も未実装である。
