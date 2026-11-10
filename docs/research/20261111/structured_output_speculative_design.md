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
