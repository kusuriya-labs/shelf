# Hy4-flash の実装

Hy4 は専用の checkpoint loader と decoder を使用する。MLA・sliding attention・EXL3 線形・WMMA の Hy4 固有実装は `models/hy4_flash` に置く。radix 選択は model 非依存の `kernels/fused/detail` を共有する。checkpoint の model_type、層数、形状、量子化形式を検証してから実行する。attention、KV、head、DFly と shared expert は rank 0 に置き、routed expert を配置表に従って各 rank へ置く。FP8 Engram はホストに保持し、必要行だけを取得する。GPU の一覧は ROCr の可視 device 順を使用し、gfx1201 以外を拒否する。

expert の転送は peer access を有効にし、送信側 rank の kernel から受信側 rank の領域へ書く push を既定とする。宛先バッファの再利用と転送完了は event で順序付ける。peer access を利用できない場合は pinned host buffer を使用する。`SHELF_HY4_EP_TRANSPORT=host` で host 経由、`peer` で受信側 rank の kernel による peer 読み出しを選べる。各 rank は expert ID 順に BF16 で部分和を計算し、rank 0 が rank 順に BF16 で合算する。配置表や EP 数を変えると加算の括り方が変わり、greedy ID が変わることがある。予約時に各 rank の実空き容量と配置表を照合する。収まらない場合は `Status::unsupported` を返す。配置作成手順は[操作方法](../user/hy4_flash.md)を参照する。

`--expert-parallel` と `--tensor-parallel` で EP × TP の構成を選ぶ。rank は `ep * tp_size + tp` の順で並び、配置表の値は expert の所有 EP group を表す。TP では次の matrix axis を分割する。

- N 分割: `q_b_proj`、`kv_b_proj`、`linear_gate`、MLP と shared expert の `gate_proj` / `up_proj`、expert の `gate_proj` / `up_proj`
- K 分割: `o_proj`、MLP と shared expert の `down_proj`、expert の `down_proj`
- replicated: `q_a_proj`、`kv_a_proj_with_mqa`、indexer、router、embedding、norm、hyper-connection、`routed_expert_down_proj` / `up_proj`、lm_head、DFly drafter

N / K の分割境界は 128 要素単位に揃える。attention は head を TP lane に割り当て、`learnable_sink_param` は lane ごとの head 範囲を使う。row-parallel の出力は TP group 内で bf16 の all-reduce を行い、routed expert の出力は TP group 内の reduce の後に EP group 間で reduce する。TP と EP は別の受信バッファを使う。all-reduce の順序が変わるため、EP 構成との logits 一致は bit 一致ではなく相対誤差で判定する。TP lane は全 rank が attention と KV を持つため、workspace は TP 時に 1 GiB、KV と index は全 rank に確保する。2Mi 予約は TP=2 でも成立するが、`generate` の予約長上限は 1Mi である。

Attention は展開前の KV latent を用い、query と value に fold 済みの KV 射影を適用する。行数にかかわらず、各 query で参照する KV の列を確定してから、同じ順序で射影、スコアの正規化、値の集約を行う。線形射影の K 方向の分割数も行列の形状から決める。4 bpw の 3072 → 1536 射影は、6 行では K の各分割が使う 3 グループの符号とスケールを LDS に置く。同じ 4 bpw 形状の TP=1 の共有 FFN は 16 行以下で同じ入力に対する gate と up をまとめて起動し、出力回転と SwiGLU を融合する。部分和の加算順序、ゲートと up の BF16 丸め位置は個別の射影と同じである。

Prefill の固定 EXL3 射影は 256 行以上で、K-major の trellis を N-major の INT8 タイルへ作業域内で展開する。同じタイルを複数の入力行で共有し、128 個の K 要素ごとに INT32 の積和へ入力スケールを掛ける。出力回転までの K の分割と BF16 丸め位置は小さい行数の経路と共通である。

Prefill MoE は最大 1024 入力行ずつ処理する。GPU で top-16 を expert 順に並べ、rank ごとのビットマスク、expert ごとの件数と開始位置を求める。同じ expert への入力を最大 32 行にまとめ、gate と up の入力回転、SwiGLU と down の入力回転をそれぞれ融合する。expert 内の作業行と元の加算位置の対応を GPU に保持する。push と peer では宛先に経路を持つ行の latent と、その rank の選択情報だけを転送する。戻りは push では送信側が rank 0 の領域へ直接書き、peer では rank 0 が受信側の部分和を直接読んで加算する。host 転送には方向ごとの pinned buffer と event を使う。

疎な層の Indexer は各 query の直前までに完了した 4-token グループから 511 個を選び、現在グループの causal な末尾を加える。位置 2047 までは全履歴を使う。位置 4096 以降の prefill では 16 ヘッドと 16 グループの BF16 内積を WMMA で求める。スコアの上位 16 ビットから 511 位を含む区間を決め、query と key の最大絶対値、head weight の絶対値和から求めた丸め誤差の余裕を両端に加える。この区間のスコアを scalar 経路と同じ順序で再計算してから、同点をグループ番号順に扱う radix 選択を行う。decode、verify と 6 行以下の入力は scalar 経路を使う。

EP の prefill は最大 4096 行を受け付ける。attention の一時行列は最大 1024 query 行ずつ再利用し、疎な長文では 16 行ずつ処理する。疎な選択結果の領域は予約長と実行できるチャンク長の小さい方に合わせて作業域内に確保する。EP の作業域は rank 0 が 2 GiB、peer が 512 MiB である。TP は 1 GiB の作業域に合わせて要求を最大 1024 行に分ける。

Target の Graph は各 MoE 層で scatter、rank ごとの expert、合流の順に実行する。scatter と expert の完了はホスト側で event を記録し、受信側の stream が待つ。合流後は同じ作業域を次の層で再利用し、最終層では head の正規化結果を DFly2 用の state に保存する。Graph キャッシュのキーには行数、verify、2048 境界までの行数、KV のバケット、作業域の開始位置を含める。単一 DFly2 は受理した行、位置、seed、入力 ID を起動ごとに更新する。サンプリング設定は DFly2 のキャッシュキーに含める。

単一 GPU の `test_hy4_graph` は本体と DFly2 の checkpoint ディレクトリを引数に取る。42 層の計算を VRAM に収めるため、第 1 層の routed expert 重みを各層から参照する。各層の router と共有 expert、Attention、Engram、head、DFly2 は元の重みを使う。3・4 論理 rank の試験では rank ごとに stream と作業域を用意し、同じ物理 GPU 内で転送と event 同期を実行する。

`test_hy4_graph BODY DFLY transport` は転送方式、DFly2、Graph 設定、バッチ件数による greedy ID の一致と、作業域を異なる値で埋めたときのチャンク別 logits を検査する。末尾に配置表を渡すと、可視 GPU に全重みを実配置して同じ検査を行う。`test_hy4_graph BODY DFLY upload OWNERS` は実機の配置表で重みを予約・転送し、全 payload、expert 配置・射影表、fold 済み KV 射影をホストの基準と照合する。BODY に FP8 Engram の 4 表が必要である。1 GPU で OWNERS に `-` を渡すか省略すると、上記の expert 共有による検証を行う。末尾に `INPUT_IDS CHUNK LOGITS_FILE` を追加すると、同じプロセスで初回 prefill の全 logits を保存し、eager で再実行した各層の hidden state のハッシュ、rank の PCI アドレスと重み・作業域のアドレス、初回 logits との一致を出力してからアップロード内容を照合する。

`test_hy4_graph BODY DFLY layers OWNERS INPUT_IDS LOGITS_FILE` は 1 台の GPU に各層固有の全 expert を順次転送し、指定の配置表で論理 EP を構成する。1〜32 token の入力についてチャンク間の各層出力のビット一致を検査し、全入力位置の logits を FP32 で保存する。

`tests/unit/test_hy4_flash.py` は CPU の Octave、Engram と expert 配置を検査する。実 checkpoint の層照合では `tools/hy4_flash/check_layer.py` で CPU 基準 fixture を作り、`test_hy4_flash_layer` に渡す。`tools/hy4_flash/check_linear.py` と `check_chain.py` は量子化線形と decoder の CPU 基準照合に使用する。DFly checkpoint の生成には `tools/hy4_flash/quant_dfly.py` を使う。

eager 実行では各 forward の開始時に 4 つの Engram 層の行を非同期に先読みする。`SHELF_HY4_ENGRAM_PREFETCH=0` で無効化できる。Graph 実行では第 0 層を起動した後、ホストで必要な Engram 行を集め、第 1 層の前に同じ stream で転送する。Engram は既定で populate しない。`SHELF_HY4_ENGRAM_POPULATE=1` を指定すると、生成前に 4 表を populate する。server の既定動作は Engram の mmap に `MADV_POPULATE_READ` を適用し、`mincore` で常駐状態を記録してから request を受け付ける。`SHELF_HY4_ENGRAM_POPULATE=0` では遅延 mmap を使用する。`SHELF_HY4_ENGRAM_MLOCK=1` は全表の memlock 上限を検査し、各 mmap を `mlock` して全ページの常駐を検証する。上限不足、システムコール失敗または固定後の常駐不足は起動を失敗させる。mmap の解除時に固定も解除される。固定なしでは populate 後の常駐は保証されない。起動時間と操作方法は[利用者向け手順](../user/hy4_flash.md)を参照する。

`SHELF_HY4_PROFILE_EP=1` では request 前後の Engram 常駐ページとプロセスの page fault 数を記録する。Hy4 server の compute 起動待ちは 900 秒である。

## 複数リクエストの実行

TP=1、EP=1〜4 では、SWA 30 層の固定長リングバッファと生成履歴をスロットごとに持ち、DSA 12 層の KV（294 B/tok/layer）と圧縮 index（64 B/tok/layer、4-token グループ単位）は `kv_page_tokens = 256` の共有ページプールとスロット別 `page_table` を介して参照する。DFly2 の隠れ状態と疎な attention の選択行もスロットに対応付ける。バッチ decode と verify は同じ rank 別 Graph を使う。EP=1 では層 0 と残りの層をまとめた 2 個の Graph を順に起動する。EP が複数の場合は rank 0 の routed expert と共有 FFN を別々の stream で実行し、合流後に部分和を加える。バッチ DFly2 は指定された `K` 候補を 1 回の Graph 起動で生成する。Target は最大 4 件、DFly2 は最大 2 件の Graph を予約単位で保持する。バッチのキーには稼働中の件数と候補数 `draft_tokens` を含め、上限を超えた場合は順番に入れ替える。疎な attention を使う行ではスロット番号と 2048 境界までの行数を区別し、draft ではスロットごとのサンプリング設定も区別する。位置、seed、受理行、token ID、Engram 入力は再生ごとに更新する。対応する転送構成と切替方法は[操作方法](../user/hy4_flash.md)を参照する。

DFly2 の候補数 `K` は 2〜5（既定 4）。`--dfly-adaptive` を有効にすると、`Hy4DflyAdaptiveController` は投機ラウンドごとに、スロットあたりの平均受理数（anchor を除く）の指数移動平均（EMA、係数 0.25）を更新する。開始時と切り替え直後は最初の観測値で EMA を初期化し、同じ `K` で 8 ラウンド観測してから次の切り替えを許可する。上昇は一度に 1 段とし、下降時は受理数に見合う段まで下げる。

| 稼働中のスロット数 | 切り替えの境界 | 上昇する平均受理数 | 下降する平均受理数 |
| --- | --- | --- | --- |
| 1〜16 | 2 / 3 | 0.90 以上 | 0.70 未満 |
| 1〜16 | 3 / 4 | 1.95 以上 | 1.65 未満 |
| 1〜16 | 4 / 5 | 3.35 以上 | 3.05 未満 |
| 17〜32（上限 4） | 2 / 3 | 1.50 以上 | 1.20 未満 |
| 17〜32（上限 4） | 3 / 4 | 2.55 以上 | 2.25 未満 |

TP>1 の予約は同時リクエスト数が 1 を超えると `Status::unsupported` を返す。

`generate_batch` は 1〜32 件を受け付ける。prefill は入力ごとに実行し、decode では処理中のスロットと位置を GPU に渡す。通常の dense・sliding decode は各層で全行のスコア計算、正規化、値の集約を 1 つのカーネルで行い、KV と index を一括更新する。疎な attention の対象になる位置はスロットごとに処理する。

バッチ decode と verify で各 EP rank の GPU 計画は所有 expert、入力行、重み、元の加算順を保持する。複数行は expert ごとにまとめ、最大 16 行の `Hy4ExpertBatch` を全 rank で gfx1201 WMMA に渡す。部分和は各入力行の expert 順で計算し、rank 0 が rank 順に加算する。転送には通常経路と同じ方向別の peer 判定と 2 組の event を使用する。

`Hy4BatchSession` は要求を FIFO で空きスロットに割り当て、Prefix Cache の完全ページをページ参照カウントで共有し、末尾の部分ページのみコピーまたは引き継ぎで再利用する。ページプールの空きが不足する場合は未使用 Prefix Cache エントリを LRU 順に解放し、それでも不足する要求は `waiting` キューで Admission Control を行う。1 回の `step()` で prefill の 1 チャンク、または準備が済んだ要求の decode を進め、`prefill_decode_interval`（既定 `1`）ごとに完了前の長文 prefill チャンク間に decode ステップを優先割り込みさせる。キャンセルは指定した要求だけを完了させ、解放したスロットに待機中の要求を割り当てる。DFly2 は各スロットの anchor と `K` 候補（`K + 1` 行）をまとめ、最大 192 行を検証する。`Residency` / `GraphInput` / `BatchGraphInput` / `sparse_top_slot` / SWA の固定バッファ容量は最大 `K=5`（6 行）分を確保し、実行時の `K` は `Target` / `Hy4BatchSession` / `GraphKey` で保持する。受理数と乱数 seed は要求ごとに管理する。いずれかの要求で KV 残量が `K + 1` 未満、または生成できる残りが 1 トークンの場合、そのステップは通常の decode を使う。`SHELF_HY4_BATCH_DFLY=0` でバッチ投機を無効にできる。`serve` の `stats` / `done` イベントには現在の `draft_tokens` と切替回数 `k_switches` を出力し、`/v1/meta` の `speculative` には `"adaptive": true/false` を含める。

`test_hy4_graph BODY DRAFT batch` は単一 R9700 上で 1〜4 論理 rank を使い、バッチの greedy 生成 ID を B=1 と照合する。単一 rank では B=2〜32 の各件数を検査する。位置 2047・2053 の混在は各 EP 構成で検査する。4 スロットのセッションではキャンセル後の再利用、FIFO、KV 残量が異なる要求、capture 失敗後の eager 復帰を検査する。

`test_hy4_graph BODY microbench` は 4 bpw の共有 expert の線形射影と routed expert の 3 射影を測る。`test_hy4_graph BODY moebench` は router、共有 expert、加重和を含む MoE 全体を測る。`test_hy4_graph BODY DRAFT stepbench B [EP]` は指定した論理 EP（既定は 3）で同じ入力に対して Graph 有効・無効の step 時間と生成 ID を照合する。投機なしの計測では DRAFT に `-` を渡す。

`test_hy4_graph BODY prefillbench LENGTH CHUNK EP OUTPUT` は第 1 MoE 層の expert を共有して指定長の prefill を実行し、最終位置の FP32 logits を OUTPUT に保存する。OUTPUT に `moe` を渡すと共有 FFN と MoE 全体、`index` を渡すと Indexer のスコアと選択集合を測定する。`capacity` はすべての経路を 1 rank に集中させ、rank 0 と末尾の peer で出力と経路数を照合する。

## `lm_head` 複数 GPU 語彙分割

`lm_head`（`[120832, 3072]`）は、総物理 rank 数 `ranks = ep * tp > 1` かつ `120832 % (ranks * 128) == 0`（例: `ranks = 2, 4`）で `SHELF_HY4_LM_HEAD_TP` が `0` 以外のとき、TP=1 / TP>1 双方で語彙方向（128 列タイル単位）に全 rank へ均等分割してロード・実行する。端数により割り切れない rank 数（例: `ranks = 3`）および `SHELF_HY4_LM_HEAD_TP=0` では rank 0（TP>1 では各 rank）に全語彙を維持する。

greedy 生成時は各 rank が担当語彙のローカル最大値とグローバル語彙 ID を計算し、`tp_argmax_pack` / `tp_argmax_reduce`（8 B/row）で集約する。sampling / logits 出力時は `tp_scatter_logits` で全語彙 logits へ組み立てる。`lm_hidden_buf` / `lm_head_recv` / `lm_logits_recv` / `lm_staging` と `lm_ready` / `lm_done` / `lm_arrived` / `lm_consumed` は、`tp_recv` / `ep_recv` / `reduce_staging` / `tp_*` / `ep_*` から物理的・論理的に完全に分離し、MoE reduction との WAR ハザードを防止する。HIP Graph 経路では `GraphHeadStep` を介して `scatter` / rank 別 `rank_exec` / reduction をイベント同期で再生する。`test_hy4_graph BODY DRAFT lmhead` は `SHELF_HY4_LM_HEAD_TP=0` と `1` の出力（prefill logits、greedy/sample、K=2..5、batch、Graph ON/OFF、push/peer/host）の一致と rank 0 の重み削減を検証する。
