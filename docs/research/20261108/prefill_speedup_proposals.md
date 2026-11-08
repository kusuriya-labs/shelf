# R9700 27B EXL3-INT8 prefill の高速化案

2026-11-08〜09。根拠・計測条件・event の限界は [測定記録](../../perf/prefill-breakdown-r9700.md) を参照。本書は実装の提案であり、候補の速度・正しさは未検証。shelf の数値は 2048 token / chunk 2048 の診断 event の 5 回中央値を使い、Saber の実サーバー trace は各条件 1 回とする。通常の GPU event 中央値は 1143.6 ms、診断は 1196.5 ms であるため、以下の各上限を通常計測にそのまま足し込まない。候補は対象区間が重なるため利得は加算できない。優先順は **対象時間と長文での影響を勘案した仮説**であり、実装後の順位ではない。

精度区分: **B0** は同じ入力で bit 一致を目標、**B1** は演算順・丸めが変わり得る、**A** は量子化や近似を変え出力分布に影響し得る。B0 も bit 一致を試験する。B1/A は同一 checkpoint / token / KV / chunk に対し `shelf-bench quality` と `tools/quality_gate.py` で全語彙 KLD（平均・p95・最大）、PPL、top-1/top-5、code・日本語・長文を含む end-task、および greedy `GENERATED_IDS` の一致と初回分岐を評価する。長文の GDN state と chunk 境界、非有限値、複数 seed も調べる。差の閾値は実験前に合意・固定し、速度だけで採用しない。

## 2026-11-09 改訂: Saber の 2048・8192 token 実測を踏まえた優先順位

以下が現時点の優先順位。後半の旧案・順位は判断の履歴として残す。Saber 側は 27B・TP=1 の **MXFP4 W4A8、FP8 KV**、shelf 側は **EXL3-INT8、BF16 KV** であり、投影の結合や GDN a/b の扱いも異なる。測定対象も Saber の実サーバー kernel trace、shelf の診断 dispatch event で完全には一致しない。速度差を同一演算の純粋な実装差と断定しない。Saber の値は到達可能性を考えるための他形式での参考値であり、EXL3 での到達保証ではない。取得方法・計測差・生ログは [比較記録](../../perf/prefill-breakdown-r9700.md#saber-との比較) を参照。

2048 token の Saber 実サーバー kernel trace は、最後の head と drafter の補助 tap を含む 764 launch、kernel 時間の合計 **461.676 ms**、先頭 kernel の開始から最後の kernel 終了まで **467.112 ms**。隙間は 5.437 ms だが、その区間の profiler 観測上の隙間であり GPU 全体の遊休時間ではない。同一区間の MXFP4 線形 304 回は計 **387.395 ms、257.5 TOPS 相当**、主要 5 形状は 253.7〜260.5 TFLOPS、小さい 96×5120 は 63.7 TFLOPS。別実行の host forward counter は profiler なし 483.131 ms、あり 488.126 ms で +1.0%（各 1 回）。op 名の対応は別実行の event 計装の順序・全 kernel 名との完全一致で確認した。shelf の EXL3 dense INT8 本体は 400 行列で 529.68 ms、約 208 TOPS。**算術的な差 142.3 ms は演算内容・形式が異なる参考差**であり、そのまま短縮予測としない。Saber の全 467.112 ms より shelf の GEMM 本体だけが長いという観測は、今回の直接 trace でも成立する。

周辺処理は shelf の入力回転・量子化 126.36 ms、rowscale / flag 47.79 ms、復号展開 51.90 ms、flag GEMM 25.70 ms、出力回転・scale 204.73 ms。Saber の対応する MXFP4 展開は GEMM 内に含まれ、同じ EXL3 回転・flag 分岐はない。Saber のその他の producer は add+rms+quant 14.657 ms、SiLU+mul+quant 19.273 ms、QK norm/RoPE 3.797 ms、KV append 0.452 ms 等で、shelf の各処理との一対一対応ではない。したがって周辺処理差の全額を削減可能とは扱わない。

GDN の conv / KKT / scan は Saber で 6.713 / 2.290 / 10.174 ms、shelf の conv / recurrence は 9.24 / 51.10 ms。scan と recurrence の内部処理・state 型は同一ではなく、引き算は最適化余地の仮説に限る。attention kernel は Saber 7.363 ms・112.0 TFLOPS、shelf 25.30 ms。Saber は FP8 KV、shelf は BF16 KV のため、この 17.9 ms 差は dtype・アルゴリズムを含む。層全体は Saber の kernel 時間の和で GDN 342.486 ms、attention 116.607 ms、shelf はそれぞれ 882.82 / 303.76 ms（両者とも MLP / norm 等を含む）。層別値と op 別値は重なる。

| 順位 | 案 / 精度 | 実測に基づく参考と割引 | 規模・主なリスク |
|---:|---|---|---|
| 1 | INT8 GEMM の shape ごとの algorithm / workspace 探索、B0/B1 | 529.68 ms を対象に、他形式 Saber の 387.395 ms を参考とする。差 142.3 ms は同形状でも同演算でもないため保証しない。例えば shelf 本体を 10% 改善できれば約 53.0 ms、診断全体の 4.4%。 | 中。現行 heuristic を含め、workspace 上限 0 と tuning off を対照にする。VRAM、tuning の初期費、整数積と flag 分岐の一致を確認。 |
| 2 | EXL3 trellis 復号と INT8 WMMA の tile 融合、B0 目標 / B1 | 展開 51.90 ms と GEMM 529.68 ms が対象。Saber の fold 込み線形 387.395 ms は「重み変換込みで高実効性能を出せる」別形式の参考。EXL3 復号は複雑であり、差を実現可能幅と見なさない。展開だけの除去上限は 51.90 ms、仮に GEMM も 10% 短縮なら合計約 104.9 ms だが未実証。 | 大。ALU / LDS / VGPR と occupancy、flag 行・scale・丸め、重み再利用が危険。1 の高速 library と同一形状で比較。 |
| 3 | 回転・scale・量子化の traffic と融合、B0 目標 / B1 | shelf の回転量子化・rowscale・出力回転の合計 378.88 ms が対象。Saber の producer は別演算ながら上記のように数十 ms 規模。20% 減の仮定は 75.8 ms、6.3%。回転演算の省略や全額削減を前提にしない。 | 大。重み固有の回転、128 要素 scale、flag 行の correction と丸め位置の保全が必要。 |
| 4 | GDN scan と補助 op、B1 | Saber の conv/KKT/scan 合計 19.176 ms 対 shelf conv/recurrence 60.34 ms。約 41.2 ms の単純差は非同等演算の参考であり、削減額ではない。shelf recurrence 半減という算術例は 25.55 ms。 | 大。状態型・scan 順序と chunk 境界に品質リスク。 |
| 5 | 長文 attention、B1/A | 2K 時点の単純差 17.9 ms は FP8/BF16 KV の差を含む。8192 token の Saber は 4 chunk 計 92.009 ms、shelf は別の 32K 実行の最初の 8192 token 分で 372.27 ms。32K の shelf attention 5865 ms も対象だが、条件差を揃えて測る。 | 大。paged KV、causal mask、FP8 KV を採用するなら精度 A。 |
| 6 | chunk と scheduling、B1 | 8192 token は Saber 実測で 4 prefill step / 8192 live rows、各 2048 行。attention は chunk ごとに 7.417 / 17.694 / 28.236 / 38.662 ms と増える。shelf の chunk 4096 は長文で速かったが品質差が報告されており、既定値を変える前に gate が必要。 | 小〜中。単独 latency と多重要求を分け、長文の state 精度を確認。 |
| 7 | launch / graph、B0 | Saber の実 server は 2048 token 764 launch、8192 token 3038 launch。前者の 467.112 ms 区間中、kernel 間は 5.437 ms（1.16%）、後者の各 chunk 内合計は 22.021 ms（1.14%）。全 GPU idle ではなく profiler の影響を含む。shelf は少なくとも約 4199 launch の静的下限で、隙間は未分離。 | 中〜大。trace で棚側の隙間を得てから順位を決める。 |

DFlash2 context の 310.7 ms は **serving 固有の別区間**として引き続き優先する。上表は非投機 target forward の順序であり、旧版の DFlash2 1 位を否定する結果ではない。旧版の EXL3 融合 2 位は維持し、BLASLt 探索を 5 位から 1 位に上げた。attention は 2K の差が線形・周辺処理より小さいため、旧 3 位から長文向けの 5 位に下げる。旧 25% GEMM 短縮例は仮定であり、Saber の今回の値は EXL3 での 25% 改善を実証しない。

### 上位 3 件の実装と検証

1. `src/shelf/inference/execution/exl3_dense_gemm.hip` の `run_dense_gemm` / `find_algo` と `SHELF_EXL3_INT8_TUNE` を対象に、M=2048、実際の N/K 毎に現行 algorithm を含め candidate を事前 warmup 後に GPU event で交互測定する。workspace の候補を 0 と有限量で試し、実行中の `hipMalloc` / `hipFree` を避けるため executor が確保する workspace の上限・寿命・cache key を設計する。単体の INT32 accumulator、flag の結果、出力回転後を現行と比較し、全 forward と long context を再測定する。workspace 費とチューニング費も別に記録。
2. `src/shelf/kernels/optimized/linear/exl3_int8.hip` の `exl3_int8_recon_kernel` と `exl3_int8_gemm_kernel` を参考に、`src/shelf/inference/execution/optimized_dispatch.hip` の `try_launch_linear_exl3_int8` に実験経路を追加する。まず既存の行 scale / flag を維持し、通常行の trellis を tile 単位で INT8 に復号して WMMA に渡す。flag 行は現行補正経路を維持する。16×16 の fragment 順と trellis の循環 window を同じ値で復元し、全 bits・端数・flag 発生入力を比較。低 M、M=2048、長文、各主要 N/K で展開＋BLASLt の合算と比較する。int32 overflow、scale、出力回転と最終 logits を検査する。
3. 同じ dispatch と `exl3_int8_rotate_quantize_kernel`、`exl3_int8_rowscale`、`exl3_int8_rotate_out_i32_kernel` を対象に、中間 buffer ごとの read/write と時間をまず検証し、各行の 128 要素の回転・scale と flag 判定、出力の `svh` と Hadamard の依存を保った融合候補を試す。重みごとに入力回転が異なる場合は共有しない。融合前後の中間値と非有限値・flag 行を照合し、品質 gate と end-to-end で評価する。

### EXL3 を LDS で復号する設計の概算

Saber の W fold は MXFP4 の block scale を FP8 byte に折り込む方式で、EXL3 の循環 trellis 復号をそのまま置き換えることはできない。shelf の 16×16 trellis tile は 32 lane が各 8 個を bit window と整数乗算・byte sum で復号し、既に WMMA B fragment の順を持つ。候補は各 workgroup が **M=128、N=128、K=64** の slab を扱い、例えば 4×8 個の 16×16 重み tile を協調復号し、INT8 の B slab 128×64 = 8 KiB を LDS に置く。A slab 128×64 = 8 KiB と合わせて単一 buffer 16 KiB、両方 double buffer なら **32 KiB**（padding・alignment・barrier 等を除く）。実装可能性を示す容量計算であり、実機の割当・bank conflict・occupancy は未測定。

128×128 の出力を INT32 で保持する最低量は workgroup 全体で 64 KiB、256 thread に均等配分できれば 64 個の 32-bit accumulator/thread、512 thread なら 32 個/thread。これは理想的な算術下限であり、WMMA fragment の複製、アドレス、復号の `cur`/`nxt`、activation、scale や一時値、compiler 配置を含まないため、実 VGPR 使用数・spill は必ず compiler と profiler で測る。EXL3 復号には重みごとの整数 ALU と bit extraction が必要で、B slab を M 方向へ再利用しても復号 ALU、LDS 帯域、barrier が支配する可能性がある。スラブの K=64 を跨ぐ 128 要素 scale の扱いに注意し、通常行は既存 rowscale で全 K の整数累積を再現、flag 行は各 128-K group の scale を維持する。flag 行に通常行の scale を適用したり、別々の group を単一 INT32 累積に混ぜて後で 1 scale を掛けたりしてはならない。まず通常行のみ融合し、flag は既存経路で残して正しさと実速度を確認する。

## 旧版の優先候補（履歴）

| 順位 | 案・精度 | 対象と期待・上限 | 規模 / 依存 / 主なリスク |
|---:|---|---|---|
| 1 | DFlash2 prefill の特徴投影・context ring 構築、B0 目標 / B1 | serving 限定。2K 診断で target 後の append が 310.7 ms、診断 TTFT 1472.2 ms の 21.1%。仮に半減なら約 155 ms、10.6%。完全除去は算術上限で、context は必要。非投機 `pp` には効かない。 | 中〜大。FC と 5 層 K/V 投影、layout と ring、prefix cache、state 整合が依存。精度と長文の位置合わせを必須検証。 |
| 2 | EXL3→INT8 復号と行列積の tile 融合、B0 目標 / B1 | dense GEMM 529.7 ms と展開 51.9 ms。展開のみ完全除去なら診断全体の最大 4.3%、さらに GEMM を仮に 25% 短縮なら両方で約 184 ms、15.4% が算術例。実測予測ではない。融合が GEMM を遅くすれば負ける。 | 大。trellis 復号・WMMA operand 配置・正しい scale、長い K に対する occupancy が必要。B1 は累積順変更時。 |
| 3 | BF16 KV の既存 prefill WMMA の tile・KV 再利用改善、B1 | 2K attention 25.3 ms、32K 診断 5 回の p50 は 5865.0 ms。仮に 32K の attention を半減すれば 2932.5 ms、同診断全体 24576.2 ms の 11.9% 短縮。完全除去上限は 24.0%。別 process 5 回の event 診断であり、通常実行の速度ではない。2K の半減は 12.7 ms に限る。 | 大。既存 kernel は 32 query 行・8 wave・BF16 WMMA で動作する。causal mask / paged KV / online softmax・部分結果の精度と GQA の対応が必要。矩形 tile の不要な計算や LDS 容量が逆効果になり得る。 |
| 4 | 出力回転・活性準備を融合し中間 traffic を減らす、B0 目標 / B1 | 出力回転 204.7 ms、活性回転量子化 126.4 ms、rowscale 47.8 ms。仮にこれらを一律 20% 短縮で 75.8 ms、全体の 6.3%。完全除去は 378.9 ms だが演算自体は必要で到達不能な上限。 | 大。行と列の変換順、flag 行の補正、BF16 中間丸めを保つこと。残差や gate / up 入力を共有できるかは重み固有 `suh` を確認しなければならない。 |
| 5 | hipBLASLt algorithm / tile と workspace の形状別探索、B0/B1 | GEMM 529.7 ms。25% 短縮の仮定なら 132.4 ms、11.1%。探索候補には現行 heuristic を必ず含め、すべての形状に勝つ保証はない。 | 中。`exl3_dense_gemm.hip` は既定 workspace 上限 0、INT8 tuning 既定 off。初回 tuning をリクエスト外に置く、VRAM と cache key・再現性の検証が必要。整数積が同一でも前後の変更は B1 の可能性。 |
| 6 | dense 展開済み重みの部分 cache、B0 | 展開 51.9 ms のうち cache hit に対応する分が理想上限。全層の展開済み値は 22.66 GiB 追加で、既存圧縮重みを残す設計では 32 GB に収まらない見込み。部分 cache で hit した分だけ節約できるが、帯域圧迫や初回展開費で逆転もあり得る。 | 中〜大。arena / KV 予算と優先度・所有権、複数 request、解放時の安全性が必要。EXL3 を捨てる方式は decode 性能と実際のメモリ差を別途評価。 |
| 7 | GDN chunked scan と補助 op の改善、B1 | recurrence 51.1 ms、conv 9.2 ms。recurrence 半減の仮定で 25.5 ms、2.1%。48 層分で高頻度。状態表現・scan の積算順が変われば B1。 | 大。`gdn_recurrence_dispatch.hip` と `kernels/optimized/gdn/recurrence.hip` の prefill 契約、状態受け渡しと複数 request の検証。 |
| 8 | chunk 4096 の候補化、B1 | 今回の 4096 / 8192 / 16384 / 32768 token は chunk 4096 でそれぞれ 6.6% / 7.0% / 5.8% / 5.9% 短縮したが、交互 A/B ではない参考値。過去の独立測定は 8192 / 32768 token で約 7.6 / 6.5% 短縮だが、chunk 2048→4096 の平均 KL 0.264、top-1 一致 95.2% という報告がある。速度だけで既定変更しない。 | 小。原因切り分け、GDN state、token ごとの差、長文とモデル横断の品質 gate が先。過去の値は別入力・別 revision。参照は `feature/prefill-chunk-defaults` の `docs/perf/prefill-chunk-reference.md`、revision `5376403`、PR #18。 |
| 9 | CPU submit / 同期・graph、B0 | 全体 event と各 dispatch event の差を host 待ちと同一視できないため、上限は未特定。graph で削れるのは露出した launch 待ちで、GEMM・recurrence の GPU 演算時間ではない。 | 中〜大。現行 `executor.hip` の graph 条件は `ExecutionRole::Decode` のため prefill は対象外。prefill の新たな capture 設計、chunk ごとのポインタと KV / state、初回 capture、shape 変化と fallback を検証する。 |
| 10 | 複数 request の prefill batch、B1 | 複数要求で共通重みを再利用し throughput 改善の可能性。単一 request の 2K latency の上限からは予測不可。scheduler の混合 batch、待ち時間、padding、GDN / KV の独立性を測る。 | 大。`ScheduledBatch` 契約・メモリ上限・公平性が依存。単独 TTFT が増える可能性。 |
| 11 | INT8 から FP8 WMMA / 異なる weight format、A | 現行 GEMM 529.7 ms が対象だが、AMD 公称は INT8 383 TOPS、FP8 383 TFLOPS と同じで、dtype を替えるだけの理論速度向上はない。実効性能や圧縮展開が改善する可能性は未測定。 | 大。EXL3 の trellis 値・scales と量子化誤差を再定義、異なる checkpoint なら重み品質を分離する必要。 |

## 上位 3 案の具体策

1. **DFlash2 context**: `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp` の `dflash2_spec_prefill` と `src/shelf/models/qwen35/dflash2/executor.hip` の `dflash2_append_target_taps`、`dflash2_project_target_features`、`dflash2_context_append` を対象に、まず concat / FC / norm と 5 層の K/V 投影・RoPE・ring 書込みを event で別々に計り、BF16 GEMM の形状と中間 traffic を調べる。2K の append 310.7 ms は合計値で、個々の原因は未分離。現行の `max_rows=8` により 256 分割・FC と K/V 計 2816 GEMM 呼び出しになる（静的な loop の計数）。prefill 専用に例えば 256 行をまとめれば 8 分割・88 GEMM 呼び出しになる計算だが、行数変更で selector と FP32 累積順・BF16 丸めが変わる可能性があり B1 とする。`select_bf16_gemm` の shape allowlist と BF16 kernel が 256 行の FC / K / V を受け入れるか未検証で、必要なら selector と kernel を拡張する。decode 用 `max_rows` をそのまま 2048 に上げると巨大な `[rows,vocab]` の logits / head accum まで拡大するため、context 専用の上限・scratch を分離する。各 F32 buffer は 2048×248320×4 B = 約 1.89 GiB で、両方なら約 3.79 GiB（その他の拡大分は別）。5 tap の concat、FC 出力、正規化後 feature の BF16 3 buffer だけなら 256 行で計 17.5 MiB、2048 行で 140 MiB（alignment や他の buffer を除く）。KV 投影用 scratch は現状すでに `max_context_rows=2048` で確保される。実際の arena・同時 request の追加費は設計後に確認する。FC や K/V の GEMM を batch 化・tile 調整し、必要な BF16 丸め位置を保つ経路を先に試す。長文では sliding window 2048 に残らない位置の処理省略も候補だが、prefix cache の中間境界保存・ring `next_position`・複数 chunk と request・復元を保てる条件を先に証明する。state と KV ring の byte 一致、最初と後続の `GENERATED_IDS`・受理数、長文と prefix hit/miss を検証。演算順変更時は B1 の品質 gate、量子化を変える場合は A を別評価する。
2. **融合**: `src/shelf/inference/execution/optimized_dispatch.hip` の `try_launch_linear_exl3_int8` と `exl3_dense_gemm.hip` の `exl3_dense_gemm_int8` を起点に、`src/shelf/kernels/optimized/linear/exl3_int8.hip` の trellis decode を tile 内で INT8 operand に展開する opt-in 経路を設ける。Saber のような LDS slab 再利用・double buffer は設計の参考であり、MXFP4 の block scale と EXL3 の trellis は異なる。trellis の依存と chunk 境界・tile の分割、復号重複、LDS/VGPR を調べ、最初に形状別 microbenchmark で現行の展開＋Lt 全体と A/B する。整数積・flag 行・出力の bit 一致から始め、split や還元順を変更したら B1 の品質 gate を追加する。400 行列の N/K と小行数の fallback、冷温 cache と E2E を検証する。
3. **attention**: `src/shelf/inference/execution/paged_attention_dispatch.hip` の `launch_paged_attention_physical`、`src/shelf/kernels/optimized/attention/` の `paged_prefill.hip` の既存 BF16 WMMA と selector へ、query ごとの causal 範囲を保ちつつ複数 query 間や GQA head 間の KV 再利用・page loop を改善する variant を追加する。現行は 32 query 行、8 wave、16 token/page で、PV は高位と残差の 2 回の BF16 WMMA を使う。2 回を 1 回に減らすなら近似変更なので A として個別評価する。現行は 16 層・各 chunk の event で集計済み。2048 / 8192 / 32768 の各位置、非整列長、page 境界、GQA、KV 保存と複数 request で同じ出力・state を比較し、WMMA の変換・softmax の誤差は B1 gate で評価する。全 tile で上下限マスクと last chunk を検査する。
4. **準備・出力**: `src/shelf/kernels/optimized/linear/exl3_int8.hip` の `launch_exl3_int8_rotate_quantize`、`launch_exl3_int8_rowscale`、`launch_exl3_int8_rotate_out_i32` と `try_launch_linear_exl3_int8` を対象に、rowscale と入力準備の中間出力、出力変換と後続の要素演算の read/write を計数する。flag 行を含めて演算順と BF16 丸め点を維持できる境界から opt-in で融合し、各中間 buffer・最終 logits・state を bit 比較する。中間丸めを省いた variant は B1 として別 gate にする。直列の stage event と E2E の両方を確認し、行列ごとに異なる回転を無条件に共有しない。

## 当面採らない案・判断保留

- **全重みの INT8 常駐**: 展開先だけで 22.66 GiB、圧縮版・embedding・lm_head・KV・作業領域も必要。全常駐を前提にした設計は 32 GB では成立しない。部分 cache は別途可能性を測る。
- **Saber の MXFP4 kernel をそのまま移植**: checkpoint・形式・尺度・計算が違い、EXL3 の trellis を置き換えられない。LDS の発想を仕様として独立に評価する。第三者ソースは読み取りのみ。
- **chunk 4096 の即時既定化**: 速度が良くても既存の大きな logits 差の原因が未解明。これは性能比較の否定ではなく、品質 gate が未完了という判断。
- **FP8 が INT8 より公称 2 倍速いという前提、あるいは rocprofv3 の elapsed を通常速度とする案**: 前者は R9700 の公称に反し、後者は launch 計測負荷が大きい。どちらも採用根拠にしない。
- **340 ms を server の host overhead と決め打ち**: 既存 TTFT と bench は入力・revision・投機処理・計測境界が異なる。同一入力の server 側時刻と GPU 区間を揃えるまで要因別の推計はしない。
