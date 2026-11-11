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

## chunk verify の限定実験

逐次 verify は fp16 丸めに加え、分岐時の親 state 読み戻し、行ごとの recurrent
history 書き込み、compact log 生成を担う。chunk 内の三角解法で得られる各行の
`delta` と元の `key`、各行の減衰係数を既存形式で記録すれば、compact commit で
受理 prefix を再生できる。ただし commit は行ごとに fp16 に丸め、chunk 内部では
丸めないため両者は完全一致しない。WMMA 自体にも lossy と 4 回 WMMA の差がある。

実験版は `SHELF_GDN_VERIFY_CHUNKED=1` かつ deferred state store、完全な compact
log、recurrent history なし、tree parent なしの場合だけ使う。既定と、それ以外の
経路は serial のままである。現在の DFlash2 batched verify は compact replay を
使うが同時に recurrent history を要求するため、この条件では実験版を選ばない。
単一 request の chain も deferred store ではない。既存の呼び出し契約を暗黙に
破らずに両経路を有効化するには、利用者が recurrent history を不要と明示する
変更や、chunk 版が要求された history を生成する方式が必要である。tree は親の
history を必要とするため別途設計が必要である。

基準と同じゼロ入力・hipEvent、3 warmup、20 回平均。以下は compact log あり、
最終 state 書き込みなしの serial と chunk、および通常 decode。µs/層。

| request × 行 | decode lossy | serial lossy | chunk lossy | 速度比 serial/chunk |
|---|---:|---:|---:|---:|
| 1×8 | 19.074 | 105.055 | 19.438 | 5.40 |
| 1×10 | 19.214 | 128.705 | 20.186 | 6.38 |
| 1×16 | 19.954 | 177.171 | 19.762 | 8.97 |
| 4×8 | 57.587 | 347.353 | 57.515 | 6.04 |
| 8×8 | 110.767 | 695.556 | 108.862 | 6.39 |
| 16×4 | 207.377 | 705.526 | 192.679 | 3.66 |

同じ条件の 4 回 WMMA（exact）の実測は次のとおり。lossy との各行は別実行で、
クロック等の差を含む。

| request × 行 | decode exact | serial exact | chunk exact |
|---|---:|---:|---:|
| 1×8 | 21.912 | 124.338 | 22.584 |
| 1×10 | 21.650 | 154.862 | 22.846 |
| 1×16 | 22.178 | 215.019 | 22.136 |
| 4×8 | 62.145 | 421.951 | 65.677 |
| 8×8 | 115.719 | 817.661 | 120.270 |
| 16×4 | 219.085 | 809.258 | 215.471 |

数値比較は 1 request × 16 行、27B geometry、固定 seed の合成非ゼロ入力。
q と k は一様乱数に 0.1 を掛けた。serial と chunk は同じ入力と fp16 初期 state
を使い、deferred verify 後に 16 行の compact log を replay する。相対最大は
最大絶対誤差を参照出力の絶対値の最大で割った値、相対平均は絶対誤差の平均を
参照の絶対値の平均で割った値。state は replay 後の serial との差である。

| WMMA | 出力 最大絶対誤差 | 出力 平均絶対誤差 | 出力 相対最大 | 出力 相対平均 | state 最大絶対誤差 | state 相対最大 |
|---|---:|---:|---:|---:|---:|---:|
| lossy | 3.9302e-5 | 1.9860e-6 | 1.4269e-4 | 1.5604e-4 | 6.1035e-5 | 2.4414e-4 |
| exact | 3.1047e-5 | 1.6704e-6 | 1.1271e-4 | 1.3125e-4 | 6.1035e-5 | 2.4414e-4 |

exact は出力誤差が小さいが、この入力では lossy も小さく、実測時間は lossy が
短い。compact commit は元から lossy WMMA であり、exact verify の選択だけでは
replay まで exact にはならない。各行の bit 一致は保証されない。履歴が要求された
ときの serial への切替と deferred verify による live state 非更新はテストで確認した。
この合成入力だけで実モデルの logit 誤差を保証することはできない。

関連 ctest は recurrence、exact history、spec history、compact commit、selector の
5 件が通過した。server core は 18 pass、1 skip（`test_plamo3_template.py`）、
0 fail であり、skip を含むため全件通過ではない。全回帰は実行していない。

実モデルの shell/code/reasoning 各 256 token の first-mismatch、logit 差、受理長、
round 時間、decode tok/s と、同時 4/8/16/32 の速度・出力一致率は未測定である。
現行 chain/batch のどちらでも実験版が選ばれないため、そのまま測っても新 variant
の比較にならない。既定を変更する判断は、呼び出し側が history の要否を明示できる
ようにした後、実モデルの数値ゲートと速度を測定してから行う必要がある。

## 実モデル batch 経路の再確認

前節の「batch が recurrent history も要求する」という記述は誤りだった。
`create_dflash2_spec_batch_context` は `create_gdn_spec_history(..., false)`
で conv history のみを確保し、`build_batch_gdn_options` は全 compact log と
`defer_state_store` を指定している。verify 後は受理 prefix の compact log を
live state に replay し、conv は必要な行の history から復元する。従って recurrent
history の要求元は batch にはなく、verify dispatch の tree parent は通常の chain
では null である。条件を満たす batch の WMMA verify は元から環境変数を有効にすれば
chunk を選択する。先の接続不能という推測を実モデルの測定前に訂正する。

空の recurrent tensor に対する `data()` の呼び出しを避け、conv-only history view が
recurrent pointer と stride を明示的に null / 0 にするようにした。conv history と
capture 行数は維持し、この契約の単体テストを加えた。通常の batch では元から空の
pointer であり、この修正を chunk 有効化による性能改善と混同しない。

単発 `dflash2_spec_step` は部分受理時に snapshot へ戻して compact log を replay
するが、全受理時は verify が更新した live state をそのまま採用する。deferred 化には
全受理でも replay する変更が必要であり、今回は共有経路を広げず、max-concurrent 1
の batched serve を単一 request の実測に用いる。環境変数を指定しなければ serial の
ままである。

### batch の実測

`/tmp/agentx/conc3.py` の benchmark prompt、FP8 KV、greedy、各 request 最大
160 token、K=7、`--max-num-batched-tokens 2048`、prefix cache 無効。全員が
生成中の共通区間に出た token 数 / 区間長であり、全処理時間で割った速度ではない。
各条件 2 回、単位 tok/s。短い共通区間（約 2.2〜3.1 秒）と request ごとの生成終了、
生成列の変化による影響を含む。

| 同時数 | serial 2回（平均） | chunk 2回（平均） | 平均比 |
| ---: | --- | --- | ---: |
| 4 | 194.15, 178.64（186.40） | 212.18, 212.93（212.56） | +14.0% |
| 8 | 220.71, 246.90（233.81） | 295.45, 295.41（295.43） | +26.4% |
| 16 | 237.67, 238.23（237.95） | 274.75, 274.59（274.67） | +15.4% |
| 32 | 291.19, 297.10（294.15） | 333.15, 332.23（332.69） | +13.1% |

過去の非投機値は別の測定回であり、同時実行の比較ではない。提示された値との比較は
以下のとおり。差は各値 / 非投機 - 1 で、改善幅は serial から chunk への差の変化。

| 同時数 | 過去の非投機 | 今回 serial（差） | 今回 chunk（差） | 差の改善 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 286 | 237.95（-16.8%） | 274.67（-4.0%） | +12.8 pp |
| 32 | 420 | 294.15（-30.0%） | 332.69（-20.8%） | +9.2 pp |

実モデルの verify 中、8 request × 8 行の最初の 64 行を別の診断ビルドで捕捉した。
semantic timing の hipEvent で GDN recurrence 48 層を合算すると、serial 34.282 ms、
chunk 5.804 ms（5.91 倍、各 1 回）だった。各計時区間の event 投入と同期を含む
診断で、上記 throughput 測定とは別である。両方とも `rows=64 requests=8` を確認した。

### 実モデルの数値ゲートと単一 request の速度

R9700、Qwen3.8 27B EXL3 4bpw mul24、DFlash2 K=7、BF16 KV、greedy、seed 0、
最大生成 256 token。`draft_forest.md` の shell / code / reasoning と同じ prompt
を使った。非投機は `shelf-compute --dump-logits`、serial と chunk は `--serve-stdio
--max-concurrent-requests 1` の batched verify を用いた。各方式の生成 token を
非投機と比較し、最初の不一致位置の非投機 logits の top-1 と top-2 の差を求めた。
位置は **0 始まり**、一致は 256 token 全部が一致したことを表す。
serial と chunk は各 3 回とも方式内の生成列が一致した。

| prompt | serial 対非投機の最初の不一致 / 差 | chunk 対非投機の最初の不一致 / 差 | serial と chunk の先頭一致長（割合） |
| --- | --- | --- | --- |
| shell | 一致 / — | 37 / 0.051245 | 37 / 256（14.5%） |
| code | 一致 / — | 118 / 0.020216 | 118 / 256（46.1%） |
| reasoning | 一致 / — | 53 / 0.030224 | 53 / 256（20.7%） |

最初の不一致位置では 3 件とも指定された logit 差 0.1 以下だった。これは同じ
prefix における最初の分岐だけの検査であり、以後の異なる生成文脈での品質や
全入力に対する一致を保証しない。3 prompt とも chunk は bit 一致しない。

単一 request の各方式 3 回。受理長は draft token / round、round 時間は
`DFLASH2_ROUND_MS / DFLASH2_ROUNDS` の各回平均、decode tok/s は serve の最初と
最後の token 通知の時刻差で 255 token を割った各回の平均。通知や host 側処理を
含むので単発 CLI の過去表とは直接同一視しない。括弧内は各回の tok/s。

| prompt | 方式 | 受理/round | ms/round | decode tok/s 平均（各回） |
| --- | --- | ---: | ---: | --- |
| shell | serial | 3.048 | 58.790 | 68.74（68.51, 68.90, 68.80） |
| shell | chunk | 3.180 | 55.956 | 75.00（75.36, 67.91, 81.74） |
| code | serial | 3.636 | 57.686 | 79.29（79.29, 79.24, 79.33） |
| code | chunk | 3.113 | 54.836 | 74.06（74.58, 72.98, 74.61） |
| reasoning | serial | 3.322 | 58.760 | 73.43（73.51, 73.37, 73.40） |
| reasoning | chunk | 3.113 | 55.457 | 74.03（73.99, 74.05, 74.06） |

chunk は全 prompt で round 時間が短いが、出力の分岐後に受理長が変化する。code の
平均 decode 速度は約 6.6% 低下し、全 prompt の速度改善は得られなかった。shell の
chunk 速度は 67.91〜81.74 tok/s とばらつきが大きい。

### 検証

ビルド成功。`test_qwen35_gdn_recurrence_selector`、`test_gdn_recurrence`、
`test_gdn_recurrence_exact_history`、`test_gdn_spec_history`、
`test_gdn_compact_log_commit`、DFlash2 関連など計 10 ctest は全 pass。
server `--group core` は 18 pass、1 skip（`test_plamo3_template.py`）、0 fail。
全回帰 `--all` は実行していない。既定を変えるか、bit 一致を諦めるかの判断は
この限定的な gate と速度結果を踏まえて別途必要である。

### 追加調査: 受理長の同値と分布比較の未完了事項

上記の既存実行ログを再集計した。code と reasoning の chunk の 3.113 は
丸めだけが同じなのではなく、どちらも 193 accepted / 62 rounds である。ただし
位置 bucket 別の内訳は異なる（code: 0–256 で 160/52、256–512 で 33/10、
reasoning: 164/52、29/10。位置は生成 token の offset ではなく実行時の絶対位置）。
したがって同じ総数から各 round の受理長が同じとは言えず、同値の理由が偶然か
どうかも、この集計だけでは判断できない。両実行は別の生成文面であり、これを
固定文面の比較に用いることはできない。

| 参考値 | mean KL | 条件 |
| --- | ---: | --- |
| Qwen3.8 27B 4bpw mul24 INT8 対 BF16 | 0.02167 | rocm_exl3、8 ケース・2632 位置、[既存記録](../20261105/exl3_int8_qwen38_27b.md) |
| chunk verify 対非投機 decode | 未測定 | 同一 prefix の logits が必要 |
| batch 1 対 batch 8 の非投機 decode | 未測定 | 同一 prefix の logits が必要 |

上記の量子化値は別の実装・コーパスの参考値であり、今回の chunk の品質を推定
する値ではない。`shelf-compute --dump-logits` は DFlash2 と併用不可、既存
`shelf-bench quality` は固定 corpus を逐次 decode するが speculative verify の
各行を dump しない。従って、既存の生成列とログから KL や固定文面の受理率を
計算することはできない。`SHELF_GDN_RECURRENCE_EXACT=1` は exact WMMA の選択で、
compact replay や chunk と逐次間の fp16 丸め差が消えるという意味ではない。

残作業は、固定 prefix 上で draft と serial / chunk の verify 行 logits を同じ
位置・同じ token 列で採取し、逐次 decode の参照と対応付ける診断を設けること、
同じ入力で非投機 batch 1/8 を採取すること、および十分な corpus で KL の平均・
p99・最大、top-1 一致、位置ごとの最大 logit 差を検証することである。受理長の
変化が文面差か数値差か、chunk のずれが batch による数値ノイズと同程度か、
bit 一致を手放してよいかは未判定である。
