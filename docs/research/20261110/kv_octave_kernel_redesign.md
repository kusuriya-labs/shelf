# Octave KV kernel 再設計

## 律速

対象は Qwen3.8 27B、KV head 4、query head 24、head dimension 256、wave32 である。
32 Ki context で 1 layer、1 decode token が読む packed K/V は c4v4 が
`32768 * 4 * 260 = 34,078,720 B`（32.5 MiB）、c5v5 が 40.5 MiB、
b4n3v3 が 26.5 MiB である。BF16 は 128 MiB、FP8 は約 65 MiB なので、
Octave は本来 DRAM 帯域上有利である。QK と PV は合わせて
`32768 * 24 * 256 * 2 = 402,653,184 FMA`、約 0.805 GFLOP/layer である。

現行 kernel は 1024 threads を 4 token lane に分け、token 4 個ごとに workgroup
全体の barrier を繰り返す。各 code を scalar shift と fp32 LDS LUT で復号し、6 query
head の dot と online softmax を scalar ALU で行う。32 Ki の実測差から attention の
追加時間は約 1.2 ms/layer で、c4v4 の理想 DRAM 時間約 54 us の 20 倍以上である。
主因は次の複合と考える。

- 32 wave の 1024-thread workgroup が token ごとの LDS barrier で直列化する。
- code ごとの fp32 LUT load、整数 shift、fp32 FMA が matrix core を使わない。
- 1 KV token の復号を共有しても、6 head の QK/PV と softmax が scalar loop のままである。
- 1024-thread workgroup と約 33 KiB の LDS、thread ごとの 18 個前後の状態が occupancy と
  wave scheduler の自由度を落とす。

## Decode / verify tile

既定の新 kernel は 1 workgroup = 256 threads（8 wave）、1 KV head、1 row、1 split とする。
KV tile は 16 token、query tile は 16 row で、実 query 6 本を詰めて残りを zero fill する。
wave 0 が fp16 WMMA で QK、wave 1〜4 がそれぞれ V の 64 次元を fp16 WMMA で処理し、
wave 5〜7 を含む全 wave が cooperative unpack/load を行う。Q は int8 化せず、符号反転と
H256（compact）または 4 個の H64（block）を wave 内 shuffle だけで行って fp16 にする。
fp32 accumulator を維持する。

KV token 16 本について K/V を fp16 LDS tile に展開する。decode は single buffer で、
K/V が各 `16 * 264 * 2 = 8448 B`、Q が `16 * 264 * 2 = 8448 B`、確率と状態を含めて
約 27 KiB である。QK wave は block K の 64 次元ごとに別の accumulator を持ち、各 token
の fp16 scale を部分和へ掛けた後で加算する。compact K は 256 次元の積へ scale を 1 回だけ
掛ける。V scale は `exp(score-max) * v_scale` として確率 fragment に畳み込み、PV 後に
改めて掛けない。

online softmax は query row ごとの `(max, sum)` と PV accumulator を wave/LDS 内で更新する。
barrier は 16-token tile の load 完了、QK 完了、PV 完了だけであり、1024-thread barrier は
使わない。PV accumulator は各 wave 4 個の WMMA fragment（64 output dims）なので、QK の
4 scale accumulator と合わせても概ね 64 VGPR/thread 未満を狙う。

split は既定 16 とする。rows=1 でも grid は `4 KV head * 16 = 64 workgroup` となり 64 CU を
埋める。rows が増えた場合も同じ kernel を使い、rows 1〜32 を device 配列の position と
sequence slot から解決する。通常 visibility、block mask、tree ancestor を同じ item iterator
へ正規化する。まず全 row shape を正しく扱い、その後 verify の同一 request に属する 2〜4
row を 16 query row へ詰める variant を追加する。tree row ごとに visibility が違う場合は
score mask だけを分け、K/V tile は共有する。

reduce は split の `(max, sum, rotated V)` を query head ごとに統合する。最後の H256 と固定
符号の逆変換は reduce kernel 内に残し、追加 dispatch を作らない。旧 scalar kernel は
`SHELF_OCTAVE_DECODE_LEGACY=1` で選択できるようにし、parity と交互 A/B の基準にする。

## Codebook 復号

Lloyd-Max level は bit 幅ごとに最大絶対値を 127 に対応させた signed int8 table と fp16
係数に分ける。3-bit は `v_perm_b32` 1 回、4-bit は 2 回と high-bit select、5/6-bit は
high bits で 8-entry bank を選ぶ 2〜4 段の perm tree とする。table dword は compile-time
定数として VGPR/SGPR に置き、code ごとの global/LDS LUT access を無くす。復号値は
`half(int8_level) * half(level_scale) * half(norm)` とする。int8 近似の parity が許容値を
越える場合は codebook 値を fp16 LDS に置く fallback を bit 幅ごとに残し、品質を優先する。

この unpack と WMMA の構造は vLLM の Apache-2.0 実装
`octave_decode_wmma_rdna3.cu`、`octave_prefill_rdna3.cu`、`octave_format.cuh` を参考にする。
ただし gfx11 の lane pair/DPP operand 配置は使わず、shelf の gfx12
`paged_prefill.hip` と BF16/EXL3 WMMA kernel の `lane & 15`、`lane >> 4`、8-element fragment
配置に合わせる。最初に 16x16x16 の単体テストで A/B/C の lane 対応を固定する。

## 旧回転空間 prefill

既定実装は 1 workgroup = 1 KV head × 10 causal query row とする。M 次元を
`6 GQA head × 10 row = 60`（64 row に zero pad）へ flatten し、8 wave を
4 M tile × 2 dimension half に割り当てる。これにより K/V 16-token tile の unpack と
LDS staging を 6 query head が共有する。workgroup 数は 2048 row で 820、8192 row で
3280 となり、64 CU を埋める。

Q は小さい先行 kernel で固定符号と H256（compact）または 4×H64（block）を適用し、
既存 output buffer を一時領域として使う。main kernel は回転済み Q と codebook 値を
gfx12 fp16 WMMA / fp32 accumulator で積算するため、K の inverse WHT を行わない。
V も回転空間で PV を積算し、後続 kernel が query row ごとに H256 と固定符号を 1 回だけ
適用する。3 kernel は同一 stream 上で順序付けられ、追加 allocation と host 同期は無い。

compact 3〜6 bit は decode と同じ signed-int8 固定 codebook と係数を用いる。lane が担当する
連続 8 dimension は bit-plane word を 1〜3 回だけ読み、8 code へ展開する。K scale は score、
V scale は probability に畳み込む。block K は segment ごとの scale が必要なため exact table
fallback を使う。online softmax は fp32、probability は hi/lo fp16、PV accumulator は fp32
である。LDS は約 32 KiB で、`__launch_bounds__(256, 2)` により 2 workgroup/CU を許す。

prefix と current chunk は区別せず、append 済み packed cache 全体を causal position と
block mask で読むため、全量量子化後という意味論を維持する。この 10-row 経路は
`SHELF_OCTAVE_PREFILL_LEGACY=1` で残す。

## split-KV prefill

既定実装は KV を 2048-token block に分ける。各 block の packed K/V は KV head 4 本を
original space の BF16 scratch へ一度だけ展開し、固定符号と compact H256 または block
4×H64 の逆変換もこの段階で行う。連続 8 dimension の bit-plane word は
`octave_unpack8_direct` で一括展開する。次に既存の 256-thread BF16 WMMA prefill kernel が
その block に対して全 query tile を処理する。これにより unpack 回数は
`query tile 数 × prefix block 数` から `prefix block 数` へ減り、prefix 長に線形となる。

block 間では query head ごとの fp32 `(max, sum)` と未正規化 O を保持する。O は既存 output
buffer に BF16 または F32 で置き、最終可視 block だけで正規化する。`(max, sum)` は
`rows * 24 * 2 * 4` B である。K/V scratch は
`2 * 2048 * 4 * 256 * 2 = 8,388,608 B`、8192 rows の softmax state は 1.5 MiB で、
executor が decode partial 用に既に確保する 32 MiB workspace 内に収まる。新しい arena
allocation は無く、永続 KV token 容量は変わらない。workspace が不足する geometry は
旧回転空間経路へ fallback する。

HIP graph では host から見える最大可視 token 数を 2048 単位の block 数へ丸めて graph
signature に含める。同じ block 数なら row position、block table、mask は device pointer の
まま変化できる。capture 中の allocation と host 同期は無い。prefix cache 復元後の追い
prefill も同じ logical block table を使い、current chunk を特別扱いしないため、全量量子化後の
意味論を維持する。

## 試行結果と不採用案

- 4-row tile は GQA 共有しても workgroup 数が旧経路より増え、2048 c4v4 が 1635 tok/s と
  旧 packed-direct 1721 tok/s を下回った。
- 8-row tile は 1695 tok/s、10-row tile は 1736 tok/s まで改善した。bit-plane word の
  8-code 一括展開、固定 int8 codebook、Q-LDS の除去、fp16 WMMA により最終的に 1820 tok/s。
- 32-token KV tile は Q-LDS を output buffer へ移して収容したが、softmax/PV の 16-token
  half が 2 回必要で barrier 削減にならず、8192 c4v4 は 1590 tok/s と 16-token tile の
  1661 tok/s を下回ったため採用しなかった。
- K/V double buffer は 10-row/60-M tile の score、probability、accumulator と同居すると
  64 KiB LDS とレジスタ圧を越える。loader/compute overlap より 2 block/CU を優先した。
- A の 32-row/M192 tile は 24 wave、約 58 KiB LDS、768 thread とした。8192 c4v4 は
  1529.66 tok/s、32768 は 1008.06 tok/s で、10-row から有意に伸びなかった。1 workgroup/CU、
  barrier、wave ごとの 8 個の Q/O fragment によるレジスタ圧が unpack 償却を相殺した。
- B の KV block を 2048 から 7680 token に広げても、8192 c4v4 は 1720.25 から
  1723.35 tok/s と誤差範囲だった。block 境界の state 往復は主律速ではなく、scratch を
  8 MiB に抑えられる 2048 を採用した。
- C として split-KV scratch に M192 GQA kernel を組み合わせると parity は 0.0000051 だったが、
  8192 c4v4 は 1545.82 tok/s に低下した。大 workgroup の occupancy と LDS barrier が
  復号済み K/V の head 間共有利得を上回るため採用しなかった。
- 採用 B の FP8 比は c4v4 が 100.8% / 100.2% / 96.1%、c5v5 が
  100.4% / 100.0% / 95.9%、b4n3v3 が 100.3% / 100.2% / 96.1% で、
  2048 / 8192 / 32768 の全条件で 95% gate を満たした。32768 の旧比は
  c4v4 1.480 倍、c5v5 1.589 倍、b4n3v3 1.508 倍である。

pool 全体を BF16 化する K/V 各 1-layer scratch（合計 4096 B/token）は不要である。
固定 2048-token block だけを既存 workspace へ展開するため、arena/page footprint は永続
packed bytes のままである。

