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

## Packed-direct prefill

prefill は query 32 row × KV 16 token を基本 tile とする。8 wave のうち loader wave が
packed K/V を次の LDS buffer へ復号し、compute wave が現在 buffer で QK、online softmax、
PV を行う。K/V double buffer は約 33 KiB、Q と score/probability を合わせて 64 KiB 未満に
収める。GQA の 6 query head は同じ K/V tile を共有する。prefix と chunk を区別せず、append
済み packed cache を causal position と block mask で mask するため、全量量子化後という
現在の意味論を維持する。

この経路により pool 全体を BF16 化する K/V 各 1-layer scratch（合計 4096 B/token）と
dequant dispatch を削除する。arena/page footprint は永続 packed bytes と小さい attention
partials だけに更新する。HIP graph では host length を読まず、row position、block table、
tree 配列を device pointer のまま参照し、capture 中に allocation と同期を行わない。

