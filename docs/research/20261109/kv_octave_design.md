# Octave 低 bit KV cache 設計

## 目的と範囲

Qwen3.8 27B の full-attention 層だけを対象に、RoPE 適用済みの K と V を
3〜6 bit で保存する。既定の BF16 / FP8 経路と ABI は変更せず、
`--kv-cache-dtype octave:...` を明示した場合だけ有効にする。
初期実装は head dimension 256、rotary dimension 64、KV head 4、TP/PP なし、
MoE なしに限定し、契約外の構成は起動時に拒否する。

## format descriptor

descriptor は `k_mode`、`k_rope_bits`、`k_nope_bits`、`v_bits` を持つ。
compact K は `octave:c<K>v<V>`、block K は `octave:b<R>n<N>v<V>` で表す。
対応 bit は 3、4、5、6 である。既存名との互換用に次の別名を受理する。

- `octave_k3v3_compact` = `octave:c3v3`
- `octave_k3v3` = `octave:b4n3v3`
- `octave_k3v4` = `octave:b4n3v4`

## bit-plane layout

code の下位 bit から 4、2、1 bit plane の順に、必要な plane だけを置く。
3 bit は 2+1、4 bit は 4、5 bit は 4+1、6 bit は 4+2 である。
各 plane は 32 bit word 内に整数個の field を格納し、field を word 境界に
またがらせない。dimension index は plane 内で昇順とし、word の下位 field から詰める。
fp16 scale は code stream の直後に置く。K/V は別 pool だが、同じ token/head の
K と V を合わせた slot は 4 byte の倍数になる。pool の page 先頭は 256 byte aligned とする。

| bit | plane | 256 code bytes | 64 code bytes |
| ---: | --- | ---: | ---: |
| 3 | 2+1 | 96 | 24 |
| 4 | 4 | 128 | 32 |
| 5 | 4+1 | 160 | 40 |
| 6 | 4+2 | 192 | 48 |

compact K は 256 code と fp16 scale 1 個、V も同じ構成である。block K は
64 dimension を 4 block に分け、先頭を rope bit、残りを nope bit で保存し、
各 block に fp16 scale を 1 個置く。

| 形式 | K bytes | V bytes | K+V bytes/token/head |
| --- | ---: | ---: | ---: |
| c3v3 | 98 | 98 | 196 |
| c4v4 | 130 | 130 | 260 |
| c5v5 | 162 | 162 | 324 |
| c6v6 | 194 | 194 | 388 |
| b4n3v3 | 112 | 98 | 212 |
| b4n3v4 | 112 | 130 | 244 |

page 内は `[layer][page][token][head][slot bytes]` の順である。4 KV heads の
token stride は上表の slot の 4 倍なので 4 byte aligned となる。page copy は
K/V pool の該当 layer/page byte range をそのまま複製する。

## 量子化と kernel 構成

起動時に CPU で N(0,1) の Lloyd-Max table を 3〜6 bit 分生成し、device の
定数 table へ転送する。append は token/head ごとに 256 threads を使い、固定符号、
正規直交 WHT、最近傍 code、`ss / dot` の fp16 scale、bit-plane pack の順で処理する。
K compact は H256、K block は独立した H64 を 4 個、V は H256 を使う。

M2 は visible cache を BF16 scratch に逆量子化し、既存 BF16 paged attention を使う。
scratch は layer 間で再利用し、最大 visible token 数で arena 見積もりを行う。
M3 decode/verify は Q を K と同じ回転空間へ変換し、packed K/V と LUT を直接読む
split-KV kernel と reduce kernelを使う。V の重み付き和は回転空間で蓄積し、query ごとに
H256 と固定符号で元空間へ戻す。prefill は M2 経路を維持する。

## 既存コードへの変更点

- `kv_types.h`: `OCTAVE` と descriptor、parse、byte geometry を追加する。
- `paged_kv_pool.*`: Octave K/V byte pool、view、allocation、page copy、footprint を追加する。
- compute CLI/server/runtime: 文字列 descriptor を parse し executor と pool へ渡す。
- model dispatch state: Octave pool view と descriptor を追加する。
- `kv_append.hip` と dispatch/selector: Octave append kernel と規則を追加する。
- paged attention と dispatch/selector: M2 scratch 経路、M3 fused decode/verify 規則を追加する。
- executor memory plan: pool byte geometryと reusable BF16 scratch を見積もる。
- prefix cache: dtype に加えて descriptor の一致を検査し、Octave page bytes を複製する。
- tests/bench: CPU 参照との code/復元/NMSE、M2/M3 parity、CLI と footprint の回帰を追加する。

## 安全側の制約

descriptor は runtime の所有値として伝播させ、enum 値だけから既定形式を推測しない。
head dimension 256 と rotary dimension 64 以外、TP、PP、MoE、HIP graph、DFlash2 以外の
特殊 attention 経路は実装済みと確認できるまで起動時に明示的に拒否する。
BF16/FP8 の selector rule、layout、kernel symbol は変更しない。

## M2 実測

`s2k_0`〜`s2k_2`、context 2048、score 256、prefill chunk 2048 で、実 cache と
fake-quant の全量量子化モードを比較した。いずれも BF16 KV dump を基準とした KL 平均である。

| 形式 | 実 cache | fake-quant `PREFILL=1` | 比 |
| --- | ---: | ---: | ---: |
| c4v4 | 0.005331 | 0.005776 | 0.923 |
| c5v5 | 0.002624 | 0.002640 | 0.994 |
| b4n3v3 | 0.014150 | 0.013798 | 1.026 |

M2 は attention 呼び出しごとに該当 layer の全物理 page を 1-layer BF16 scratch へ
逆量子化する。未使用 page も変換するため遅いが、既存 BF16 attention と数値契約を共有できる。
scratch は K/V 各 1 layer 分を pool と同時に確保し、page 当たりの memory plan に含めた。

## M3 所見

packed-direct decode/verify は 16-way split-KV と専用 reduce で実装した。Q/K の dot と
V の重み付き和は回転空間の fp32 で行い、reduce 後の V 出力だけを H256 と固定符号で
元空間へ戻す。rows=1 c4v4 の M2 reference との差は最大 0.001774、rows=4 b4n3v3 は
0.002820 だった。性能と容量の実測は `docs/perf/kv-octave.md` を正本とする。

current chunk の量子化前 K/V を prefill attention へ渡す A mode は、現在の attention
primitive が append 入力を参照できないため見送った。lowering の value binding を増やす
変更は BF16/FP8 の既存 graph contract に及ぶため、既存挙動を壊さない側を選んだ。
