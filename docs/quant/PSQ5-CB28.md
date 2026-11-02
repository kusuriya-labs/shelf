# PSQ5-CB28 — 5bit Codebook量子化仕様

**対象:** PhaseShift / 独自MXFP5の検討用  
**状態:** 会話で再確認したCB28数列とFP8変換規則を記録した設計メモ。性能・精度は未実測。  
**目的:** 5bit固定長のまま、W32単位の8bit scaleを使用し、FP8 E4M3への展開をLUTなしで実現する。

## 1. コードブック

4bit magnitude index に対する **CB28** は次の16値。

```cpp
static constexpr uint8_t CB28[16] = {
     0,  1,  2,  3,
     4,  5,  6,  7,
     8, 10, 12, 14,
    16, 20, 24, 28
};
```

5bit code `q5` は、下位4bitが絶対値index、bit4が符号を表す。

```text
q5 = [sign:1bit][magnitude_index:4bit]

 0..15 : +CB28[q5 & 0x0F]
16..31 : -CB28[q5 & 0x0F]
```

`q5 = 0` と `q5 = 16` はそれぞれ `+0` と `-0` に対応する（数値としては同じゼロ）。**32コード／数値としては31種類**。

## 2. FP8 E4M3へのビット変換

以下は **OCP系 E4M3（指数バイアス7、subnormalあり）** のビット解釈を前提とする。E4M3FNUZなど異なる符号化とは混同しない。

| magnitude index | CB28整数値 | FP8 E4M3 raw（正） |
|---:|---:|---:|
| 0 | 0 | `0x00` |
| 1 | 1 | `0x01` |
| 2 | 2 | `0x02` |
| 3 | 3 | `0x03` |
| 4 | 4 | `0x04` |
| 5 | 5 | `0x05` |
| 6 | 6 | `0x06` |
| 7 | 7 | `0x07` |
| 8 | 8 | `0x08` |
| 9 | 10 | `0x0A` |
| 10 | 12 | `0x0C` |
| 11 | 14 | `0x0E` |
| 12 | 16 | `0x10` |
| 13 | 20 | `0x12` |
| 14 | 24 | `0x14` |
| 15 | 28 | `0x16` |

負の値は、同一magnitudeにFP8の符号ビット `0x80` をORするだけ。

- `q5 = 3` → `0x03`（+3 / 512）
- `q5 = 19 = 16 + 3` → `0x83`（−3 / 512）
- `q5 = 15` → `0x16`（+28 / 512）
- `q5 = 31` → `0x96`（−28 / 512）

### LUT不要の変換式（検証済み）

```cpp
// q5 は 0..31。返り値はFP8 E4M3の生ビット列。
static inline constexpr uint8_t cb28_to_e4m3(uint8_t q5) noexcept {
    const uint8_t m    = q5 & 0x0F;
    const uint8_t sign = (q5 & 0x10) << 3; // bit4 -> FP8 bit7
    const uint8_t mag  = m + ((m >> 3) * (m & 0x07));
    return mag | sign;
}
```

この式は `m=0..15` で以下を生成する。

```text
m    :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
FP8  : 00 01 02 03 04 05 06 07 08 0A 0C 0E 10 12 14 16
```

**注意:** 以前提示した `m >= 12` の場合に `16 + ((m & 3) << 2)` とする式は誤り。ここに記載した `m + ((m >> 3) * (m & 7))` が正しい。正負を含む全32コードについてCB28値との一致を検証した。

## 3. Scale Foldの数学的関係

正負ともに、E4M3ビット列を浮動小数点数として解釈すると、

```text
CB28_signed(q5) = E4M3(cb28_to_e4m3(q5)) × 512
```

W32 block の実数scaleを `s` とした場合、

```text
weight = CB28_signed(q5) × s
       = E4M3(cb28_to_e4m3(q5)) × (512 × s)
```

よって、`512 = 2^9` の係数をscale側へ取り込める。**これは数値としての恒等式であり、GPUがネイティブFP8で常に同じ結果・性能を出す保証ではない。**

特にFP8 `0x01..0x07` はsubnormalであるため、使用するGPU演算経路がsubnormalをflushする場合はこのままでは使えない。また、`512*s` がscale表現範囲を外れるケースにも注意。

## 4. 格納形式：W32 + 8bit scale

| 項目 | W32あたり |
|---|---:|
| magnitude（4bit × 32） | 16 byte |
| sign plane（1bit × 32） | 4 byte |
| scale（8bit × 1） | 1 byte |
| **合計** | **21 byte = 5.25 bpw** |

**scaleの8bit符号化形式（E8M0、UE4M3、独自形式など）は、このメモの情報だけでは未確定。** どれを採用しても21byte/W32という格納量は同じだが、精度やFoldの実装は変わる。

### 推奨物理配置：SoA

```text
Q4 magnitude stream : [16B][16B][16B] ...
S1 sign stream      : [ 4B][ 4B][ 4B] ...
S8 scale stream     : [ 1B][ 1B][ 1B] ...
```

21BのAoS strideを避け、各streamを別のアライン済み配列として置く。これによりmagnitude側を128bit単位で連続ロードしやすい。

16 block（512 weights）ごとのデータ量：

```text
magnitude : 256B
sign      :  64B
scale     :  16B
----------------
total     : 336B
```

これは帯域効率の候補であり、実際のGPUでAoS/AoSoA/SoAのどれが最速かはdecodeとGEMMを含む測定で決める。

## 5. 復号の概要

```cpp
// 概念的なW32復号。q4[16]には各byteに2つのmagnitude index、
// sign_maskには32個の符号bitが入る。
for (unsigned i = 0; i < 32; ++i) {
    const uint8_t m = (q4[i >> 1] >> ((i & 1) * 4)) & 0x0F;
    const uint8_t s = (sign_mask >> i) & 1;
    const uint8_t fp8_raw = cb28_to_e4m3(m | (s << 4));
    // 実ウェイト = decode_e4m3(fp8_raw) * (512 * scale)
}
```

FP8 rawを算出する部分はLUT不要で、q4部分とsign planeを別ロードできる。一方、**FP8のネイティブ演算経路の可否・サブノーマル保持・scale処理コスト**は別途評価が必要。

## 6. 検証項目

1. **全32コード検証** — 符号と絶対値がCB28の定義と一致すること。`+0`/`-0` も含める。
2. **scale検証** — 選択した8bit scale符号化で、`512*s` の範囲、丸め、underflow/overflowを確認。
3. **GPU decode検証** — ネイティブFP8経路でsubnormalが保持されるか、および代替のINT8/BF16展開経路の性能を測る。
4. **量子化精度検証** — 同条件のKLD・Top1をBF16、既存PSQ4、MXFP4等と比較。
5. **速度検証** — GEMV/GEMMそれぞれでbitpack方式、SoA/AoSoA、LUT有無を比較する。

## 7. 要点

- **CB28:** `{0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28}`
- **5bit:** 上位1bitは符号、下位4bitはmagnitude index。
- **FP8 raw化:** `m + ((m >> 3) * (m & 7))` と符号bit注入でLUT不要。
- **512倍Fold:** CB28整数値は対応するE4M3値の512倍。
- **W32 + 8bit scale:** **5.25 bpw**。
- **制約:** LUT不要の変換が成立しても、FP8ハードウェアのsubnormal取り扱いとscale範囲は要実測。

**まとめ:** PSQ5-CB28は、固定5bitのコード体系をFP8 E4M3の下位ビット表現と意図的に整合させ、格納効率と展開処理の単純化を狙う設計。精度・速度の優位性は今後のベンチマークで検証する。
