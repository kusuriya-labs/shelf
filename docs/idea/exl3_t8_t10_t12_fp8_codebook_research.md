# EXL3 T12 × Finegrained FP8 Codebook 研究・実装計画（不採用）

- 作成日：2026-11-05
- 改稿日：2026-11-14
- 試行日：2026-11-15
- 対象：AMD Radeon AI PRO R9700（gfx1201）、EXL3 系 Trellis 量子化、HIP
- ステータス：**試行済み・不採用（2026-11-15）**
- 目的：T12 Trellis を復号して FP8 E4M3 の重みへ展開し、128×128 ブロック単位の FP32 scale で GEMM する構成を実装・評価する

## 0. 試行結果

T12 Trellis を復号して FP8 E4M3 へ展開し、128×128 ブロック単位の FP32 scale で GEMM する構成を試したが、採用しなかった。

本文の計画は試行時の記録であり、現在の実装方針ではない。以降の実装は行わない。
T12 形式、Finegrained FP8、128×128 block scale、exllamav3 fork のいずれも、この試行の結果として採用しない。

## 1. 試行時の方針

以下は試行時に固定した方針である。

- Trellis は **T12**（12-bit state、4,096 状態）を使う。
- 復号後の重みは **FP8 E4M3** とする。
- 重みの scale は **128×128 ブロック（16,384 重み）ごとに FP32** を1つ持つ。
- 活性側は **K=128 単位**で量子化する。
- Hadamard 回転は、E2E でコストが無視できる場合に適用する。固定で必須とはしない。
- T8 / T10 は今回の対象外とし、T12 を比較・実装の中心に置く。

実装基盤は次の方針とする。

- **exllamav3 を fork し、T12 符号化と Finegrained FP8 を追加する**。
- dsx は T16 形式の仕様、独立参照実装、exllamav3 へのパッチ手順の参考として使う。
- vendor/quark は本体実装には使わない。FP8 ベースラインの生成や品質比較の補助に限定する。
- gfx1201 における hipBLASLt FP8 GEMM の成立確認は、本計画の前提条件にしない。GEMM バックエンドは復号経路と block scale の設計後に選ぶ。

## 2. 目標形式

### 2.1 重み表現

概念的には次の形にする。

\[
\widehat{W}_{k,n} = s_{kb,nb} \cdot C_{8}[z_{k,n}]
\]

- `z`：T12 Trellis の状態。12 bit、4,096 状態。
- `C8`：FP8 E4M3 の Codebook。1状態につき1バイト。
- `s`：128×128 ブロック単位の FP32 scale。`kb = k / 128`、`nb = n / 128`。

T12 の状態数は 4,096 だが、**FP8 E4M3 の符号は 256 通り、有限値は 254 種類**である。したがって 4,096 状態の Codebook は、多くの状態が同じ FP8 値へ写る多対一の写像になる。T12 は値の種類を増やすのではなく、Trellis の遷移履歴・制約の自由度を増やすために使う。

Codebook の LUT 容量は 1状態1バイトなら 4 KiB である。実際に現れる値は 254 種類でも、状態から値への対応表は 4,096 entry 必要になる。

### 2.2 ブロック scale

- 1ブロックは **N=128、K=128**、重み 16,384 個。
- scale は FP32 を1個。
- scale のオーバーヘッドは `32 / 16384 = 0.00195 bpw`。
- scale は重みの量子化前に決め、Trellis 探索では固定値として扱う。

現行 EXL3 の `suh` / `svh`（入力側・出力側の per-channel scale）は、128×128 ブロック scale と役割が重複する。今回の形式では **ブロック scale を重み scale の正本**とし、`suh` / `svh` は廃止するか、Hadamard の符号・回転用に限定する。どちらにするかは Phase 0 で確定する。

### 2.3 活性側

- 活性は K=128 ブロックごとに量子化する。
- 重みの 128×128 ブロックと K 境界が一致するため、K ブロックごとの部分和に重み scale を掛けて加算できる。
- Hadamard を適用する場合、H128 のブロック境界も 128 に揃う。

### 2.4 artifact

checkpoint には最低限、次を持たせる。

- `trellis`：T12 のビット列。
- `block_scale`：`[n/128][k/128]` の FP32。
- `marker`：T12 + FP8 Codebook を識別する専用値。
- `format_version`：T12 形式の版。
- 必要なら `codebook` の識別子。

既存 EXL3 の `mul1` marker は流用しない。`mul1` を付けると、既存の exllamav3 系 runtime が T12 を 16-bit state として復号し、**エラーを出さずに誤った重みを生成する**危険がある。T12 は必ず別 marker とする。

## 3. T12 の定義

本稿では T12 を **12-bit state（4,096 状態）** と定義する。T12 は bits per weight ではない。実際の bpw は Trellis の 1ステップ当たりのビット数 `K` で決まる。

現行 EXL3 は 16-bit state（65,536 状態）を前提にしている。

```text
edges = 65536 >> K
```

T12 ではこれを `4096 >> K` に変更する。必要な変更は以下。

- Viterbi の状態数と履歴バッファ。
- 状態 mask と出力 index の幅。
- 12-bit 窓を使う ring bitstream の packing / unpacking。
- Codebook の 4,096 entry 化。
- 復号カーネルと CPU 参照実装。

T12 にすると状態数が 1/16 になるため、Viterbi のスクラッチと計算量は減る。一方で、遷移履歴の表現力は T16 より下がる。品質は必ず実測で確認する。

## 4. オフライン量子化

### 4.1 基本パイプライン

```text
高精度の元重み
    ↓
Hadamard / 正則化（採用する場合）
    ↓
128×128 ブロック scale の決定
    ↓
FP8 E4M3 Codebook を使った T12 Viterbi 探索
    ↓
trellis + block_scale + marker
```

### 4.2 ブロック scale の決め方

各 128×128 ブロックで、まず scale を決める。第一候補は E4M3 の最大値 448 を使う。

```text
s = amax(block) / 448
```

その後、`W / s` を E4M3 の Codebook へ写す。scale を探索する場合は、ブロックごとに複数候補を試し、Viterbi 後の再構成誤差が最小になる値を選ぶ。ただし探索コストが大きいため、最初は amax 固定で品質を見る。

### 4.3 Codebook

- 初期実装は **全層共通の 4,096 entry Codebook** とする。
- Codebook の値は FP8 E4M3 の有限値 254 種類から選ぶ。
- 層ごと・ブロックごとに Codebook を変えると artifact と LUT が増えるため、初期段階では採用しない。
- 共通 Codebook で品質が不足する場合だけ、層グループ単位の複数 Codebook を検討する。

### 4.4 Viterbi 探索

- 状態数：4,096。
- 1ステップの入力ビット数：`K`。
- 誤差評価：復号値 `C8[state] * s` と元重みの二乗誤差。
- Hadamard を適用する場合、探索対象は回転後の重み `H W H` とする。
- ブロック scale は探索中に固定し、状態遷移のコストへ反映する。

### 4.5 実装基盤

#### exllamav3 fork

T12 符号化の本体は exllamav3 の fork で実装する。

- Trellis の Viterbi、LDLQ、Hadamard、packing が既にある。
- RDNA / gfx1201 向けの HIP カーネルがある。
- `tests/testlib/trellis.py` に独立した NumPy 参照実装がある。
- MIT ライセンスである。

変更対象は以下。

- `quantize_tiles_kernel.cuh` などの状態数・edge 数。
- `pack.cu` の 12-bit 窓 packing。
- `codebook.cuh` の FP8 Codebook。
- `quantize.py` の block scale と artifact 出力。
- CPU 参照実装とテスト。

#### dsx

dsx は T16 の完成品であり、T12 でも Finegrained FP8 でもない。そのまま使うのではなく、次の参考として使う。

- `docs/t16-format.md` の byte-exact な形式記述。
- `quant/t16/t16_ref.py` の独立 decoder。
- `quant/t16/make_mul16_exllamav3.sh` の exllamav3 パッチ手順。
- 専用 marker と private type id で既存形式と分離する設計。
- `from_exl3` のタイル並べ替えと byte swap。

dsx の AMD runner は MI325X（gfx942）向けであり、R9700（gfx1201）用ではない。dsx の runtime は流用しない。

#### vendor/quark

vendor/quark は本体実装には使わない。

- 2D per-block の量子化は `PerBlock2DMinMaxObserver` が placeholder で、calibration が未実装。
- T12 Trellis の符号化・復号がない。
- EXL3 の `trellis` / `suh` / `svh` / `mul1` を出力できない。
- 128×128 の FP32 block scale を生成する経路がない。

Quark は FP8 ベースラインの生成、量子化設定の参照、品質比較の補助に限定する。

## 5. 実行時

### 5.1 復号方式

比較する方式は2つ。

| 方式 | 内容 | 長所 | 短所 |
|---|---|---|---|
| D0 | load 時または層単位で T12 を FP8 dense へ展開 | GEMM が単純 | FP8 dense のメモリ帯域と展開コスト |
| D1 | GEMM カーネル内で T12 を復号 | 展開済み重みを持たない | 復号命令と LUT 参照が GEMM に乗る |

最初は D0 で品質と GEMM の上限を測り、その後に D1 を検討する。

### 5.2 block scale 付き GEMM

128×128 ブロック scale を持つため、GEMM は K ブロックごとの部分和に scale を掛けて加算する。

```text
C[n] = Σ_kb s[kb][nb] * (A[kb] @ W[kb][nb])
```

- A 側も K=128 で量子化するため、K ブロック境界が一致する。
- 各 K ブロックの部分和を FP32 で保持し、scale を掛けてから加算する。
- 単純な1回の GEMM に1つの scale を掛ける構成にはしない。

### 5.3 Hadamard

Hadamard は次の条件で適用を判断する。

- H128 のブロック境界と 128×128 scale 境界が一致する。
- A 側の H128 を K=128 の活性量子化カーネルへ融合できる。
- 出力側の H128 を epilogue へ融合できる。
- 回転を含めた E2E が、回転なしより速い。

現行 EXL3 では回転に層あたり約 0.044 ms かかっている。FP8 dense に移行しても無料にはならないため、**「タダ同然」と仮定せず必ず測る**。回転を入れる場合は、重み側の回転をオフラインで `H W H` に吸収し、実行時は A 側と出力側だけに限定する。

## 6. 現行ベースライン

参考値として、現行の EXL3 mul1 → FP16 dense → hipBLASLt FP16 GEMM の内訳を残す。

| 項目 | 参考値 |
|---|---:|
| PP2048 | 約1,390〜1,400 tok/s |
| Rotate | 0.044 ms |
| Recon | 0.140 ms |
| GEMM | 0.744 ms |
| Epilogue | 0.060 ms |
| 合計 | 0.988 ms |

これらはユーザー報告の参考値であり、今回の T12 形式の実測値ではない。

## 7. 試行時の検証段階

| Stage | 内容 | 主目的 |
|---|---|---|
| P0 | T12 形式・Codebook・block scale・marker・参照 decoder の確定 | 形式の固定 |
| P1 | exllamav3 fork で T12 + FP8 Codebook + block scale を生成 | 品質の確認 |
| P2 | T12 → FP8 dense 展開と block scale 付き GEMM | 復号・GEMM の成立確認 |
| P3 | D0 / D1、Hadamard あり / なしの比較 | 実行方式の選択 |
| P4 | PP2048 全体の tok/s | E2E の判定 |

gfx1201 の hipBLASLt FP8 GEMM 成立確認は P0 の前提にしない。P2 で必要になった時点で、利用可能な GEMM バックエンドを評価する。

## 8. 固定品質ゲート

既存のプロジェクトゲートを変更せず適用する。

| 指標 | 合格条件 |
|---|---:|
| PPL回帰 | +1.0%以下 |
| Mean KL | 0.001以下 |
| Top-1 agreement | 98%以上 |
| Top-5 overlap | 98.5%以上 |

- BetterBench v1、固定 corpus・固定ハッシュ、context 128、BF16 KV、評価 position を揃える。
- 短い 32 position と長い 1024 position の両方を記録する。
- PPL が負方向へ変化しても、KL / Top-1 / Top-5 に失敗した候補は不合格とする。
- 評価を **FP8 Codebook + FP16 GEMM** と **FP8 Codebook + FP8 GEMM** に分け、重みの FP8 化と活性の FP8 化を切り分ける。
- Codebook、block scale、marker、format metadata を含む実効 bpw を算出する。

## 9. リスクと確認事項

1. **T12 の定義**：12-bit state とするか、別の意味にするか。本稿は 12-bit state を前提とする。
2. **FP8 の値域**：T12 でも Codebook の値は E4M3 の有限値 254 種類に限られる。T12 の状態数は値の表現力を増やさない。
3. **ブロック scale の粒度**：128×128 で品質が足りるか。層・列・グループ単位へ細分化する必要があるか。
4. **Codebook の共有範囲**：全層共通で足りるか。層グループごとに分ける必要があるか。
5. **Viterbi の品質**：T16 から T12 へ状態数を減らすことで、遷移履歴の表現力がどこまで落ちるか。
6. **Hadamard のコスト**：A 側 K128 量子化と epilogue への融合で、本当に無視できるか。
7. **GEMM の scale 適用**：K ブロックごとの部分和と scale 適用が、FP8 GEMM の性能を損なわないか。
8. **marker の分離**：既存の `mul1` と混同しない専用 marker が必要。
9. **実装基盤の分離**：`../references/exllamav3` を直接改造せず、fork と参照実装を分ける。

## 10. 試行時の Go / No-Go

### Phase 0：形式の確定

- [ ] T12 の状態幅、K、Codebook、block scale の配置を確定
- [ ] marker と format version を確定
- [ ] NumPy の参照 decoder を作る
- [ ] T12 の packing / unpacking が bit-exact であることを確認

**No-Go：** 参照 decoder と符号化器が一致しない場合、または block scale の適用位置が確定しない場合は先へ進まない。

### Phase 1：量子化品質

- [ ] T12 + FP8 Codebook + 128×128 block scale を生成
- [ ] FP8 Codebook + FP16 GEMM で品質を評価
- [ ] 現行 EXL3 mul1 / FP16 と比較
- [ ] 実効 bpw を算出
- [ ] Codebook の共有範囲と scale 粒度を比較

**No-Go：** 固定品質ゲートを満たさない場合は、T12 の状態幅、Codebook、scale 粒度を見直す。

### Phase 2：復号・GEMM

- [ ] T12 → FP8 dense 展開を実装
- [ ] block scale 付き GEMM を実装
- [ ] A 側 K128 量子化を実装
- [ ] Hadamard あり / なしを比較
- [ ] D0 / D1 を比較

**No-Go：** block scale 付き GEMM が成立しない、または D0 の時点で現行方式に劣る場合は、形式または実行方式を再検討する。

### Phase 3：E2E

- [ ] PP2048 全体の tok/s を測定
- [ ] 品質ゲートを再確認
- [ ] 現行 FP16 方式との比較

**Go：** 固定品質ゲートを満たし、T12 復号・FP8 化・block scale 適用を含む E2E で現行方式を上回ること。

## 11. 試行時の暫定結論

- 目標形式は **T12 Trellis → FP8 E4M3 dense、128×128 ブロックごとの FP32 scale** とする。
- T12 は 12-bit state であり、bpw ではない。実際の bpw は `K` で決まる。
- T12 の 4,096 状態は FP8 の値の種類を増やさない。Codebook の値は E4M3 の有限値 254 種類に限られる。
- 実装の本命は **exllamav3 の fork** である。
- dsx は T16 形式の仕様、独立 decoder、exllamav3 パッチ手順の参考として使う。
- vendor/quark は本体実装に使わず、FP8 ベースラインと品質比較の補助に限定する。
- Hadamard は「タダ同然」と仮定せず、A 側 K128 量子化と epilogue への融合を含めて測る。
- 品質ゲートを最優先し、合格候補についてのみ GEMM 融合と性能最適化を進める。

---

*本稿は試行時の設計方針と実装計画であり、現在の実装方針ではない。T12 の品質・128×128 block scale の妥当性・gfx1201 の FP8 GEMM 性能を実測で保証するものではない。試行の結果、本方式は不採用とする。*
