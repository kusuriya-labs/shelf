# EXL3 T8/T10/T12 × FP8ターゲットCodebook 研究・実装検証案

- 作成日：2026-11-05
- 対象：AMD Radeon AI PRO R9700（gfx1201）、EXL3 mul1系Trellis量子化、HIP / hipBLASLt
- ステータス：**研究仮説／未検証の設計案**
- 目的：量子化率・品質を維持しつつ、オンライン復号コストおよびGEMM実行時間を削減する

## 1. 背景と問題設定

現行実装では、EXL3重みをFP16 dense行列へ展開してからhipBLASLtのFP16 GEMMを実行する方式が、従来の復号融合WMMA方式より大幅に高速だった。

現時点の参考測定値：

| 項目 | 実測値・条件 |
|---|---:|
| PP2048（FP16展開 + hipBLASLt） | 約1,390〜1,400 tok/s |
| 代表的な1層 Rotate | 0.044 ms |
| 同 Recon | 0.140 ms |
| 同 hipBLASLt FP16 GEMM | 0.744 ms |
| 同 Epilogue | 0.060 ms |
| 同 合計 | 0.988 ms |

上記はユーザー報告の参考値であり、実測条件が異なる個別ベンチマークとの直接比較には注意する。

一方、EXL3をFP16展開して単純なper-column INT8重みへ再量子化する方式（A方式）は、固定の品質ゲートに不合格となった。1024 positionでは重みのみのINT8化で mean KL≈0.0139、Top-1≈94.1%であり、特に重み再量子化の誤差が課題である。

そこで、**完成したEXL3重みを後からFP8へ丸めるのではなく、量子化探索そのものをFP8ターゲットCodebookで実施する**構成を検討する。さらにTrellisの状態数をT16からT8/T10/T12へ縮小し、ハッシュ／SAD復号を小型LUT参照に置き換える。

## 2. 提案の基本構造

```text
[オフライン量子化]
高精度の元重み
    ↓
FP8 E4M3ターゲットCodebookの設計
    ↓
T8 / T10 / T12 の状態遷移制約に基づく探索（例：Viterbi）
    ↓
圧縮済みTrellis重み + Codebook + Scale

[推論：案A / 展開方式]
圧縮済みTrellis重み
    ↓  bit/state復号
FP8 LUT参照 + FP8重み行列への展開
    ↓
hipBLASLt FP8 GEMM（対応レイアウト・Scale条件を満たす場合）
    ↓
Scale補正・既存のHadamard/SVH等の処理

[推論：案B / 融合方式]
圧縮済みTrellis重み
    ↓
FP8 LUT参照 → WMMA向けタイル配置 → FP8 WMMA
    ↓
Scale補正・Epilogue
```

案Aは現行の「展開→hipBLASLt」構造を再利用でき、まず性能の上限を見積もりやすい。案Bは展開用VRAMトラフィックを削減できる可能性がある一方、過去に問題となったWMMAタイル構成、レジスタ・LDS供給効率を再度最適化する必要がある。

## 3. T8/T10/T12のLUT容量

**以下は「Tnがn-bitのLUT索引であり、全2^n状態に1バイトのFP8値を1つ保持する」という仮定に基づく。** 実際のEXL3形式でTが表す意味（state幅、遷移制約、Codebookの索引幅）をコードで確認すること。状態数と圧縮率（bpw）は同義ではない。

| 方式 | 仮定する状態数 | FP8 LUT（1 byte/entry） | FP16 LUT（2 bytes/entry） |
|---|---:|---:|---:|
| T8 | 256 | 256 B | 512 B |
| T10 | 1,024 | 1 KiB | 2 KiB |
| T12 | 4,096 | 4 KiB | 8 KiB |
| T16 | 65,536 | 64 KiB | 128 KiB |

補足：
- 「256 B / 1 KiB / 4 KiBがキャッシュライン1本に収まる」というわけではない。キャッシュラインを128 Bと仮定すれば、それぞれ2 / 8 / 32ラインになる。**実機のラインサイズは別途確認が必要**。
- 容量としてキャッシュに乗ることと、Wave内の分散したLUT参照が高速であることは別問題。
- 複数Codebook、複数Scale、追加メタデータがある場合、実際の常駐容量は増える。
- Tを小さくすることで探索空間・遷移制約・品質が変化するため、LUT容量だけで優劣は決められない。

## 4. FP8ターゲットCodebookの設計

第一候補はE4M3を使い、**最終的にFP8で表現できる値を前提にオフライン量子化探索を実施する**方式とする。

概念的には次のような重み表現を検討する：

\[
\widehat{W}_{k,n} = s_n \cdot C_{8}[z_{k,n}]
\]

- `C8`：FP8 E4M3形式のCodebook値
- `z`：Trellis復号で得た状態（具体的な状態定義は要確認）
- `s_n`：出力列単位のScaleの例

上式は**一つの設計案**であり、既存EXL3のScale、符号化・回転・逆変換と完全互換であると確認されたものではない。ScaleをK方向にも変化させる場合、部分和ごとにScale適用が必要になり、単純な1回GEMM＋列Scaleでは扱えない。

### 品質上の要点

1. FP16展開後にFP8へ丸めるだけでは、元のTrellis量子化がFP8丸め誤差を考慮していない。
2. FP8ターゲット値を使った探索により、最終表現での誤差を直接最適化する。
3. Codebookの共通化、Scale粒度、重み分布、探索目的関数（MSE、Activation-weighted等）が品質に影響する。
4. GEMMにFP8入力Activationが必要なら、**重みのFP8化とは別にActivation由来の誤差**が生じる。これを独立に評価する。
5. 4bpw維持を目指す場合、Trellis索引、遷移符号、Scale/Codebook等のメタデータを含む**実効bpw**を算出する。

## 5. オンライン復号とLUT配置

比較する復号方式：

| 方式 | メリット | 懸念事項 |
|---|---|---|
| Global Memory LUT（キャッシュ経由） | 実装が容易 | Wave内の不規則参照、ロード遅延、キャッシュアクセス効率 |
| LDS LUT | 小型テーブルを共有できる | Bank conflict、初期ロード、LDS使用量 |
| Wave内レジスタ分散LUT（特にT8） | 外部メモリLUTロードを回避できる可能性 | Lane間shuffle/gather命令数、レジスタ圧力 |
| 現行hash/SAD | メモリLUT不要、比較基準に適する | 演算命令数と復号結果のフォーマット変換 |

T8の256 B LUTを32 laneに分散すると平均8 B/laneで保持できるが、任意の索引を取り出すコストは無視できない。**LUTが小さい＝必ず速い、ではない。**

計測対象：
- 復号1重み当たりの命令数、ロード数
- キャッシュヒット率／メモリアクセス効率
- LDS bank conflict、LDS使用量
- VGPR/SGPR使用量とOccupancy
- WMMAへのタイル供給・再配置コスト
- `Recon`単独時間と、`Recon + GEMM`の全体時間

## 6. 性能検証の段階

| Stage | 重み復号・生成 | GEMM | 主目的 |
|---|---|---|---|
| F0 | 現行EXL3 mul1 → FP16 dense | hipBLASLt FP16 | 品質・速度Baseline |
| F1 | T8 → FP8 LUT → FP8 dense | hipBLASLt FP8 | 最小LUTでの品質・速度 |
| F2 | T10 → FP8 LUT → FP8 dense | hipBLASLt FP8 | 中間候補 |
| F3 | T12 → FP8 LUT → FP8 dense | hipBLASLt FP8 | 品質寄り候補 |
| F4 | 各TのFP8 LUT出力 → FP16変換 | hipBLASLt FP16 | Codebook品質とFP8 GEMM/Activation誤差の分離 |
| F5（任意） | FP8 LUT復号をカーネル内融合 | FP8 WMMA | オンライン展開の成立性 |

**重要な順番：**

1. 対象gfx1201/ROCm環境で、E4M3のhipBLASLt FP8 GEMMが必要な形状・Transpose・Scale条件で利用できることを確認する。可能なら**事前に用意したFP8 dense重み**でGEMM単独性能を測定する。
2. 次にT8/T10/T12のオフライン再量子化を行い、GPU実装より先に品質を評価する。
3. 品質ゲートに合格した候補のみ、LUT復号の性能実装へ進める。
4. 最終的にPP2048全体のtok/sで比較する。

### 性能試算（実測ではない）

参考内訳：Rotate 0.044ms / Recon 0.140ms / GEMM 0.744ms / Epilogue 0.060ms。

仮にReconが0.100ms、FP8 GEMMがFP16比1.7倍の速度で動作し、その他が不変なら、

\[
0.044 + 0.100 + 0.744/1.7 + 0.060 \simeq 0.642 \ \mathrm{ms}
\]

現行0.988msに対して1層単位で**約1.54倍**。これは仮説上の算術試算であり、FP8変換、Activation量子化、Scale処理、アルゴリズム選択、PP全体の非GEMM処理によって変化する。**PPトークン速度を1.54倍と断定しない。**

## 7. 固定品質ゲート

既存のプロジェクトゲートを変更せず適用する。

| 指標 | 合格条件 |
|---|---:|
| PPL回帰 | +1.0%以下 |
| Mean KL | 0.001以下 |
| Top-1 agreement | 98%以上 |
| Top-5 overlap | 98.5%以上 |

- BetterBench v1、固定corpus・固定ハッシュ、context 128、BF16 KV、評価positionを揃える。
- 少なくとも既存比較で用いた短い32 positionと長い1024 positionの両方を記録する。
- PPLが負方向へ変化しても、KL / Top-1 / Top-5に失敗した候補は不合格とする。
- 評価対象を**FP8 Codebook + FP16 GEMM**と**FP8 Codebook + FP8 GEMM**に分け、Activation誤差を切り分ける。

### 参考：以前のINT8再量子化の棄却理由

- FP16展開済み重みをper-column INT8へ丸める方式は、1024 positionでKL約0.0139、Top-1約94.1%となり不合格。
- これは今回の**FP8ターゲットCodebookを用いた再探索**の不合格を意味しない。
- 逆に、FP8ターゲット化すれば合格するという保証もない。

## 8. 主要リスクと確認事項

1. **Tの定義：** 実コードでT8/T10/T12/T16と状態幅・遷移表現・実効bpwの対応を確認する。
2. **Codebook粒度：** 1テーブルで十分か、層/グループ/列ごとに複数必要かでLUT容量が変わる。
3. **FP8表現：** E4M3の正確な符号化、NaN/特殊値、ZeroとScale処理を確認する。
4. **GEMM API：** hipBLASLtのFP8入力レイアウト、スケーリング方式、Epilogueの対応状況はROCm版・gfx1201向けに実機で確認する。
5. **品質：** T縮小によるTrellis探索自由度の低下とFP8数値精度の制約が重なる。
6. **供給効率：** 過去の融合WMMA実装では復号なしでも性能が低かった。融合経路を作る場合、LUT最適化だけでは解決しない可能性がある。
7. **計測：** 単独GEMM TFLOPSとモデル全体tok/sを区別し、Recon、変換、Scale、Epilogueを含むE2Eで判定する。

## 9. 優先順位とGo/No-Go

### Phase 0：成立性確認

- [ ] Tの定義と圧縮率維持条件をコードで確認
- [ ] gfx1201のhipBLASLt FP8 GEMMを実形状でベンチマーク
- [ ] FP8 Activation単独の品質を確認

**No-Go：** FP8 GEMMがFP16とほぼ同速以下、または必要なScale方式が対応困難な場合は、FP8 GEMM移行を再評価する。

### Phase 1：量子化品質

- [ ] FP8ターゲットCodebookを使いT8を生成
- [ ] 同条件でT10、T12を生成
- [ ] FP16演算でCodebook/重み起因の誤差を比較
- [ ] FP8 GEMM込みで固定品質ゲートを実行
- [ ] Codebook/Scale込みの実効bpwを確認

**No-Go：** 全候補が品質ゲートを満たさない場合はFP8移行を保留する。

### Phase 2：復号・性能

- [ ] Global LUT、LDS LUT、T8レジスタ分散LUTの小規模ベンチ
- [ ] F0〜F4を同一Linear・同一shapeで比較
- [ ] PP2048全体のtok/sを確認
- [ ] 必要ならF5の融合FP8 WMMAを検討

**Go：** 固定品質ゲートを満たし、展開・FP8量子化・Scaleを含むE2Eで現行FP16方式を上回ること。

## 10. 暫定結論

- **T10を第一候補、T8とT12を比較対象**とする。ただしこれはLUT容量と表現力のバランスに基づく研究上の優先順位であり、品質実測による結論ではない。
- EXLR8型のFP8ターゲット探索を、Trellis状態数縮小および小型LUT化と組み合わせることに価値がある。
- 最初は現行の「展開→hipBLASLt」アーキテクチャを維持し、FP8 dense展開でFP8 GEMMの実利を測定する。
- **品質ゲートを最優先**し、合格候補についてのみオンラインLUT／FP8 WMMA融合を最適化する。

---

*本書は議論に基づく研究計画であり、T8/T10/T12の実装可否・品質・gfx1201のFP8性能を実測で保証するものではない。*
