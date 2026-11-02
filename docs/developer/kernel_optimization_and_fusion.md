# カーネル最適化と融合の手順

カーネル最適化・カーネル融合を扱うときの標準手順を定義する。
対象は AMD RDNA4（gfx1201）/ ROCm / HIP のみ。
性能値の正本は `docs/perf/`、計測原則は `docs/perf/methodology.md`、
採否の R&D 記録は Git history にある。本ドキュメントは手順とルールの
現在の契約を書く。

## 原則

1. **測ってから決める。** 直感だけの変更はマージしない。
2. **正しさと性能を混同しない。** 正しさゲートと性能ゲートを分離する。
3. **効果は影響 share で頭打ちになる（Amdahl）。** 個別の高速化倍率ではなく
   「share × (1 − 1/倍率)」で価値を見積もる。
4. **数値値の変更は承認制。** WMMA/倍精度維持の数式等価変更（tolerance 内・
   決定論的・非ロス）は事前承認済み。quantization 段階の変更・ロス変更・
   等価でない変更は必ずエスカレーションする。

## 手順の全体像

```
① 現状計測と ISA 収集  →  選定表（share × 律速分類 × レバー可否）
② レバー選択  →  個別改善ルール または 融合ルールで実装
③ 検証ゲート  →  正しさ + 性能。不合格なら ① に戻る
```

## 1. ① 計測と ISA 収集

### 収集する項目

| 項目 | 手段 | 用途 |
| --- | --- | --- |
| duration / dispatch 数 / share | rocprofv3 kernel-trace（1 回の run で全 dispatch） | 影響 share の特定 |
| VGPR / SGPR / LDS / scratch | kernel-trace の資源フィールド | レジスタ圧・occupancy 算定 |
| grid / workgroup size | kernel-trace | 並列性・wave 飢餓の判定 |
| 実効転送速度 | 静的 bytes モデル ÷ duration | 律速分類（下記） |
| occupancy | 資源値からの静的算定 | レバー可否 |
| 命令クラス / `s_barrier` 数 | コードオブジェクトの disassembly grep | 命令昇格・同期の余地 |
| host gap（dispatch 間隔） | trace の時間列 | launch 律速の露出度 |

### gfx1201 での制約と算定式

- PMC カウンタは 0 を返すため**ハードウェア直読は不可**。すべて導出する。
- driver の occupancy API はソフトウェアモデルなので参考値扱いとし、
  次の定数から静的に算定する:
  - 実効 VGPR file = 47,104 dwords/SIMD（= 188,416 dwords/WGP）。§7.88 の
    co-residency 実測境界から逆算した値。compiler が仮定する 196,608 dwords/WGP は
    実機とずれており、`launch_bounds` の min-blocks が効かない原因になる
  - wave 上限 = 32 wf/SIMD、wave32、32 WGP × 2 CU = 64 CU（driver の
    `multiProcessorCount` は WGP 数）
  - 算定式: `vgpr_r = ceil(VGPR/8)×8`、
    `budget_waves = min(4 × floor(47104 / (32 × vgpr_r)), 64)`、
    `blocks = min(budget_waves / (wg/32), 131072 / LDS, 64 / (wg/32))`。
    §7.88 の実測表を全行再現する。単一の正は
    `docs/rnd/kernel_baseline_2026-10.md` §occupancy
  - 拘束条件 = min(register, LDS, wave 上限, grid)。**先にどれが拘束かを判定してから
    レジスタ圧レバーを語る。**
- 実効転送速度の判別目安（streaming read 実測 636GB/s との比較）:
  - **>100%**: L3/L2 吸収の証明。転送削減は DRAM 面で無料
  - **550–636GB/s 帯**: DRAM 境界
  - **数 GB/s 台**: レイテンシ / 並列不足
- standalone ベンチのランキングは L3 再利用で反転しうる（単一重みバッファ等）。
  順位は kernel trace と interleaved E2E で判定する。

### 選定表

kernel 家族ごとに「share、実効 GB/s、資源値、occupancy 拘束条件、
命令クラス、`s_barrier` 数、中間バッファ vs LDS 容量」を並べ、
**② の律速分類 × ④ のレバー可否**を機械的に読める形にする。
融合・個別改善の候補はこの表から選ぶ。表のない候補議論はしない。

## 2. 律速の 4 分類

| 分類 | 識別 | 有効なレバー | 無効なレバー |
| --- | --- | --- | --- |
| ① DRAM 帯域律速 | 実効 550–636GB/s 帯 | bytes の実減少（アルゴリズム削減）、launch 削減 | 計算効率・occupancy・命令昇格 |
| ② L3 / pass 律速 | 吸収フラグ、転送無料で ALU/pass 支配 | 冗長計算の消去、命令昇格、同期消去 | 転送削減のみの融合（pass しか消えない） |
| ③ レイテンシ / 並列不足 | 実効が低値、grid 小、wave 飢餓 | occupancy 上昇、GEMM 統合、命令昇格、同期消去 | — |
| ④ launch / host 律速 | gap の wall 露出度 | dispatch 削減、HIP graph | 計算効率系すべて |

- gap の実額露出は prefill のみ（decode は queue に隠れる）。decode の
  launch 削減を見積もって wall に出ると仮定しない。
- 候補の優先順位は「share 大 × ②③ × (occupancy / 命令昇格 / 同期消去) が
  成立する構造」。①と④を計算効率の主題に選ばない。

## 3. ②a 個別改善ルール

1. **冗長な命令を消す。** dead work の削除、代数簡約、計算の表参照化。
   命令数は ISA で数える。
2. **ロード・ストアの回数を減らす。** 1 要素を複数回読む pass を畳む。
   中間結果の再読は L1/L2 経由を検討する。ただし再読バイトが容量を超えれば
   実額コストとして計上する。
3. **レジスタ圧を下げて並列性・実効性を上げる。** 前提として拘束条件を
   判定する（①で律速なら occupancy レバーは効かない）。occupancy 段階が
   変わらない低減は、unroll 余地・spill 回避としてのみ効く。
   occupancy 段階の有無は資源対比表で必ず示す。
4. **高効率な命令（WMMA / packed dot 等）に置き換える。** 常に resource
   対比を併記する（高効率命令は VGPR を増やし occupancy を落としうる）。
   累加順が変わる命令は数値規則（原則 4）の適用対象。share の大きい
   ②・③ クラスでのみ意味を持つ。
5. **1 回のアクセスで触るキャッシュライン数を減らす。** coalesced な
   vector load/store（aligned・vec 幅）、同一ラインへの重複アクセス排除。
   境界は scalar fallback に落とす。
6. **計算途中結果は極力書き込まない。** レジスタ → LDS → VRAM の順で
   書き込みを避ける。materialize は消費者がレジスタ/LDS に収まらない
   とき、または別の kernel にまたがるときのみ。
7. **同期を消す・格下げする。** grid-wide 同期（kernel 境界）→ kernel 内
   barrier → 自スレッド完結の順で格下げを狙う。成立条件は「中間データが
   レジスタ/LDS に収まるか、own-slice 構造で read-before-write が
   thread-local になるか」。同期を staging 構造に変換して occupancy を
   落とす方向は禁止。
8. **依存を緩めて ILP を上げる。** unroll・ループ pipeliningは resource
   予算（VGPR/LDS）内で行う。予算超過で occupancy に跳ね返る変更は
   resource 対比で棄却する。
9. **dispatch / launch を減らすのは prefill 主体で見積もる。**
   replay できる平坦な純 launch 列は HIP graph と併用を検討する。

## 4. ②b 融合ルール

### 融合の価値源と成立条件

融合は「複数の dispatch を 1 本にすること」自体に価値があるのではない。
効果源は次の 4 つに分解して会計する。

1. dispatch / launch 削減（pp でのみ実額）
2. 消費側が行っていた**仕事そのものの消去**（再読・再エンコード等）
3. 中間 materialize の削減（容量超過分のみ実額。L3 容量内の分は会計除外）
4. 同期の消去・格下げ（kernel 境界 = grid-wide 同期の消去）

素材カーネルが L3/pass 律速（分類 ②）で、計算が削れないなら
**融合だけでは速くならない**（primitive が 1.00×）。まず選定表で
素材の律速を確認する。

### tier ゲート

融合セルフアセスメントが tier を宣言し、主指標で下限を満たすことを要求する。

| tier | 内容 | 下限（主指標） |
| --- | --- | --- |
| T0 | dispatch 結合のみ（kernel 本体不変） | ≥0.3% |
| T1 | 新 fused kernel 1 本（body は template 直接呼び出し・single call site） | ≥0.7% |
| T2 | matcher / topology の大掛かりな変更 | ≥1.5% |
| T3 | 分岐の重い融合 | ≥2% |

共通条件:

- byte-exact（文書化した等価性で代替可）
- occupancy 段階変化なし（資源対比表を報告）
- L3 容量内の intermediate 消去は会計除外
- 判定は interleaved A/B のペア別中央値

### matcher / topology の役割分担

- **matcher** は「適用してよいか」の証明器（安全装置）である。融合後パスの
  前提（overlap なし、geometry 一致等）が真であることを検証してから
  launch する。
- **topology** は「融合後に計算パスをどう組み直すか」の設計である。
  kernel 境界（grid-wide 同期＋materialize＋再読）を、own-slice 完結 /
  LDS 経由 / materialize 維持のどれに置き換えるかを選び、レバーの成立
  可否（特に同期消去）をここで決める。
- **カーネル本体** はその中で式・命令クラス・resource を実装する。
- **byte-exact は構造変更の自動結果ではない。** per-element の演算列・
  順序・中間精度段階を保存するように設計したときにのみ成立する
  （norm 数学の複製や縮約順・累加順の変化は数学的等価でも 1 ULP の差を生む）。
  保存できない変更（WMMA への命令昇格、ロスの導入など）は byte-exact を
  目指さず、数値規則（原則 4）へ分岐して tolerance・決定論・非ロスと
  token ゲートの更新を承認条件にする。byte-exact は必ず focused test と
  token ゲートで検証し、仮定しない。

### matcher ルール

- **applicability を完全証明してから launch する。** 階接・slot 連結・
  geometry（dtype / k / rows / profile）・workspace レンジを検証する。
  kernel 側の validation エラーが matcher をすり抜け、dispatch を
  エラー終了させる経路を作らない。
- **部分 overlap は reject。** 完全 identical のみ条件付き許容
  （読む側が全スレッドで完了してから書く側に進む構造がある場合のみ）。
- alias は topology 構造で保証できるなら range 検査の代わりに構造で
  証明する（own-slice なら read-before-write が thread-local）。

### topology ルール

- single call site。chains は dispatch ブランチに直列に置く。
- group-slice / own-slice 構造を優先する（phase 遷移の barrier を増やさない）。
- **行数ガード**: row-wide な処理を含む融合は `rows == 1` で split を
  維持する（grid が 1 CTA に直列化し split の方が速い）。
- **共有 norm body は template 直接呼び出しのみ。** 数学の warp-local 複製は
  禁止（縮約順序の 1 ULP 差が多層で蓄積しトークンが崩れる）。
- occupancy 対比は resource 表で示す（計算には gfx1201 の定数を使う）。

## 5. ③ 検証ゲート

### 正しさ（順に実施）

1. focused test（adversarial 入力・denormal・巨大値・alias ケース）を byte-exact で
2. E2E token ゲート（greedy token 集計・生成 ids の一致・投機的推論の統計量）
3. `ctest` の cpu と GPU スイート。required は skip = fail

### 性能

- interleaved A/B（old/new 交互）を 3 ペア以上。GPU の汚染を KFD の
  プロセス監視で検知し、汚染ペアは棄却する。
- 主指標 1 つと副指標（非悪化確認）を事前に決める。
- 判定はペア別中央値。**ノイズ幅未満のマージンなら判定ルールを事前に固定して
  ペアを追加し、データで決着する。** 事後的な基準変更をしない。
- kernel accounting（影響 kernel の時間変化から機関的に見積もった値）と
  wall 効果を突き合わせる。乖離があれば原因（queue 隠れ・L3 吸収・
  遊休の食い潰し）を特定してから採否を決める。

### 時間計測の禁則

- `PHASESHIFT_OP_DUMP=1` の付いた run の時間は使わない（per-op 同期が入る）。
  時間は kernel trace と E2E が正。
- standalone の小ベンチだけで E2E を予測しない（L3 再利用で順位が反転する）。
- 採否に必要な数値は E2E interleaved で確定する。

## 6. 禁止パターン

- 共有 device body を逐次処理へ書き換える融合
- 複数 tile / 複数 work を 1 CTA に束ねて occupancy を落とす変更
- grid-wide barrier を必須にする融合
- workspace 部分 overlap を無視する matcher
- launch 削減のみで wall 効果が確定不能な融合（特に decode の短い連鎖）
- 数学の複製（norm body 等の shared body）による「独立した同じ計算」
- L3 容量内の中間消去を実効 bytes として計上すること
- decode の queue 隠れ gap を wall 減少として見積もること
- 「再構成」を理由にした実装の全面撤去の反復（差分は stash に残して採否を記録する）

## 関連

- 計測原則: `docs/perf/methodology.md`
- 現在の性能値: `docs/perf/current.md`
- カーネル配置・分割: `docs/developer/kernels.md`
- テスト: `docs/developer/testing.md`
- 過去の融合候補調査・採否・総括的知見（当時は `docs/rnd/` に配置）: Git history
