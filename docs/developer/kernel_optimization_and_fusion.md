# カーネル最適化の手順

カーネル最適化を扱うときの標準手順を定義する。
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
② レバー選択  →  個別改善ルールで実装
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

## 3. 個別改善ルール

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

## 4. 融合ルール（未実装）

現在、実装済みのカーネル融合は存在しない（`src/shelf/kernels/optimized/` に
fused kernel を置かない）。融合を行う場合は次の手順を通す。

1. 素材 kernel の律速分類（§2）を計測で確定する。分類 ②（L3 / pass 律速）で
   計算を削れないなら、融合だけでは速くならない。
2. dispatch 削減だけでなく、消費側仕事の消去・中間 materialize 削減・
   grid-wide 同期の消去のどれで稼ぐかを数値で見積もる。
3. 成立条件（byte-exact または文書化した等価性、occupancy 段階変化なし、
   launch 1 本）を focused test と interleaved A/B で検証する。
4. 採用時に限り、融合 kernel と matcher を追加し、`cmake/tests.cmake` の
   required に focused test を登録する。

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

- `SHELF_OP_DUMP=1` の付いた run の時間は使わない（per-op 同期が入る）。
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
