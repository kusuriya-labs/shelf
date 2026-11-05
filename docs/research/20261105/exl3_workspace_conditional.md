# EXL3 実行 workspace を無条件で確保していた件（2026-11-05）

実験 ID: `EXL3-WS-1` / 結論: **ACCEPT**（確保を program set と weight 列の条件つきへ変えた）

## 1. 経緯

`main` の EXL3 W16A16 系（`docs/research/20261105/exl3_w16a16.md`）を取り込む際、実行系が
無条件に大きく変わることを発見した。`create_executor` の execution workspace に

- xrot `80 MiB`
- dense 復号平面 `192 MiB`（`kExl3DenseBytes`）
- accum `rows * 17408 * 4` bytes（`exl3_dense_accum_bytes`）
- rowsum `rows * 4` bytes

が追加され、executor footprint の容量式（`qwen35_executor_reserve_bytes`）にも同じ項が入った。
いずれも **program が `LINEAR_EXL3_W16A16` を走らせるかどうかを見ていない**。

row bucket ごとの上乗せ（定数からの計算値）:

| row bucket | 上乗せ |
| ---: | ---: |
| 512 | 306 MiB |
| 2048 | 408 MiB |
| 4096 | 544 MiB |
| 8192 | 816 MiB |

## 2. 症状（merged tree、`main` 側の line で再現）

- `test_executor_footprint`（required）: `reserve stays within 15 percent of the measurement`
  が失敗。rows4096 の見積もりが 1.82 GiB 台から 2.37 GiB へ跳ね、`< 2048 MiB` の upper bound を
  越えた（lower bound 側は gfx1201 の実測 0.25 / 0.91 / 1.78 GiB なので、上乗せ分は**実測に
  含まれていない**）。
- `test_qwen35_tp_execution`（gpu2）: `arena capacity exceeded: capacity 1.00 GiB used 0.75 GiB
  requested 0.31 GiB`。fixture は `arena_bytes = 1 GiB` のまま（テストも fixture も main で
  未変更）。**dense TP の実行 workspace 確保が 0.31 GiB 増えた**ための起動失敗。
- 本番側は KV pool が同じ額だけ減る。`pp --prompt-tokens 8192` の row bucket 8192 では
  1 rank あたり 0.8 GiB が KV pool から消える計算になる。

どちらも main 単体で赤（実行系を変えたのは main 側だけなので、競合解消とは無関係）。

## 3. 採否判断

候補:

1. fixture の arena を 2 GiB へ上げ、upper bound を緩める
2. **EXL3 の領域を、実際に EXL3 linear を走らせる実行に対してだけ確保する**

2 を採用。理由は AGENTS の「計測可能なオーバーヘッドは置かない」に沿うもので、
BF16 / HF FP8 / MXFP4 の target で 0.3〜0.8 GiB / rank を常時失うのは、KV pool の容量と
page 数に直接効く。1 は症状を隠すだけで、本番の無駄は残る。

dispatcher 側はもともと `ctx.exl3_xrot != nullptr` / `ctx.exl3_dense != nullptr` を見て
 dense recon・pre-rotate 経路の on/off を決めている（`optimized_dispatch.hip`）ため、
 領域を置かないことは contract 上も安全で、**EXL3 を使わない model での挙動変化は無い**。

## 4. 実装の形

- **確保側は program set を張る**（正本）。`create_executor` で built program の dispatch を
  走査し、`LINEAR_EXL3_W16A16` が無ければ 4 つの領域を置きしない（`Executor::exl3_workspace`）。
  確保と使用が同じ条件になるため、領域が足りないという状態は作れない。
- **見積もり側は weight 列を張る**。容量式は KV pool を切る前に呼ぶので program がまだ無い。
  `qwen35_executor_reserve_bytes(..., bool exl3_linear_workspace)` に引数を増やし、呼び出し側は
  load 済みの weight の `MatrixEncoding::Exl3` を見る（`qwen_hybrid::layer_uses_exl3_linear` +
  各 family の `uses_exl3_linear`）。TP では `TpRankModel::uses_exl3_linear()`。
- 領域サイズは `exl3_dense_workspace.h` の `exl3_workspace_bytes(rows)` に一本化し、
  容量式と `create_executor` が同じ式を使う。
- 見積もりが確保より少ない場合（weight 列に Exl3 があるのに program が走らせない、の逆方向では
  起きない）は起動失敗になるが、その場合は program が EXL3 dispatch を持つので確保もされる。
  逆（weight に Exl3 が無く program が持つ）は lower_to_primitives の経路上発生しない。

## 5. 影響の確認

| tree | required | gpu2 |
| --- | --- | --- |
| merged（`0e526f0`、gate 前） | 167/168（`test_executor_footprint` 失敗） | `test_qwen35_tp_execution` 失敗 |
| gate 後（`ef80238`） | **168/168** | **6/6**（3 skip は model dir 未指定） |

- `test_executor_footprint` に `exl3_workspace_bytes` と一致するかの check を追加した
  （条件を外して項を残す、または項目を落とす変更を弾く）
- 確保条件そのものは `program_set_needs_exl3_workspace` を公開して分岐両方を test にした
  （bf16 dispatch だけ → false、1 bucket でも `LINEAR_EXL3_W16A16` があれば true）
- 見積もり側（weight 列）は確保側（program 列）の情報を含む。program の `LINEAR_EXL3_W16A16`
  dispatch は load 済み weight の encoding からしか生まれないので、weight 列が false のときに
  program 列が true になる方向は起きない（起きれば reserve < 確保 で起動失敗になる）
- **実 model で EXL3 経路を確認する**: checkpoint はリポジトリ直下
  `models/Qwen3.8-27B-exl3`（16 GiB、worktree には `models/` が無い）。`main` の記録条件
  （`pp --prompt-tokens 2048 --runs 3 --warmup 1 --arena-gib 28`、decode は
  `tg --mode forward --context 128 --tokens 16 --warmup 1`）で再走らせた。§6

## 6. gate が EXL3 prefill を 2 倍遅くしていた（移動でメンバが落ちる）

実 model 再走で捕まった。症状（`--device 0`、他の GPU は別テナント）:

| 条件 | 記録（`ca5af8f`、main） | gate 後の初測 |
| --- | ---: | ---: |
| prefill 2048 | 1389.5 / 1390.4 / 1398.8 tok/s | **714.1 / 715.2 / 716.0 tok/s** |
| decode 128 | 20.09 / 20.16 / 20.47 tok/s | **20.69 tok/s** |

decode（融合 W16A16）は無傷で、prefill（`rows >= 256` の dense recon + hipBLASLt）だけが
**約 1.96 倍**遅い。run 揺れではなく stddev 3.8 ms / 2863 ms。fallback 行数は 0 なので、
correctness path へは落ちていない。つまり EXL3 kernel は走っているが、その内部で
dense recon 経路が選ばれていない。

原因は確保条件を `Executor::exl3_workspace` という **メンバに持たせた**こと。`Executor` の
move constructor / move assignment はメンバを手で張っており、追加した bool が写されない。
移動後は既定値の false なので、

- workspace の**確保**はされている（`create_executor` 内の局所 flag で決まる）
- execution context の**切り出し**はメンバを見ており、移動後の executor では走らない
- `ctx.exl3_xrot / exl3_dense / exl3_accum / exl3_rowsum` は null のまま
- dispatcher は null 判定で dense recon・pre-rotate を諦め、fused 経路で組む

という形になり、領域を確保したまま速い経路だけ失った。required / gpu2 は绿色のまま
（EXL3 checkpoint を executor 経由で通す test が無い）で、捕まえたのは **perf baseline との
比較**だけだった。

修正はメンバをやめて**使用側から導出**する形:

- `program_needs_exl3_workspace(const Program&)` を公開し、`build_host_execution_context` は
  実行する program を見て切り出す（`program_set` は move ops に含まれているので移動で壊れない）
- 確保側は `program_set_needs_exl3_workspace`（同じ predicate の総当り）で size を決める。
  program が使うなら確保もある、が構造で保証される
- 切り出しは program 実行 1 回につき dispatch 一覧 1 走査（実 model で ~1.5k 比較）。
  decode 1 step の budget は ~15 ms なので hot path のコストとしては測定範囲外

`test_executor_footprint` には per-program / per-set の両方を入れた（bf16 dispatch だけ →
false、1 つでも `LINEAR_EXL3_W16A16` を持つ program / bucket がある set → true）。

修正後の再測（同じ `--device 0`、`--arena-gib 28`）:

| 条件 | gate 導入直後 | 導出へ変えた後 | 記録（main） |
| --- | ---: | ---: | ---: |
| prefill 2048 | 715.2 | **1439.4** | 1389.5 / 1390.4 / 1398.8 |
| decode 128 | 20.69 | **20.70** | 20.09 / 20.16 / 20.47 |
| fallback 行数（`pp 512`） | 0 | 0 |  |

prefill は記録帯より 3 % 上に出た。device も tree も違うので**高速化の主張としては使わない**
（使ったのは「dense 経路が復活した」ことの確認まで）。MoE 側は導出後も動かず、
`pp --prompt-tokens 8192 --prefill-chunk 2048` で TP=1 **3783.8** / TP=2 **6511.6** tok/s、
`kv_reserved_bytes` **18.64 GiB**（gate 導入直後の測值と同一）。required 168/168・gpu2 6/6。

採否の記録として残す教訓: `Executor` にメンバを追加したら move ops を張替えなければならない。
cold path の flag でも、この class の失敗は required テストでは捕まらない（実 model + executor
経路の coverage が無い）。性能主張に baseline を残しておく意義はこの種の silent regression を
唯一捕まえられる手段であること。
- MoE TP=2（`Qwen3.6-35B-A3B`、`pp --prompt-tokens 8192` の row bucket 8192）を組み直した tree
  で測り直した:

  | | 合体前 | 合体後（gate あり） |
  | --- | ---: | ---: |
  | executor footprint / rank | 1.31 GiB | **1.33 GiB** |
  | auto reserve / rank | 1.96 GiB | **2.17 GiB**（auto slack 0.20 GiB を表示に含む） |
  | `kv_reserved_bytes` | 18.86 GiB | **18.64 GiB** |

  footprint の +0.02 GiB は main の scratch 増分（8192 → 3145728 float）だけ。EXL3 項は載って
  いない。gate を外せば bucket 8192 で **+816 MiB / rank**（xrot 80 + dense 192 + accum 544 +
  rowsum 0.03。定数からの計算で、その形の実測はしていない）が reserve に乗り、KV pool は
  17.8 GiB 台へ落ちる
- prefill throughput は gate 前の同一条件と run 揺れの範囲（TP=1 3755.3 / 3782.8 / 3618.7、
  TP=2 6204.3 / 6510.8 / 6439.1、chunk 1024 / 2048 / 4096）。`docs/perf/current.md` が正本
