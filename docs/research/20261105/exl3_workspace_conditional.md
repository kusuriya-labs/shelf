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

- required / gpu2 の再測（同じ tree）: 表参照
- `test_executor_footprint` に `exl3_workspace_bytes` と一致するかの check を追加した
  （条件を外して項を残す、または項目を落とす変更を弾く）
- MoE TP=2 の prefill / decode を組み直した tree で測り直し、KV pool の回復を
  `kv_reserved_bytes` で確認した: <!-- KV_DELTA -->
