# MoE（Qwen3.6-35B-A3B）backbone 線形の allowlist 登録

decode の fallback 内訳（`docs/research/20261104/moe_gate_d_slot_layout.md` §11）で、
correctness へ落ちる 171 本 / token が backbone の bf16 線形だけだと確定した。登録の前に
`shelf-bench gemm` で shape ごとの最適化 kernel と correctness path を実測した。

## 計測条件

```sh
shelf-bench gemm --dtype bf16 --shapes 8192:2048,4096:2048,32:2048,2048:4096,248320:2048 \
  --rows 1,16,64,128 --variant gemm --impl auto --check --warmup 20 --samples 30 --launches 10
shelf-bench gemm --dtype bf16 --shapes <same> --rows 1,16,64,128 --variant correctness \
  --warmup 5 --samples 20 --launches 5
```

- GPU: Radeon AI PRO R9700（`--device 0`、測定時は device 0 / 1 idle）
- `--check` は production correctness path と突き合わせ（`variant=gemm` 側で実施）

## 実測（p50、µs）

| shape | rows | optimized | correctness | 倍率 | `--check` |
| --- | ---: | ---: | ---: | ---: | --- |
| `8192:2048` | 1 | 33.2 | 377.7 | 11.4× | max_rel 0.0020、nan/inf 0 |
| `8192:2048` | 16 | 47.6 | 1783.5 | 37.5× | max_rel 0.0039 |
| `8192:2048` | 64 | 161.3 | 6523.3 | 40.4× | max_rel 0.0039 |
| `8192:2048` | 128 | 177.3 | 12655.8 | 71.4× | max_rel 0.0078 |
| `4096:2048` | 1 | 24.6 | 403.9 | 16.4× | max_rel 0.0020 |
| `4096:2048` | 16 | 40.3 | 1000.8 | 24.9× | max_rel 0.0039 |
| `4096:2048` | 64 | 49.7 | 3602.9 | 72.5× | max_rel 0.0078 |
| `4096:2048` | 128 | 122.2 | 7363.0 | 60.3× | max_rel 0.0078 |
| `32:2048` | 1 | 4.8 | 335.9 | 70.5× | max_rel 0.0020 |
| `32:2048` | 128 | 7.1 | 351.4 | 49.6× | max_rel 0.0078 |
| `2048:4096` | 1 | 19.0 | 550.0 | 28.9× | max_rel 0.0020 |
| `2048:4096` | 16 | 70.2 | 980.1 | 14.0× | max_rel 0.0039 |
| `2048:4096` | 64 | 70.0 | 3711.0 | 53.0× | max_rel 0.0039 |
| `2048:4096` | 128 | 214.2 | 7500.9 | 35.0× | max_rel 0.0074 |
| `248320:2048` | 1 | 1692.8 | 11158.3 | 6.6× | max_rel 0.0020 |
| `248320:2048` | 16 | 3536.2 | 171325.1 | 48.4× | max_rel 0.0039 |
| `248320:2048` | 64 | 11560.8 | 637641.3 | 55.2× | max_rel 0.0078 |
| `248320:2048` | 128 | 8191.3 | 966203.8 | 117.9× | max_rel 0.0078 |

`--check` の不一致はいずれも bf16 の丸め差の範囲（最大 max_rel 0.0078 = 2^-7 弱）で、NaN / inf は
0。最適化 kernel は rows が伸びると 248320:2048 で 8 ms 台へ収まる（correctness は 1 s 級）。

## token 当たりの削減推算（rows=1、decode）

fallback 本数（`moe_gate_d_slot_layout.md` §11 の実測内訳）× correctness の実測時間:

| 内訳 | 本数 / token | correctness 合計 | optimized 合計 |
| --- | ---: | ---: | ---: |
| GDN in_proj_qkv / z / a / b + out_proj | 150 | 60.1 ms | 2.59 ms |
| attention q_proj / o_proj | 20 | 9.3 ms | 0.52 ms |
| lm_head | 1 | 11.2 ms | 1.69 ms |
| 合計 | 171 | **80.5 ms** | **4.8 ms** |

 optimized 側の合計は bench の実測値で、bench は同一 weight を反復するため weight が L3 に残る。
model 全体を流れる実測（4.4 GB / token、DRAM 636 GB/s）ではこれらの線形は合計 5.6 ms が
帯域下限なので、bench 値（4.8 ms）と同オーダーである。

## 登録と E2E 実測

登録した shape（`linear_selector.cpp` の `kLinearShapeRules` に 5 本追加、`kFamilyBf16` のみ）:
`{8192,2048}`・`{4096,2048}`・`{32,2048}`・`{2048,4096}`・`{248320,2048}`。

E2E は `shelf-bench tg --mode greedy --context 64 --tokens 32 --warmup 1` を 5 rep（同一 GPU、
device 0、測定時は device 0 / 1 idle）。

| | linear fallback（当時 counter） | gpu_ms | tok/s p50 | 生成 anchor |
| --- | ---: | --- | ---: | ---: | --- |
| 登録前 | 5472 | 5490.07 / 5511.22 / 5514.72 / 5521.73 | 5.81 | `GREEDY_TOKEN_SUM=15232` |
| 登録後 | **0** | 3126.98 / 3127.28 / 3130.21 / 3130.54 / 3150.59 | **10.22** | `GREEDY_TOKEN_SUM=2987` |

- 5 rep の spread は 0.7%（methodology の 30% 判定をパス）
- 削減量は推算（80.5 → 4.8 ms / token、差 75.7 ms）に対し実測は 172.0 → 97.8 ms / token
  （差 74.2 ms）で一致
- `shelf-compute`（`--max-num-batched-tokens 64 --max-concurrent-requests 1`、8 decode）の
  `DECODE_TOKENS_PER_SEC` は 8.00 → 19.64
- fallback 列は当時の counter で、**linear しか数えていなかった**。同じ run を kernel 単位の
  counter で読み直すと linear は 0 のまま、残りは `RMS_NORM` 2592・`PAGED_ATTENTION` 320・
  `EMBEDDING_LOOKUP` 32（32 token 区間）だった。counter 統合の経緯は
  `moe_correctness_path_counters.md`
- 生成列は変わった（`15232` → `2987`）。backbone が correctness interpreter から最適化 WMMA
  kernel へ移り、丸め順が変わるためである。5 rep 間で生成列は同一（再現性は保たれている）。
  bf16 丸め差としての妥当性は `--dump-logits` の A/B（登録前 binary と登録後 binary を同一
  prompt / seed / `--max-new-tokens 8` で走らせた logits 9 行、vocab 248320）で見た:

| row | top1 pre → post | rel_l2 | maxabs |
| --- | --- | ---: | ---: |
| 0 | 3966 → 3966（一致） | 0.05448 | 0.85 |
| 1 | 3966 → 3966（一致） | 0.05448 | 0.85 |
| 2 | 51 → 198 | 0.12570 | 1.46 |
| 3 以降 | 分岐 | 0.75 〜 1.21 | 13 〜 26 |

- prefill 終了地点（row 0 / 1）は top1 が一致し、相対差は 0.054。micro bench の max_rel
  0.0078 が 40 層で累積した値として妥当な範囲である
- row 2 以降は sampled token が分岐した後の**別 token 地点**の logits なので、乖離は数値誤差の
  拡大ではなく入力違いである。生成列の同一性はこの A/B では確認できない（5 rep 一致までを
  確認済み）
- HF との数値照合 gate（`--hidden-taps` の層別比較）は未実施のまま残る

