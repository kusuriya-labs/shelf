# Qwen3.5-9B (BF16) の greedy 出力が参照と一致しない問題

`shelf-compute --model-dir models/Qwen3.5-9B` の生成列が無意味になる。重みロードは
byte 一致で、逸脱は **層 0（GDN）の計算結果**から始まる。今回は切り分けまで。
修正は行っていない。

## 1. 再現

```bash
# Shelf (build は gfx1201, SHELF_BUILD_OPTIONAL_TESTS=ON)
mkdir -p /tmp/taps && printf '1299\n' > /tmp/one.txt
SHELF_DEBUG_HIDDEN_FILE=/tmp/taps ./build/shelf-compute \
    --model-dir models/Qwen3.5-9B --input-ids-file /tmp/one.txt \
    --max-new-tokens 1 --hidden-taps 0,1,2,3
# -> GENERATED_IDS=2062, taps は <tag>_L<layer>.bin に rows x 4096 の f32 で出る

# 参照 (torch 2.13.0+rocm10.0.0, transformers 5.20.2, bf16, greedy)
# hidden_states[0] = embedding, hidden_states[L+1] = 层 L の後
```

prompt `[3710, 369, 279, 6511, 314, 9338, 30, 21134, 25]`
（= "What is the capital of France? Answer:"）に対する 9 token:

```text
HF   : 11751,13,271,3710,...  -> " Paris.\n\nWhat is the capital of Germany? Answer: Berlin..."
Shelf: 95839,98900,88792,15729,3709,67,13,...  -> 無意味列
```

## 2. 定量局在（1 token prompt, rows=1）

`yS = Shelf の层 L 出力 - embedding`、`yH = HF の层 L 出力 - embedding`（= 层の加算寄与）。

| layer | 位相 | \|yS\| | \|yH\| | cos(yS,yH) | 要素比 median | 要素比 p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | GDN | 11.17 | 6.86 | 0.51 | 4.70 | 23.9 |
| 1 | GDN | 22.35 | 10.19 | 0.43 | 5.42 | 27.9 |
| 2 | GDN | 28.01 | 16.93 | 0.33 | 4.32 | 23.4 |
| 3 | full attn | 38.95 | 16.91 | 0.60 | 4.72 | 27.0 |

- **層 0 から既に逸脱**している（履歴が要る conv / recurrence の順序問題ではない）
- 值頭（head_v=128）ブロック別のノルム: Shelf `[2.11, 1.93, 2.12, 1.75, 1.81, 1.95]` /
  HF `[0.41, 0.46, 0.56, 0.43, 0.42, 0.45]`。**Shelf は頭間でほぼ一定、HF はばらつく**
  （= `linear_attn.norm.weight` の頭別・ channel 別の重みが効いていない様子）
- 倍率は一律ではなく channel 依存（median 4.7、p90 24）

## 3. 切り分けて却下した仮説

| 仮説 | 検証 | 結果 |
| --- | --- | --- |
| lm_head 别名修正が誤った tensor を選ばせた | 9B の index に lm_head は 1 本だけ。`test_qwen35_mtp_weight_real` の body 検証で embed / lm_head は source と一致 | 却下 |
| GDN weight が GPU に正しく載っていない | 层 0 の `in_proj_qkv / z / a / b / out_proj`（行列 5 本）と `conv1d / norm / dt_bias / A_log`（4 本）を source byte と比較。**全部一致**（F32 の `norm.weight [128]` と `A_log [32]` は bf16 変換後に一致）。check 自体を `test_qwen35_mtp_weight_real` に恒久化 | 却下 |
| BF16 path が dtype を検証せず F32 を再解釈している | `load_bf16_tensor` は `spec.dtype == F32` で `upload_tensor_f32_to_bf16` を通す（変換自体は正しい）。9B の F32 tensor は `linear_attn.norm.weight` と `A_log` の 2 種だけ | 却下 |
| conv1d が rank-3 `[8192,1,4]` で壊れる | kernel は `conv_weight[ch*4]` の lineare 読み、dispatch の検証は byte 数 `conv_dim*4` のみ。load 済み tensor の d0=8192, elems=32768 で一致 | 却下 |
| rope 設定の取り違え（`rope_parameters` しか読まない問題） | 9B / 27B / 35B とも `text_config.rope_parameters = {rope_theta: 1e7, partial_rotary_factor: 0.25, rope_type: default}`。Shelf の ROPE profile は theta 1e7 / rotary_dim 64 で一致 | 却下 |
| topology 規則（layer_types / interval）の不一致 | 9B は `layer_types` を 32 個宣言、`full_attention_interval=4` と一致。load 位相と HF の位相（24 GDN / 8 full attn）も一致 | 却下 |
| optimized kernel 側のバグ | `SHELF_QWEN35_KERNEL_MODE=correctness` の taps と optimized の taps の差は bf16 丸め水準（层 0 で maxabs 0.0039、层 3 で 0.031）。greedy ids も同一 | 却下（**両 path が同じ誤った値を出している**） |
| GDN 実装自体が壊れている | `test_qwen35_correctness_fixture`（f64 oracle fixture、27B-FP8）が PASS。层 taps 8 本の相対 L2 ≤ 2e-2 と greedy 64 token 一致 | 却下（**27B geometry では正しい**） |

## 4. 残っている仮説（未検証・推測）

9B は **GDN の value features が hidden_size と一致する**（value_heads 32 × head_v 128 =
4096 = hidden 4096）。他の checkpoint では一致しない:

| model | hidden | value features | conv features |
| --- | --- | --- | --- |
| Qwen3.5-9B | 4096 | **4096** | 8192 |
| Qwen3.8-27B-FP8 | 5120 | 6144 | 10240 |
| Qwen3.6-35B-A3B MoE | 2048 | 4096 | 8192 |

`gdn_geometry.h` の shape 検証は extent を明示している（`z = [value_features, hidden]`、
`out = [hidden, value_features]`）が、9B では両者が同一なので、
**extent の取り違えがあっても shape 検証を通過してしまう**。27B-FP8 ではその取り違えが
shape 不一致で露見する（＝通らない）ため、9B だけ silent になりうる。
加えて head 間ノルムがほぼ一定という症状は、`linear_attn.norm.weight`（head_v 長の
channel 重み）が期待通りに効いていないことと整合する。

**これは仮説であり、該当箇所は特定できていない。**

## 5. 追跡の入口（次recommended）

1. HF 側で GDN module の sub-step（in_proj 出力 → conv → recurrence → RMSNorm(×norm.weight)
   → gate z → out_proj）を hook で取得し、Shelf の同等中間値と突き合わせる。Shelf 側は
   层境界の tap しか出せないため、**层内を 1 層だけの合成 checkpoint**（`test_qwen35_lowering_contract`
   系の fixture で GDN 1 层）にして中间 buffer を dump するのが速い
2. 9B 相当 geometry（hidden 4096、key 16×128、value 32×128、conv 4）の GDN 単体テストを
   追加して、27B geometry（value 48）と両方通す。現状の GDN kernel テストがどの geometry を
   覆盖しているかを先に確認する
3. 9B のように `value_features == hidden` の合成 checkpoint を fixtures に加える
   （実 checkpoint が無いため required 化しやすい）

## 6. 本次に追加した回帰テスト

`test_qwen35_mtp_weight_real`（`gpu1;optional;external_files`）に **hybrid body の GDN
tensor 検証**を追加した。层 0 の行列 5 本と small vector 4 本を checkpoint byte と比較し、
F32 宣言の tensor は bf16 変換後に比較する。実 checkpoint 3 種で PASS:

```text
Qwen3.5-9B                    Passed   (BF16, F32 norm.weight / A_log を変換して一致)
Qwen3.8-27B-Quark-AWQ-MXFP4   Passed
Qwen3.8-27B-FP8               Passed
```

つまり **ロード側は contract 通り**で、この問題は実行系（lowering / kernel / state）に残る。
