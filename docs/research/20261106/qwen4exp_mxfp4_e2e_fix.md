# Qwen3.8-Flash-Next MXFP4 / FP8 TP4 の E2E 修正

対象: `models/Qwen3.8-Flash-Next-Quark-MXFP4`、R9700 4 枚（TP4）、`main` の
`5a3a259` 以降。E2E の生成が `The capital of France is` に対して無意味な列になり、
実行間で先頭 token が変わる問題を、層単位の参照比較で切り分けた記録である。

現在の contract は [../../../developer/qwen4exp.md](../../../developer/qwen4exp.md)、
性能値は [../../../perf/current.md](../../../perf/current.md) を正本とする。

## 1. 参照の作り方

- PLE: 実 checkpoint の FP8 code と BF16 `weight_scale` を Python で復号し、行 RMS を
  EXL3 解析時の値（行 0 で 0.01570）と比較した。
- 層 0: `Qwen4ExpTextDecoderLayer` を PyTorch で構築し、checkpoint の重みを
  MXFP4 / FP8 から float32 へ復号してロードした。GDN は HF の
  `causal_conv1d_fn` / `torch_chunk_gated_delta_rule` の reference 実装を使い、
  hyper connection / MoE も同じ入力で計算した。
- 実行側は `SHELF_LAYER_DUMP_DIR` で層出力と中間値を BF16/F32 のまま書き出し、
  同じ層・同じ入力の値を比較した。

この比較で、層 0 の attention 出力は RMS 0.113 で一致する一方、hyper apply 後が
4 倍、MoE shared expert が group 行を読んでいることを特定した。

## 2. 修正した不具合

### 2.1 PLE の per-tensor scale を F16 として読んでいた

`PleTable::load_fp8` は `ngram_embedding.weight_scale`（BF16）の 16 bit を
`f32_from_f16_bits` で変換していた。実データの BF16 は `0x3951` で、
正しい値は `0.0001993`、F16 として読むと `0.6646` になり、PLE が約 3334 倍に
なっていた。`bf16_bits_to_f32` を追加して修正した。

検証: 行 0 の復号 RMS は 0.01587、行 123456 は 0.00929 で、EXL3 解析時の
0.01570 / 0.00933 と整合する。

### 2.2 FP8 PTPC が活性 per-token scale の row stride を無視していた

`gemm_fp8_ptpc_kernel` は `activation_scales[r]` を読んでいた。workspace の
scale は `align_up_16(ceil(k/128) * 4)` バイト間隔（k=2560 で 80 バイト）なので、
2 行目以降が別の scale を読んでいた。`activation_scale_stride_bytes` を
カーネル引数に追加し、`r * stride / 4` で読むようにした。

検証: `test_fp8_ptpc`（rows=5、k=256）が CPU 参照と一致する。

### 2.3 複製される GDN / attention の出力を TP で足していた

`emit_gdn` と `emit_gated_attention` は o_proj の出力に `tp_combine = 1` を
無条件に付けていた。Qwen4Exp の attention / GDN は全 rank に複製されるため、
TP4 では出力が 4 倍になっていた。`GdnLoweringGeometry::tp_combine` と
`GatedAttentionLoweringGeometry::tp_combine` を追加し、Qwen4Exp の lowering で
false にした。Qwen4Exp で TP 集約するのは MoE combine だけである。

検証: 層 0 の `HYPER_APPLY` 直前の attention 出力が参照と一致し、
`test_qwen4exp_lower` が 1 層 1 combine だけを固定する。

### 2.4 grouped 行と token 行で活性 workspace を共有していた

`post_norm` は routed expert（group 行、800 行）と shared expert（token 行、5 行）
の両方が読む。program builder は入力 `ValueId` だけで activation workspace を
共有していたため、先に作られた group 行 plane を shared expert が読んでいた。
`act_ws_key` の bit 63 で token plane と group plane を分離した。

検証: `test_program_activation_planes` が同じ入力に対して 2 つの
`ACTIVATION_QUANTIZE_MXFP4_ROWREF` と異なる workspace を要求する。

### 2.5 MXFP4 dense の preshuffle staging が非同期コピーだった

`load_quantized_view` の FP8 / MXFP4 preshuffle は関数ローカルの staging buffer から
`hipMemcpyAsync` していた。pageable H2D は呼び出し後に完了し得るため、buffer の
破棄と競合する。さらに MXFP4 は function-local `static` を共有していたため、
TP rank の並行ロードで互いのデータを壊す可能性があった。staging 専用の同期コピー
`copy_to_device_blocking` を追加し、static をローカルにした。routed expert の
MXFP4 も 1 expert ごとに同期する。

### 2.6 shared expert の形状が MXFP4 allowlist になかった

`640 x 2560` と `2560 x 640` は BF16 としてだけ allowlist にあり、shared expert は
全層で correctness fallback に落ちていた。`kFamilyMxfp4` を追加した。
decode は 11.2 → 26.8 tok/s、prefill は 165.9 → 106 ms になった。

## 3. DFlash2

- target tap は「完了した層出力を hyper stream 4 本で平均したもの」とした
  （DFlash2 checkpoint の patch の定義）。`MeanStreamsNode` を追加し、
  Qwen4Exp lowering の `hidden_taps` で external output の後ろへ追加する。
- `--dflash2-model-dir models/Qwen3.8-Flash-Next-DFlash2` で TP4 の target と
  組み合わせられる。CLI と serve の両方で target-only greedy と一致する。
- `--dflash2-drafts 7`（block_size 8 の 8 行 verify）は 64 token で 46 token 目に
  target-only と分岐した。drafts 1〜6 は一致する。Qwen4Exp では 6 に cap し、
  理由を stderr に出す。8 行 verify の losslessness は未解決課題である。

性能（5 token prompt、64 new token、bf16 KV、`--arena-gib 31`）:

| 構成 | tok/s | mean accepted |
| --- | ---: | ---: |
| target-only | 26.77 | - |
| DFlash2 drafts=6 | 61.13 | 2.15 |
| DFlash2 drafts=7（参考、不一致） | 59.71 | 2.15 |

## 4. 残っている課題

- 8 行 verify の losslessness。target verify の logits が target-only と
  bit-exact でない可能性があり、near-tie で token が変わる。
- prefill 用 FP8 PTPC の 2D tile、MXFP4 grouped の tile 選択。
- Qwen4Exp の PLE は host gather のままである。
