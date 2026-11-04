# Gate D 実施記録 第二段 Step 1a: rank-3 weight slot と batch stride の単一源

MoE FFN を graph / kernel に配する前の下ごしらえ。`WeightSlot`（program の weight 記述）を
rank-3 expert 平面に対応させ、expert 間 stride の式を 1 箇所に集約した。
現在形の contract は `docs/developer/qwen35moe.md` §3。

## 1. 実験 D-2a: stride 式の重複を layout contract 側へ寄せる

配線前に、同じ式を 4 箇所が別々に書いていた。

```text
moe_weight_plan.cpp      (rows/16) * blocks * 256        ... arena 割当
moe_expert_load.cpp      tiles * native.codes_row_stride  ... 実平面の stride
mxfp4.hip launcher       (weight_rows>>4) * nb * kWeightBlockBytes ... kernel の受理条件
Mxfp4NativeView::validate batch * tiles * blocks * 256    ... host payload 検証
```

host 側の weight table 検証（`validate_host_weight_slot`）は rank-3 を一切見ておらず、
kernel の受理条件と検証条件が食い違っても launch まで気づかない。

採用: 式を `include/shelf/quantization/mxfp4/mxfp4_native.h` の
`mxfp4_native_codes_batch_stride_bytes` / `mxfp4_native_scale_batch_stride_bytes` に置き、
上記 4 箇所と `validate_host_weight_slot` の 5 箇所すべてがそれを呼ぶ。
常量も `mxfp4_native_block_bytes` / `mxfp4_native_output_tile` を使い、
`kernels/optimized/linear/mxfp4.hip` に
`static_assert(kMxfp4GemmRowTile == mxfp4_native_output_tile)` を入れた。

却下: 式を `kernels/optimized/linear/mxfp4.h` に置く案。grouped GEMM の受理条件である前に
native SoA の layout contract なので、配置は quantization 側が正しい
（`models/qwen35moe` が `kernels/optimized` を include する辺も消える）。

## 2. 実験 D-2b: plan は k_padded が 128 の倍数でないと拒否する

`moe_weight_plan.cpp` の padding は 64 単位（checkpoint の declared padding と一致させる値）の
まま、`k_padded % 128 != 0` の plan を `Status::unsupported` にした。grouped GEMM は K を
128 単位で読む（`ng = k_padded >> 7`）ため、64 単位の平面は kernel が読めない。
従来は `hidden_size = 2064` のような geometry で plan が通り、load まで進んで
launch 時に `hipErrorInvalidValue` になっていた。

この判断で `test_moe_weight_plan` の既存 assert
（`hidden 2064 → k_padded 2112` を受理）は契約変更になった。代わりに
`moe_intermediate_size = 200 → w2_k_padded 256`（切り上げは残る）を受理し、
`2064` を拒否する形にしている。

## 3. 実験 D-2c: weight table は構造体そのものを写す

`upload_host_weight_table` は字段を 1 つずつ写しており、`bytes` と `compute_spec` が
device 側の table に落ちていた（`WeightSlot::bytes` は host 側 program build
（`program.cpp` の bf16 検証）でしか効いていないため、実害は出ていなかった）。
字段追加と同じ穴があくので `host[i] = s;` の構造体代入に変えた。
device 側 kernel が `bytes` / `compute_spec` を読む箇所は無い（grep 0 件）ため、
dense の挙動は変わらない。

## 4. fail-first 実演

`tests/unit/test_weight_slot_validation.cpp`（CPU、required）を先に置き、検証本体を
変えない状態で実行した。

```text
FAIL experts 0 rejected
FAIL rank-3 with zero codes batch stride rejected
FAIL rank-3 with zero scales batch stride rejected
FAIL rank-3 with tampered codes batch stride rejected
FAIL rank-3 with tampered scales batch stride rejected
FAIL rank-3 with output rows not a multiple of 16 rejected
FAIL rank-3 with k_padded not a multiple of 128 rejected
FAIL rank-3 bf16 encoding rejected
（+ expert count 不一致の 1 件、後述の理由で削除）
FAILED 9 checks
```

rank-3 検査を入れた後に 0 FAIL。`experts` 数と arena 平面長の突合は、slot 自体が
plane byte 数を持たないため表現できない（grouped launcher も expert 数を読まない）。
これは loader 側の `moe_ffn_weights_match` が plan と突合して固定するものとして、
slot 検証の case からは外した。

## 5. 実測

`build/`（`SHELF_BUILD_OPTIONAL_TESTS=ON`、gfx1201）で実施。

required acceptance（`python3 tests/run_required_acceptance.py --build-dir build`）:

```text
Passed: 149  Failed: 0  Errors: 0  Skipped: 0
```

Step 1a 前の baseline が 148 PASS で、増えた 1 件が `test_weight_slot_validation`。
`test_weight_load` / `test_qwen35_mtp_weight_load` / `test_qwen35_tp_weight_load` は
そのまま PASS（dense の slot 記述は変化していない）。

実 checkpoint の MoE load（`test_qwen35moe_model_load_real`、optional / external files）:

```text
load 25.4 s, arena used 20.527 GiB, expert planes 15.967 GiB (plan sum), peak RSS 21.40 GiB
fold-exact planes: 80 / 80 (both planes in 40 layers)
```

arena・expert 平面・fold-exact 数は第一段の実測（20.527 GiB / 15.967 GiB / 80 本）と同一。
load 時間の差（32.7 s → 25.4 s）は同一 build 系の再現実行のばらつきで、
今回の変更による短縮としては計測していない。
