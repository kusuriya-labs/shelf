# EXL3-INT8 expert の host offload

## 目的（ID: A）

`SHELF_MOE_HOST_EXPERTS` による routed expert の host 退避を MXFP4 に加えて EXL3-INT8
（mul24）でも使えるようにする。Qwen4Exp は 48 層・hidden 2560・experts 512・top_k 10 で、
TP=2 の 1 rank あたり 256 expert のうち冷たい expert を pinned host へ置く。

## 配置

MXFP4 の host plane と同じ `MatrixWeight` の host ポインタを流用する。

| EXL3 plane | host 側の格納先 | 1 expert の byte（gate/up） |
| --- | --- | ---: |
| trellis | `host_codes` | 819,200 |
| suh | `host_scales` | 5,120 |
| svh | `host_row_ref` | 1,280 |
| marker | VRAM の 1 枚を共有 | 4 |

marker は loader が全 expert で同一であることを検査済みで、GEMM が参照しないため
host 側には置かない。`MatrixWeight::host_exl3_marker` が VRAM marker の先頭を指す。

slot map、route freq、`MoeExpertCache` の plan / swap / hit 集計は形式非依存で変更しない。
EXL3 の swap も trellis / suh / svh の 3 plane を MXFP4 と同じ経路でコピーする。

## 変更点

- `qwen4exp_load.cpp`: EXL3 の plane を `local_experts - host_experts` 個の VRAM と
  `host_experts` 個の host に分けて確保・upload する。marker の一貫性検査は全 expert に対して行う。
  `expect_experts()` の EXL3 shape 検査を `local_experts` ではなく `w.experts` に対して行う。
- `weight_slot.cpp`: `host_resident` の EXL3 では `exl3_suh` / `exl3_svh` / `exl3_marker` を
  host 側のポインタで埋める。VRAM Tensor からの代入がこれを上書きしないよう else 側へ移す。
- `program.cpp`: EXL3 の grouped linear binding に `host_weight_index` を設定する。
- `group_plan.h` / `types.h`: `group_row_base` を追加する。miss 側の group index を
  `miss_base` へ写像し、`row_token` は `base - group_index` 行だけずらす。既存の
  VRAM 経路では `base == group_index` なので変化しない。
- `moe_dispatch.hip`: EXL3 の grouped GEMM を hit / miss の 2 launch にする。miss 側は
  host plane を使い、`expert_group_begin` を null にして group fuse を切る。

## 検証

- `test_exl3_mul24_grouped`（全ケース）: PASS。`group_row_base` 追加による回帰なし。
- `test_qwen4exp_load --skip-ple --rank 0 --device 2` を
  `SHELF_MOE_HOST_EXPERTS=64` で実行: PASS。VRAM / host の split、host plane の常駐、
  VRAM と host の両方の trellis が元 safetensors とバイト一致、
  `validate_host_weight_slot()` が rank-3 plane として受理することを確認した。
- 生成 ID の一致（TP=2、device 2,3、prompt 8 token と 16 token）:

  | activation | `moe-host-experts` | generated ids |
  | --- | ---: | --- |
  | FP16（既定） | 0 | `350,1186,248046` |
  | FP16（既定） | 64 | `350,1186,248046` |
  | INT8（`SHELF_EXL3_INT8_MOE=1`） | 0 | `350,1186,248046` |
  | INT8（`SHELF_EXL3_INT8_MOE=1`） | 64 | `350,1186,248046` |

  prompt 8 token では 3 通りすべて `248046` で一致した。

## 採否

- ACCEPT: MXFP4 と同じ slot map / cache を共有し、EXL3 では plane の格納先だけを
  `MatrixWeight` の既存フィールドへ割り当てる。生成 ID が FP16 / INT8 の両方で一致した。
- ACCEPT: `group_row_base` を `resolve_group_plan()` 内で処理し、grouped kernel 本体を
  変更しない。既存の VRAM 経路の回帰テストが通った。
- 未検証: mul1（W16A16）の EXL3 expert での host offload。今回は mul24 のみ確認した。

## group fuse との両立（ID: B）

最初の実装では host offload が有効なとき group fuse（`expert_group_begin`）を切っていた。
fuse の range が元の group 空間を指し、hit/miss plan が group を圧縮するため、
swap で slot map が変わると index が食い違い、miss 側が不正な行を読んで non-finite になった。

- hit plan と miss plan に専用の `hit_expert_group_begin` / `miss_expert_group_begin` を
  `MoePlanScratch` に追加し、`moe_slot_scan_kernel` が圧縮後の index で expert ごとの
  最初の group を書く。
- `resolve_group_plan()` の fuse 終端は `expert_group_begin[expert + 1]` ではなく、
  `group_expert` が変わる位置までとした。これで元の group 空間でも圧縮後でも同じ
  ロジックで動く。

4096 token、TP=2、warmup 8、runs 3:

| 構成 | gpu_ms_median |
| --- | ---: |
| host 0 | 1433.7 ms |
| host 224、fuse なし（回避策） | 1799.6 ms |
| host 224、fuse あり（修正後） | 1683.6 ms |

host 224 の生成 ID は host 0 と一致した（512 token prompt、16 token 生成、FP16）。

## INT8 活性と FP16 活性（ID: C）

4096 token prefill（host 0、TP=2、warmup 8、runs 3）:

| activation | gpu_ms_median |
| --- | ---: |
| FP16（既定） | 1438.4 ms |
| INT8（`SHELF_EXL3_INT8_MOE=1`） | 1447.7 ms |

精度は 512 token prompt、16 token 生成で比較した。

- FP16: `5513,271,256,318,8839,25,279,3294,369,35007,47622,310,8204,16347,26,279`
- INT8: `5513,12,4830,5513,12,4830,...`（3 token の繰り返し）

INT8 は速度でも FP16 を上回らず、実モデルの生成が破綻した。既定を FP16 にする判断を支持する。
