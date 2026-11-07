# pp2048 TP4 上位カーネルの ISA 調査

## 1. 対象と成果物

`Qwen3.8-Flash-Next-Quark-MXFP4` / TP=4 / 2048 token prefill の GPU time 上位 4 カーネルを
gfx1201 の ISA に落とし、律速要因を調べた。

ISA と補助データは build 配下に置いた（commit 対象ではない）。

```text
build/isa/pp2048_tp4/
  attention_paged_bf16.s
  gemm_bf16_wmma_4.s
  qsa_topk_batch.s
  gemm_fp8_ptpc.s
  metadata.txt
  instruction_mix.txt
```

抽出は build の per-TU object から行った。

```sh
B=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin
$B/llvm-objcopy --dump-section .hip_fatbin=out.fatbin <object>.o
$B/clang-offload-bundler --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
  --input=out.fatbin --output=out.gfx1201.o
$B/llvm-objdump -d --disassemble-symbols=<symbol> out.gfx1201.o
```

対象 object と source:

| kernel | object | source |
| --- | --- | --- |
| `shelf_attention_paged_bf16` | `attention/paged_decode.hip.o` | `src/shelf/kernels/optimized/attention/paged_decode.hip` |
| `gemm_bf16_wmma_kernel<4>` | `linear/bf16.hip.o` | `src/shelf/kernels/optimized/linear/bf16.hip` |
| `qsa_topk_batch_kernel` | `qsa.hip.o` | `src/shelf/kernels/optimized/qsa.hip` |
| `gemm_fp8_ptpc_kernel` | `linear/fp8_ptpc.hip.o` | `src/shelf/kernels/optimized/linear/fp8_ptpc.hip` |

## 2. 測定条件

通常の 5-run p50 は 1202.36 / 1208.92 ms、1703.31 / 1694.07 tok/s。カーネル別の
share と時間は [../../perf/current.md](../../perf/current.md) の pp2048 節にある。
ISA 調査用の trace は `rocprofv3 1.3.5 --kernel-trace` の 1 run で、trace 下の wall は
1280.83 ms（profiling overhead で通常より 6.5 % 大きい）。上位 4 カーネルの share は
attention 18.7 %、BF16 WMMA 17.0 %、QSA top-k 14.6 %、FP8 PTPC 12.5 % である。
4 カーネルとも SGPR/VGPR spill は 0 で、register pressure による溢れは起きていない。
問題は spill ではなく、CTA の並列度、K/V や tile の再読、選択アルゴリズムにある。

## 3. 所見サマリ

| kernel | 問題 | 直接の対策 |
| --- | --- | --- |
| `shelf_attention_paged_bf16` | token mask があるため prefill kernel が選ばれず、1 (row, q_head) 単位の decode kernel が KV を q_head ごとに再読する | mask 対応の prefill kernel、または q_per_kv 個の q_head を 1 CTA で共有 |
| `gemm_bf16_wmma_kernel<4>` | 16 出力 × 64 行の tile で activation を出力 tile ごとに再読する。K 境界の masked load 制御がホットループに残る | 出力 tile を広げて activation を LDS/L2 で再利用。K 整列時の専用経路 |
| `qsa_topk_batch_kernel` | 1 work-item が visible=512 個から k=512 個を選択ソートする。今回は全 block が選ばれるため選択自体が不要 | `visible <= k` なら 0..visible-1 を直接書き、残りを -1 にする |
| `gemm_fp8_ptpc_kernel` | 16 × 16 tile で A と W を CTA ごとに読む。出力 tile 数だけ再読する | LDS tiling と CTA あたりの出力 tile 拡大 |

## 4. 個別の調査

### 4-1. `shelf_attention_paged_bf16`（share 18.7 %）

Qwen4Exp の QSA は attention に per-query-row の token mask を渡す。
`plan_paged_attention` は `token_mask != nullptr` のとき `use_prefill=false` にして
tiled prefill と split decode を無効化するため、2048 token の prefill でも
per-(row, q_head) の decode kernel が使われる。

この kernel は 1 CTA = 8 wave、1 CTA が 1 つの (row, q_head) を担当する。
各 wave は visible token を 8 個おきに処理し、K/V を global から読む。
q_heads=24、kv_heads=2、head_dim=256 なので、同じ KV head を 12 q_head が
それぞれ独立に読む。

1 layer あたりの KV read 量（1 rank）:

```text
rows 2048 × q_heads 24 × visible 2048 × head_dim 256 × (K+V) × 2 B
  ≈ 96 GiB / layer
12 attention layer ≈ 1.13 TiB / rank
```

実測は 12 layer で約 204 ms/rank（1 layer 約 17 ms）なので、L2 帯域に換算すると
約 5.6 TiB/s になる。KV 4 MiB/layer は L2 8 MiB に収まるため、DRAM ではなく
L2 再読が律速である。

ISA でも K/V を LDS に staging する `ds_load` は見当たらず、hot path は
`global_load_u16` と `v_fmac_f32` である。`ds_*` は softmax の reduction 用で、
K/V の共有には使っていない。

対策は、mask を適用できる prefill kernel を用意し、1 CTA で複数の q_head
（q_per_kv = 12）と複数 row を処理して K/V tile を LDS で共有すること。
既存の `shelf_attention_paged_prefill_bf16` に mask を追加するのが最短である。

### 4-2. `gemm_bf16_wmma_kernel<4>`（share 17.0 %）

1 CTA = 1 wave、出力 16 列 × 64 行、LDS 0、VGPR 64。`kRowTiles=4` で weight を
4 行 tile 分共有するが、出力方向の tile は 16 列のままである。

hyper connection の down は rows=2048、out_features=320、k=10240。
activation は 2048 × 10240 × 2 B = 40 MiB で、grid.x = 320 / 16 = 20。
1 出力 tile ごとに 40 MiB を読むため、activation read は約 800 MiB。
実測約 1.0 ms は 800 GB/s で、DRAM 640 GB/s を超える分は L2 hit と見られる。

up は out_features=10240、k=320 で、activation は 1.3 MiB と小さいが、
grid.x = 640 なので同じく約 800 MiB を読む。こちらは L2 に収まるため
実測約 0.6 ms（約 1.3 TiB/s）と down より速い。

ISA は静的 3516 命令のうち scalar 2091、global load 204、WMMA 10。
`s_cbranch_execz` 200、`s_and_saveexec_b32` 179、`s_wait_alu` 498 が目立つ。
`load_bf16x4_masked` の境界判定と整列判定が K tile ごとに残っており、
K が 16 の倍数で整列している production shape でも分岐マスクを実行している。

また layer 1 の PLE key projection に 1 回 4.5 ms の呼び出しがある。
rows=32768、k=160 の形で、WMMA tile に対して K が短く、output が大きい。
これは上位 4 カーネルの集計には埋もれるが、単体では別に調べる価値がある。

対策は、出力 tile を 16 列より広げて activation を CTA 内で再利用すること、
K が整列・全長のときの masked load なし経路を用意することである。

### 4-3. `qsa_topk_batch_kernel`（share 14.6 %）

metadata は `max_flat_workgroup_size=1`、VGPR 5、LDS 512 B。
1 CTA = 1 work-item が 1 row を担当する。1 CTA が 1 lane なので、
1 CU あたり最大 32 CTA でも 32 lane しか埋まらず、memory latency を隠せない。

Flash Next の indexer は `indexer_budget=2048`、`indexer_compress_ratio=4` なので
2048 token では visible=512、k=512 になる。`visible <= k` の fast path は
`used[512]` を LDS に置き、512 回の選択それぞれで 512 block を走査する。
1 row あたり約 262k 比較、2048 row で約 537M 比較である。

実測は 1 呼び出し 13.3 ms、12 layer で約 160 ms/rank。
ISA は 246 命令のうち scalar 186、vector 15、LDS 34 で、1 lane のループである。

重要なのは、`visible <= k` のとき選ばれる集合は visible block の全部になることである。
`qsa_expand_mask_kernel` は selected 配列を membership 判定にしか使わないため、
順序は結果に影響しない。したがって `visible <= k` では
`out[j] = j`（j < visible）、残り -1 でよく、スコアの選択ソートは不要である。
これでこの workload の 14.6 % をほぼ消せる。

順序付きの selected 列が将来必要になった場合も、1 work-item の選択ソートではなく、
wave 並列の bitonic sort か radix select に置き換えるべきである。

### 4-4. `gemm_fp8_ptpc_kernel`（share 12.5 %）

1 CTA = 1 wave、16 行 × 16 出力、LDS 0、VGPR 38。
A（activation）と W（weight）を CTA ごとに読む。出力 tile をまたいだ再利用はない。

例: GDN の in_proj_qkv は rows=2048、out_features=10240、k=2560。
activation は 5.2 MiB で、grid.x = 10240 / 16 = 640。
activation だけで 640 × 5.2 MiB ≈ 3.3 GiB を読む。weight も別に読む。
実測 1.6 ms は合計で数 TiB/s の L2 帯域に相当する。

ISA は 463 命令のうち WMMA 4、global load 29、scalar 232。
128-bit の A/W load はできているが、CTA 間で同じ A/W tile を共有していない。
LDS tiling か CTA あたりの出力 tile 拡大で再読を減らす必要がある。

## 5. 優先順位

1. **QSA top-k の `visible <= k` 高速化**。実装が小さく、14.6 % を狙える。
2. **mask 対応 prefill attention**。18.7 % の主因で、K/V の q_head 共有が本質。
3. **BF16 / FP8 GEMM の tile 再利用**。合計 29.5 % で、効果は大きいが実装量も大きい。

## 6. 注意

- ISA の命令数は静的コード全体の数であり、動的な実行回数ではない。
- 帯域の見積りは shape と読み込み回数からの概算であり、実測 counter ではない。
- trace は profiling overhead を含む 1 run であり、通常の 5-run p50 とは区間が違う。

## 7. 実装後の状態

この調査の後、次の 4 点を実装した。

1. QSA top-k: `visible <= k` の short path と radix select。selected の順序は
   score 降順・同点は小さい index を保つ。
2. attention: bf16 prefill kernel に per-row token mask を適用し、masked prefill を
   有効にした。fp8 KV は従来の decode kernel のまま。
3. BF16 WMMA: 出力 tile を 4 倍（workers >= 512 のとき）にし、activation の再読を
   減らした。小さい rows は従来 tile に戻す。
4. FP8 PTPC: 出力 tile を rows に応じて 1 / 4 / 8 倍にし、activation の再読を減らした。

各段階の p50 と最終的な kernel 内訳は
[../../perf/current.md](../../perf/current.md) の pp2048 節を正本とする。
ISA の再取得は `build/isa/pp2048_tp4/README.md` の手順で行える。
