# Qwen4Exp MoE expert offload と頻度 cache

## 目的

`models/Qwen3.8-Flash-Next-Quark-MXFP4` の TP=2 は、MTP を除き LM head を量子化した
本体が約 68.2 GiB になる。1 rank あたり約 34.1 GiB で、実測 VRAM 31.87 GiB を超える。
routed expert の一部を pinned host へ逃がし、VRAM の slot と頻度で入れ替える。

## 配置

- layer ごとに `SHELF_MOE_HOST_EXPERTS=N` で cold expert を pinned host に置く。
  残りの `local_experts - N` は VRAM の slot に置く。
- TP=2 の `local_experts` は 256。MXFP4 の 1 expert は gate/up/down 合計
  2,611,200 B（2.49 MiB）である。
- host plane は `hipHostMalloc(hipHostMallocMapped)`。GPU の grouped GEMM が PCIe 越しに
  読む。CPU GEMV は持たない。
- slot map は device int32 `[layers][local_experts]`。`>= 0` は VRAM slot、
  `-1 - slot` は host slot。router 後の plan scan がこの map で group を hit/miss に分ける。
- hit と miss は同じ grouped GEMM を 2 回起動する。`group_row_base` で元の group 行を
  指すため、activation plane と output は 1 本の plan のまま共有する。
- miss の重みは host plane から読む。cache の入れ替えは step 間だけで、推論中の
  host 同期は増やさない。

## 頻度 cache

- router は所有 expert ごとの route 数を device の `expert_route_freq` に積む。
- batch 完了後、`Executor::after_batch` が cache の `step()` を呼ぶ。
- `frequency += batch routes`、peak が `SHELF_MOE_CACHE_DECAY`（既定 4096）を超えたら
  全体を半分にする。
- hot host expert と、その step に使われていない cold VRAM expert を交換する。
  1 step の交換数は `SHELF_MOE_CACHE_INSERTS`（既定 0 = 交換しない）。
- 交換は「demoted を VRAM から host へ退避 → promoted を host から VRAM へ upload →
  demoted を host slot へ書く」の順で行う。VRAM slot と host slot は同じ index で
  gate/up/down の 3 plane をそろえる。
- `SHELF_MOE_CACHE_LOG_INTERVAL`（既定 64）ごとに hit rate と swap 数を stderr に出す。

## メモリ

`test_qwen4exp_load --skip-ple --rank 0 --device 2 --arena-gib 31` の実測:

| host experts / layer | rank の VRAM 重み | arena used |
| ---: | ---: | ---: |
| 0（全 VRAM） | 約 34 GiB（TP=2 では 31.87 GiB に載らない） | - |
| 48 | 27.572 GiB | 30.709 GiB |
| 64 | 25.702 GiB | 28.839 GiB |
| 96 | 21.961 GiB | 25.098 GiB |

`host_experts=96` の shelf-compute では KV に約 4.59 GB を確保できた。
`host_experts=64` では約 0.57 GB だった。

## 正しさ

prose prompt 38 token、greedy 8 token で、`host_experts=64` と `96` の生成 ID が一致した。

```text
8068,21473,45543,13771,942,19670,11106,4744
```

配置が違っても同じ重みを読むため、結果が一致することを確認した。
`test_qwen4exp_load` は 64 host / 96 host の双方で PASS した。

## hit rate

`host_experts=96`（VRAM 160 / host 96）で `SHELF_MOE_CACHE_LOG_INTERVAL=1` の 8 step:

```text
step=1 hit=61.09% swaps=4
...
step=8 hit=62.82% swaps=32
```

VRAM 在籍率 160/256 = 62.5% とほぼ同じで、8 step では頻度 swap の効果はまだ出ていない。

### routing の偏り（TP=1、`host_experts=384`）

`context 3072` / 128 token を 152 step 回し、累積の routing から配置の上限を測った。
1 step の routing は 48 層 × top_k 10 × 128 token = 61 440。

| 指標 | 値 |
| --- | ---: |
| VRAM 在籍率 | 25.0 %（層あたり 128 / 512） |
| ロード時の配置（expert 0..127）のヒット率 | 21.2 % |
| 層ごとに上位 128 を選んだ場合 | 95.4 % |
| 全体で上位 6 144 pair を選んだ場合 | 97.8 % |
| 上位 2 % / 5 % / 10 % / 25 % の pair が占める routing | 48.1 / 69.2 / 85.0 / 97.7 % |

一様なら 25 % になるはずのヒット率が 21.2 % で、偏りは非常に強い。**在籍率ではなく
置き場所が律速**である。同じモデルで 20.8 % という値が `docs/PREFILL_HEAD_ALIAS.md` に
あるが、あれは 2 % 在籍（508 / 24 576 pair）での値で、比較できる数字ではない。

### 交換の実装をやり直した

- `swap()` が slot map を逆に更新していた。VRAM に入るのは host 側の expert なのに、
  退避した expert を VRAM 在籍として記録していた。このため交換してもヒット率が動かず、
  交換後の計算は誤った重みを読んでいた（`INSERTS` を使った以前の計測はすべて無効）。
- 交換の判断材料を「その step に使われたか」から「累積頻度の順位」に変えた。
  128 token の step では層あたり 1 280 回の routing が 512 expert に散り、
  「未使用の VRAM expert」がほぼ存在しないため、旧方式では交換候補が作れなかった。
- 交換を step あたりの本数ではなく、**層ごとのランキング差分**としてまとめて計画し、
  複数 step に分けて消化するようにした。

| 設定 | 既定 | 内容 |
| --- | ---: | --- |
| `SHELF_MOE_CACHE_REBALANCE_STEPS` | 4 | 再評価の間隔（step） |
| `SHELF_MOE_CACHE_SWAP_BUDGET` | 96 | 1 step の交換数 |
| `SHELF_MOE_CACHE_BURST_SWAPS` | 512 | batch が大きい step（routes > 層×expert）の交換数 |
| `SHELF_MOE_CACHE_MIN_GAIN` | 150 | 交換に必要な頻度比（%） |
| `SHELF_MOE_CACHE_DECAY` | 4096 | 累積頻度のピークがこれを超えたら全体を 0.7 倍 |

2026-11-19 の修正で、route 数が 層×local expert を超える batch は計画した交換をその
step で全数消化するようになり、`SHELF_MOE_CACHE_BURST_SWAPS`（`--moe-cache-burst`）は
削除された。経緯は
[../20261119/moe_cache_chunk_dependent_decode.md](../20261119/moe_cache_chunk_dependent_decode.md)。

結果（`--tokens 128`、交換後）: ヒット率 21.2 % → 累積 69.3 %（直近区間 92.8 %）、
per-token 68.9 ms → 38.0 ms。`--tokens 256` では累積 70.1 %（直近 92.6 %）/
38.6 ms。再配置は約 4 900 組、PCIe 約 26 GB の一括コスト（約 1.0 s）で、
prefill 中に消化される。

- ACCEPT: 累積頻度の順位による再配置。上限 95.4 % に対し 92-93 % に達する。
- ACCEPT: 4 step ごとの再評価と min-gain 1.5。収束後は交換が発生せずコストが無い。
- REJECT: その step の使用有無による 1 対 1 交換。候補が作れず、map のバグと相まって
  ヒット率を上げられなかった。
- REJECT: `SHELF_MOE_CACHE_INSERTS` を増やす方向（上限 64 でも改善しない）。
- 未検証: 層をまたいだ全体ランキング（層ごとの上限 95.4 % に対し全体は 97.8 %）。
  現状の在籍は層ごとに固定なので、層ごとの割り当てを変える設計が要る。

### CPU 側で miss を計算する場合の上限（AVX512）

`moe_intermediate 640` / `hidden 2560` の MXFP4 expert（2.46 MB）を、DDR5 上の
512 expert からランダムに 36 個読んで 1 行 GEMV する条件の実測:

| スレッド | 帯域 | 36 expert の処理 |
| ---: | ---: | ---: |
| 1 | 13.8 GB/s | 6.81 ms |
| 2 | 26.7 GB/s | 3.52 ms |
| 4 | 50.5 GB/s | 1.86 ms |
| 6 | 68.5 GB/s | 1.37 ms |
| 8 | 68.6 GB/s | 1.37 ms |

PCIe（25 GB/s）で同じ 36 expert を読むと約 3.5 ms なので、帯域だけ見れば CPU が
1.9-2.6 倍有利である。1 スレッドではレイテンシ律速で伸びない。
ただし後段の実測のとおり、node 単位の直列化と CPU 仕事量そのもののために
この帯域差は per-token の短縮に結び付かなかった（[CPU で miss を計算する案](#cpu-で-miss-を計算する案reject)）。

## 制約

- 現状は **MXFP4 expert のみ**。EXL3 expert は `SHELF_MOE_HOST_EXPERTS>0` で
  `unsupported` を返す。EXL3 は trellis / suh / svh の host plane と grouped kernel の
  split 対応が必要になる。
- 長文 KV、prefix cache、DFlash2、KLD / top-1 は未測定。
- offload は既定で無効（`SHELF_MOE_HOST_EXPERTS=0`）。

## CPU で miss を計算する案（REJECT）

PCIe（実測 25 GB/s）より CPU の DRAM 帯域（4 スレッド 50 GB/s）が広いため、
host 常駐 expert の GEMV を AVX512 で肩代わりさせる案を実装して測った。
AVX512 GEMV と同期経路は GPU 経路とビット一致まで詰めたが、性能が出ないため撤去した。

| 方式 | per-token（`--context 256 --tokens 4`、host expert 384） |
| --- | ---: |
| GPU のみ（host plane を PCIe で読む） | 68.5-79.5 ms |
| CPU 同期（行ごとの blocking copy + host 同期） | 226.2 ms |
| CPU 非同期（pinned 先読み + 1 block wait） | 127.4-134.0 ms |

- CPU 仕事量そのものが 4 token で 315 ms（約 79 ms/token、574 node、0.55 ms/node）。
  GPU のみの 1 token 予算と同程度で、重ねられても parity が上限になる。
- 重ねられるのは node 内の hit plane の区間だけである。stream は
  `begin`（staging + ready）→ hit plane → `wait` → `scatter` の順に進むので、
  CPU が hit plane より長い分はそのまま加算される。stream が `wait` で止まるため
  CPU が先の node へ走ることもできない。
- staging は `row_capacity` 全体（1 token で 1.3 MB）を転送しており、必要な行
  （10 行で 80 KB）の 16 倍だった。plan を device で読む gather にすれば削れるが、
  上の 2 点（CPU 仕事量と窓）が支配的なので撤去を選んだ。

### 途中で確定した実装上の規則

1. 全グリッドの GPU spin は他ストリームの D2H コピーを止める。1 block なら
   121 us で完了するコピーが、2048 block のスピン中は解放まで 3.0 s 待たされる。
2. pinned へ `hipMemcpyAsync`（DMA）で書いた内容は CPU から見えない。
   8 KB のコピーで全語が不一致になり、1 launch 前の値が見えることもある。
   `hipHostMallocCoherent` / `Mapped` / `NonCoherent` のいずれでも変わらない。
3. kernel ストアは CPU から見えるが、コピー kernel とフラグ kernel を分けると data は
   見えないままになる。同一 kernel 内で `__threadfence_system()` の後にフラグを
   書いた場合だけ一致する（8 KB、4 round、mismatch 0）。
4. `hipExtMallocWithFlags(..., hipMallocSignalMemory)` はこの環境で
   `invalid argument` を返す（`hipDeviceAttributeCanUseStreamWaitValue` は 1）。
   signal memory を使う `hipStreamWaitValue32` 方式は採れない。

## Qwen4Exp で DFlash2 を動かすまでの修正

このモデル（Qwen3.8-Flash-Next）で DFlash2 を有効にすると、次の 4 箇所で止まった。

| 症状 | 原因 | 対応 |
| --- | --- | --- |
| `proposal shard head encoding is unsupported`（TP=2） | lm_head が BF16。sharded proposal は Int8W32 / Exl3 のみ | lm_head を load 時に INT8 W32 化（既定 ON、`--lm-head-int8 0` で無効）＋ shard に BF16 exact-rows 経路を追加 |
| `Qwen4Exp matrix has an unexpected encoding: lm_head`、`lm_head is not EXL3 or BF16` | Qwen4Exp の load 検証が Int8W32 を拒否 | 検証を Int8W32 許容に拡張 |
| `the PLE input gather needs host token ids`（TP=1） | speculative verify の batch が device 側 token ids を使う | PLE モデルでは host token ids を強制（`DFlash2SpecDecoderConfig::host_token_ids` を Qwen4Exp で有効化） |
| `Qwen4Exp requires --tensor-parallel 2 or 4` | compute runtime の制限 | TP=1 を許可（tg と同じ単 GPU 構成） |

結果（prompt 4096、`octave:c5v5`、drafts 6、tree-budget 0、単一リクエスト）:

| 構成 | decode | 受理 |
| --- | ---: | ---: |
| TP=2 spec なし | 15.19 tok/s（65.8 ms/token） | - |
| TP=2 DFlash2 | 38.70 tok/s（25.8 ms/token） | full 0.636 / 平均 4.18 drafts |
| TP=1 DFlash2 | 46.58 tok/s（21.5 ms/token） | full 0.889 / 平均 5.33 drafts |

prompt が同一 token の繰り返し（id=1）なので受理率は実運用より高い。TP=2 の DFlash2 は
`DFLASH2_D2H_MS` が 11 round で 111 ms あり、host token ids の往復が次の削減対象である。

## host plane を別 stream で重ねる案（REJECT）

miss（host plane）は PCIe 律速、hit（resident plane）は演算律速なので、2 本の stream に
分けて重ねれば PCIe を隠せると考えて実装した（executor に aux stream と 2 つの event、
`--moe-miss-overlap`）。計測は `shelf-bench tg --context 3072 --tokens 128
--moe-host-experts 384`。

| 実行 | per-token | p50 | LOGITS |
| --- | ---: | ---: | --- |
| overlap=0 | 37.76 ms | 37.47 ms | max=21.077881 sum=-552621.438 |
| overlap=1 | 37.91 ms | 37.53 ms | max=21.826017 sum=-580326.312 |
| overlap=0 | 37.65 ms | 37.42 ms | max=21.077881 sum=-552621.438 |
| overlap=1 | 37.86 ms | 37.58 ms | max=21.077881 sum=-552621.438 |

- 性能差は ±0.5 % で、重ねても PCIe は隠れなかった。miss の読み出しは hit と同じ
  kernel（同じ SM 資源）を奪い合うため、帯域ではなく SM が律速になっている。
- 同じ config の overlap=1 で LOGITS が 1 回だけ変わった。event 待ちが効かず plan を
  読み違える競合が疑わしい（`shelf-bench` は legacy default stream を使う）。
  再現性が無いため不採用とし、実装は stash に退避した。
- 副産物: 最初の測定（96.7 ms/token）は同一ホストで走っていた別の pp 計測による
  PCIe 競合で汚染されていた。単独実行では再現しない。
