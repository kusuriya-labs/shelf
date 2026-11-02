# Resident Model Infrastructure

GPU model weight の lifetime を compute process から分離するための基盤。
Resident Model Host が model weight を GPU memory 上に保持し、compute/runtime
process は HIP IPC で attach する。主目的は model reload の削減と、
compute failure から weight lifetime を分離することである。

test infrastructure は本機構の consumer の一つであり、GPU test session では
同一 resident weight を複数 worker から再利用する。

## failure domain

```
Model lifetime   ≠   Compute lifetime   ≠   Request lifetime
```

| layer | 所有するもの |
|---|---|
| Resident Model Host | target の Qwen weights、TP rank weights、MTP weights、DFlash2 / EAGLE4 drafter weights、read-only metadata |
| Compute / runtime process | KV cache、GDN state、SequenceSlotPool、Executor、workspace、activation、drafter executor state、sampling state、stream / event |
| Request | server request state、batch state |

target model は必ず host 経由でロードする。EAGLE4 drafter も host 経由でロードし、
compute process は HIP IPC で attach する。DFlash2 drafter は compute process が
disk から local load して自分の ephemeral arena で保持する。
したがって同一 target を複数の compute process が共有する構成
（例: Resident → Compute1（DFlash2）/ Resident → Compute2（EAGLE4)）が成立する。

host 側は test session runner 用に dflash2 kind の cache を保持できる
（`--warmup-dflash2`）。production の compute 経路はこれを attach しない。
EAGLE4 は `--warmup-eagle4` で rank 0 の host に常駐させる。

compute process が crash / watchdog timeout / assert / SIGKILL / logic hang で
死んでも weight は host 側に残り、新しい compute process は HIP IPC attach で即復帰する。

GPU device reset、ROCm driver reset、device lost、GPU context 全破壊、
health check 不能時は host 自身の allocation も失われるため、
session を fail-closed に終了して model reload を許可する。

## 構成

```
Resident Model Infrastructure
│
├─ Resident Model Host          （internal runtime component）
│    ├─ weight load
│    ├─ compact resident block
│    ├─ HIP IPC
│    ├─ model cache
│    ├─ HEALTH / STATS
│    └─ RELEASE / SHUTDOWN
│
├─ ModelSource                  （runtime capability）
│    ├─ SafetensorsModelSource  … 既存どおり disk から load
│    └─ ResidentModelSource     … host から attach
│
├─ Runtime Client
│    ├─ shelf-compute
│    ├─ host backend executable
│    └─ TpRankRuntime
│
└─ Test Session Runner          （consumer）
     ├─ CTest / regression orchestration
     ├─ worker timeout
     ├─ SIGKILL
     ├─ profile grouping
     └─ disk-load-count assertion
```

### shelf-model-host

`shelf-model-host` は internal runtime component の executable である。
`cmake/apps.cmake` の 4 user-facing entrypoint（`shelf-compute` /
`shelf-launcher` / `shelf-server` / `shelf-bench`）には含めない。
`shelf-launcher` と session runner が起動し、`shelf-server` と
`shelf-compute` が起動することはない。

必須ではなく、従来経路のまま選択できる。

```
従来:   shelf-compute  → 自分で safetensors load
resident: shelf-model-host → HIP IPC → shelf-compute
```

`shelf-launcher` → `shelf-model-host` と
`shelf-launcher` → `shelf-server` → `shelf-compute` という関係であり、
server / compute に model 管理機能は戻さない。

```
Model Host  = GPU weight ownership
Compute     = 計算状態
Server      = HTTP / API
```

### ModelSource

`Qwen35ComputeRuntime::initialize()` や `TpRankRuntime::create()` は
`ps::models::ModelSource` を通して **target model** を得る。
ModelSource は Qwen35 target のみを扱う。EAGLE4 drafter は
`ResidentModelAttachment::take_eagle4()` で別途 attach する。

```cpp
ModelLoadContext context;
context.qwen_model_dir = ...;
context.model_host_socket = ...;   // nullopt = local load
auto source = ModelSource::create(context);
...
arena = GpuArena::create(device, source->ephemeral_bytes(arena_bytes));
model = source->load_qwen35(arena, stream, options);
```

`model_host_socket` の有無で `SafetensorsModelSource` と `ResidentModelSource`
が選ばれる。choice は process ごとに 1 回だけ解決され、
`initialize()` 内で環境変数を読まない。選択は startup path のみで、
decode hot path に branch を追加しない。

drafter の扱いは kind で異なる。EAGLE4 は compute process が
`ResidentModelAttachment` を取得して attach する。DFlash2 と、EAGLE4 が
resident 不可の場合は compute process が local load して保持する。
`--dflash2-model-dir` / `--eagle4-model-dir` は target の source 選択と独立である。

## 利用方法

### launcher

`shelf-launcher` が host lifetime の owner である。launcher は rank ごとに
host を起動し、すべての host が protocol ping に応答した後に child
（`shelf-compute` / `shelf-server` / `shelf-bench`）を
`--model-host` 付きで起動する。child の終了時に全 host を shutdown する。

ターゲットモデルは必ず host 経由でロードされる。EAGLE4 drafter は rank 0 の host に
常駐し、compute が attach する。DFlash2 drafter は child が local load する。
[user/launcher.md](../user/launcher.md) を参照。

### compute

`shelf-compute` は host の socket を `--model-host` で直接指定できる。
`--model-host` を指定しなければ `SHELF_MODEL_HOST_SOCKET` を見る。
どちらも無ければ従来どおり自分で safetensors を load する。

### server

`shelf-server` は `--model-host` と `--tp` を受け取り、
`shelf-compute` へ透過する。launcher 経由で使うのが通常である。

### environment

| 変数 | 役割 |
|---|---|
| `SHELF_MODEL_HOST_SOCKET` | model host の socket。単一 path か `0=<path>,1=<path>` の index map |
| `SHELF_DISABLE_RESIDENT_MODEL` | 設定すると resident を無効化し常に local load |
| `SHELF_TEST_STATS_SOCKET` | test 検証専用。worker が STATS を読む先 |

index は `tp_size > 1` なら `tp_rank`、それ以外は worker の device index である。
`HIP_VISIBLE_DEVICES` の remap を考慮し、worker は host の physical device 番号を
自分の device へコピーしない。IPC handle の open は worker 自身の device 行上で行う。

## compact resident block

`GpuArena::create()` の非 VMM 経路は `hipMalloc(capacity_bytes)` であり、
`used()` ではなく `capacity()` 全体を物理確保する。そこで model host は
staging arena を使い、used prefix だけの compact block を作る。

1. 既存 loader で staging arena に model をロードする。
2. `GpuArena::used_prefix_view()` で used prefix の base / bytes を得る。
3. すべての weight tensor / `MatrixWeight` descriptor を収集する
   （`resident_format.cpp` が walk 順を単一の定義として使う）。
4. used prefix を host bounce buffer へ D2D→H2D で退避する。
5. staging arena を shutdown する。
6. `round_up(used, 2MiB)` の `hipMalloc` block を作り、bounce buffer を H2D する。
7. descriptor から `Qwen35ModelWeights` / `DFlash2Weights` を block 上に再構築する。
8. 再構築物から descriptor を再収集し、手順 3 のものと byte 単位で比較して検証する。
9. `hipIpcGetMemHandle` を取得し、block だけを保持する。

手順 4-6 の bounce buffer は D2D copy の 2×used ピークを避けるためである。
`GpuArena::base_` は private のままであり、production code へは公開しない。
used prefix を返す view のみを追加している。

## ResidentModelKey

model host は model directory だけで cache key を決めない。

| 項目 | 内容 |
|---|---|
| canonical model directory | `std::filesystem::weakly_canonical` 済み |
| model kind | `qwen35` / `dflash2` / `eagle4` |
| load flags | verify CRC、load MTP layers、preshuffle、quantized dir |
| tp_size / tp_rank | Tensor Parallel の rank 指定 |
| device | key を解決した host の device |
| fingerprint | 上記の FNV-1a |

fingerprint が一致した場合のみ同じ resident block を返す。
host は key の fingerprint を再計算して検証し、異なる load option を
同じ resident block として誤共有しない。

## protocol

Unix domain socket の test にも production にも使える小さな protocol。
HTTP / gRPC は使わない。

frame は 24-byte header（magic、version、command、status、payload_size）と payload。

| command | 内容 |
|---|---|
| `HELLO` | proto version、host device、host 合計 resident bytes |
| `ACQUIRE_MODEL` | key を受け取り IPC handle + manifest を返す（未 load ならここで load） |
| `RELEASE_MODEL` | key（または STATS の `entry key=` 文字列）の entry を解放 |
| `HEALTH` | device に対する小さな HIP 操作の成否 |
| `STATS` | key、disk load count、attach count、resident bytes、device、last health |
| `SHUTDOWN` | host を終了する |

manifest は versioned binary で、C++ object の raw `memcpy` は使わない。
Tensor / MatrixWeight / config を `resident_format.cpp` が field 単位で serialize する。

## ephemeral budget

resident mode の worker は次の式で ephemeral arena を作る。

```
ephemeral = requested_budget - persistent_bytes
persistent_bytes = max(自分が attach した resident bytes,
                       HELLO で受け取った host 合計 resident bytes)
```

`requested_budget` は従来使っていた arena 容量そのものであり、
`--arena-gib` の CLI semantics は production 向けに変更していない。
resident mode の内部導出のみである。`--arena-gib auto` では
`requested_budget = persistent_bytes + (空き VRAM - 1 GiB)` として、
resident 分を capacity に含めたうえで ephemeral を導出する。

`persistent_bytes` が `requested_budget` 以上になる場合は fail-closed で error にする。

drafter weights の扱いは kind で異なる。EAGLE4（約 5.8 GiB の checkpoint）は
resident 分として `persistent_bytes` に含まれ、ephemeral arena を消費しない。
DFlash2（約 3.6 GiB の checkpoint）は resident 分ではないため、
compute process の ephemeral arena から消費する。

EAGLE4 の checkpoint は host が完全な形（embed / lm_head を含む）で保持する。
compute は `--eagle4-lm-head draft` のとき target の private lm_head を要求しない。

## health check と crash 隔離

1. worker 起動
2. deadline 超過 / crash / assert / hang
3. worker process group のみ `SIGKILL`
4. ephemeral allocation / context は OS と ROCm driver に委ねる
5. model host は kill しない
6. health check（host alive、HIP runtime call、resident allocation 有効、
   短い stream 操作の完了、timeout 付き）
7. healthy なら次 worker を attach で起動
8. unhealthy なら session を終了する

model host は test kernel を実行しない。health check は `hipMalloc` /
`hipMemset` / `hipStreamSynchronize` だけを使う。

## model cache と release

自動 LRU は作らない。host は key 単位で cache し、`RELEASE_MODEL` で解放する。

- 同じ key なら disk load は 1 回、attach は何回でも可
- profile / load option が変わると別 key になり、必要なら load が増える
- memory pressure で load が失敗した場合、host は既存 entry を
  明示的に release して 1 回だけ retry する（LRU ではない）
- cache は無制限に増えない

test session では runner が test を model profile ごとに group 化し、
profile 切替時に release する。

## 段階導入

### Phase 1（現在）

Resident Model Infrastructure を実装し、test session runner が使う。
大量の GPU テストと server regression が reload から解放される。

### Phase 2

`shelf-compute --model-host ...` による任意利用。
従来の local load 経路はそのまま残すため互換性がある。

### Phase 3

安定後、server 構成の標準運用候補とする。
server が model host を起動することはない。

## 対象 test

resident fast path の対象は「同じ正しい重みを何度も読み直してから runtime を
検証する」test である。

- host backend の runtime / E2E / perf external test
- DFlash2 gate 系、target taps、TP E2E
- `shelf-compute`（`--serve-stdio`、single-shot、server 経由）

対象外:

- loader correctness test（`test_weight_load`、`test_qwen35_tp_weight_load`、
  `test_qwen35_mtp_weight_load`、`test_qwen35_mtp_weight_real`、
  `test_dflash2_weight_real`、`test_dflash2_weight_contract`）
  load path 自体が検証対象なので従来どおり safetensors を読ませる。
  これらは `SHELF_DISABLE_RESIDENT_MODEL=1` でも強制できる。
- model を読まない CPU contract / small kernel / substrate test
  `shelf-required-tests` は resident 非依存のまま自己完結を保つ。

## 関連

- test session runner の使い方は [testing.md](testing.md)
- inventory・採否判断・計測は `docs/rnd/resident_model/resident_model.md`
