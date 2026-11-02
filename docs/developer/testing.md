# Testing

## required acceptance

自己完結: repository + ROCm + 1 GPUのみ。外部ファイル不要。
canonical source は `tests/run_required_acceptance.py` である。runner は
`shelf-required-tests` target を build し、`required` label の CTest を実行する。

required の集合は `cmake/tests.cmake` の `required` label と
`shelf-required-tests` target が正本である。件数は build 構成
（benchmark build の有無）で変わるため、ここに固定しない。

一覧の取得:

```bash
ctest --test-dir build -L required -N
```

実行:

```bash
python3 tests/run_required_acceptance.py --build-dir build
```

**skip / failure / error はすべて acceptance failure である。** GPU 不足などで
test binary が exit 77 を返した場合も required acceptance では失敗として扱う。

required は CPU contract test（safetensors writer、FPX layout、量子化 payload /
codec、selector、architecture boundary 等）と、gfx1201 限定の GPU kernel /
runtime test、bench `--help` smoke からなる。
FP8 kernel には ISA 資源契約があり、`tools/check_kernel_spills.py` が
build の compile flags で kernel を assembly へコンパイルし、VGPR / SGPR spill と
private segment（local memory）使用が 0 であることを検証する
（`test_fp8_kernel_spill_contract`、label `cpu;required`）。
arch 非適合の kernel test は非適合 arch では CTest レベルで DISABLED にし、
external model 不在・GPU 不足などによる exit 77 は `SKIP_RETURN_CODE 77` で
CTest の Skipped として扱って、plain `ctest` が誤った FAIL を出さないようにしている。
Skipped は開発者ローカル向けの表示にすぎず、required acceptance は JUnit の
`<skipped>` を failure として数えるため通らない。

## test tree の区分

- `tests/unit` / `tests/kernels` — current の contract / regression / kernel 単体。
- `tests/server` — Python 経路の contract test。reasoning は
  `tests/server/test_reasoning_parser.py`（GPU / model 不要）と
  `test_reasoning_request.py`（model のみ）、E2E が `test_server_reasoning*.py` である。
  推論予算と thinking 予算は `test_inference_budget.py`（protocol / model のみ）と
  `test_compute_thinking_budget.py`（compute E2E）が固定する。
  `test_compute_thinking_budget.py` の DFlash2 経路は `SHELF_DFLASH2_MODEL_DIR` と
  `SHELF_TEST_TP_DEVICES`（例 `0,1`）を設定した時だけ実行され、未設定では skip を表示する。
  27B FP8 target に drafter を足すと 1 枚では収まらないため、DFlash2 経路は TP=2 の
  GPU 2 枚が必要な test である。KV auto-fit が drafter 分を確保計算に織り込むので、
  `--working-set-reserve-gib` の計算式は [runtime.md](runtime.md) の arena budget。
  この case は production と同じ 4 GiB を明示する。1 GiB まで下げると drafter ではなく
  plan 後の executor workspace が不足して起動しない。
  この case は `--arena-gib` を渡さず auto にする。TP2 の rank arena で固定値を渡すと
  物理 commit と競合して weight load が失敗するためである。
  decode 経路と DFlash2 経路の token 列は比較しない。TP の reduction 違いで
  分岐しうるため、each 経路で thinking 予算の contract だけを検証する。
- `tests/correctness` — CPU fixture と GPU correctness path の突き合わせ。
  `SHELF_MODEL_DIR` を要求する optional test である。
- `tests/quality` — quality regression の runner。`quality` label のみで、
  external model を要求する。性能 test とは分離する。

R&D Gate / 診断用 test は旧 `tests/rnd` に置いていたが削除済みである。
経緯は Git history を参照する。

## quality regression

性能 regression（`perf` label）とは別に、model の品質を測る `quality` label を置く。
速度の数値と混ぜず、品質の合否だけを扱う。

正本は `tools/quality_gate.py` である。BF16 KV と FP8 KV を同一 corpus・同一位置で
比較し、次を出す。

- teacher-forced perplexity（`shelf-bench quality --compare`）
- mean KL（BF16 の分布を p、FP8 を q とする）
- top-1 一致率 / top-5 overlap
- greedy 生成 token の prefix 一致率
- 固定 reasoning / code / gsm8k end-task の正解数

固定データ:

- corpus: `vendor/BetterBench/corpus/v1` の 8 category（chat / code / file_edit /
  json / math / prose / reasoning / summarization）を chat template で token 化して
  連結する。token corpus は commit せず、`run` が work dir に生成する
- end-task: `tests/data/quality/reasoning.json`（10 問、数値の完全一致）および
  `tests/data/quality/gsm8k.json`（openai/gsm8k test split 全1319問。既定は先頭50問、
  `--gsm8k-limit` で変更、末尾の #### 数値で照合）

実行:

```bash
SHELF_MODEL_DIR=models/Qwen3.8-27B-FP8 \
  python3 tests/quality/run_quality_gate.py

ctest --test-dir build -L quality
```

`SHELF_MODEL_DIR` が無い場合は exit 77 で skip する。GPU 30 GiB を予約し、
`test_quality_fp8_kv` として実行する。

report には corpus の token 数・SHA256・source を残す。BetterBench corpus と
tokenizer の版が変わると tuple が変わるため、perplexity の絶対値ではなく
同一 run 内の BF16 KV との差分を gate にする。corpus だけを生成する場合:

```bash
python3 tools/quality_gate.py corpus \
  --model-dir models/Qwen3.8-27B-FP8 --out build/quality/corpus.psktok
```

合否は絶対値ではなく BF16 に対する差分で判定する。既定の閾値は
ppl 劣化 1% / mean KL 0.05 / top-1 0.95 / 生成 prefix 一致 0.90 /
end-task は reasoning / gsm8k それぞれで BF16 から後退を許容し、
既定の許容問数は reasoning 1 問、gsm8k 2 問である。実測 baseline を採った後は
その値を記録し、必要なら閾値を厳しくする。
実行時間・tok/s は品質 gate の対象ではなく、性能値として引用しない。

dump 形式と PPL / KL / top-1 の計算は `test_quality_metrics`（required）が固定し、
fixture の回答照合と parse は `test_quality_gate_selftest`（required）が固定する。

### 採点位置の固定

`--context` を変えると既定の採点範囲（`context+1` 以降）も変わるため、ctx 間の
数値をそのまま比較できない。`--score-start-position N`（1 始まりの絶対 position）で
採点開始位置を固定すると、context だけを変えた比較ができる。prefill 後、採点開始
位置までは同じ corpus を decode で進め、ログ採点だけを遅らせる（KV / GDN 状態を
飛ばさない）。

```bash
./build/shelf-bench quality \
  --model-dir models/Qwen3.8-27B-FP8 --tokens-file "$CORPUS" \
  --context 1600 --score-start-position 1601 --positions 448 \
  --kv-cache-dtype fp8 --dump /tmp/fp8.psq --compare /tmp/bf16.psq
```

dump header の `reserved` に採点開始位置を記録する（version は 1 のまま）。
`--compare` は baseline の値と一致しない場合にエラーにする。`reserved` が 0 の
旧 dump は未記録として扱い、警告のみ出す。出力には `QUALITY_SCORE_START` と
`QUALITY_NLL_DELTA_MEAN` を含める。

### fp8 paged attention の単体テスト判定

`test_paged_attention` は bf16 / fp8 の両方で次を固定する。

- K を全て 0 にした一様 attention（V の layout と正規化の検査）
- V を全て 0 にしたケース（出力が 0 で非有限値を含まないこと）
- 長文脈 decode（`visible=8192`）の同条件

`test_paged_attention_split_reduce_exact` は全 split の max が `-FLT_MAX`、
sum が 0 の partial を与え、reduce が出力を書かず非有限値も残さないことを固定する。

`test_paged_attention` の `fp8 PREFILL` ケースは、参照計算に Q の行単位量子化を
含めたうえで、行単位の相対 L2（`||got-ref|| / max(1,||ref||)`）で判定する。
基準は mean 1e-2 以下かつ max 0.6 以下、および非有限値なしである。P の量子化は
実装仕様であり、要素単位の許容値では表現できない。非有限値の混入は必ず失敗させる
（NaN は要素単位比較が常に偽になるため、明示的な検査が必要）。

## CLI option 表の同期

`docs/user/` の option 表は `tools/gen_cli_docs.py` が各 entrypoint の `--help` から
生成する。手で表を編集しない。option を追加・変更したら次を実行する。

```bash
python3 tools/gen_cli_docs.py --build-dir build
```

`test_cli_docs_sync`（required）が compute / bench の表を、`test_cli_docs_sync_python`
（optional）が cli / server の表を `--check` で検査する。

## GPU 予約

`GPU_COUNT > 0` のテストは `shelf-gpu-test-runner` で wrap され、
GPU を論理 VRAM で予約してから実行する。予約できないままで
`SHELF_TEST_GPU_RESERVE_TIMEOUT`（既定 900 秒）を経過すると失敗する。

- 予約量は `GPU_COST_GB`（既定 1 GiB）。上限は各 GPU の実 VRAM
  （`/sys/class/drm/<node>/device/mem_info_vram_total`）から求める。
  明示指定は `--budget-gb` と `SHELF_TEST_GPU_BUDGET_GB`。
- iGPU は候補から外す。判定は `/sys/class/drm/<node>/device/firmware_node` の有無で、
  iGPU のみの環境では外した後の空リストを避けて iGPU を候補にする。
- `SHELF_TEST_GPUS` / `HIP_VISIBLE_DEVICES` が指定されている場合は
  明示指定として扱い、iGPU 除外を適用しない。

### GPU 負荷をかけている間の監視制約

GPU に負荷をかけている間は **SMU metrics を取得してはならない**。
`rocm-smi` / `amd-smi metric` / `/sys/class/drm/*/device/gpu_metrics` の読み出しは
SMU へ metrics export を要求し、応答が無いと次が出る。

```text
SMU: No response msg_reg: 12 resp_reg: 0
Failed to export SMU metrics table!
```

このとき得られた異常値から `GPU over temperature range(SW CTF)` が宣言され、
`System is going to shutdown due to GPU SW CTF!` でマシンが落ちる事故を観測している。

- 温度は **負荷実行の直前と直後だけ** で取る。実行中は打たない。
- 温度取得の正本は `amd-smi metric -t`（root 不要。EDGE / HOTSPOT / MEM を返す）。
  `rocm-smi --showtemp` は dGPU を返さない（iGPU のみ）ため正本にしない。

```bash
amd-smi metric -t        # 実行前
<負荷テスト>
amd-smi metric -t        # 実行後
```

### GPU ハング時の収手順

GPU が wedge して `devcoredump` が作られたら、**再起動する前に** 取る。再起動で消える。
`card` は PCI アドレスで選ぶ。

| HIP device | PCI | drm |
|---|---|---|
| 0 | `0000:07:00.0` | `card1` |
| 1 | `0000:0a:00.0` | `card2` |
| 2 | `0000:0f:00.0` | `card3` |
| 3 | `0000:12:00.0` | `card4` |
| iGPU | `0000:83:00.0` | `card5` |

```bash
for c in /sys/class/drm/card*/device; do printf '%s -> ' "$c"; readlink -f "$c"; done
sudo cat /sys/class/drm/card<N>/device/devcoredump/data \
  > amdgpu-devcoredump-$(date +%Y%m%d-%H%M%S).bin
```

あわせて kernel log を保存する。再起動後は `journalctl -k -b -1` で前 boot のログが読める。

```bash
journalctl -k -b -1 > amdgpu-kernel-prev-boot.log
```

## optional tests

`SHELF_BUILD_OPTIONAL_TESTS=ON` で external model を要する heavy test
（DFlash2 gate、TP、MTP weight contract 等）が build される。
manifest は `cmake/tests.cmake` が正本であり、件数は固定しない。

external test の model directory はすべて環境変数で指定する。
`SHELF_MODEL_DIR`（Qwen3.5 系の既定）、`SHELF_MODEL_DIR_MTP`、
`SHELF_MODEL_DIR_DFLASH2`、`SHELF_MODEL_DIR_DFLASH2_TARGET`、
`SHELF_TP_MODEL_DIR`。cmake option に model directory は持たない。

Qwen3.5-4B full application E2E は削除済みである。外部 model が無い環境で
skip を垂れ流さないため、実行できない suite は repository に残さない。
generation stop token contract（`test_server_eos_contract.py`）は model の
`generation_config` / tokenizer と実際の generation だけを使い、model 固有の
oracle fixture に依存しない。

## server regression

server / compute E2E は `tests/server/` の Python suite で検証する。
`shelf-server` は `shelf-compute` を subprocess として起動するため、
追加の frontend は要らない。model directory は `SHELF_MODEL_DIR` で指定する。

canonical runner は `tests/server/run_server_regression.py` である。glob ではなく明示 manifest を
持ち、group 単位で subprocess として順次実行する。non-zero return は失敗として扱い、
skip で通さない（自己判定した skip は exit 77 で報告される）。

```bash
SHELF_MODEL_DIR=$PWD/models/Qwen3.8-27B-FP8 \
SHELF_TEST_DEVICE=1 \
SHELF_TEST_ARENA_GIB=31 \
SHELF_TEST_WORKING_SET_RESERVE_GIB=1 \
  python3 tests/server/run_server_regression.py --all
python3 tests/server/run_server_regression.py --group tools --group agent
python3 tests/server/run_server_regression.py --list
```

`SHELF_MODEL_DIR` は runner が test を別 directory で起動するため絶対 path で渡す。
既定の harness は 24 GiB arena を前提にするため、27B の FP8 重みを載せるホストでは
`SHELF_TEST_ARENA_GIB` で容量を合わせる。working set は compute 側の auto（`0`）に
委ねるため、通常 `SHELF_TEST_WORKING_SET_RESERVE_GIB` は設定しない。明示した場合は
harness が compute へ渡す。値を変えたら起動 log の `MEMORY_PLAN` で実際の割当を確認する。
KV auto-fit が入れる drafter 分の計算式は `tests/unit/test_dflash2_footprint.cpp`、
auto の working set reserve は `tests/unit/test_executor_footprint.cpp`（いずれも cpu）が
固定する。

group と test 一覧は `tests/server/run_server_regression.py` の `GROUPS` が正本である。
現在の group は `core` / `tools` / `reasoning` / `unsupported` / `concurrency-cancel` / `agent`。
`concurrency-cancel` の `test_server_context_boundary.py` は 32K BF16 KV を前提にするため、
32 GiB の GPU 1 枚に 27B FP8 を載せる構成では起動しない。

prefix cache の JSONL contract は
`tests/server/test_compute_prefix_contract.py` が固定する。

## 実行層

2 層を混同しない。

```
required acceptance   self-contained C++ / GPU suite（tests/run_required_acceptance.py）
server regression      model + compute E2E（tests/server/run_server_regression.py）
```

`support.py` は共有 helper であり test ではない。manifest へ入れない。
