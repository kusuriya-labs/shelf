# GDN state の f32 丸め修正の検証

実験 ID: `GDN-STATE-F32-ROUNDING-1`。判定: **REJECT（push 不可）**。

`origin/main` の `fdaa0ad46b5dbcb2fb5f01fe638e248644f4347c` を基準に、`8dea2173` のテスト配列の型修正と recurrence / commit の fp16 丸め除去だけを適用した。テスト修正のみでは exact-history が 0/7、両修正では 7/7 だった。モデル品質、投機生成の完全一致、速度のすべてを満たすことを受入条件とする。chunk gate は平均 KL ≤0.01、top-1 一致 ≥99%、速度は暖機後 3 回の中央値でおおむね ±2% を使用した。BF16 参照への KL / top-1 / PPL が悪化しないことも必要である。

R9700 1 枚、ROCm 10.2、gfx1201、Release、LLVM 23 の libc++、BF16 KV。すべての GPU 実行は `flock /tmp/r9700.lock`、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、`HIP_VISIBLE_DEVICES=0` を使用した。`SHELF_LOAD_LIBSTDCXX_FIRST` は基準 revision では未実装で CMake に無視されるため、両 build に `-DCMAKE_EXE_LINKER_FLAGS=/lib/x86_64-linux-gnu/libstdc++.so.6` を指定した。`readelf -d` により libstdc++、libc++abi、libc++ の順の直接依存を確認した。ソースにはこの設定を加えていない。

## テスト

| CTest | 変更前 | 変更後 |
| --- | ---: | ---: |
| 全 241 件 | 失敗 12、skip 11 | 失敗 11、skip 11 |
| `test_gdn_recurrence_exact_history` | 失敗 | 通過 |
| `-R gdn` | exact-history と LFS fixture の 2 件失敗 | LFS fixture の 1 件失敗、12 件通過 |

全件比較の失敗は変更後、新規追加なし。両方の失敗は `test_exl3_mul1_grouped`、`test_mxfp4_linear`、`test_weight_load`、`test_qwen35_mtp_weight_load`、`test_mtp_reserve`、`test_moe_router`、`test_moe_grouped_linear`、`test_moe_combine`、`test_moe_layer_e2e`、`test_qwen35_gdn_gate_reference`、`test_moe_program_execute`。全件比較時の GDN reference の失敗は safetensors が Git LFS ポインターのままであることによる。対象の 2 ファイルを LFS から取得し SHA-256 を照合して再試行すると、両版とも `lm_head_tied disagrees with lm_head storage` で失敗した。untied embedding と lm head の両方を host に置く経路で、検査が GPU payload の同一性を見るため両方 null と判定されることが原因と推定される。元の LFS ポインターは試行後に復元した。GDN 全件通過の条件は満たさない。skip も両版で同じ 11 件で、複数 GPU や外部データを必要とする試験を含む。実行結果は skip を通過として扱わない。

## 品質

`shelf-bench quality`、EXL3 4bpw mul24、64 位置、chunk 4096。変更前を基準に変更後を比較した。

| 入力 | context | 前 PPL | 後 PPL | 平均 KL | top-1 一致 | logit 最大絶対差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 128 | 13.6539462 | 13.6539462 | 0 | 100% | 0 |
| BetterBench | 2048 | 7.95453781 | 7.95453781 | 0 | 100% | 0 |
| chat | 2048 | 2.33631814 | 2.33631814 | 0 | 100% | 0 |
| code | 2048 | 1.04048019 | 1.04048019 | 0 | 100% | 0 |
| 日本語 | 2048 | 1.46906751 | 1.46906751 | 0 | 100% | 0 |
| long | 8192 | 2.73108229 | 2.73108229 | 0 | 100% | 0 |

long の chunk 4096 を基準として chunk 2048 を比較すると、変更前後とも KL 0.0868127414、top-1 96.875%、PPL 2.73108229 → 1.96857606、logit 最大絶対差 16.0024118。両版で同値だが gate は不合格。chunk 2048 同士の前後比較も KL 0、top-1 100%、最大 logit 差 0。`SHELF_GDN_RECURRENCE_EXACT=1` で lossy 分解を無効にしても、変更後の 4096→2048 は KL 0.0202467294、top-1 96.875%、PPL 1.8344386 → 2.11117386、最大 logit 差 14.5265916 で gate 不合格だった。lossy 分解の無効化だけでは解消しないが、原因は未確定。これは EXL3 同士の比較である。

BF16 モデルは約 52 GiB で、R9700 単体の `arena auto` 31.15 GiB では `layers.43.self_attn.o_proj.weight` の確保に失敗した。BF16 参照に対する KL / top-1 / PPL は未測定であり、悪化しないとは判定できない。

## 生成と速度

最大 256 token、greedy、tree budget auto、drafts 7。初回差は 0 始まり。非投機は logit をダンプした。

| 入力 | 変更前: 投機と非投機の初回差 / 受理長 | 変更後: 同左 | 非投機の前後差 |
| --- | --- | --- | --- |
| shell | 45 / 3.397 | 37 / 3.250 | 256 token 一致 |
| code | 118 / 3.636 | 256 token 一致 / 3.722 | 256 token 一致 |
| 日本語 | 60 / 1.318 | 60 / 1.217 | 256 token 一致 |

変更後 shell の初回差は非投機 ID 10966、投機 ID 17789。非投機側の両 logit は 23.341351 と 23.188080、差 0.153271。日本語は ID 95882 と 246922、logit 15.839705 と 15.833183、差 0.006522。変更前 shell は ID 321 と 6600、logit 23.886116 と 23.829987、差 0.056129。ダンプなしの非投機も、3 入力の前後両版すべてでダンプありの生成列と 256 token 一致した。したがって上記の投機との差の位置はダンプの有無に依存しなかった。

| 速度、token/s | 変更前 3 回、中央値 | 変更後 3 回、中央値 | 中央値変化 |
| --- | --- | --- | ---: |
| pp2048 | 1126.89 / 1123.83 / 1121.71、1123.83 | 1127.23 / 1124.61 / 1122.54、1124.61 | +0.07% |
| 非投機 tg128 | 31.67 / 31.69 / 31.69、31.69 | 31.66 / 31.70 / 31.69、31.69 | 0% |
| DFlash2、shell、256 token | 85.10 / 85.13 / 85.25、85.13 | 82.25 / 82.42 / 82.38、82.38 | -3.23% |

pp は process 内 warmup 1、3 runs。tg と DFlash2 は別 process で 1 回捨て、3 回を測定。tg は各 process でも warmup 1。固定順、非交互、clock 固定なし。DFlash2 の受理長は各 3 回すべて変更前 3.397、変更後 3.250。非投機と投機の生成列が一致しないため、この速度差は同一生成列での純粋な kernel 性能差を意味しない。

MTP は最初、既定の設定と同時要求 1 の設定のいずれでも `mtp drafter create failed`（arena 31.29 GiB のうち 31.16 GiB 使用、0.14 GiB の確保失敗）となった。`--max-concurrent-requests 1 --max-num-batched-tokens 512 --prefill-chunk 512 --working-set-reserve-gib 4 --prefix-cache 0 --mtp --mtp-drafts 7` では起動した。非投機側にも MTP 以外の同一引数を指定し、各 256 token を比較した。

| 入力 | 変更前 MTP と非投機 | 変更後 MTP と非投機 | 受理長 前 → 後 |
| --- | --- | --- | ---: |
| shell | 一致 | 一致 | 3.397 → 3.322 |
| code | 一致 | 一致 | 2.806 → 2.806 |
| 日本語 | 6 番目から不一致 | 一致 | 1.576 → 1.525 |

`tests/server/test_compute_dflash2.py` は変更前 93 通過・6 失敗、変更後 97 通過・2 失敗。変更前は BF16 / FP8 / Octave c4v4 のすべてで `serve == target-only greedy` と `serve == single-shot dflash2` が失敗した。変更後は BF16 と Octave c4v4 の `serve == target-only greedy` が失敗し、FP8 の両比較および 3 形式の single-shot 一致は通過した。このテストの DFlash2 は chain（tree budget 0）である。

## 入力

入力の SHA-256（PSKLDTOK ファイルのバイト列）:

| 入力 | パス | SHA-256 |
| --- | --- | --- |
| short / BetterBench | `/tmp/shelf-quality-betterbench.psktok` | `99fe6aa90b4f52a9427361391f0dc3e17377391d67793c1c716e890d8c28799d` |
| chat | `/tmp/shelf-quality-chat.psktok` | `549e627ad1cea95ed6aa68d78198de33f0ebe6686da8a230c1c943ba0aaef8cd` |
| code | `/tmp/shelf-quality-code.psktok` | `f50e07af305fb1b3e25b14a349761b53c9b31b61ad8cdefcadce95dbc8ae634f` |
| 日本語 | `/tmp/shelf-quality-ja.psktok` | `25d9482d999ad1c1b8f7f5c367505093631ca66de0ec4b6245985e2dab32fd50` |
| long / 速度 | `/tmp/shelf-prefill-32768.psktok` | `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb` |
| shell 生成 | `/tmp/gdn-fix-evidence/shell.ids` | `2bd565fb5b2a2a96c03757322ffd990211aa86d13b88e1bb38b7ecd7c17b5916` |
| code 生成 | `/tmp/gdn-fix-evidence/code.ids` | `854540fe33a5a458d20acf95be61bf8bfc454d928ce538a4406ced78b6807f22` |
| 日本語生成 | `/tmp/gdn-fix-evidence/japanese.ids` | `ff19891057cea55bddfc96a871c4f151325463bc2c0c94ff3636889ece22f653` |

これらの長文・日本語・chat・code は反復 corpus であり、一般的な実タスクの代表とは限らない。生成入力は各 JSON の `input_ids` を空白区切り、末尾改行で書いた。

## コマンド

以下の `B` は `build-gdn-base` または `build-gdn-fix`、`MODEL` は `/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24`、`DRAFT` は同ディレクトリの `dflash2-drafter-exl3-4bpw-mul24`、`LIB` は `/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib`。以下の実行はすべて `flock /tmp/r9700.lock env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/core-10.2/lib:$LIB` を前置した。

```sh
cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.2/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.2/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.2 \
  -DSHELF_ROCM_ROOT=/opt/rocm/core-10.2 \
  -DSHELF_LIBCXX_LLVM_ROOT=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23 \
  -DSHELF_LOAD_LIBSTDCXX_FIRST=ON \
  -DCMAKE_EXE_LINKER_FLAGS=/lib/x86_64-linux-gnu/libstdc++.so.6 \
  -DSHELF_BUILD_TESTS=ON -DSHELF_BUILD_BENCHMARKS=ON
cmake --build "$B" -j 16
ctest --test-dir "$B" --output-on-failure --timeout 180
ctest --test-dir "$B" -R gdn --output-on-failure --timeout 180
"$B/shelf-bench" quality --model-dir "$MODEL" --tokens-file "$INPUT" \
  --context "$CONTEXT" --positions 64 --prefill-chunk "$CHUNK" --kv-cache-dtype bf16 \
  --dump "$DUMP" # 変更後と chunk 比較は基準 dump を --compare で指定
"$B/shelf-compute" --model-dir "$MODEL" --input-ids-file "$IDS" --max-new-tokens 256 \
  --max-seq-len 512 --arena-gib auto --device 0 --kv-cache-dtype bf16 \
  --prefill-chunk 4096 --temperature 0 --top-p 1 --top-k 0 --seed 0 \
  --dflash2-model-dir "$DRAFT" --dflash2-drafts 7 --dflash2-tree-budget auto
"$B/shelf-bench" pp --model-dir "$MODEL" --mode forward --prompt-tokens 2048 \
  --prefill-chunk 4096 --kv-cache-dtype bf16 --page-tokens 16 --warmup 1 --runs 3 \
  --tokens-file /tmp/shelf-prefill-32768.psktok
"$B/shelf-bench" tg --model-dir "$MODEL" --mode greedy --context 2048 --tokens 128 \
  --prefill-chunk 4096 --kv-cache-dtype bf16 --page-tokens 16 --warmup 1 \
  --tokens-file /tmp/shelf-prefill-32768.psktok
```

追加の chunk 診断では同じ quality コマンドの chunk 2048 dump を基準に前後比較し、さらに `SHELF_GDN_RECURRENCE_EXACT=1` で chunk 4096 dump と 2048 を比較した。

BF16 の試行は `MODEL` を `Qwen3.8-27B-BF16`、context 128、chunk 128、位置 64 とした。非投機生成は DFlash2 引数を除き `--dump-logits` を加えた。MTP の比較は上記 compute コマンドから DFlash2 引数を除き、同時要求 1 と working-set 予約を上記のとおり指定し、MTP 有無で各 1 回実行した。server regression は `SHELF_BUILD_DIR=$PWD/$B SHELF_MODEL_DIR=$MODEL SHELF_DFLASH2_MODEL_DIR=$DRAFT SHELF_TEST_DEVICE=0 python3 -u tests/server/test_compute_dflash2.py`。同じ非投機コマンドを `--dump-logits` なしでも再実行した。測定ログと入力は `/tmp/gdn-fix-evidence/` に保存した一時ファイルである。

## 追試と受入条件の訂正（2026-11-20）

ユーザーの訂正により、必須条件は **DFlash2 と非投機 greedy の全入力での完全一致、GDN テスト通過、速度の悪化がないこと**とする。上記の chunk の KL / top-1 閾値と BF16 参照は採否条件から外す。以下は元の記録を残したまま追加した追試である。判定は **push 保留**。今回の対象の完全一致と GDN 全件通過を確認できていない。

### ROCm と変更前後の比較

| 項目 | 7.14 変更前 | 7.14 変更後 | 10.2 変更前 | 10.2 変更後 |
| --- | --- | --- | --- | --- |
| GDN CTest、13 件 | 11 通過、exact-history と fixture 失敗 | 12 通過、fixture 失敗 | 11 通過、exact-history と fixture 失敗 | 12 通過、fixture 失敗 |
| chunk 4096 → 2048、KL / top-1 | 0 / 100% | 0 / 100% | 0.0868127414 / 96.875% | 同左 |
| 3 入力、投機 == 非投機 | 0 / 3 | 1 / 3 | 0 / 3 | 1 / 3 |
| server regression | 93 通過 / 6 失敗 | 97 通過 / 2 失敗 | 93 / 6 | 97 / 2 |
| pp2048 tok/s、中央値 | 1903.52 | 1891.11 | 1123.83 | 1124.61 |
| 非投機 tg128 tok/s、中央値 | 31.72 | 31.65 | 31.69 | 31.69 |

GDN fixture は Git LFS ポインターで失敗する。LFS 実体を使った 10.2 の前回試行でも `lm_head_tied disagrees with lm_head storage` で失敗したため、fixture を取得するだけでは全件通過しない。未取得のまま実行した 7.14 については、この二次的な失敗を再検証していない。テストを skip や通過とは数えない。7.14 の tg 変更前 3 回は 31.76 / 23.73 / 31.72 で、1 回の外れ値を含む。pp 変更後は約 -0.65%、tg は約 -0.22%。10.2 はそれぞれ +0.07%、0%。

3 入力の初回分岐は両 ROCm で同一だった。0 始まり、256 は全 256 token 一致を示す。

| 入力 | 変更前: 初回位置（非投機 / 投機 ID） | 変更後: 同左 |
| --- | --- | --- |
| shell | 45（321 / 6600） | 37（10966 / 17789） |
| code | 118（23238 / 348） | 256（一致） |
| 日本語 | 60（95882 / 246922） | 60（95882 / 246922） |

server の変更前の失敗は BF16 / FP8 / Octave c4v4 各々の serve 対 target-only greedy と single-shot DFlash2。変更後は BF16 と Octave の serve 対 target-only greedy のみ失敗した。これは改善であり完全一致ではない。server regression は `SHELF_GDN_VERIFY_CHUNKED=0` を指定している。

7.14 の build は `fdaa0ad4` の `git archive` と `704b753d` を含む作業ツリーから別々に作成し、`build-rocm714` の compiler、gfx1201、ROCm root、host libc++ 23、vendor include を踏襲した。両方に `/lib/x86_64-linux-gnu/libstdc++.so.6` を EXE linker flags で先頭に追加した。基準 revision では `SHELF_LOAD_LIBSTDCXX_FIRST` が未実装なので、同一のリンク順にするためである。実行時 `LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LIB` で `libamdhip64`・HSA・hipBLASLt が 7.14 から解決されることを確認。GPU は全実行で `flock /tmp/r9700.lock`、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、`HIP_VISIBLE_DEVICES=0`。

chunk 4096 の 64 位置の dump は 7.14 / 10.2 の前後 4 条件で SHA-256 `ce6ce45de88a56062e1c966ef02d1cf217e73676d0f413a94df44c9968951cd5` と一致した。chunk 2048 側は 7.14 と 10.2 で異なるため、このモデル・入力での chunk 差は ROCm compiler/runtime を含む環境に依存する。ただし原因の部品は未確定であり、速度で既知の INT8 WMMA compiler 差が品質差の原因という証拠はない。これは今回の採否条件ではない。過去の f32 通過記録 `792ab093` は Quark AWQ MXFP4、prose 16,384 token、context 8192、評価 512 位置であり、今回の EXL3 と異なる。元記録に入力ファイルの特定情報と完全なコマンドはなく、同モデルも探索範囲で見つからなかったため、完全に同条件での再現はできていない。

### 一致しない経路の診断

10.2、修正版、上記 3 入力、256 token。以下の位置は各設定の非投機と比較した初回差で、256 は一致。exact 設定では非投機にも同じ `SHELF_GDN_RECURRENCE_EXACT=1` を指定した。

| 条件 | shell | code | 日本語 |
| --- | ---: | ---: | ---: |
| tree auto、既定 | 37 | 256 | 60 |
| chain（tree budget 0）、既定 | 47 | 118 | 60 |
| chain、`SHELF_GDN_VERIFY_CHUNKED=0` と `SHELF_GDN_RECURRENCE_EXACT=1` | 256 | 256 | 60 |
| tree auto、同 exact 設定 | 30 | 256 | 60 |
| mul24 の split-K を診断用に常時 1、tree auto | 30 | 246 | 39 |

診断用 split-K 変更は別バイナリのみで、ソースは元に戻した。tree を切っても不一致は残り、exact GDN と serial chain は一部を改善するが全入力を解決しない。split-K 固定も全入力で不一致だった。したがって tree の rollback だけ、または split-K の違いだけが原因とは言えない。異なる verify 行数と 1 行 decode の線形演算経路は、別の実験でも同一入力の投影値に差が出ている（`docs/research/20261111/dflash2_adaptive.md`）。今回の個々の演算での最初の差は未特定である。

shell について診断用バイナリで tree の verify logits をコピーした。初回分岐直前の共通履歴に対応する position 74 の row 2 で、非投機の ID 10966 / 17789 の logits は **23.341351 / 23.188080**、verify は **23.330748 / 23.351402**。前者では 10966、後者では 17789 が最大で、target logits 自体の順位が逆転している。非投機 logit 差は 0.153271、verify は逆方向に約 0.020654。診断時に stream 同期とコピーを追加しており、速度の測定には使わない。

共通 runtime の `gdn_spec_history.hip` は recurrent の確保・コピーに `GdnStateScalar`（float）を使い、修正後の recurrence と compact commit の state load/store も float。`correctness/detail/gdn.inc` の f16 helper は当該ファイル内で呼ばれていない。WMMA 入力分解には `_Float16` が残るが、これを state の保存丸めの取りこぼしと見なす証拠はない。未解決の複数行 verify と decode の数値差を解消し、全入力一致と速度を同時に満たす変更は特定できなかったため、追加の推測的な kernel 修正は加えていない。

### DFlash2 の速度と受理長

最大 256 token、tree budget auto、drafts 7、BF16 KV、各条件 3 process の中央値。単位は tok/s、括弧内は `DFLASH2_MEAN_ACCEPTED`。前後は交互に実行したが、7.14 の shell・code・日本語の最初の 1 回は先行する一致確認で取得した。clock 固定は行っていない。

| 入力 | 7.14 変更前 | 7.14 変更後 | 変化 | 10.2 変更前 | 10.2 変更後 | 変化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shell | 87.95 (3.397) | 85.46 (3.250) | -2.83% | 85.64 (3.397) | 82.88 (3.250) | -3.22% |
| code | 92.95 (3.636) | 94.81 (3.722) | +2.00% | 90.43 (3.636) | 91.95 (3.722) | +1.68% |
| 日本語 | 29.74 (1.318) | 30.21 (1.217) | +1.58% | 45.75 (1.318) | 43.67 (1.217) | -4.55% |
| chat | 78.82 (2.923) | 78.83 (2.923) | +0.01% | 76.54 (2.923) | 76.42 (2.923) | -0.16% |
| debug | 79.33 (2.984) | 82.96 (3.113) | +4.58% | 76.75 (2.984) | 80.09 (3.113) | +4.35% |
| math | 124.50 (5.220) | 124.36 (5.220) | -0.11% | 120.60 (5.220) | 120.64 (5.220) | +0.03% |

7.14 の日本語は前 47.35 / 29.03 / 29.74、後 45.13 / 28.12 / 30.21 と大きく変動した。10.2 でも日本語の前 1 回が 32.02、chat 後 1 回が 45.36、debug 前 1 回が 57.71 と外れた。中央値でもこのような揺れを取り除ける保証はない。shell では両 ROCm で約 3% 低下が再現し、code / debug は上昇、chat / math はほぼ横ばい。生成列と受理長の変化を伴う入力依存の差であり、一律の速度低下でも純粋な kernel 性能差でもない。一方、shell の低下は実測上存在するため、全入力で速度悪化なしとは判断できない。

追加入力は `docs/research/20261111/gdn_recurrence_data.json` の `base-11`（分散ジョブ queue）、`base-12`（C++ memory leak 調査）、`base-18`（二次方程式）から `input_ids` を使用。PSKLDTOK ではなく、空白区切りの ID と末尾改行。SHA-256 は順に `1e3dc44c3ca1893cfd6b2dbaf5d6269296f0c16eaa848155fe9fbce1d8d957b6`、`fcac825bf8ed0a3dc7fe108683eb298292f41692d40e80273159953ad637a359`、`e77d1125c1e9ee9f74c2c8363dbab34c5bcfbe9b6a7f23990af7c668fc49fbb9`。

追加 3 入力の初回分岐も 7.14 / 10.2 で位置と ID が一致した。chat は変更前後とも 256 token 一致、debug は変更前 171（70 / 2）→変更後 35（40357 / 4499）、math は変更前後とも 135（332 / 9568）。したがって 6 入力の一致は変更前 1 / 6、変更後 2 / 6。debug では最初の不一致が早くなっており、修正が全入力で一致を改善するわけではない。

### 推奨

今回の変更全体は push 保留を推奨する。両 ROCm で投機と非投機の完全一致が未達、GDN reference fixture が未通過、shell の DFlash2 は両方で約 3% 低下した。exact-history の改善と server の 4 件改善は確認できるが、全入力の品質と速度の受入条件を満たす根拠にはならない。テスト配列の型修正だけは領域外書き込みを除去する独立の修正として分離可能だが、それだけでは exact-history は失敗するため、失敗を許容する運用がなければ先行 push も推奨しない。次に必要なのは、同一 prefix の 1 行 decode と複数行 verify の層別中間値・GDN state・線形投影を実測し、最初に食い違う演算を特定した上で、全 6 入力と server regression、GDN fixture、速度を再測定することである。

診断ログ、各回の速度、検証用スクリプトと一時的な logits dump は `/tmp/gdn-compare/` に保存した。元の入力・モデル・共通コマンドは上記「入力」「コマンド」に記載。7.14 は `B=build-gdn-base-714` / `build-gdn-fix-714`、`LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LIB` とし、それ以外は同条件。tree / chain は compute の `--dflash2-tree-budget auto` / `0` で切り替えた。多入力速度の `chat` / `debug` / `math` は上記 ID に差し替え、各 256 token、同じ greedy 設定で実行した。

7.14 の日本語は上表の揺れを受けて、前後交互に各 3 回を追加実行した。前 47.07 / 47.11 / 47.15、後 45.01 / 45.10 / 45.05 tok/s、中央値 47.11 → 45.05（-4.37%）。受理長は各回それぞれ 1.318 / 1.217。この再測定では 10.2 と同様の速度低下が見られた。上表の初回測定を置き換えず両方を報告する。
