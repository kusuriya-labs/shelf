# hipFile / SSD 環境の適合性調査

- ID: QWEN4EXP-G0-HIPFILE
- 実施日: 2026-10-24
- 対象: Shelf worktree `.worktrees/qwen4exp`、R9700 (gfx1201) ×4

## 1. バージョン（固定できたもの）

| 項目 | 値 |
| --- | --- |
| HIP | `7.15.26333-0000000`（hip-clang AMD clang 23.0.0git） |
| ROCm | pip 配布 SDK（`rocm-10.0.0`）。`HIP_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel` |
| ROCk kernel module | 7.1.3.31500000 |
| HSA runtime | 1.21 |
| OS / kernel | Linux `7.0.0-38-generic`（Ubuntu 24.04 系） |
| GPU | AMD Radeon AI PRO R9700（gfx1201, DID 0x7551）×4、VRAM 34,208,743,424 B 各 |
| 追加デバイス | DID 0x164e の 512 MiB デバイス 1 枚（iGPU。候補から除外される） |
| Python / torch | 3.12.3 / `2.13.0+rocm10.0.0` |
| transformers | `5.14.1`（`qwen4_exp` モジュールは**非搭載**） |

`/opt/rocm` には `core-7.14` のみが在り、実運用の HIP は venv SDK 配下にある。
`docs/developer/testing.md` の記載と整合する。

## 2. hipFile の実在と API

```
libhipfile.so.0 => /opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhipfile.so.0
  （libhipfile.so.0.4.0、2025-08-29 build、約 755 KB）
ヘッダ: _rocm_sdk_devel/include/hipfile.h, hipfile-api-trace.h
CMake: _rocm_sdk_devel/lib/cmake/hipfile/
tool:  ais-check / ais-stats（SDK bin 配下。PATH 未登録）
```

手順書 §7.1 の `hipconfig --version` / `rocminfo` / `ais-check` は実行できた。
`ais-check -v` は sudo なしで成功（診断のみ）。

`hipfile.h` の主要 API（G2 で使うもの）:

| API | 用途 |
| --- | --- |
| `hipFileDriverOpen` / `hipFileDriverClose` | ドライバ open/close |
| `hipFileHandleRegister` / `hipFileDeregister` | file handle 登録 / 解除 |
| `hipFileBufRegister` / `hipFileBufDeregister` | GPU buffer 登録 / 解除 |
| `hipFileRead` / `hipFileReadAsync` | 同期 / 非同期読み込み |
| `hipFileStreamRegister` / `Deregister` | stream 登録 |
| `hipFileGetVersion` / `hipFileDriverGetProperties` | バージョン / property 取得 |

→ **I/O ライブラリとしては実在し、最小実装に必要な API は揃っている。**

## 3. ais-check の実測結果

```
libamdhip64.so.7 (AIS supported)
マウント:
  /     nvme1n1p5 ext4 ordered   O_DIRECT=unverified  HIPFILE=unverified
  /opt  nvme0n1p1 ext4 ordered   O_DIRECT=unverified  HIPFILE=unverified
  sda3  ext4 scsi                O_DIRECT=yes         HIPFILE=no
AIS support:
  Kernel P2PDMA=True
  HIP runtime=True
  amdgpu=True
  hipFile-capable volume: False
```

**重要な判定:**

- `hipFile-capable volume: False`。`/opt`（models を載せている NVMe / ext4）は
  現状のマウント形態では fastpath 対象として**未成立**。
- `O_DIRECT` も `unverified`。整列単位 `A` は**未実測**（ext4 + fs block 4096 なら
  4096 が第一候補だが、これは仮説であり G2 で実測する）。
- 手順書 §7.1 の注意どおり、**`ais-check` 成功と fastpath 利用は別問題**であり、
  本調査では fastpath 未成立と判定する。

→ **G2-Direct は現状未達。** G2-Functional（数値一致）とは独立に扱う。

## 4. ストレージ構成

| デバイス | 容量 | マウント | 備考 |
| --- | --- | --- | --- |
| nvme0n1 | 931.5 GB | `/opt`（ext4 ordered） | models はここ。使用 523 G / 空き 347 G |
| nvme1n1 | 1.8 TB | `/`（ext4） | |
| sda | 1.8 TB | sda3 = `/media/zen/...`（ext4） | |

- NVMe logical / physical block size = 512 B、I/O scheduler `none`、fs block = 4096。
- PLE 全体は約 95.4 GiB。`/opt` の空き 347 G には**収まる**。

## 5. GPU の可用性

調査時点で **GPU[0] / GPU[1] は他プロセスが VRAM 約 33.6 GB（98%）使用中**。
GPU[2] / GPU[3] は空き（63 MB 使用）。

`shelf-gpu-test-runner` の予約台帳（ロックファイル）は自プロセスの予約のみを
追跡するため、外部プロセスの使用は反映されない。その結果、
GPU0/1 へ割り当てられたテストは `hipMalloc failed: out of memory` で失敗する。

**ベースライン取得時の対応**: `SHELF_TEST_GPUS=2,3` を指定して空き GPU に限定した。
この指定下で required acceptance は 143/143 PASS（`gates.md` 参照）。

`docs/developer/testing.md` の制約「GPU 負荷中は rocm-smi / amd-smi / gpu_metrics を
叩いてはならない（SMU 無応答 → SW CTF shutdown の観測あり）」を遵守し、
rocm-smi の実行はテスト非実行時のみとした。

## 6. G0 への判定

| 項目 | 状態 |
| --- | --- |
| hipFile ライブラリ / ヘッダ / tool の実在 | **達成** |
| ROCm / HIP / GPU / OS の revision 固定 | **達成**（上表） |
| O_DIRECT 整列単位 `A` の実測 | **未達**（G2 で実測） |
| hipFile fastpath の成立 | **未達**（`hipFile-capable volume: False`） |
| SSD / GPU 構成の記録 | **達成** |
