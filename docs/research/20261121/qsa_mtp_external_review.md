# 外部エンジンの QSA / MTP 参照（2026-11-21）

手順書 §9 の参照ソースを確認した記録である。取得は 2026-11-21、記載 SHA は取得時点の
default branch HEAD。外部コードはコピーせず、アルゴリズムとテスト設計の参考に限定する。

## 1. vLLM

- repo: `vllm-project/vllm`
- SHA: `bd53af0bbec2`
- file: `vllm/models/qwen4_exp/amd/ops/qsa.py`
- 確認した関数:
  - `_qsa_mqa_paged_kernel`
  - `_qsa_mqa_paged_rows_kernel`
  - `_qsa_sparse_paged_gqa_splitk_kernel`
  - `qsa_select_paged_tokens`
  - `expand_qsa_block_indices_cuda`

所見:

- `_qsa_mqa_paged_rows_kernel` は `row` を program_id に取り、`token_to_req` で
  request を引く。複数 row / request を 1 launch の grid で処理する。
- `visible` は `query_position` と request ごとの `sequence_length` から row 単位で
  計算し、request の未来 token を混ぜない。
- `qsa_select_paged_tokens` は host sync なしで score → top-k → block 展開を行い、
  workspace から `rows_per_chunk` を決める。
- split-KV の GQA 経路は request ごとの KV page table を引く。

採用:

- Shelf の Wavefront verify で「同じ深さの request を 1 launch にまとめる」考え方は
  この rows kernel と一致する。
- QSA score の launch 集約 (Phase 3) では `token_to_req` 相当の
  `row_sequence_slots` と request ごとの visible block 数を使う。

不採用:

- Triton の reduction order / FMA 順序を gfx1201 の WMMA にそのまま移植しない。
- host sync なしの選択は参考にするが、Shelf の既存 QSA mask / top-k の同点規則を
  変えない。

## 2. SGLang

- repo: `sgl-project/sglang`
- SHA: `8396d60a14ef`
- file: `python/sglang/srt/layers/attention/qsa/qsa_indexer.py`
- 確認した関数: `project_qk`、`_fused_compress_store`、
  `update_key_state_and_compress`、`_compress_decode_cuda_graph`、
  `_qsa_prefill_row_chunk_size`

所見:

- Q norm + RoPE + store、compress + norm + RoPE + store を融合している。
- CUDA Graph 用に固定形状の writer と bounded score workspace を持つ。
- prefill row chunk を固定し、Graph capture 時の形状を安定させる。

採用:

- QSA key compression の launch 削減 (Phase 6) の設計参考。
- Graph-safe 化では固定 shape / bounded workspace の考え方を使う。

不採用:

- Graph 化を先に有効化しない。Shelf の
  `docs/research/20261116/qsa_topk_graph_fault.md` の stale block_count 問題が
  未解決であり、correctness gate を満たすまで既定にしない。

## 3. hipfire

- docs: `https://hipfire.dev/learn/sparse-attention/`
- 取得時点で GitHub API の default branch 名を確定できなかったため、SHA は未記録。
- 参照した設計:
  - 1 KV group / 12 Q heads の K/V 再利用
  - decode / verify 8 行以下と prefill 16 行の分離
  - R9700 の gather + F16 WMMA prefill
  - Graph 安定な scratch

採用:

- QSA sparse attention の decode / verify と prefill で dispatch を分ける。
- K/V の staging 再利用 (Phase 5) の比較対象にする。

不採用:

- hipfire が gather F16 WMMA を bit-exact でないと明記しているため、exact 経路には
  採用しない。近似 mode にする場合も target-only ID gate と分離する。

## 4. TensorFold

- repo: `ashhart/TensorFold`
- SHA: `5954c6c6dcb2`
- files:
  - `zig/src/families/flashnext/cuda_mtp_plan.zig`
  - `zig/src/families/flashnext/cuda_mtp_draft.zig`
  - `zig/src/families/flashnext/cuda_mtp_roll.zig`
  - `zig/kernels/hip/flashnext/sparse.hpp`

所見:

- `Draft` は head cache / KV / window scores / draft logits / GDN late rows を
  request ごとの state として持つ。
- draft は chain を `drafted` で管理し、window の一部だけを残す rollback を持つ。
- rollback は GDN の conv / history と PLE n-gram を同じ graph で戻す。
- MTP の draft と target verify の forward 粒度はコード側で別に管理している。

採用:

- Shelf の `MtpDraftShared` + request ごとの MTP KV / hidden / last_tokens は
  この state 分離と同型である。
- Phase 2 の partial accept では GDN / PLE / QSA を受理 prefix に戻す既存契約を
  維持する。

不採用:

- TensorFold の CUDA graph / rollback 実装をそのまま移植しない。Shelf の
  `accept_verified` と spec history の契約を正本とする。

## 5. Strata

- repo: `Niko1221/Strata`
- SHA: `3142b66b96e9`
- files:
  - `engine/src/kernels/cuda/qsa_select.cu`
  - `docs/QSAROWS.md`
  - `engine/src/kernels/qsa_rows_bench.cpp`
- 確認した関数:
  - `block_scores_rows_kernel<NQ>`
  - `block_topk_kernel`
  - `block_topk_reg_kernel`
  - `block_topk_wide_kernel`

所見:

- `block_scores_rows_kernel` は 1〜4 query を register に保持し、K/V を 1 回読んで
  複数 query で共有する。
- top-k は radix threshold、同点は小さい index、cell 昇順で決める。
- 資料中で WMMA / TF32 経路は warp kernel と bitwise でないと明記されている。

採用:

- QSA score の row 方向集約 (Phase 3) で、複数 row の query を register に置く
  設計を参考にする。
- top-k の同点規則は「小さい block index 優先」を維持する。

不採用:

- tensor-core scorer の非 bitwise 経路を exact 既定にしない。
- CUDA の occupancy / shuffle 数を R9700 の性能値として引用しない。

## 6. 今回の実装への反映

- Phase 1: vLLM の rows kernel と TensorFold の request 別 state を参考に、
  draft を depth 単位、verify を最大 4 request の Wavefront DECODE batch にした。
- Phase 2 以降: vLLM の host-sync なし選択、SGLang の Graph-safe workspace、
  hipfire の decode/prefill 分離、Strata の row scorer を比較対象にする。
- 数値は外部実装と一致しない前提とし、target-only greedy ID を最終 gate にする。
