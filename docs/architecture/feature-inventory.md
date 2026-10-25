# shelf — 推論エンジン機能インベントリ（XeStrata-exl3 / Strata 全列挙）

> 目的: Qwen3.5-4B 動作化で**契約の縫い目が実証済み**なった今、リファレンス 2 本の
> 推論エンジン機能を漏れなく列挙し、**機能契約の取捨選択**の材料にする。
> **本ドキュメントは設計入力であり、実装約束は含まない。**
> 調査日: 2026-10-26（両リポジトリ read-only 監査。Strata 134 項目 / XeStrata 93 項目を統合）

凡例:
- **X / S**: XeStrata-exl3 / Strata に実装あり（●=既定 ON・◐=opt-in・○=dev/実験・–=なし）
- **Seam**: shelf 側の既存継ぎ目の状態（✅ あり / 🟡 部分的 / ❌ なし）

## A. サービング / API（エンジン ABI の外側）

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| HTTP サーバー（エンジン常駐＋stdio） | ● | ● | JSON ⇄ token 列 line protocol | tokenizer/frontend 側 | ❌ |
| OpenAI Chat Completions（stream/tools） | ● | ● | messages+sampling → SSE deltas | session, cache | ❌ |
| OpenAI Responses API（stateless） | ● | ● | input items → typed SSE | 同上＋conv cache | ❌ |
| Anthropic Messages + count_tokens | ● | ● | Anthropic JSON ⇄ 同一経路 | reasoning 制御 | ❌ |
| 組み込み Web UI / チャット CLI | ● | ● | ブラウザ⇄API | — | ❌ |
| エンジン line protocol（GEN/BGEN/BT/BDONE/BYIELD/SAVE…） | ● | ● | **token ids+key → token ids+stats** | サーバーが detok 所有 | 🟡 shelf-run の decode ループが実質この雛形 |
| ワンショット CLI 生成（--greedy/--dump-logits） | ● | ● | ids → ids/logits | — | ✅ shelf-run |
| load/unload・VRAM 返却（lazy/idle-unload/reserve） | ◐ | ● | HTTP 制御 → プロセス/VRAM 状態 | VMM/cache tier | ❌ |
| セキュリティ（api_key/Host allowlist/CORS/offline） | ● | ● | Origin/Host/key → allow/deny | サーバーのみ | ❌ |
| observability endpoint（/health /status /slots /metrics + vLLM 名 Prometheus） | ● | ● | 内部状態 → JSON/text | engine INFO | ❌ |
| リクエストモニター＋ハードウェア telemetry | ◐ | ● | 環境 → ダッシュボード | OS 依存 | ❌ |
| 構造化出力（schema プロンプト＋検証。grammar 非対応） | – | ◐ | JSON schema ⇄ 検証済み reply | chat path | ❌ |
| Codex compaction cache（/compact でも prefix cache を生存させる） | – | ◐ | turn metadata → 再描画再利用 | Responses route, checkpoints | ❌ |
| キュー/ Admission（429+Retry-After、short-first） | ● | ● | backlog → 拒否/合流 | batch slots | ❌ |
| モデル差し替え（llama-swap 手順 / aliases） | ○doc | ○doc | プロキシ⇄設定 | プロセス単位 | ❌ |
| マネジメントブリッジ（typed JSON actions） | – | ◐ | {action,params} → result | CLI parity | ❌ |
| レプリカ（データ並列 N エンジン×1 front） | ◐ | ◐ | request → replica 割当 | multi-GPU | ❌ |

## B. セッション / 状態管理

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| Session 状態オブジェクト（KV+GDN+HC+PLE 履歴の所有者） | ● | ● | token 位置列 → 進化した状態 | 全 decode | ✅ executor の state_dev_（1 文脈限定） |
| **Checkpoint チェーン**（turn 境界/16K 毎/system root pin／**radix ツリーなし**） | ● | ● | prefix tokens → 再開オフセット | KV+recurrent 保存 | ✅ 方針決定済み（paged-kv-abi.md rev2.1、BlockKey=ID 専用） |
| checkpoint spread / near-tail / message-boundary | ◐ | ◐ | 保持戦略（間隔・末尾・編集耐性） | prompt cache | ❌ |
| pin 共有 prefix（strata_prefix / pin=N） | ● | ● | client 宣言 → 固定 checkpoint | parking | ❌ |
| **会話パーキング**（RAM に丸ごと退避、prefix 完全一致で復元、borrow） | ◐ | ● | 会話 ⇄ host snapshot | RAM guard, 状態形式 | ❌ |
| セッション永続化（save/restore ファイル、finger print 照合、圧縮） | ◐ | ● | 状態 ⇄ disk | slots API | ❌ |
| ホストメモリ guard（MemAvailable/cgroup/self limit） | ● | ● | OS 統計 → 許/拒 | parking/save/arena | ❌ |
| 会話 set-aside を disk スロットへ（DeepSeek ds4-slots: TTL/cap/shared prefix auto-pin） | – | ◐ | turns ⇄ slot files | engine 固有機能 | ❌ |
| slots=会話キャッシュ（バッチ slot が会話を保持、次 turn 復元） | ● | ● | slot 所有マップ | batch | ❌ |
| BYIELD（長い prefill が短い要求に道を開ける） | ● | ● | キュー公正性 | batch+cache | ❌ |
| NaN logits ガード（窓検出→破棄→透明リトライ、cache も捨てる） | ● | ● | logits → 中止信号 | 全 state | ❌ |
| watchdog / stall 復旧 / プロセス再 supervise | ● | ● | heartbeat → 停止/再起動 | lifeline | ❌ |
| 決定的再現実行（seed Philox・canonical CPU==GPU・exact compare recipe） | ◐ | ◐ | 同一設定 → 同一 token | sampler/canon kernels | 🟡 greedy bit 一致テストのみ |

## C. KV キャッシュ / ページング

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| ページ KV（page table＋位置 arena、K/V 2 プール） | ● | ● | 位置 → 物理ページ | attention kernel | ✅ **ABI rev2（canonical SeparatePools）** |
| QSA sparse attention（indexer top-k、選択ページのみ読む） | ● | ● | query+scores → 選択ブロック | KV pool | ❌（IR にも op なし） |
| KV 精度（fp16/int8/q4_0+Hadamard/k8v4） | ● | ● | cell → 量子化ページ | per-format kernel | ❌ |
| KV ストリーミング（権威コピー pinned RAM、VRAM=常駐スロット、**evict=CLOCK second-chance（XeStrata）**） | ● | ◐ | ページ需要 → 常駐 map | graphs/pinned | ❌ |
| 弾性 KV 成長（VMM map/unmap、アドレス不変=graph 生存） | ◐ | ◐ | VRAM 圧力 ⇄ KV | expert cache とトレード | ❌ |
| KV prefetch（次層の streamed 転送を現層 prefill と重ねる） | – | ◐ | staging copy | 2nd stream | ❌ |
| PLE/Engram n-gram ティア（数十 GB テーブルを SSD→RAM→row cache） | ● | ● | token 窓 → gather rows | I/O probe | ❌（モデル固有 op に注意） |
| expert hot profile（（layer,expert）ランキング→事前配置） | ● | ● | 統計 ⇄ profile.bin | cache tier | ❌ |
| ※ Strata の expert 追い出しは **CLOCK ではなく decayed-count(LFU 系)**。パーキングは LRU/spread | ● | ● | — | — | 追記: rev2.1 文書の「CLOCK」は KV ティア限定で正しい |

## D. バッチ / スケジューリング

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| **prefill チャンク**（chunk 8K、ring transfer、host planner） | ● | ● | token chunk → KV+state | cache tier | 🟡 rows≤64 固定、chunk 計画なし |
| short-read（≤64 token は decode 窓で読む） | ● | ● | tail tokens | prompt cache | ❌ |
| バッチスロット（N 会話×1 token/窓、exact 一致保証） | ◐ | ● | N session → 行=slot 窓 | session/copy | 🟡 多行 kernel は同一文脈のみ |
| batch-mtp（slot ごとに draft 検証） | ◐ | ◐ | 窓 rows×2 | spec | ❌ |
| batch-groups（layer split GPU 間で slot グループ回転） | ◐ | ◐ | groups → 並行窓 | split | ❌ |
| admission/yield/solo↔slot 移行 | ● | ● | キュー⇄slot | cache | ❌ |
| 窓 kernel 統合（ATTN_ROWS/GDN_GROUPS 1 launch 化） | ◐ | ◐ | 行を 1 launch に | kernels | ✅ 相当（rows 対応済み） |

## E. MoE / Expert 階層化（CPU・GPU 協調）

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| 3 層メモリ（VRAM cache / RAM arena(pinned,THP) / file 直読み） | ● | ● | (layer,expert) → slot/miss | expert_layout() | ❌（Phase 7 の本体） |
| 追随型 residency（decay counts で hottest in/coldest out、async） | ● | ● | ルーティング統計 → 配置 | graphs | ❌ |
| per-layer slot budget | ◐ | ● | budget → 層割当 | cache | ❌ |
| CPU expert pool（AVX2/512・canonical 数学=GPU と bit 一致） | ● | ● | x,ids → out rows | exchange buffers | ❌ |
| **misses の CPU ⇄ PCIe 分割**（実測で自動配分） | ● | ● | miss 集合 → {stream, cpu-compute} | probes | ❌ |
| doorbell（GPU 窓が host flag スピン待ちで CPU 行を合成） | ● | ● | ring → merge | graphs/device caps | ❌ |
| prefill CPU share（短い chunk で低ルーティング expert を CPU） | ● | ● | 行配分 | pool | ❌ |
| helper expert-cache GPU / remote experts / opt | ◐ | ◐ | hit on helper → rows | pinned host | ❌ |
| peer tier（P2P/NVLink で 2nd GPU が cache+計算） | – | ◐ | rows over P2P | link | ❌ |
| exchange buffer rotation（evict を residency に戻す） | ◐ | ◐ | pinned buffer pool | resident mode | ❌ |
| batched DMA（cudaMemcpyBatchAsync） | – | ◐ | copy 集合 | CUDA13 | ❌ |
| route tail skip（miss rank≥7 を 0 扱い）/ route bias | ◐ | ◐ | router logits → 計算 subset | **出力が変わる** | ❌ |
| foresight swap（予測で追加 slot 先埋め→計測遅いので off） | – | ○ | 予測 | — | ❌（反例として記録） |
| cross-process expert arena 共有（/dev/shm+lock+identity） | ◐ | ◐ | arena file → 2nd 起動短縮 | Linux | ❌ |
| low-RAM モード（resident budget / mmap page cache / SSD stager） | ● | ● | free RAM → tier 方針 | guard | ❌ |
| EXPDEC（窓 expert kernel、bit 一致共有メモリ stage） | ◐ | ◐ | blobs×rows | CUDA | ❌ |
| MoE route/merge の演算意味論と配置決定の分離（設計原則） | ● | ● | router→expert set / placement は別責務 | — | 🟡 IR に moe_route/moe_merge、実行系なし |

## F. 量子化 / Weight 形式

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| GGUF reader + inspect（ヘッダのみで bpw 表示） | ● | ● | file/URL → catalog | fingerprints | 🟡 safetensors のみ（artifact/common はある） |
| EXL3（archive→MatrixView descriptors→**dot 内 trellis 展開 kernel**、remap 非再量子化） | ◐Dev | – | model dir → descriptors+blob reads | quality gates | ❌ |
| i-quant 網羅（IQ1_M…IQ4_XS、MMVQ、CPU ggml parity） | ● | ● | blocks → dot | parity oracles | ❌ |
| K-quant / Q8 expert（prefill は FP16 dequant+BLAS、MMQ opt-in） | ● | ● | shards 直読み | RAM tier | ❌ |
| NVFP4（GPTQ/NVIDIA/QAD 3 系統、w4a8/w4a4、PLE 変種） | – | ● | blobs+scales | Blackwell caps | ❌ |
| MXFP4（DeepSeek 側、vendored ExL3 kernels） | – | ● | GGUF 直読み | dsv41 | ❌ |
| Gyro trellis（TQK6/7、ggml overlay、SYCL/Metal 移植） | – | ◐ | GGUF | overlay build | ❌ |
| 活性化量子化（q8_1/Q8_K activation image × int8 MMQ） | ● | ● | fp rows → 量子化積 | device caps | ❌ |
| HC/PLE 等の Q8_0/BF16 計時選択・compat-bf16 pack | ● | ● | 形式 → 実行選択 | pack tools | ❌ |
| 品質枠（**KL(P_BF16‖P_cand) 単一基準**・bpw 整合・speed gate・per-config 基準） | ● | ● | teacher vs cand → 表 | fixtures | ✅ 概念（f64 oracle parity。KL 計測は未） |
| KV 量子化（C 参照） | ● | ● | — | — | ❌ |

## G. マルチ GPU / ハードウェア

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| **layer split（stage 分割、token は 1 回/窓 host 経由 hand-off）** | ● | ● | GPUs+測定値 → 分割 K | session/KV per stage | ❌ |
| auto placement（VRAM/link/RAM/Zipf で予測、helper vs stage 価格比較） | ● | ● | 装置測定 → 計画 | probes | ❌ |
| pipeline-windows（窓を跨ぎ stage 重叠、状態コピーで roll back） | ◐ | ◐ | guessed rows → commit/revert | recurrent copy | ❌ |
| prefill pipe（chunk 計画 b/a 比、handoff events） | ◐ | ◐ | chunk plan | split | ❌ |
| prefill head alias（読み込み中 head 退避→大きな chunk） | ◐ | ◐ | head VRAM ⇄ buffers | — | ❌ |
| trim stage weights（自層のみロード） | – | ◐ | residency | split | ❌ |
| device enum/pin（PCI・順序規則・reserve 政策・エミュレーション env） | ● | ● | device list → 配置+予算 | tests seams | 🟡 --gpu N のみ |
| backend 網羅（CUDA/HIP/SYCL(ARC)/Metal/CPU floor） | ●(SYCL+HIP/CUDA) | ●(all) | device report → kernel family | **ベンダー文字列禁止ポリシー** | 🟡 HIP 実装・SYCL 保留（指示済み） |
| arch 既定セット（bit-exact 既定を arch 別に凍結） | ◐ | ● | arch → flag sets | all | ❌ |
| link/SSD/VRAM probes → 計画インプット | ● | ● | 実測 GB/s → plans | tier/scheduler | ❌ |
| UMA/APU/iGPU 対策（carve-out 計算、segmented 窓、770 動作記録） | ● | ● | device flags → headroom | — | ❌ |
| 2 モデル×2 カード（独立エンジン） | – | ◐ | per-card process | app | ❌ |
| OS メモリ特性対策（WDDM budget/pin cap、pagefile commit 警告、UMA register） | ◐ | ● | allocation policy | Windows/APU | ❌ |

## H. 投的デコード（推測）

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| **verify window**（1〜8 行を全層 1 回、per-row argmax=逐次 greedy と bit 一致、採択外で recurrent rollback） | ● | ● | last+k drafts → committed ids | state copy/restore | ❌（IR の rows 土台のみ） |
| MTP drafter（モデル自前層、2〜3 先出し、受容率 .85-.89、draft KV） | ● | ● | hidden → proposals＋draft K/V | VRAM budget | ❌（4B に MTP 層なし要確認） |
| suffix/prompt-lookup drafter（trigram index、コストゲート、零 GPU 費用） | ● | ● | history → candidates | per-bucket acceptance EMA | ❌ |
| lookup chain / extra draft sources レジストリ | ◐ | ◐ | 連結 | spec path | ❌ |
| **経済コントローラ**（E[committed]/T で source×k 毎ステップ選択、+5% 未満 k=0） | ● | ● | stats+cost model → {source,k} | window size | ❌ |
| coupled draft sampling / Gumbel 変種 | ◐ | ◐ | seed → drafts+受容 | sampler | ❌ |
| draft vocab subset（小さい head で VRAM 節約） | ◐ | ◐ | mask → 縮退 head | EXL3 非対応 | ❌ |
| DFlash2 / DSpark 外部 draft モデル | – | ◐ | draft pack → proposals | engines | ❌ |
| state rollback（窓破棄時の recurrent/KV 巻き戻し） | ● | ● | checkpoint copy → revert | verify | 🟡 reset 相当は zero-only |
| 窓ハッシュ/--spec-follow 決定的検証ハーネス | – | ◐ | run → hashes | CI | ❌ |

## I. サンプリング / トークン化

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| device sampler chain（penalty→top-k→top-p→min-p→temp→Philox(seed,pos)） | ● | ● | logits+params → ids | batch 行一致 | ❌（argmax のみ。kernel 資産あり） |
| logit_bias（±100、フィルタ前、device array） | ● | ● | ids→bias | batch/sycl 不可 | ❌ |
| logprobs/top_logprobs（pre-sampling 値 stream） | ◐ | ◐ | logits → JSON | API | ❌ |
| penalties の batch 窓制限（一致検証のため off） | ● | ● | — | batch | — 教訓 |
| penalties/seed の llama.cpp alias 互換 | ● | ● | key map | API | ❌ |
| tokenizer（BPE byte-level/SentencePiece。エンジン側は id のみ） | ●(Py) | ●(Py) | text ⇄ ids | pack dir | ❌（外側責務と明示済み） |
| chat template（Jinja HF 完全一致・golden test）＋stream 解析（thinking/tool 分離、部分 tag 送出禁止、early-close） | ● | ● | messages → ids / tokens → typed deltas | template caps | ❌ |
| reasoning 制御（effort/budget/auto-room、loop recovery、preserve） | ● | ● | request → 指示＋server 切断 | template | ❌ |
| shared-prefix 事前符号化（server 側の前 turn prefix から再トークン化） | ● | ● | text+history → 部分 ids | prompt cache | ❌ |
| EOS/stop/finish_reason（stop/length/cancel/nonfinite） | ● | ● | ids → 終了理由 | parser | 🟡 eos break のみ |

## J. ツール / MCP / マルチモーダル

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| ネイティブ tool call（XML envelope、whitelist、stream 安全解析、parallel） | ● | ● | text ⇄ tool_calls | template | ❌ |
| tool_call recovery（崩れ envelope 救済、opt-in） | – | ◐ | text → calls | parser | ❌ |
| DSML（DeepSeek tool 記述→共通形翻訳） | – | ● | text → calls | parser | ❌ |
| MCP client（モデルが外部 tools） | ● | ● | servers → tool defs/back | web path | ❌ |
| MCP server（エンジンを操作される側で提供） | ● | ● | MCP ⇄ setup/server 制御 | CLI parity | ❌ |
| Vision input（mmproj ViT helper、M-RoPE (t,h,w)、image 同一性キーで cache 一致） | ◐ | ● | pixels → embedding rows+2D pos | GENI/BGENI | ❌ |
| 画像生成（DiT+VAE daemon、/v1/images、GPU 共有ポリシー） | ◐ | ● | prompt → PNG | daemon | ❌ |
| embeddings/pooling endpoint | – | **–（意図的不在）** | — | — | 棚上げ妥当 |
| control vector / speed projection（残差 steer、安全性示唆、既定 off） | ○Exp | ◐Exp | vector GGUF → 活性化調整 | cache drop | ❌（要注意枠で記録） |

## K. 運用 / 診断 / 開発ツール

| Feature | X | S | 契約（入力→出力） | 結合 | Shelf Seam |
|---|---|---|---|---|---|
| CPU⇄GPU⇄ggml parity スイート（oracle vector、exhaustive CB2 等） | ● | ● | kernel → bit/近差 check | CI | ✅ 相当（hip kernel tests+oracle parity 文化） |
| bench ハーネス（tier 別 tok/s、upstream A/B judge gate） | ● | ● | config → record.json | — | 🟡 bench_gemv_hip 残存 |
| calibrate/tune（--pcie-frac/spec-min-p/workers 自動計測、証拠保存） | ● | ● | runs → tuned keys | probes | ❌ |
| 診断（trace env、stage timing、timeline JSON、--check、inspect） | ● | ● | env → logs/JSON | — | ❌ |
| memory planner（FITS/REFUSE 算術計画、device selftest） | ● | – | sizes+device → 計画 | 共有 sizing | ❌ |
| packaging（deb/rpm/docker/podman、signed updates） | ● | ● | src → artifacts | — | ❌ |
| モデル取得（SHA pin/fingerprint/resume/HF mirror/inspector） | ● | ● | repo → 検証済み local set | pack | ❌ |
| 文書規約（ja/en 対、由来注記、LOCAL.md 実測事実のみ、ベンダー文字列禁止） | ● | ● | — | 文化 | ✅ provenance 文書・境界テスト |
| 機能証拠としてのテストハーネス（parity/bench/batch/parking/serve mock…GPU 非依存層あり） | ● | ● | run → pass/fail | CI | ✅ 文化同一（unit/parity/integration の区別） |
| agent 診断 playbook / safety posture（local-first、pinned downloads、SECURITY） | ● | ● | — | — | ❌ |

---

## shelf 側の「契約として空いていた」縫い目（取捨選択の候補単位）

インベントリから導出される、**shelf が先に契约を切るべき単位**（実装ではない）:

1. **Session ABI**（B20/26/28/29）: 状態一式（paged KV + recurrent/conv + future: HC/PLE）の
   snapshot/restore/pin 契約。rev2.1 の checkpoint 方針の直下。**Seam 一部あり**
2. **PrefixCache/Checkpoint policy**（B21–25, C top）: 保持戦略（turn/spread/tail/boundary/pin）を
   ポリシーオブジェクトとして KV ABI から分離。**方針のみ**
3. **Batch/Scheduler contract**（D 全体, A10/14）: 「1 窓 = (rows, per-row (session_key, position))」
   に executor を一般化すると prefill/continuous-batch/spec-window が同じ契約に載る（現状 rows は
   単一 session 前提 — ここが最大の設計分岐）
4. **Verify-window / Spec contract**（H）: state rollback（copy-on-write checkpoint at row boundary）+
   accept/reject。IR の attention_gqa_step/gdn_step は rows 対応済みで**下地あり**
5. **ExpertTier + Placement**（E）: Phase 7 の 3 層+tier+CPU pool。canonical 数学（bit 一致）が
   parity 前提条件 — XeStrata/Strata 双方の最重量機能
6. **QuantSpec/Codec seam**（F）: `src/quant/` 新設。学習は EXL3 系: descriptors（展開せず
   kernel へ直行）+ 品質ゲート（KL per-config）。**§10 checklist 未成立項目の本体**
7. **Topology/Probe contract**（G）: device/link/VRAM/RAM 実測 → plan。multi-GPU は
   layer-split（pipeline）+ replicas（data-parallel）の 2 契約に分解可能
8. **Token/text boundary**（A/J/I）: エンジン ABI は token 列のみ（両者一致）— shelf-run で
   既に同じ形。**サーバー/tokenizer/template はエンジン契约の外**と明文化できる
9. **Guard/Telemetry**（B34/35, K）: NaN ガード・watchdog・parity/bench・determinism recipe。
   低コスト高価値
10. **Admission/Queue contract**（A10, D30/29/52）: 「拒否(429)/待機/slot 化/割り込み(BYIELD)」
    はサーバー層契約。エンジン ABI（=token 列 line protocol）の外に置ける — 
    shelf が API 層を持つときの最上位 seam
11. **Vision/multimodal input**（A6 GENI, J18/118）: 「embedding rows + (t,h,w) positions を
    注入する」だけの狭い契約。エンジン ABI の拡張点として先に型だけ決められる

## 調査で確定した訂正事項（過去の前提への影響）

- **ORCA.md はプロトコルではない**（OrcaRouter GGUF の packing 手順）
- **LLAMA_SWAP.md は expert offload ではない**（モデル切替 HTTP プロキシでの運用）
- **radix ツリー不在は両者一致**: XeStrata `program/conv_cache.hpp` 自身が
  "a radix cache's tree collapsed onto one branch" と明言 → **rev2.1 決定の追認**
- **追い出し**: KV ストリーミング（XeStrata）=CLOCK second-chance、Strata の expert tier=
  decayed-count LFU 系、parking=LRU/spread。「CLOCK 追い出し」の一般化に注意（rev2.1 文書を
  本日の監査結果で更新済み）
- Strata に **embeddings endpoint なし**（意図的）/ **grammar デコードなし**（schema-prompt+検証）
- 「GYRO は speculative」は誤解（**量子化ファミリー**。投機は MTP+suffix lookup+コントローラ）
