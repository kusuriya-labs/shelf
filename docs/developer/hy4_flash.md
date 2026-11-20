# Hy4-flash の実装

Hy4 は専用の checkpoint loader と decoder を使用する。MLA・sliding attention・EXL3 線形・WMMA の Hy4 固有実装は `models/hy4_flash` に置く。radix 選択は model 非依存の `kernels/fused/detail` を共有する。checkpoint の model_type、層数、形状、量子化形式を検証してから実行する。attention、KV、head、DFly と shared expert は rank 0 に置き、routed expert を配置表に従って各 rank へ置く。FP8 Engram はホストに保持し、必要行だけを取得する。GPU の一覧は ROCr の可視 device 順を使用し、gfx1201 以外を拒否する。

expert の転送は peer access を有効にし、送信側 rank の kernel から受信側 rank の領域へ書く push を既定とする。宛先バッファの再利用と転送完了は event で順序付ける。peer access を利用できない場合は pinned host buffer を使用する。`SHELF_HY4_EP_TRANSPORT=host` で host 経由、`peer` で従来の受信側 stream による `hipMemcpyPeerAsync` を選べる。各 rank の部分和は rank 0 で合算する。予約時に実空き量を確認し、必要に応じて expert を再配置する。収まらない場合は `Status::unsupported` を返す。配置作成手順は[操作方法](../user/hy4_flash.md)を参照する。

`--expert-parallel` と `--tensor-parallel` で EP × TP の構成を選ぶ。rank は `ep * tp_size + tp` の順で並び、配置表の値は expert の所有 EP group を表す。TP では次の matrix axis を分割する。

- N 分割: `q_b_proj`、`kv_b_proj`、`linear_gate`、MLP と shared expert の `gate_proj` / `up_proj`、expert の `gate_proj` / `up_proj`
- K 分割: `o_proj`、MLP と shared expert の `down_proj`、expert の `down_proj`
- replicated: `q_a_proj`、`kv_a_proj_with_mqa`、indexer、router、embedding、norm、hyper-connection、`routed_expert_down_proj` / `up_proj`、lm_head、DFly drafter

N / K の分割境界は 128 要素単位に揃える。attention は head を TP lane に割り当て、`learnable_sink_param` は lane ごとの head 範囲を使う。row-parallel の出力は TP group 内で bf16 の all-reduce を行い、routed expert の出力は TP group 内の reduce の後に EP group 間で reduce する。TP と EP は別の受信バッファを使う。all-reduce の順序が変わるため、EP 構成との logits 一致は bit 一致ではなく相対誤差で判定する。TP lane は全 rank が attention と KV を持つため、workspace は TP 時に 1 GiB、KV と index は全 rank に確保する。2Mi 予約は TP=2 でも成立するが、`generate` の予約長上限は 1Mi である。

`tests/unit/test_hy4_flash.py` は CPU の Octave、Engram と expert 配置を検査する。実 checkpoint の層照合では `tools/hy4_flash/check_layer.py` で CPU 基準 fixture を作り、`test_hy4_flash_layer` に渡す。`tools/hy4_flash/check_linear.py` と `check_chain.py` は量子化線形と decoder の CPU 基準照合に使用する。DFly checkpoint の生成には `tools/hy4_flash/quant_dfly.py` を使う。

server の既定動作は Engram の mmap に `MADV_POPULATE_READ` を適用し、`mincore` で常駐状態を記録してから request を受け付ける。`SHELF_HY4_ENGRAM_POPULATE=0` では遅延 mmap を使用する。`SHELF_HY4_ENGRAM_MLOCK=1` は全表の memlock 上限を検査し、各 mmap を `mlock` して全ページの常駐を検証する。上限不足、システムコール失敗または固定後の常駐不足は起動を失敗させる。mmap の解除時に固定も解除される。固定なしでは populate 後の常駐は保証されない。起動時間と操作方法は[利用者向け手順](../user/hy4_flash.md)を参照する。

`SHELF_HY4_PROFILE_EP=1` では request 前後の Engram 常駐ページとプロセスの page fault 数を記録する。Hy4 server の compute 起動待ちは 900 秒である。
