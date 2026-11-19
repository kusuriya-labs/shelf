# Hy4-flash の実装

Hy4 は専用の checkpoint loader と decoder を使用する。checkpoint の model_type、層数、形状、量子化形式を検証してから実行する。attention、KV、head、DFly と shared expert は rank 0 に置き、routed expert を配置表に従って各 rank へ置く。FP8 Engram はホストに保持し、必要行だけを取得する。GPU の一覧は ROCr の可視 device 順を使用し、gfx1201 以外を拒否する。

expert の転送には peer access を優先し、利用できない場合は pinned host buffer を使用する。各 rank の部分和は rank 0 で合算する。予約時に実空き量を確認し、必要に応じて expert を再配置する。収まらない場合は `Status::unsupported` を返す。配置作成手順は[操作方法](../user/hy4_flash.md)を参照する。

`tests/unit/test_hy4_flash.py` は CPU の Octave、Engram と expert 配置を検査する。実 checkpoint の層照合では `tools/hy4_flash/check_layer.py` で CPU 基準 fixture を作り、`test_hy4_flash_layer` に渡す。`tools/hy4_flash/check_linear.py` と `check_chain.py` は量子化線形と decoder の CPU 基準照合に使用する。DFly checkpoint の生成には `tools/hy4_flash/quant_dfly.py` を使う。

server の既定動作は Engram の mmap に `MADV_POPULATE_READ` を適用し、`mincore` で常駐状態を記録してから request を受け付ける。`SHELF_HY4_ENGRAM_POPULATE=0` では遅延 mmap を使用する。`SHELF_HY4_ENGRAM_MLOCK=1` は全表の memlock 上限を検査し、各 mmap を `mlock` して全ページの常駐を検証する。上限不足、システムコール失敗または固定後の常駐不足は起動を失敗させる。mmap の解除時に固定も解除される。固定なしでは populate 後の常駐は保証されない。起動時間と操作方法は[利用者向け手順](../user/hy4_flash.md)を参照する。

`SHELF_HY4_PROFILE_EP=1` では request 前後の Engram 常駐ページとプロセスの page fault 数を記録する。Hy4 server の compute 起動待ちは 900 秒である。
