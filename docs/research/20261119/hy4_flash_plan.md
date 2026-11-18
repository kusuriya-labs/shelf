# Hy4-flash 専用 fork の計画

対象は Hy4-flash のテキスト生成。原本の text_config と tensor header、rocm_exl3 の組立仕様を照合する。最初は decoder 平均 4 bit、head 6 bit の EXL3-INT8、Engram はホスト側 FP8 表から必要行を集める。KV は Octave を使う。

| 段階 | 実装・検証 | 完了の判定 |
|---|---|---|
| 入力 | 原本から Engram 4 表を分割読み込みして FP8 に変換し、行単位で復元 | 各表の形状・丸め・ゼロ行・出力サイズの検査 |
| CPU | iHC の縮約と展開、sliding と疎な MLA の causal 境界、indexer の未完グループ、router と latent expert、Engram の履歴を原実装に合わせる | 短い固定列を一括・逐次で比較 |
| EP | R9700 が attention、shared expert、出力を担当。V620 は WMMA に依存しない EXL3-INT8 routed expert を担当。各 GPU に expert を分割する | 両 ISA へのオフラインコンパイル、実機で各 rank の部分和と単一入力の参照値を比較 |
| 本体 | checkpoint の全 tensor と量子化付帯情報を照合。42 層、層別 KV、4 stream の実行を既存 CLI に接続 | prefill、decode の最終 logits と greedy 列を比較 |
| DFly | 本体の量子化後に付属 5 層を追加量子化。層 25・29・33・37・41 の KV/index を渡し、候補 5 token を検証 | 全採択、部分採択、全棄却の KV・Engram 履歴・位置を通常生成と照合 |
| 受入 | 重み・各 rank の作業域を確保後、2,097,152 token 分の予約を試す。失敗時は解放して 1,048,576 で再試行 | rank 別の要求量・空き容量・結果を記録。推論上限は config の 1,048,576 |
| 最終 | 4 bit 版の正しさと性能検証後に 3.5 bpw を作成 | 同じ入力で品質、実保存量、速度を再測定 |

## 実機解放後の順序

本体の conversion_report.json と head_complete を確認する。1 分 load average が 8 未満のとき、/tmp/r9700.lock と V620 の /tmp/v620.lock、/tmp/v620-76a08.lock を同一順序で保持し、ROCR_VISIBLE_DEVICES に対象 UUID を指定する。PCIe peer access、転送帯域、実際の空き容量を先に測る。小形状で正しさを確認し、全文脈予約、pp2048、DFly 有効時の decode の順に測る。予約成功と生成成功は別々に記録する。

性能目標は pp2048 で 500 tok/s、decode で 40 tok/s。ウォームアップ後、同一入力、同一出力長で複数回実行し、生成済み token 数を実時間で割る。DFly の受理長分布も記録する。各層の選択頻度と rank ごとの容量から expert 配置を決める。最初は均等頻度を仮定し、実測で hot expert を高帯域 rank に寄せる。送るのは latent activation と routing 情報、戻すのは rank 内で加重和を取った出力。prefill は rank ごとにまとめた転送、decode は shared expert との重畳を比較する。計測には HIP event と壁時計を用いる。

## 実装に合わせた検証手順

現在の段階と測定値は [進捗記録](hy4_flash_progress.md) にまとめる。専用構成は SHELF_HY4_ONLY=ON。R9700 を rank 0、V620 を rank 1・2 とし、4 表はホストに置く。本体の配布形式では量子化しなかった indexer の pooling の線形だけは原本の精度を使う。

1. 本体完了前は tools/hy4_flash/check_chain.py で、完成済み層まで同じ 4 token の CPU 出力を保存する。配置表も保存して同じ条件で再開する。本体完了後、同じコマンドへの --body の指定で 6 bit head の FP32 logits も保存する。
2. 本体の head_complete と実出力の conversion_report.json を確認し、原本、組立済み本体、校正 work、DFly 出力先をこの順に quant_dfly.py に渡す。量子化側と同じ mul24 用 rocm_exl3 の Python 環境で実行する。コンテナ使用時はホストで locked.py v620 を取り、docker run に -e ROCR_VISIBLE_DEVICES を渡す。
3. ホストで locked.py ep3 を経由して qualify.py を起動する。引数は shelf-compute、本体、Engram ディレクトリ、DFly、配置 JSON、CPU 照合結果ディレクトリ、校正 safetensors、実測記録先。温度 0 の logits と生成列、正の温度で同じ seed の DFly 生成の再現性、実予約の順に検査する。
4. 同一常駐モデルで pp2048 と decode を 4 回測定し、初回を除いて中央値を取る。PCIe 転送・top-k のホスト同期、kv_b_proj の再展開、R9700 上の行列演算を分けて評価する。直列にかかる時間が支配的な部分から高速化し、各変更後に CPU 照合を繰り返す。
