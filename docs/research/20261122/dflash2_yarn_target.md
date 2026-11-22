# DFlash2 と Static YaRN の併用（Gate 5）

- 実施日: 2026-11-22
- target: `models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`
- drafter: `models/kusuriya-labs/Qwen3.8-27B-DFlash2-EXL3-INT8-4bpw`
- GPU: Radeon AI PRO R9700（gfx1201）× 2（`--device 0,1`、TP=2）
- build: HEAD `438879e2` + 作業ツリー
- 温度 0、`--dflash2-drafts 6`、`--arena-gib 30`

## 目的

Gate 5 では、Target の Static YaRN（factor 1 / 2 / 4）と DFlash2 を
併用したときの acceptance rate と tokens/s を未測定としていた。Drafter が target の
factor を継承せず、checkpoint の `rope_theta` と `rope_type=default` で動くことの確認を
含める。

## 短文（51 token prompt、128 token 生成）

prompt は chat template を通した 51 token。各 factor を 2 run ずつ実行した。

| factor | run | decode tok/s | mean accepted | rounds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 68.62 | 1.822 | 45 |
| 1 | 2 | 68.47 | 1.822 | 45 |
| 2 | 1 | 67.13 | 1.761 | 46 |
| 2 | 2 | 67.07 | 1.761 | 46 |
| 4 | 1 | 68.59 | 1.822 | 45 |
| 4 | 2 | 68.64 | 1.822 | 45 |

起動ログは factor 2 で `ROPE profile: type=yarn ... attention_factor=1.069314718`、
factor 4 で `attention_factor=1.138629436`、drafter は全 factor で
`DFLASH2 rope: type=default ... source=checkpoint` を出した。target の factor が
drafter へ伝播していないことを起動ログで確認できる。

factor 2 だけ mean accepted が 1.761 へ下がり、decode は約 2% 遅い。factor 1 と 4 は
同じ受理数だった。greedy の生成列は全 factor・全 run で一致した
（`GENERATED_IDS` の sha1 = `1524af36c720f586e37b24b9eed162357f4010f5`）。
factor 2 は target の logits が変わることで受理数だけが変わり、確定 token は同じだった。

## 長位置（262,000 token prompt、32 token 生成）

2,203 token の corpus を繰り返して 262,000 token の prompt を作り、
`--max-seq-len 262144`、`--kv-cache-dtype fp8` で実行した。繰り返し corpus のため
受理率は自然文の値ではない。位置依存の起動・prefill・decode が成立することの確認に
限定する。

| factor | prefill | decode tok/s | mean accepted | rounds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 323,215 ms | 22.34 | 2.875 | 8 |
| 2 | 323,262 ms | 19.76 | 2.444 | 9 |
| 4 | 325,715 ms | 19.76 | 2.444 | 9 |

262,000 token の生成列は全 factor で一致した
（`GENERATED_IDS` の sha1 = `9b613cfae22d62c18aa6c50dc354ef9031dd673a`）。

## 524,288 位置（1 run、32 token）

factor 2、524,000 token prompt、`--max-seq-len 524288`、32 token 生成、FP8 KV の 1 run。

| prefill | prefill tok/s | decode tok/s | mean accepted | rounds |
| ---: | ---: | ---: | ---: | ---: |
| 1,090,479 ms | 481 | 10.20 | 2.100 | 10 |

`GENERATED_IDS` の sha1 は `793fcaf0546301f915c116d25b7312cf2692e41d`。32 token の生成は
`4% driven mostly by a new analytics pipeline that was not decommissioned after its pilot.
Storage costs were flat. The team recommended shutting down idle staging clusters`
で、反復崩壊や文字化けは無かった。8 token の先行 run（prefill 1,091,051 ms、
decode 10.86 tok/s）も同じ傾向である。1 run のみで run 間のばらつきを評価していないため、
性能 baseline ではなく容量と経路の成立確認として扱う。262,000（22.34 tok/s）から
524,000（10.20 tok/s）へ decode が半分以下になるのは、KV の可視長が 2 倍になり、
split-KV の page 列挙と attention の読み出しが増えるためである。この切り分けは未計測で、
仮説として記録する。

## 結論

- Target の factor 1 / 2 / 4 と DFlash2 の併用は起動・生成とも成立する。
- Drafter は target の factor を継承せず、checkpoint の default RoPE を使う。
- 262,144 位置でも prefill と投機 decode が完了する。524,288 位置も 32 token 生成で
  完了し、prefill 1,090 s、decode 10.20 tok/s、生成文は発散しなかった。
- この環境の DFlash2 checkpoint は `rope_type=default` のみで、`yarn` を持つ
  checkpoint は無い。drafter の RoPE を default に限定する現在の契約を維持する。
