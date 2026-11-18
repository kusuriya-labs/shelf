# Shelf — Hy4-flash 専用 fork

Hy4-flash のテキスト生成を AMD Radeon AI PRO R9700 1 台と V620 2 台で実行する。expert parallel は 3 rank、本体は EXL3-INT8、Engram は FP8、KV は Octave。付属の DFly2 で 5 token を先読みできる。

## クイックスタート

[このホストのビルド手順](docs/research/20261119/hy4_flash_plan.md#このホストでのビルドと入力)で `build-hy4/shelf-compute` を作成する。入力は空白区切りの token ID を保存したテキストファイル。次の例は校正列の先頭 4 token から生成する。

```bash
quantroot="$HOME/datapool/rocm-exl3-rdna2"
pilot="$quantroot/hy4-flash-pilot"
printf '%s\n' '120000 802 15063 2576' > prompt-ids.txt
python3 tools/hy4_flash/locked.py ep3 build-hy4/shelf-compute hy4 generate \
  --model-dir "$quantroot/models/Hy4-flash-exl3-4bpw-mul24" \
  --engram-dir "$pilot/engram-e4m3fn" \
  --dfly-dir "$pilot/shelf-dfly/Hy4-flash-dfly-4bpw" \
  --placement "$pilot/shelf-placement/owners-4bpw.json" \
  --input-ids-file prompt-ids.txt --max-new-tokens 128 --chunk 1024
```

生成した token ID は `GENERATED_IDS` に出力される。GPU の予約量と速度は[性能記録](docs/perf/current.md#hy4-flash-専用-fork-の-ep3)を参照。

## Contributors

- **jyohukuchan** — EXL3-INT8 開発 / EXL3-INT8 カーネル最適化 / 構造化出力 etc

## License

MIT — [LICENSE](LICENSE)
