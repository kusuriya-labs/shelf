# third_party

外部依存のライセンス・由来をこのディレクトリ内で管理する（依存の追加は
`cmake/Dependencies.cmake` と docs/architecture/porting-provenance.md の更新を必須とする）。

## nlohmann/json

- 版本: 3.12.0（単頭ファイル、`include/nlohmann/json.hpp` / `json_fwd.hpp`）
- ライセンス: MIT（SPDX: `Niels Lohmann`, MIT.txt は nlohmann 本体ヘッダ内に表記）
- 用途: `src/artifact/safetensors` のヘッダ JSON パースのみ
- 導入経路: PhaseShift `vendor/nlohmann/` と同一ファイル（改変なし）
- 使用範囲: artifact モジュールの PRIVATE 依存。公開ヘッダに json 型を漏らさない。
