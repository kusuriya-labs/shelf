# research

R&D 記録の正本である。実験の経緯、採否判断、棄却の根拠をここに残す。

確定した contract は `docs/developer/` へ、現在の性能値は `docs/perf/` へ移す。
`docs/research/` に contract や性能値の複製を残さない。

## 配置規約

- `docs/research/yyyyMMdd/****.md` の形式で置く。
  `yyyyMMdd` はその実験を実施した日とする。
- 複数ファイルにまたがる研究は、日付ディレクトリ配下にサブディレクトリを置く
  （例: `20261022/prefill_kernel_optimization/`）。
- 単発の調査・横断的な知見は、日付ディレクトリ直下に置く
  （例: `20261022/optimization_findings.md`）。

## 記録規約

- 実測値のみを記録する。未検証の仮説は仮説と明示する。
- 個別実験には ID を振り、ACCEPT / REJECT の結論と根拠を残す。
- 測定原則は `docs/perf/methodology.md` を正本とし、ここに複製しない。
