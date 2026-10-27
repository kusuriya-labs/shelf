#!/usr/bin/env python3
"""shelf-validate 用 入力 fixture の生成（開発時のみ使用）。

モデル同梱の tokenizer.json（+ チャットテンプレート）でプロンプトを
テンプレート適用し、完成形の token ids を C++ ヘッダとして書き出す。
CI と実行はこのスクリプトに依存しない（成果物のヘッダだけを同梱する）。

使い方（リポジトリルートで実行）:
    python3 apps/fixtures/make_fixture.py

仕様: draft-docs/shelf-validate.md §3
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

# 入力プロンプト。prefill 256 tok に厳密一致させ、かつ decode が
# 128..256 tok の範囲で StopToken に到達するよう「約 150 語で即答」を指示する
# （思考モード OFF と組み合わせる。長さは --tokens と厳密一致させること）。
PROMPT = (
    "Please answer the following question about the history of celestial navigation "
    "and its influence on the age of exploration. Cover the ancient use of the sun and "
    "stars for orientation, the development of the astrolabe, the quadrant, the "
    "cross-staff and the backstaff, how sailors determined longitude at sea, the marine "
    "chronometer, and the transition to radio and satellite navigation. Include concrete "
    "historical examples, dates and instrument names. "
    "For each instrument, state the practical problem it solved and how it was used on a "
    "moving deck, and assess its accuracy compared with what replaced it. Mention the "
    "navigators and voyages associated with each instrument and the years major navies "
    "adopted it. Keep the chronology clear and avoid repeating anecdotes. "
    "Answer in about 150 words, in plain prose: no planning, no outline, no headings, "
    "and no preamble."
    " The reader knows nothing about nautical terms, so explain them briefly."
    " The answer will be printed on a single page of a printed handout."
    " Treat the question as an exam prompt and give only the final answer text."
    " Assume the reader has heard of Columbus but not of Harrison or Halley."
    " Do not mention any modern software, websites, or online encyclopedias."
)

HEADER_TEMPLATE = """\
// shelf-validate 用 入力 fixture（自動生成ファイル — 手動編集禁止）。
//
// 生成コマンド（開発時のみ・モデル同梱の tokenizer.json を使用）:
//     python3 apps/fixtures/make_fixture.py
//
// チャットテンプレート適用済みの完成形。同一の ids を exact / approx の
// 両経路にそのまま適用する（docs/spec/validate.md §4）。
#pragma once

#include <cstddef>
#include <cstdint>

namespace shelf::apps {{

// 末尾はアシスタント生成開始位置（テンプレート適用後の全長）。
inline constexpr std::size_t kPrefillFixtureSize = {size};
inline constexpr std::int64_t kPrefillFixture[kPrefillFixtureSize] = {{
{body}
}};

}} // namespace shelf::apps
"""


def main() -> int:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-dir", default=str(repo / "models" / "Qwen3.5-4B"))
    ap.add_argument("--tokens", type=int, default=256, help="fixture の長さ（既定 256）")
    ap.add_argument("--out", default=str(here.parent / "prefill_fixture.hpp"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=False)
    # 思考モード OFF（テンプレートが空の   block を前置する）。
    # ON だと decode が思考だけで予算を消費し StopToken に到達しないため。
    out_ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # transformers 5.x は BatchEncoding（UserDict 系）で返すので input_ids を取り出す
    ids = out_ids if isinstance(out_ids, list) else out_ids["input_ids"]
    if len(ids) != args.tokens:
        # 長さを合わせるのは PROMPT の編集で行う（末尾切り詰めは
        # アシスタント開始トークンを失うため禁止）。
        print(
            f"error: 長さが不一致: 生成 {len(ids)} tok != 目標 {args.tokens} tok"
            f"（PROMPT を調整して再実行してください）",
            file=sys.stderr,
        )
        return 1

    rows = []
    line = "    "
    for i, t in enumerate(ids):
        piece = f"{t}, "
        if len(line) + len(piece) > 96:
            rows.append(line.rstrip())
            line = "    "
        line += piece
    rows.append(line.rstrip().rstrip(","))
    body = "\n".join(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER_TEMPLATE.format(size=len(ids), body=body), encoding="utf-8")

    # 生成先の末尾が「アシスタント生成開始」になっていることを目視用に表示。
    print(f"wrote {out} ({len(ids)} tokens)")
    print("tail text:", repr(tok.decode(ids[-8:])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
