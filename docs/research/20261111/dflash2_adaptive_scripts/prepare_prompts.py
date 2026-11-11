import json
import pathlib
import sys

repo = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / 'tests/server'))
from support import prompt_ids

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
high = json.loads((repo / 'docs/research/20261110/batch_decode_scripts/rows_cases.json').read_text())
high = [case for case in high if len(case['input_ids']) >= 16]
japanese = [
    '日本の四季について、それぞれの特徴を説明してください。',
    '量子コンピュータの基本原理を平易に説明して。',
    'Pythonでクイックソートを実装して解説して。',
    'コーヒーの淹れ方のコツを教えてください。',
    'ブラックホールとは何か説明して。',
    '健康的な朝食のメニューを提案して。',
    '機械学習の過学習とは何か。',
    '京都の観光名所を5つ挙げて。',
]
texts = {
    'japanese': japanese,
    'shell': ['Write a detailed, long tutorial with many shell command examples for finding, sorting, inspecting, and safely managing large files in a directory.'],
    'code': ['Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests.'],
    'reasoning': ['Explain in detail why the square root of two is irrational, then develop several different proofs and discuss related mathematical examples at length.'],
}
cases = {'benchmark': high}
for name, prompts in texts.items():
    cases[name] = [{'id': f'{name}-{i}', 'input_ids': prompt_ids(text)} for i, text in enumerate(prompts)]
cases['mixed'] = [case for pair in zip(high, cases['japanese'] * len(high)) for case in pair]
for name, entries in cases.items():
    (out / f'{name}.json').write_text(json.dumps(entries, ensure_ascii=False))
    print(name, len(entries), [len(entry['input_ids']) for entry in entries[:3]])
