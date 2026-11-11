import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
fit = pathlib.Path(__file__).with_name('fit_cost.py')
base_root = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else root
result = subprocess.run([sys.executable, '-I', str(fit), str(root), str(base_root)], text=True,
                        capture_output=True, check=True)
lines = result.stdout.splitlines()
if lines[0] != 'baseline = {' or lines[2] != '}' or lines[3] != 'spec = {' or lines[-1] != '}':
    raise RuntimeError('unexpected cost fit format')
baseline = [float(value) for value in lines[1].split(', ')]
spec = [[float(value) for value in line.strip().removeprefix('{').removesuffix('},').split(', ')]
        for line in lines[4:-1]]
if len(baseline) != 65 or len(spec) != 8 or any(len(row) != 7 for row in spec):
    raise RuntimeError('unexpected cost fit shape')
text = ['#pragma once', '', '#include <shelf/models/qwen35/runtime/dflash2_adaptive.h>', '',
        'namespace shelf::qwen35::runtime::adaptive {', '',
        'inline CostModel qwen38_27b_exl3_fp8_cost_model() {', '    CostModel model;',
        '    model.baseline_ms = {',
        '        ' + ', '.join(f'{value:.4f}' for value in baseline), '    };',
        '    model.speculation_ms.resize(9);']
for sessions, row in enumerate(spec, 1):
    points = ', '.join(f'{{{sessions * (k + 1)}u, {value:.4f}}}'
                       for k, value in enumerate(row, 1))
    text.append(f'    model.speculation_ms[{sessions}] = {{{points}}};')
text += ['    return model;', '}', '', '}']
out.write_text('\n'.join(text) + '\n')
print(result.stderr, end='', file=sys.stderr)
