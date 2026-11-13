import os
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', SHELF_EXL3_STAGE_TRACE='1',
           LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',
           PATH='/tmp/venv-so/bin:' + env['PATH'])
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
draft = '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
exe = folder / 'shelf-compute.trace'
cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '150', str(exe),
       '--model-dir', model, '--input-ids-file', str(root / 'docs/research/20261113/exl3_general/prompt_code.ids'),
       '--max-new-tokens', '24', '--max-seq-len', '512', '--temperature', '0',
       '--dflash2-model-dir', draft, '--dflash2-drafts', '7', '--dflash2-tree-budget', '0']
result = subprocess.run(cmd, env=env, text=True, capture_output=True)
(folder / 'trace-c1-fixed.log').write_text(result.stdout + '\n' + result.stderr)
print('c1 fixed', result.returncode, flush=True)
if result.returncode:
    raise RuntimeError(result.stderr[-2000:])
for concurrency, spec in [(2, False), (3, False), (5, False), (6, False), (7, False), (2, True)]:
    tag = f'c{concurrency}-{int(spec)}-extra'
    log = folder / f'trace-{tag}.log'
    cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '-k', '10', '--signal=INT', '150',
           '/tmp/venv-so/bin/python', '-I', 'docs/research/20261113/exl3_general/exl3_batch.py', '--compute', str(exe),
           '--model', model, '--concurrency', str(concurrency), '--runs', '1', '--tokens', '16',
           '--trace', '--log', str(log), '--prompts', 'docs/research/20261113/exl3_general/prompt_code.ids',
           'docs/research/20261113/exl3_general/prompt_prose.ids', 'docs/research/20261113/exl3_general/prompt_ja.ids']
    if spec:
        cmd += ['--spec', '--drafter', draft]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (folder / f'trace-{tag}.json').write_text(result.stdout)
    print(tag, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
