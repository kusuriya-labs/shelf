import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=Path('build-gfx1201'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
        PATH='/tmp/venv-plamo/bin:/tmp/venv-so/bin:' + env['PATH'],
        SHELF_BUILD_DIR=str(args.build_dir.resolve()),
        SHELF_PLAMO_MODEL_DIR='/home/homelab1/datapool/ai_models/safetensors/plamo-3-translate-2b',
        SHELF_PLAMO_EXL3_MODEL_DIR='/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    results = []
    for test in ['test_plamo3_server.py','test_plamo3_exl3_real.py','test_plamo3_exl3_synthetic.py',
                 'test_plamo3_template.py','test_plamo3_startup.py']:
        command = ['flock','/tmp/r9700.lock','/tmp/venv-plamo/bin/python',str(root/'tests/server'/test)]
        with (args.output / f'{test}.log').open('w') as log:
            result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append({'test':test,'exit_code':result.returncode})
        (args.output/'tests.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
        if result.returncode:
            raise RuntimeError(f'{test} failed')


if __name__ == '__main__':
    main()
