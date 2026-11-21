import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MULTIPLIER = 0x00DCD12D


def verify_extension():
    import torch
    sys.path.insert(0, str(ROOT))
    ext = importlib.import_module('exllamav3_ext')
    value = ext.exl3_mul1_multiplier()
    if value != MULTIPLIER:
        raise RuntimeError(f'拡張の multiplier が不正です: {value:#x}')
    path = Path(ext.__file__).resolve()
    if path.parent != ROOT:
        raise RuntimeError(f'想定外の拡張が読み込まれました: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'multiplier': value, 'path': str(path), 'sha256': digest}


def main():
    env = os.environ.copy()
    env.update(EXL3_BACKEND='rocm', EXL3_MUL1_MULT='0x00DCD12D', PYTORCH_ROCM_ARCH='gfx1201', MAX_JOBS='8')
    env.pop('EXLLAMA_NOCOMPILE', None)
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'], cwd=ROOT, env=env, check=True)
    info = verify_extension()
    print(f"multiplier: {info['multiplier']:#010x}\n拡張: {info['path']}\nSHA-256: {info['sha256']}")


if __name__ == '__main__':
    main()
