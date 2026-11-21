import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
QUANT = Path(__file__).resolve().parent
EXT = QUANT / 'exl3_int8'
sys.path.insert(0, str(EXT))
from build_ext import verify_extension


def run(*args, env=None):
    subprocess.run([str(x) for x in args], check=True, env=env)


def convert(args):
    import torch
    info = verify_extension()
    started = datetime.now(timezone.utc).isoformat()
    env = os.environ.copy()
    env.update(SHELF_QUANTIZE_SEED=str(args.seed), OMP_NUM_THREADS=str(args.threads))
    cmd = [sys.executable, EXT / 'convert.py', '-i', args.model, '-o', args.output,
           '-w', args.work, '-b', args.bpw, '-cb', 'mul1']
    for option, value in [('-cd', args.calibration), ('-cr', args.cal_rows), ('-cc', args.cal_cols)]:
        if value is not None:
            cmd.extend([option, value])
    run(*cmd, env=env)
    if args.ngram != 'none':
        with tempfile.TemporaryDirectory(prefix='shelf-ngram-', dir=args.work) as temp:
            target = Path(temp) / 'bundle'
            run(sys.executable, QUANT / 'ngram_quantize.py', 'convert', args.model, target,
                '--profile', args.ngram, '--model-family', args.ngram_model_family, '--threads', args.threads)
            for item in target.iterdir():
                dest = args.output / item.name
                if dest.exists():
                    raise FileExistsError(dest)
                shutil.move(str(item), str(dest))
    run(sys.executable, QUANT / 'validate_exl3.py', args.output)
    report = {'model': str(args.model.resolve()), 'output': str(args.output.resolve()),
              'work': str(args.work.resolve()), 'bpw': args.bpw, 'ngram': args.ngram,
              'ngram_model_family': args.ngram_model_family, 'threads': args.threads,
              'seed': args.seed, 'calibration': str(args.calibration.resolve()) if args.calibration else 'bundled',
              'cal_rows': args.cal_rows or 250, 'cal_cols': args.cal_cols or 2048,
              'shelf_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_commit': 'b922ccbd558237b13cc2e3bb25b87bc7cb2cbfe3',
              'pytorch': torch.__version__, 'rocm': torch.version.hip, 'extension': info,
              'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat()}
    (args.output / 'quantize_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def quality(args):
    with (args.reference / 'config.json').open() as f:
        config = json.load(f)
    vocab = config.get('vocab_size') or config.get('text_config', {}).get('vocab_size')
    if not isinstance(vocab, int) or vocab <= 0:
        raise ValueError('参照モデルの vocab_size を取得できません')
    with tempfile.TemporaryDirectory(prefix='shelf-quality-') as temp:
        baseline, candidate = Path(temp) / 'bf16.dump', Path(temp) / 'candidate.dump'
        common = ['quality', '--tokens-file', args.tokens, '--context', args.context,
                  '--positions', args.token_count, '--prefill-chunk', max(256, args.context),
                  '--arena-gib', 'auto', '--device', args.device, '--kv-cache-dtype', 'bf16']
        run(args.bench, *common, '--model-dir', args.reference, '--dump', baseline)
        print(f'語彙サイズ: {vocab}', flush=True)
        run(args.bench, *common, '--model-dir', args.model, '--dump', candidate, '--compare', baseline)


def main():
    p = argparse.ArgumentParser(description='EXL3-INT8 mul24 量子化ツール')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('build-ext', help='HIP 拡張をビルドして検証')
    c = sub.add_parser('convert', help='モデルを量子化して検証')
    for name in ('model', 'output', 'work'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--bpw', type=float, required=True)
    c.add_argument('--ngram', choices=['none', 'ngram4', 'ngram5'], default='none')
    c.add_argument('--ngram-model-family', choices=['auto', 'hy4', 'qwen4exp'], default='auto')
    c.add_argument('--threads', type=int, default=8)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--calibration', type=Path)
    c.add_argument('--cal-rows', type=int)
    c.add_argument('--cal-cols', type=int)
    v = sub.add_parser('validate', help='bundle を検証')
    v.add_argument('bundle', type=Path)
    q = sub.add_parser('quality', help='BF16 と KLD / top-1 を比較')
    q.add_argument('--model', type=Path, required=True)
    q.add_argument('--reference', type=Path, required=True)
    q.add_argument('--tokens', type=Path, required=True)
    q.add_argument('--token-count', type=int, required=True)
    q.add_argument('--context', type=int, default=256)
    q.add_argument('--device', type=int, default=0)
    q.add_argument('--bench', type=Path, default=ROOT / 'build/shelf-bench')
    args = p.parse_args()
    if args.command == 'build-ext':
        run(sys.executable, EXT / 'build_ext.py')
    elif args.command == 'validate':
        run(sys.executable, QUANT / 'validate_exl3.py', args.bundle)
    elif args.command == 'convert':
        if args.threads < 1 or args.seed < 0:
            p.error('threads は正数、seed は 0 以上にしてください')
        convert(args)
    else:
        quality(args)


if __name__ == '__main__':
    main()
