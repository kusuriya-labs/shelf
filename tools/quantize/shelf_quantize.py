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


def normalize_gdn_weights(output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    converted = set()
    for path in sorted(output.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as source:
            names = [name for name in source.keys()
                     if name.endswith(('.linear_attn.norm.weight', '.linear_attn.A_log'))
                     and source.get_slice(name).get_dtype() == 'F32']
            metadata = source.metadata()
        if not names:
            continue
        tensors = load_file(path, device='cpu')
        for name in names:
            tensors[name] = tensors[name].to(torch.bfloat16)
        with tempfile.TemporaryDirectory(prefix='.shelf-', dir=output) as temp:
            target = Path(temp) / path.name
            save_file(tensors, str(target), metadata=metadata)
            os.replace(target, path)
        converted.update(names)
    config_path = output / 'quantization_config.json'
    if converted and config_path.exists():
        config = json.loads(config_path.read_text())
        for module in config.get('tensor_storage', {}).values():
            for name, tensor in module.get('stored_tensors', {}).items():
                if name in converted:
                    tensor['dtype'] = 'torch.bfloat16'
                    tensor['n_bytes'] //= 2
        config_path.write_text(json.dumps(config, indent=4) + '\n')


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
    normalize_gdn_weights(args.output)
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


DRAFT_ARCHITECTURE = 'DFlash2DraftModel'
DRAFT_LINEAR_SUFFIXES = (
    '.self_attn.q_proj.weight', '.self_attn.k_proj.weight',
    '.self_attn.v_proj.weight', '.self_attn.o_proj.weight',
    '.mlp.gate_proj.weight', '.mlp.up_proj.weight', '.mlp.down_proj.weight',
)


def is_draft_linear(name):
    return name == 'fc.weight' or name.endswith(DRAFT_LINEAR_SUFFIXES)


def convert_draft(args):
    with (args.model / 'config.json').open() as f:
        config = json.load(f)
    if config.get('architectures') != [DRAFT_ARCHITECTURE]:
        raise ValueError(f'DFlash2DraftModel ではありません: {config.get("architectures")}')
    if 'quantization_config' in config:
        raise ValueError('既に量子化された checkpoint です')
    if not args.bpw.is_integer() or not 1 <= args.bpw <= 8:
        raise ValueError('bpw は 1 以上 8 以下の整数にしてください')
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    info = verify_extension()
    started = datetime.now(timezone.utc).isoformat()
    device = f'cuda:{args.device}'
    shards = sorted(args.model.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'.safetensors がありません: {args.model}')
    out = {}
    expected = set()
    produced = set()
    quantized = 0
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                if name in expected:
                    raise ValueError(f'重複した tensor: {name}')
                expected.add(name)
                tensor = source.get_tensor(name)
                if not is_draft_linear(name):
                    out[name] = tensor.contiguous()
                    produced.add(name)
                    continue
                weight = tensor.t().contiguous().float()
                k, n = weight.shape
                if k % 128 or n % 128:
                    raise ValueError(f'128 の倍数でない linear: {name} {tuple(weight.shape)}')
                H_data = {
                    'H': torch.empty(k, k, dtype=torch.float32, device='meta'),
                    'first_key': name,
                    'count': 0,
                    'finalized': False,
                    'num_total': 0,
                    'inf_nan': torch.zeros(2, dtype=torch.long, device=device),
                    'device': torch.device(device),
                    'L': None,
                }
                quant_args = {
                    'seed': quantized + args.seed,
                    'K': int(args.bpw),
                    'devices': [device],
                    'device_ratios': None,
                    'apply_out_scales': True,
                    'mul1': True,
                }
                _, error, tensors = quantize_exl3(
                    weight, H_data, quant_args, False,
                    progress_str=f' -- <step>: {name}',
                )
                del weight
                prefix = name[: -len('.weight')]
                out[prefix + '.trellis'] = tensors['trellis'].to('cpu').contiguous()
                out[prefix + '.suh'] = tensors['suh'].to('cpu').contiguous()
                out[prefix + '.svh'] = tensors['svh'].to('cpu').contiguous()
                out[prefix + '.mul1'] = tensors['mul1'].to('cpu').contiguous()
                produced.add(name)
                quantized += 1
                print(
                    f' -- Quantized: {name}  k={k} n={n}  bpw={quant_args["K"]}  '
                    f'rmse={error:.6f}  [{quantized}]',
                    flush=True,
                )
    if produced != expected:
        raise ValueError(f'tensor が一致しません: {sorted(expected ^ produced)}')

    args.output.mkdir(parents=True)
    save_file(out, str(args.output / 'model.safetensors'))
    config['quantization_config'] = {
        'quant_method': 'exl3',
        'version': '1.5.0',
        'bits': args.bpw,
        'codebook': 'exl3-int8',
        'multiplier': info['multiplier'],
        'calibration': None,
        'out_scales': 'always',
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    run(sys.executable, QUANT / 'validate_exl3.py', args.output)
    report = {'model': str(args.model.resolve()), 'output': str(args.output.resolve()),
              'bpw': args.bpw, 'seed': args.seed, 'device': args.device,
              'calibration': None, 'quantized_linears': quantized,
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
    d = sub.add_parser('convert-draft', help='DFlash2 drafter を calibration なしで量子化')
    for name in ('model', 'output'):
        d.add_argument('--' + name, type=Path, required=True)
    d.add_argument('--bpw', type=float, required=True)
    d.add_argument('--seed', type=int, default=0)
    d.add_argument('--device', type=int, default=0)
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
    elif args.command == 'convert-draft':
        if args.seed < 0 or args.device < 0:
            p.error('seed と device は 0 以上にしてください')
        convert_draft(args)
    else:
        quality(args)


if __name__ == '__main__':
    main()
