import os, pathlib, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[4]
out=pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
prompt_dir=pathlib.Path(sys.argv[1])
script=root/'docs/research/20261111/dflash2_adaptive_scripts/compare.py'
for name in ('benchmark','japanese'):
 for n in (2,4,8,12,16,20,24,32):
  for rep in (1,2):
   for mode in ('off','fixed'):
    tag=f'{name}_{n}_r{rep}_{mode}'
    env=dict(os.environ, ERRF=str(out/f'{tag}.err'),GENF=str(out/f'{tag}.json'))
    env.pop('SHELF_GDN_VERIFY_CHUNKED', None)
    env.pop('SHELF_GDN_VERIFY_CHUNKED_EXACT', None)
    args=['flock','/tmp/r9700.lock','timeout','1500',sys.executable,'-I',str(script),str(n),'160',str(prompt_dir / f'{name}.json'),'--kv-cache-dtype','fp8','--max-num-batched-tokens','2048']
    if mode=='fixed': args += ['--dflash2-model-dir',os.environ['SHELF_DFLASH2_MODEL_DIR'],'--dflash2-drafts','7']
    print('start',tag,flush=True)
    res=subprocess.run(args,env=env,text=True,capture_output=True)
    (out/f'{tag}.out').write_text(res.stdout+res.stderr)
    print('end',tag,res.returncode,res.stdout.strip(),flush=True)
    if res.returncode: sys.exit(res.returncode)
