import collections
import json
import re
import statistics
import subprocess
from pathlib import Path


def stats(values):
    return {'n':len(values),'mean':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,'median':statistics.median(values),'min':min(values),'max':max(values)} if values else None


def analyze_asm(text,name):
    m=re.search(r'^'+re.escape(name)+r':.*?^\.Lfunc_end\d+:',text,re.M|re.S)
    if not m:return {}
    body=m[0];instructions=[]
    for line in body.splitlines():
        op=re.match(r'^\s+([a-z][a-z0-9_]+)\s',line)
        if op:instructions.append(op[1])
    counts=collections.Counter(instructions)
    positions=[i for i,op in enumerate(instructions) if op.startswith('v_wmma')]
    gaps=[b-a for a,b in zip(positions,positions[1:])]
    blocks=[]
    for block in re.split(r'(?=^\.LBB\d+_\d+:)',body,flags=re.M):
        ops=[m[1] for m in re.finditer(r'^\s+([a-z][a-z0-9_]+)\s',block,re.M)]
        if any(x.startswith('v_wmma') for x in ops):blocks.append({'label':block.splitlines()[0],'opcodes':dict(collections.Counter(ops)),'instructions':len(ops)})
    return {'instruction_count':len(instructions),'nontemporal_load_hints':body.count('TH_LOAD_NT'),'wait_positions':[f'{i} {op}' for i,op in enumerate(instructions) if op.startswith('s_wait')],'opcodes':dict(counts),'wmma_instruction_distance':stats(gaps),'wmma_blocks':blocks}


def main():
    root=Path(__file__).resolve().parents[3];build=root/'build-gfx1201/plamo-linear2';out=root/'docs/research/20261112'
    base=json.loads((build/'compiler/baseline/resources.json').read_text())
    resource=[]
    tus=base['translation_units']
    for variant in ('accepted-prefill','rowscale-prefill','final-nocap-prefill'):
        resource_path=build/'compiler'/variant/'resources.json'
        if resource_path.exists():
            for tu in json.loads(resource_path.read_text())['translation_units']:
                tu['resource_variant']=variant
                tus.append(tu)
    for tu in tus:
        family=Path(tu['source']).stem
        resource_dir=build/'compiler'/tu.get('resource_variant','baseline')/family
        assembly=next(resource_dir.glob('*-gfx1201.s')).read_text()
        for entry in tu['remarks']:
            name=entry['function'];demangled=subprocess.check_output(['/opt/rocm/core-7.14/lib/llvm/bin/llvm-cxxfilt',name],text=True).strip()
            resource.append({'family':family,'resource_variant':tu.get('resource_variant','baseline'),'name':demangled.split('(')[0].strip() if '(anonymous namespace)' not in demangled else demangled.replace('shelf::kernel::(anonymous namespace)::','').split('(')[0].strip(),**entry,'isa':analyze_asm(assembly,name)})
    linear=json.loads((out/'plamo3_exl3_linear2.json').read_text());effective=[]
    body_nk=24*(3072*2560+2*512*2560+2560*3072+2*10240*2560+2560*10240);head_nk=107520*2560
    for event in linear['semantic_events']:
        if event.get('within_measurement_window') and event['outputs']>0 and event['file'].startswith(('mix-before-exact-','mix-final2-','mix-prefill-','mix-rowscale-','mix-nocap-')):
            ops=2*(event['rows']*body_nk+event['outputs']*head_nk);us=event['us']['LINEAR_EXL3_INT8'];tops=ops/(us*1e6)
            effective.append({'file':event['file'],'concurrency':event['concurrency'],'rows':event['rows'],'outputs':event['outputs'],'band':event['band'],'useful_ops':ops,'linear_us':us,'effective_tops':tops,'dense_peak_percent':100*tops/383.0})
    groups={}
    for phase in ('before-exact','final2','prefill','rowscale','nocap'):
        for c in (64,128):
            for band in ('1-64','65-128','129-255','256-511','512-2048'):
                es=[e for e in effective if e['file'].startswith('mix-'+phase+'-') and e['concurrency']==c and e['band']==band]
                if es:groups[f'{phase}-c{c}-{band}']={'rows':stats([e['rows'] for e in es]),'linear_us':stats([e['linear_us'] for e in es]),'tops':stats([e['effective_tops'] for e in es]),'peak_percent':stats([e['dense_peak_percent'] for e in es])}
    variants={}
    for directory in sorted((build/'compiler').iterdir()):
        if not directory.is_dir() or directory.name=='baseline':continue
        data={}
        if (directory/'build.json').exists():data['build']=json.loads((directory/'build.json').read_text())
        if (directory/'micro.jsonl').exists():
            data['runs']=[json.loads(x) for x in (directory/'micro.jsonl').read_text().splitlines() if x.endswith('}')]
            for r in data['runs']:r['before_summary']=stats(r['before_us']);r['candidate_summary']=stats(r['candidate_us'])
        log=directory/'compile.log'
        if log.exists():
            text=log.read_text();data['compiler_error']=[line for line in text.splitlines() if 'LLVM ERROR:' in line or 'error:' in line or 'Assertion' in line][:8]
            remarks=[]
            for chunk in re.split(r'(?=^remark:.*?Function Name:)',text,flags=re.M):
                match=re.search(r'Function Name:\s*(\S+)',chunk)
                if match:remarks.append({'function':match[1],**{key:int(value) for key,value in re.findall(r'(TotalSGPRs|VGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s*(\d+)',chunk)}})
            data['remarks']=remarks
            assemblies=list(directory.glob('*-gfx1201.s'))
            if assemblies:
                assembly=assemblies[0].read_text()
                selected_isa=[]
                for entry in remarks:
                    name=entry['function']
                    demangled=subprocess.check_output(['/opt/rocm/core-7.14/lib/llvm/bin/llvm-cxxfilt',name],text=True).strip()
                    if any(part in demangled for part in ('group_gemm<4, 1','group_gemm<4, 2','group_gemm<4, 4','group_gemm<5, 4','group_gemm_out<4','group_gemm_out<8')):
                        selected_isa.append({'function':demangled,**analyze_asm(assembly,name)})
                data['selected_isa']=selected_isa

        if data:variants[directory.name]=data
    prefill={}
    for directory in sorted((build/'prefill').glob('*')):
        if directory.is_dir() and (directory/'micro.jsonl').exists():
            runs=[json.loads(x) for x in (directory/'micro.jsonl').read_text().splitlines() if x.endswith('}')]
            for r in runs:r['summary']=stats(r['us'])
            prefill[directory.name]={'runs':runs,'checks':(directory/'checks.log').read_text() if (directory/'checks.log').exists() else '', 'build':json.loads((directory/'build.json').read_text()) if (directory/'build.json').exists() else {}, 'run_error':(directory.parent/(directory.name+'.log')).read_text()[-1800:] if (directory.parent/(directory.name+'.log')).exists() else ''}
    data={'peak_int8_dense_tops':383.0,'peak_derivation':{'ops_per_wmma':8192,'cycles_per_wmma':8,'simds':128,'boost_ghz':2.92,'derived_tops':8192/8*128*2.92/1000},'body_sum_nk':body_nk,'head_nk':head_nk,'resources':resource,'step_effective':effective,'band_effective':groups,'compiler_variants':variants,'prefill_variants':prefill}
    (out/'plamo3_exl3_linear2_compiler.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# PLaMo linear 第2弾: large-M とコンパイラ解析','',
        'INT8 dense の公称 peak は 383 TOPS。[AMD 公式仕様](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) の非 sparse 値を使用する。',
        'WMMA の 8192 ops / 8 cycles、128 SIMD、最大 2.92 GHz から 382.73 TOPS を得る。[RDNA4 builtin の公式 throughput 表](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html#instruction-throughput) と整合する。','',
        '## step の有効 INT8 演算率','',
        f'body の sum(N*K)={body_nk:,}、head の N*K={head_nk:,}。演算量は `2*(M*body_sum_nk + output_rows*head_nk)`。head に prefill 全行を掛けない。',
        'この演算量を LINEAR_EXL3_INT8 の event 合計で割る。量子化・復号・回転・flag の重複 GEMM と host/launch 区間を含む有効値であり、WMMA busy の実測値ではない。','',
        '| phase | c | M 帯 | samples | M 中央値 | linear ms 中央値 | 有効 TOPS 中央値 | 公称 peak 比 |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,g in groups.items():
        match=re.match(r'(.*)-c(\d+)-(.*)',key);phase,c,band=match.groups()
        lines.append(f"| {phase} | {c} | {band} | {g['tops']['n']} | {g['rows']['median']:.0f} | {g['linear_us']['median']/1000:.3f} | {g['tops']['median']:.2f} | {g['peak_percent']['median']:.2f}% |")
    lines+=['','## baseline の静的資源','',
        'ROCm 7.14 の clang に `-Rpass-analysis=kernel-resource-usage -save-temps=obj` を指定し、device ELF と ISA を保存した。occupancy は compiler の waves/SIMD 上限であり、実行中の達成値ではない。動的 LDS は別に加算する。','',
        '| family / kernel | SGPR | VGPR | LDS B/block | scratch B/lane | SGPR/VGPR spill | waves/SIMD |','| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in resource:
        name=r['name'];selected=('group_gemm<' in name or 'prefill_wmma_kernel<' in name or 'exl3_mul24_gemm_kernel<4,' in name or 'exl3_mul24_gemm_kernel<5,' in name or r['family']=='projection_prefill' or r['family']=='grouped' and '<true,' in name and name.endswith(('4u>','5u>')))
        if selected:lines.append(f"| {r['family']} ({r['resource_variant']}) / `{name}` | {r.get('TotalSGPRs',0)} | {r.get('VGPRs',0)} | {r.get('LDS Size [bytes/block]',0)} | {r.get('ScratchSize [bytes/lane]',0)} | {r.get('SGPRs Spill',0)} / {r.get('VGPRs Spill',0)} | {r.get('Occupancy [waves/SIMD]',0)} |")
    lines+=['','## ISA','',
        '全 opcode と WMMA を含む basic block を JSON に保存した。WMMA の距離は ISA 上の命令数であり、実行時の cycle 間隔ではない。GFX12 の s_wait_loadcnt / s_wait_dscnt 等も集計する。','',
        '| kernel | mul24 | dot4 | alignbit | WMMA | global b32/b64/b128 | scratch load/store | WMMA命令距離 中央値 |','| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |']
    for r in resource:
        if r['family']!='projection_prefill' and (r['family']!='projection' or not any(t in r['name'] for t in ('group_gemm<4, 2','group_gemm<4, 4','group_gemm<5, 4'))):continue
        isa=r['isa'];op=isa.get('opcodes',{});gaps=isa.get('wmma_instruction_distance');v=lambda prefix:sum(n for key,n in op.items() if key.startswith(prefix))
        lines.append(f"| `{r['name']}` | {v('v_mul_u32_u24')} | {v('v_dot4')} | {v('v_alignbit')} | {v('v_wmma')} | {v('global_load_b32')}/{v('global_load_b64')}/{v('global_load_b128')} | {v('scratch_load')}/{v('scratch_store')} | {gaps['median'] if gaps else 0} |")
    lines += ['', '旧4bit MT4は静的 ISA に scratch_store_b32 3個、scratch_store_b64 1個と対応する load を含む。8 waves の指定を満たすための spill を確認した。',
              '旧 group_body は128 K groupあたり64個の mul_u32_u24 と64個の dot4で64個のINT8値/laneを復号する。',
              '新 direct も同じ復号量である。初回と次 group 用のコードを別々に生成するため、関数全体の静的カウントは128個ずつになる。cached GEMMからは復号命令がなくなり、一時展開 kernel が担う。',
              '旧4bit MT4は global b64 で activation fragmentを32回読む。新MT4はLDSから ds_load_b128 で2回分のWMMA Aを読み、global入力をworkgroup内で共有する。cached Bもglobal_load_b128で読む。',
              '旧MT4のWMMA間の静的命令距離中央値は5、新MT4は2。発行burstは密になるが、これは実行cycleの測定ではない。',
              's_wait_loadcntはtrellis / activationの利用前に現れる。新kernelのs_wait_dscntとbarrier waitにはGEMMと出力FWHTのLDS依存が含まれる。関数全体のwait個数をstall時間とは扱わない。']
    lines+=['','## compiler A/B','',
        'baseline は通常の静的 library、candidate だけ別 TU にして flag を指定する。同じ process で AB/BA を交互に7回測定し、出力と padding の bit 一致を確認する。全体 flag は変更しない。','',
        '| variant | M / N×K / bit / count | before p50 µs (SD) | candidate p50 µs (SD) | 比 |','| --- | --- | ---: | ---: | ---: |']
    for name,v in variants.items():
        if v.get('compiler_error'):lines.append(f"| {name} | compiler error | — | — | {v['compiler_error'][0].replace('|','/')} |")
        for r in v.get('runs',[]):
            b=statistics.median(r['before_us']);a=statistics.median(r['candidate_us']);lines.append(f"| {name} | {r['m']} / {r['n']}×{r['k']} / {r['bits']} / {r['count']} | {b:.2f} ({statistics.stdev(r['before_us']):.2f}) | {a:.2f} ({statistics.stdev(r['candidate_us']):.2f}) | {a/b:.3f} |")
    lines += ['', '| variant | 設定 |', '| --- | --- |']
    for name, variant in variants.items():
        build_settings = variant.get('build', {})
        settings = list(build_settings.get('flags', []))
        if build_settings.get('occ') is not None:
            settings.append('amdgpu_waves_per_eu min=' + str(build_settings['occ']))
        if build_settings.get('nontemporal'):
            settings.append('__builtin_nontemporal_load')
        if settings:
            lines.append('| ' + name + ' | `' + ' '.join(settings) + '` |')
    lines+=['','## large-M 試作','',
        '入力と scale を swizzle 付き LDS に二重 buffer で置く。INT8 fragment の2回分を128 bitで運び、GEMMと出力回転を融合する。',
        'cached は trellis を lane / K-pair 順の INT8 へ展開し、direct は元の trellis を使う。いずれも per-128-K の scale と累積順を維持し、旧 per-group reference と BF16 出力・padding の bit 一致を確認する。',
        'exact版は256行以上の従来RowScale経路と丸めが変わる。M320の875位置/方向でtop1低下基準を超えたため不採用。最終候補はRowScale保持版である。','',
        '数値の全 p50・分散・資源・opcode は [JSON](plamo3_exl3_linear2_compiler.json) を参照。']
    lines += ['', '| variant | M / N×K / bit | dense before µs | direct µs (SD) | cached µs (SD) |', '| --- | --- | ---: | ---: | ---: |']
    for name, variant in prefill.items():
        shapes = sorted({(r['m'], r['n'], r['k'], r['bits']) for r in variant['runs']})
        for shape in shapes:
            runs = [r for r in variant['runs'] if (r['m'],r['n'],r['k'],r['bits']) == shape and r['stage'] == 'linear']
            before = [v for r in runs if r['variant']==0 for v in r['us']]
            values = []
            for mode in (2,3):
                samples = [v for r in runs if r['variant']==mode for v in r['us']]
                values.append(f"{statistics.median(samples):.2f} ({statistics.stdev(samples):.2f})" if len(samples)>1 else '—')
            m,n,k,bits = shape
            lines.append(f"| {name} | {m} / {n}×{k} / {bits} | {statistics.median(before) if before else 0:.2f} | {values[0]} | {values[1]} |")
    lines += ['', '非一時的 load は単体の一部で短縮したが、継続投入で安定した優位を示さず、通常 build に採用しない。',
              '`mt4-expert` は M=320, N=10240, K=2560 の direct 出力で 3,216,714 要素の bit 不一致を検出し、中断した。品質不合格として不採用。',
              '`max-ilp` は MachineScheduler のコンパイル中に Segmentation fault で失敗し、実行していない。',
              '旧 projection の occupancy=6 は MT4 の spill を減らすが、新 selector では該当 body の M>=129 を新 kernel に置換するため採用しない。',
              'unroll-count=1/2 は内側ループの最適化を失い大幅に悪化。LDS swizzle の除去も大幅に悪化。通常 build の全体 flag は変更しない。']
    (out/'plamo3_exl3_linear2_compiler.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'resources':len(resource),'effective_steps':len(effective),'compiler_variants':len(variants),'prefill_variants':len(prefill)}))


if __name__=='__main__':main()
