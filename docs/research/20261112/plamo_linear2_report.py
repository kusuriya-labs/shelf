import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


def summary(values):
    return {'n':len(values),'mean':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,'median':statistics.median(values),'min':min(values),'max':max(values)} if values else None


def band(rows):
    return '1-64' if rows<=64 else '65-128' if rows<=128 else '129-255' if rows<256 else '256-511' if rows<512 else '512-2048'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('build-gfx1201/plamo-linear2'))
    parser.add_argument('--output',type=Path,default=Path('docs/research/20261112/plamo3_exl3_linear2.json'))
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    root=args.root
    data={'prior_commit':'3d72e00','device_uuid':'GPU-a8e9ddefa2d60f55','dtype':'EXL3-INT8 4.05bpw / BF16 KV','closed_loop':{},'micro':{},'diagnostic':{},'quality':{},'tests':{}}
    runs=collections.defaultdict(list)
    for path in sorted((root/'runs').glob('*.json')):
        r=json.loads(path.read_text())
        tag=path.stem.rsplit('-',2)[0]
        runs[(tag,r['concurrency'])].append({'file':path.name,**{k:v for k,v in r.items() if k!='records'}})
    for (tag,c),rs in runs.items():
        data['closed_loop'].setdefault(tag,{})[str(c)]={'tok_s':summary([r['completion_tokens_per_s'] for r in rs]),'runs':rs}
    for phase in ('before','after','exact','capped','final'):
        pooled={}
        for c in (1,4,16,64,128):
            rs=[r for (tag,cc),rr in runs.items() if cc==c and tag.startswith({'exact':'final','capped':'accepted','final':'nocap-final'}.get(phase,phase)+'-') for r in rr]
            pooled[str(c)]={'tok_s':summary([r['completion_tokens_per_s'] for r in rs]),'runs':rs}
        data['closed_loop'][phase]=pooled
    for path in sorted(root.glob('*.jsonl')):
        rs=[json.loads(line) for line in path.read_text().splitlines() if line.endswith('}')]
        for r in rs:
            if 'us' in r:r['summary_us']=summary(r['us'])
        data['micro'][path.stem]=rs
    trace_contexts=[]
    for path in sorted((root/'diagnostic').glob('*.server.log')):
        jpath=path.with_name(path.name.replace('.server.log','.json'))
        if not jpath.exists():continue
        run=json.loads(jpath.read_text())
        rows=[]
        for line in path.read_text().splitlines():
            if 'STEPWALL ' in line:rows.append({k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)})
        if not rows:continue
        exact_window='measurement_start_monotonic_s' in run
        start=run.get('measurement_start_monotonic_s',rows[0]['t']+run['elapsed_before_measurement_s'])
        trace_contexts.append({'tag':path.name.rsplit('-',2)[0],'concurrency':run['concurrency'],'rows':rows,'start':start,'end':start+run['measurement_s'],'file':path.name})
        rows=[r for r in rows if start<=r['t']<start+run['measurement_s']]
        total=sum(r['total'] for r in rows)
        bins={}
        for b in sorted({band(r['tokens']) for r in rows}):
            rs=[r for r in rows if band(r['tokens'])==b]
            bins[b]={'steps':len(rs),'step_ms':summary([r['total'] for r in rs]),'total_ms':sum(r['total'] for r in rs),'wall_share':sum(r['total'] for r in rs)/total,'decode_steps':sum(r['decode']>0 for r in rs),'rows_histogram':dict(sorted(collections.Counter(int(r['tokens']) for r in rs).items()))}
        data['diagnostic'][path.stem]={'window_origin':'exact monotonic measurement window' if exact_window else 'approximate: first_step_t + elapsed_before_measurement_s','concurrency':run['concurrency'],'bands':bins,'steps':len(rows),'step_total_ms':total}
    events=[]
    for path in sorted(root.glob('mix*.csv')):
        m=re.search(r'-m(\d+)-o(\d+)-p(\d+)\.csv',path.name)
        if not m:continue
        totals=collections.defaultdict(float)
        counts=collections.Counter()
        linear=[]
        consumed=[]
        for r in csv.DictReader(path.open()):
            totals[r['kernel_id']]+=float(r['duration_us']);counts[r['kernel_id']]+=1
            if r['kernel_id']=='LINEAR_EXL3_INT8':
                linear.append(float(r['duration_us']));consumed.append(int(r['logical_consumed']))
        projections=collections.defaultdict(float)
        if consumed==[3,1,2,1]*24+[1]:
            categories=['qkv','o','gate_up','down']*24+['head']
        elif consumed==[1]*169:
            categories=['qkv','qkv','qkv','o','gate_up','gate_up','down']*24+['head']
        else:categories=[]
        for category,us in zip(categories,linear):projections[category]+=us
        parsed=re.fullmatch(r'mix-(.*)-(\d{6})-m(\d+)-o(\d+)-p(\d+)\.csv',path.name)
        phase,seq,actual_rows,outputs,prefills=parsed.groups()
        seq,actual_rows,outputs,prefills=map(int,(seq,actual_rows,outputs,prefills))
        phase={'wide-c64':'wide','final':'fusion-f32'}.get(phase,phase)
        matches=[]
        for context in trace_contexts:
            if context['tag']!=phase or seq>len(context['rows']):continue
            step=context['rows'][seq-1]
            if int(step['tokens'])==actual_rows and 0<=step['rows']-outputs<=prefills:
                matches.append((step['t'],context))
        origin=max(matches,key=lambda item:item[0]) if matches else None
        context=origin[1] if origin else None
        within_window=bool(origin and context['start']<=origin[0]<context['end'])
        events.append({'file':path.name,'rows':int(m[1]),'outputs':int(m[2]),'prefills':int(m[3]),'band':band(int(m[1])),'concurrency':context['concurrency'] if context else None,'within_measurement_window':within_window,'trace_context':context['file'] if context else None,'us':dict(totals),'projection_us':dict(projections),'dispatch_counts':dict(counts),'head_us':linear[-1] if linear and int(m[2])>0 else None})
    data['semantic_events']=events
    for name in ['quality','quality-final','quality-final3','quality-prefill','quality-prefill-320','quality-prefill-192','quality-rowscale','quality-rowscale-192','quality-rowscale-320','quality-rowscale-first','quality-uncapped-320','quality-final-nocap','quality-final-nocap-192','quality-final-nocap-320']:
        p=root/name/'quality.json'
        if p.exists():data['quality'][name]=json.loads(p.read_text())
    if (root/'quality-final-nocap-320/quality.json').exists():
        rows_equal={}
        for candidate in (root/'quality-final-nocap-320/candidate').glob('*.candidate'):
            old=root/'quality-rowscale-first/before'/candidate.name
            rows_equal[candidate.name]=old.exists() and hashlib.sha256(candidate.read_bytes()).digest()==hashlib.sha256(old.read_bytes()).digest()
        data['quality']['nocap320_vs_old_first']=rows_equal
    for name in ['cold','cold-final','cold-final3','cold-prefill','cold-rowscale','cold-uncapped','cold-final-nocap']:
        p=root/name/'cold.json'
        if p.exists():data[name]=json.loads(p.read_text())
    for path in root.glob('*test*.log'):
        data['tests'][path.name]=path.read_text()
    for folder in ('server-tests','server-tests-final3','server-tests-accepted','server-tests-prefill','server-tests-rowscale','server-tests-final-nocap'):
        for path in (root/folder).glob('tests.json'):
            data['tests']['server']=json.loads(path.read_text())
    real_unit=root/'prefill/unit-final-nocap.log'
    if real_unit.exists():data['tests']['prefill-real-shapes']=real_unit.read_text()
    paired_final=root/'paired_positions-final-nocap.json'
    if paired_final.exists():data['paired_positions-final-nocap']=json.loads(paired_final.read_text())
    data['protocol_exclusions']=[{'file':p.name,'reason':'計測窓後に親の出力待ちで lock 解放が遅延したため除外。before-c の同条件再測定に置換。'} for p in (root/'runs/discarded-lock-delay').glob('*.json')]
    event_groups={}
    for variant in ('before-exact','final2','final3','prefill','rowscale','nocap'):
        for c in (1,64,128):
            for b in ('1-64','65-128','129-255','256-511','512-2048'):
                es=[e for e in events if e['file'].startswith('mix-'+variant+'-') and e['band']==b and e['concurrency']==c and e['within_measurement_window'] and e['outputs']>0]
                if es:
                    event_groups[f'{variant}-c{c}-{b}']={'samples':len(es),'projection_us':{key:summary([e['projection_us'][key] for e in es if key in e['projection_us']]) for key in ('qkv','o','gate_up','down','head')},'linear_us':summary([e['us'].get('LINEAR_EXL3_INT8',0) for e in es]),'head_us':summary([e['head_us'] for e in es if e['head_us'] is not None]),'stage_us':{key:summary([e['us'].get(key,0.0) for e in es]) for key in sorted(set().union(*(e['us'].keys() for e in es)))}}
    data['semantic_band_summary']=event_groups
    paired=root/'paired_positions.json'
    if paired.exists():data['paired_positions']=json.loads(paired.read_text())
    cap_five=root/'cold-rowscale/cold-five.json'
    if cap_five.exists():data['capped_cold_first_five']=json.loads(cap_five.read_text())
    rejected_five=root/'cold-final3/cold-five.json'
    if rejected_five.exists():data['rejected_input_slice_cold_first_five']=json.loads(rejected_five.read_text())
    data['accepted_variant']='final2'
    if (root/'FINAL_NOCAP_ACCEPTED').exists():data['accepted_variant']='nocap'
    smoke=root/'quality96-smoke.log'
    if smoke.exists():data['quality96_smoke']={line.split('=',1)[0]:line.split('=',1)[1] for line in smoke.read_text().splitlines() if line.startswith('QUALITY_') and '=' in line}
    data['candidate_decisions']={
        'prefill_exact':{'accepted':False,'reason':'M320の同条件before対比でtop1がJA1.83pt/EN1.49pt低下。KL基準は満たすがtop1基準を超えるため不採用。coldは改善するが採用しない。','closed_loop_tags':['final-a','final-b']},
        'rowscale_capped':{'accepted':False,'reason':'M320全36dumpはbit一致、速度も改善したがcold10回のrun間差mean5.18→6.87/median6→7/max9→11で不採用。'},
        'rowscale_uncapped':{'accepted':data['accepted_variant']=='nocap','reason':'RowScaleとenergy分類を維持し、compact不要になったflag行数制限を除去。通常行全K int32とflag行groupFMAを同じGEMMで処理。'},
        'mixed_split2':{'accepted':False,'reason':'c=64 の30秒探索が2854.97/2815.37 tok/sと不安定。拡大品質未評価で採用しない。'},
        'input_slice':{'accepted':False,'scope':'M=1、4bit の QKV/O/down。K split の入力 group のみをLDSへ回転。','validation':'c=1 90秒窓156.12/156.80、全80dump bit一致。cold10回の平均6.02は同じだが中央値6→7、最大11→13のため不採用。'},
        'greedy_argmax':{'implemented':False,'reason':'BF16丸めと公開logits出力のcontractを維持。今回の融合はGEMM+outまで。'}}
    if args.require_complete:
        final_selected = data['accepted_variant'] == 'nocap'
        phases = ('before', 'final') if final_selected else ('before', 'after')
        for phase in phases:
            for c in ('1', '4', '16', '64', '128'):
                runs_for_c = data['closed_loop'][phase][c]['runs']
                if len(runs_for_c) < 2:
                    raise ValueError(f'{phase} c={c}: fewer than two runs')
                if any(r['measurement_s'] != 90 or r['warmup_after_first_completion_s'] != 20 for r in runs_for_c):
                    raise ValueError(f'{phase} c={c}: unexpected measurement protocol')
        selected_quality = data['quality']['quality-final-nocap' if final_selected else 'quality-final']
        for c in ('1', '64'):
            if not selected_quality[c]['all_logits_bit_equal'] or len(selected_quality[c]['logits']) != 40:
                raise ValueError('standard quality dumps differ or are incomplete')
        if len(data['tests'].get('server', [])) != 5 or any(r['exit_code'] for r in data['tests']['server']):
            raise ValueError('server tests incomplete')
        focused = 'focused-final-nocap-tests.log' if final_selected else 'focused-accepted-tests.log'
        if '100% tests passed' not in data['tests'].get(focused, ''):
            raise ValueError('focused tests incomplete')
        if final_selected:
            large = data['quality']['quality-final-nocap-320']
            q = large['variants']['candidate']['summary']
            for direction in ('ja-en', 'en-ja'):
                if q[direction]['positions'] != 875 or q[direction]['mean_kl'] >= 0.02:
                    raise ValueError('large-M quality KL/positions failed')
                old = large['variants']['before']['summary'][direction]['top1_agreement']
                if q[direction]['top1_agreement'] <= old - 0.01:
                    raise ValueError('large-M top1 failed')
            first_equal = data['quality']['nocap320_vs_old_first']
            if len(first_equal) != 36 or not all(first_equal.values()):
                raise ValueError('M320 row permutation check failed')
            exact = data['quality']['quality-final-nocap-192']['bit_equal']
            if len(exact) != 4 or not all(exact.values()):
                raise ValueError('M192 full-model dump mismatch')
            cold = data['cold-final-nocap']
            for variant in ('baseline', 'final'):
                if len(cold[variant]['cold64_vs_c1']) != 10:
                    raise ValueError('cold repetitions incomplete')
                if cold[variant]['c1_differ_between_repeats']:
                    raise ValueError('c1 cold repetitions differ')
            if cold['final_vs_baseline_c1']:
                raise ValueError('c1 cold changed from baseline')
            for key in ('pairwise_mean', 'pairwise_median', 'pairwise_max'):
                if cold['final'][key] > cold['baseline'][key]:
                    raise ValueError('cold output variation worsened: ' + key)
            if 'PASS' not in data['tests'].get('prefill-real-shapes', ''):
                raise ValueError('real grouped shapes test failed')
            if 'paired_positions-final-nocap' not in data:
                raise ValueError('final per-position KL missing')
            pp = root / 'paired_positions320.json'
            if not pp.exists():
                raise ValueError('large-M per-position KL missing')
            data['paired_positions320'] = json.loads(pp.read_text())
        elif 'cold-final' not in data or 'paired_positions' not in data:
            raise ValueError('prior quality or cold records incomplete')

    data['binaries']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('shelf-*') if path.is_file()}
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
