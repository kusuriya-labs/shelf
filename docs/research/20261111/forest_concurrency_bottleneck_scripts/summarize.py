#!/usr/bin/env python3
"""Attach reproducible summaries from diagnostic logs to the trial JSON.

Round end timestamps and token receipt timestamps use the same monotonic clock.
Rounds overlapping an interval boundary are assigned by their end time; token
receipt is asynchronous, so the window match is approximate.
"""
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'docs/research/20261111/forest_concurrency_bottleneck_results.json'
PAT=re.compile(r'(\w+)=([^\s]+)')
KEYS=('setup','draft_submit','selection_wait','metadata','verify_submit','decision_wait',
      'accept','replay_submit','gather_append_submit','single','sync_wait')

def fields(line):
    result={}
    for k,v in PAT.findall(line):
        try: result[k]=float(v) if '.' in v else int(v)
        except ValueError: result[k]=v
    return result


def parse(path):
    rounds=[]; slots=[]; current=None
    for line in path.read_text(errors='replace').splitlines():
        if 'DFPHASE_SLOT ' in line: slots.append(fields(line.split('DFPHASE_SLOT ',1)[1]))
        elif 'DFPHASE sessions=' in line or 'DFPHASE time_ns=' in line:
            current={'phase':fields(line.split('DFPHASE ',1)[1]), 'slots':slots, 'requests':[]}
            slots=[]; rounds.append(current)
        elif 'DFCOST sessions=' in line and current:
            current['cost']=fields(line.split('DFCOST ',1)[1])
        elif 'DFPHASE_REQUEST ' in line and current:
            current['requests'].append(fields(line.split('DFPHASE_REQUEST ',1)[1]))
        elif 'DFHOST sessions=' in line and current:
            current['host']=fields(line.split('DFHOST ',1)[1])
    return rounds


def analyze(trial):
    log=ROOT/trial['log']; rounds=parse(log)
    if not rounds: raise RuntimeError(f'no DFPHASE records in {log}')
    missing=sum('cost' not in r or 'host' not in r for r in rounds)
    if missing: raise RuntimeError(f'{missing} incomplete rounds in {log}')
    records=trial['records']; lo=max(r['timestamps_ns'][0] for r in records);hi=min(r['timestamps_ns'][-1] for r in records)
    active=[r for r in rounds if lo<=r['phase'].get('time_ns',-1)<=hi]
    req={}
    for r in rounds:
        for q in r['requests']:
            x=req.setdefault(str(q['id']),{'rounds':0,'accepted':0,'emitted':0,'drafts':0})
            for key in ('accepted','emitted','drafts'): x[key]+=q[key]
            x['rounds']+=1
    for line in log.read_text(errors='replace').splitlines():
        match=re.search(r'\[req (\d+)\] prompt=', line)
        if match and match[1] in req:
            req[match[1]]['compute_log']=fields(line)
    for v in req.values():
        v['emitted_per_round']=v['emitted']/v['rounds']; v['accepted_per_round']=v['accepted']/v['rounds']
    def summary(rs):
        if not rs: return None
        n=len(rs)
        result={'rounds':n,'wall_ms':sum(r['cost']['wall'] for r in rs),
                'host_ms':sum(r['host']['wall'] for r in rs),
                'emitted':sum(q['emitted'] for r in rs for q in r['requests']),
                'accepted':sum(q['accepted'] for r in rs for q in r['requests']),
                'draft_gpu_ms':sum(r['cost']['draft_gpu'] for r in rs),
                'verify_gpu_ms':sum(r['cost']['verify_gpu'] for r in rs),
                'phase_ms':{k:sum(r['phase'].get(k,0) for r in rs) for k in KEYS},
                'sync_errors':sum(r['phase'].get('sync_errors',0) for r in rs),
                'sessions_hist':{},'rows_hist':{},'verify_gpu_by_rows':{}}
        for r in rs:
            for field in ('sessions','rows'):
                k=str(r['cost'][field]);h=result[field+'_hist'];h[k]=h.get(k,0)+1
            if r['cost']['verify_gpu']>0:
                k=str(r['cost']['rows']);result['verify_gpu_by_rows'].setdefault(k,[]).append(r['cost']['verify_gpu'])
        for k,v in result['verify_gpu_by_rows'].items():
            result['verify_gpu_by_rows'][k]={'count':len(v),'median_ms':statistics.median(v),'mean_ms':statistics.mean(v)}
        return result
    return {'requests':req,'all_rounds':summary(rounds),'common_interval':summary(active),
            'round_event_count':len(rounds), 'request_event_count':sum(len(r['requests']) for r in rounds)}


def main():
    data=json.loads(PATH.read_text())
    for trial in data['trials']:
        trial['compute_request_logs'] = {}
        for line in (ROOT / trial['log']).read_text(errors='replace').splitlines():
            match = re.search(r'\[req (\d+)\] prompt=', line)
            if match:
                trial['compute_request_logs'][match[1]] = fields(line)
        if trial['diagnostic']:
            trial['diagnostic_summary']=analyze(trial)
    tmp=PATH.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(PATH)
    print('trials',len(data['trials']),'diagnostic',sum(bool(t['diagnostic']) for t in data['trials']))
if __name__=='__main__':main()
