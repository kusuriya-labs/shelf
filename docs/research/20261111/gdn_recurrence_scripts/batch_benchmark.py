import json, os, subprocess, sys, time
from pathlib import Path
N=int(sys.argv[1]); NEW=int(sys.argv[2]); extra=sys.argv[3:]
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
cases=json.loads((Path(__file__).resolve().parent / 'benchmark_cases.json').read_text()); cases=[c for c in cases if len(c['input_ids'])>=16]
args=[os.environ['SHELF_BENCH_BINARY'],'--model-dir',model,'--serve-stdio','--device','0','--max-seq-len','2048','--max-concurrent-requests',str(max(N,1)),'--prefix-cache','0','--temperature','0']+extra
p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(os.environ.get('ERRF','/dev/null'),'w'),text=True,bufsize=1)
def send(x):p.stdin.write(json.dumps(x)+'\n');p.stdin.flush()
def recv():
    l=p.stdout.readline()
    if not l: raise RuntimeError('ended')
    return json.loads(l)
send({'op':'ping'})
while recv().get('event')!='pong': pass
# warmup
send({'op':'generate','request_id':900,'input_ids':cases[0]['input_ids'],'max_new_tokens':8,'temperature':0})
while True:
    e=recv()
    if e.get('request_id')==900 and e.get('event') in('done','error'): break
# N concurrent, distinct 256-token prompts
gen={}; ts={}; t0=time.monotonic(); first={}; done={}; toks={}
for i in range(N):
    send({'op':'generate','request_id':i+1,'input_ids':cases[i%len(cases)]['input_ids'],'max_new_tokens':NEW,'temperature':0})
while len(done)<N:
    e=recv(); r=e.get('request_id')
    if e.get('event')=='token':
        toks[r]=toks.get(r,0)+1; first.setdefault(r,time.monotonic()); ts.setdefault(r,[]).append(time.monotonic())
    elif e.get('event')=='done': done[r]=time.monotonic(); gen[r]=e.get('generated_ids')
    elif e.get('event')=='error': raise RuntimeError(str(e))
t1=max(done.values()); f0=min(first.values())
lo=max(first.values()); hi=min(done.values())
cnt=sum(sum(1 for x in v if lo<=x<=hi) for v in ts.values())
win=hi-lo
tot=sum(toks.values())
print(json.dumps({'all_active_tps':round(cnt/win,2) if win>0.5 else None,'win_s':round(win,2),'N':N,'extra':extra,'total_tokens':tot,'ttft_max_s':round(max(first.values())-t0,2),'decode_window_s':round(t1-f0,2),'agg_tps_decode_window':round((tot-N)/(t1-f0),2),'per_req_tps':round((tot-N)/(t1-f0)/N,2)}))
json.dump(gen,open(os.environ.get('GENF','/dev/null'),'w'))
send({'op':'shutdown'});p.stdin.close(); p.wait(timeout=60)
