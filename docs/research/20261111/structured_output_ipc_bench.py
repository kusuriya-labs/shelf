import base64, json, os, statistics, subprocess, sys, time
if len(sys.argv)>1:
    for line in sys.stdin.buffer:
        obj=json.loads(line)
        mask=base64.b64decode(obj['mask'], validate=True)
        sys.stdout.write(json.dumps({'token_id': len(mask)})+'\n'); sys.stdout.flush()
else:
    mask=base64.b64encode(bytes([165])*((248320+7)//8)).decode()
    data=(json.dumps({'request_id': 1, 'mask': mask},separators=(',',':'))+'\n').encode()
    p=subprocess.Popen([sys.executable, __file__,'child'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    samples=[]
    for i in range(1100):
        t=time.perf_counter_ns(); p.stdin.write(data); p.stdout.readline(); dt=(time.perf_counter_ns()-t)/1e6
        if i>=100: samples.append(dt)
    p.stdin.close(); p.wait()
    samples.sort()
    print(f'mask_bytes={(248320+7)//8} json_bytes={len(data)} median_ms={statistics.median(samples):.4f} p95_ms={samples[949]:.4f} p99_ms={samples[989]:.4f} n={len(samples)}')
