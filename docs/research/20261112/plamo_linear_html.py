import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('docs/research/20261112/plamo3_exl3_linear.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/research/20261112/plamo3_exl3_linear.html'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())['shape_profile']
    payload = []
    names = {3072:'Q',512:'K / V',10240:'gate / up',107520:'lm_head'}
    for row in data:
        name = names.get(row['n'],'down' if row['k']==10240 else 'O')
        payload.append({'m':row['m'],'name':name,'n':row['n'],'k':row['k'],'bits':row['bits'],
                        'before':row['baseline']['gemm']['median'],'after':row['final']['gemm']['median'],
                        'roof':row['weight_only_roofline_us']})
    html = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;color:var(--foreground);font:14px/1.6 var(--font-sans,system-ui)}
.controls{display:flex;gap:8px;align-items:center;margin:4px 0 12px;flex-wrap:wrap}
button{font:inherit;color:var(--foreground);background:var(--secondary);border:1px solid var(--border);border-radius:var(--radius,8px);padding:5px 16px;cursor:pointer}
button[aria-pressed=true]{background:var(--primary);color:var(--primary-foreground);border-color:var(--primary)}
.hint{color:var(--muted-foreground);font-size:12px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:9px 6px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}th{font-weight:500;font-size:12px;color:var(--muted-foreground)}
.shape{display:block;font:11px var(--font-mono,monospace);color:var(--muted-foreground)}
.after{color:var(--success);font-weight:600}.ratio{width:92px;max-width:18%}
.track{display:flex;gap:2px;height:5px;margin-top:4px}.beforebar{height:5px;background:var(--muted-foreground);opacity:.5}
.afterbar{height:5px;background:var(--chart-2,var(--success))}
.notes{font-size:12px;color:var(--muted-foreground);margin:12px 0 0}
@media(max-width:440px){body{font-size:12px}th,td{padding:8px 3px}.shape{font-size:10px}.ratio{width:60px}}
</style>
<div class="controls" role="group" aria-label="linear の行数"><button data-m="1" aria-pressed="false">M = 1</button><button data-m="64" aria-pressed="true">M = 64</button><span class="hint">projection 単体の GEMM・µs</span></div>
<div class="scroll"><table><thead><tr><th>projection / N × K / bit</th><th>before</th><th>after</th><th>重みのみ<br>下限</th><th class="ratio">after / 下限</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="notes">5 run の p50 中央値。HIP event、cold cache、seed 17 の合成 trellis。QKV / gate-up の launch 統合はこの表に含みません。下限は trellis bytes ÷ 636 GB/s で、activation、FP32 partial、回転、launch を含みません。</p>
<script>
const rows=PAYLOAD;
function render(m){
  const selected=rows.filter(r=>r.m===m);
  document.querySelector('#rows').innerHTML=selected.map(r=>{
    const ratio=r.after/r.roof;
    return `<tr><td>${r.name}<span class="shape">${r.n} × ${r.k} · ${r.bits} bit</span></td><td>${r.before.toFixed(2)}</td><td class="after">${r.after.toFixed(2)}</td><td>${r.roof.toFixed(2)}</td><td class="ratio">${ratio.toFixed(2)}×<div class="track"><span class="beforebar" style="width:${Math.min(46,r.before/r.roof*4)}px"></span><span class="afterbar" style="width:${Math.min(46,ratio*4)}px"></span></div></td></tr>`;
  }).join('');
  document.querySelectorAll('button[data-m]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.m)===m));
  console.assert(selected.length===9,'shape count');
}
document.querySelectorAll('button[data-m]').forEach(b=>b.onclick=()=>render(Number(b.dataset.m)));
console.assert(rows.every(r=>[r.before,r.after,r.roof].every(v=>Number.isFinite(v)&&v>0)),'finite times');
render(1);render(64);console.log('M=1 / 64 の全18 shapeを検証');
</script></html>'''
    args.output.write_text(html.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False)))


if __name__ == '__main__':
    main()
