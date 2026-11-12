import argparse
import hashlib
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def stats(values):
    return {'n':len(values), 'mean':statistics.mean(values), 'median':statistics.median(values),
            'stdev':statistics.stdev(values) if len(values)>1 else 0,
            'min':min(values), 'max':max(values), 'samples':values}


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=Path('build-gfx1201/plamo-linear'))
    parser.add_argument('--output', type=Path, default=Path('docs/research/20261112/plamo3_exl3_linear'))
    args = parser.parse_args()
    work = args.workdir
    shape_rows = []
    records = load_lines(work/'final-selected.jsonl')
    shapes = sorted({(d['m'],d['n'],d['k'],d['bits']) for d in records})
    counts = {(3072,2560,4):18,(3072,2560,5):6,(512,2560,4):36,(512,2560,5):12,
              (2560,3072,4):18,(2560,3072,5):6,(10240,2560,4):48,(2560,10240,4):24,
              (107520,2560,8):1}
    for m,n,k,bits in shapes:
        row = {'m':m,'n':n,'k':k,'bits':bits,'calls_per_step':counts[n,k,bits],
               'weight_bytes':n*k*bits//8,'weight_only_roofline_us':n*k*bits/8/636000}
        for variant,name in [(0,'baseline'),(9,'final')]:
            row[name] = {d['stage']:stats(d['us']) for d in records if
                (d['m'],d['n'],d['k'],d['bits'],d['variant'])==(m,n,k,bits,variant)}
            row[name]['gemm_roofline_ratio'] = row[name]['gemm']['median']/row['weight_only_roofline_us']
            row[name]['linear_roofline_ratio'] = row[name]['linear']['median']/row['weight_only_roofline_us']
        shape_rows.append(row)
    e2e = []
    for path in sorted((work/'runs').glob('*.json')):
        data = json.loads(path.read_text())
        lengths = [r['output_tokens'] for r in data['records']]
        record = {k:v for k,v in data.items() if k!='records'}
        record.update(name=path.stem,output_tokens=stats(lengths))
        record['output_tokens'].pop('samples')
        e2e.append(record)
    throughput = {}
    for c in [1,4,16,64,128]:
        before = [r['completion_tokens_per_s'] for r in e2e if r['concurrency']==c and r['name'].startswith('baseline-')]
        after = [r['completion_tokens_per_s'] for r in e2e if r['concurrency']==c and r['name'].startswith('final-')]
        throughput[str(c)] = {'baseline':stats(before),'final':stats(after),
                             'mean_change_percent':(statistics.mean(after)/statistics.mean(before)-1)*100}
    profiles = json.loads((work/'profiles-final/profile.json').read_text())
    profile_summary = {}
    for c in [1,64]:
        profile_summary[str(c)] = {}
        for variant in ['baseline','final']:
            values = [r for r in profiles if r['concurrency']==c and r['variant']==variant]
            events = sorted({k for row in values for k in row['events']})
            summary = {k:stats([r['events'].get(k,{}).get('event_us',0) for r in values]) for k in events}
            wall = []
            for r in values:
                log = (work/'profiles-final'/f'{variant}-{r["repeat"]}-c{c}.log').read_text()
                times = [float(t) for n,t in re.findall(r'STEPWALL rows=(\d+).*? total=([\d.]+)',log) if int(n)==c]
                wall.append(statistics.median(times))
            profile_summary[str(c)][variant] = {'events_us':summary,
                'event_total_us':stats([r['event_total_us'] for r in values]),'stepwall_median_ms':stats(wall)}
    quality = json.loads((work/'quality-final/quality.json').read_text())
    for batch in ['1','64']:
        assert quality[batch]['all_logits_bit_equal'] and len(quality[batch]['logits']) == 40
        for direction in ['ja-en','en-ja']:
            q = quality[batch]['summary'][direction]
            assert q['positions'] == 875 and q['mean_kl'] < 0.02
    cold = json.loads((work/'cold-final/cold.json').read_text())
    source_paths = ['src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip',
                    'src/shelf/inference/execution/optimized_dispatch.hip',
                    'src/shelf/inference/execution/program_executor.hip',
                    'src/shelf/models/plamo3/model/lower_to_primitives.cpp']
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_commit':'d3a8a7b','implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model':'/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24',
        'gpu':'R9700 gfx1201 GPU-a8e9ddefa2d60f55','rocm':'7.14','gpu_lock':'/tmp/r9700.lock',
        'weight_only_bytes_per_step':sum(n*k*bits//8*count for (n,k,bits),count in counts.items()),
        'roofline_bandwidth_gb_s':636,'source_sha256':{p:digest(Path(p)) for p in source_paths},
        'binary_sha256':{name:digest(work/name) for name in ['shelf-compute.baseline','shelf-compute.final3','shelf-bench.final3']},
        'shape_profile':shape_rows,'group_profile':load_lines(work/'final-groups-selected.jsonl'),
        'exploration_mean20_us':{name:load_lines(work/f'{name}.jsonl') for name in ['stages','sweep','extra','groups']},
        'wide_candidate_p50_us':load_lines(work/'paired-selected.jsonl'),
        'closed_loop_runs':e2e,'throughput':throughput,'step_profile':profile_summary,'step_profile_raw':profiles,
        'quality':quality,'cold':cold,'server_tests':json.loads((work/'server-tests-final/tests.json').read_text()),
        'focused_ctest_log':(work/'focused-tests-final.log').read_text()}
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines = ['# PLaMo 3 EXL3-INT8 linear の最適化', '',
        '2026-11-12、R9700 / gfx1201、ROCm 7.14、BF16 KV、greedy、投機なし。対象は 2B / 4.05 bpw / 8 bit head のみ。',
        'RMSNorm、attention、scheduler、stall 対策は変更していない。push と PR 作成は行っていない。', '',
        '## 継続投入', '',
        '既存 `plamo_closed_loop.py` の seed 17・32 入力・最大出力 320 token・20 秒 warmup・90 秒窓を使用した。',
        '各 GPU 実行を `flock /tmp/r9700.lock` と対象 UUID で保護し、run ごとに lock を解放した。',
        '採用判定には診断なしの実行だけを使う。表示は平均 ± run 間標本標準偏差。JSON に各 run の値と完了数を保存する。', '',
        '| c | before tok/s | after tok/s | 変化 | run 数 before / after |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for c,d in throughput.items():
        b,a=d['baseline'],d['final']
        lines.append(f'| {c} | {b["mean"]:.2f} ± {b["stdev"]:.2f} | {a["mean"]:.2f} ± {a["stdev"]:.2f} | {d["mean_change_percent"]:+.2f}% | {b["n"]} / {a["n"]} |')
    lines += ['', '## shape 別 HIP event', '',
        'M は行数、N は出力、K は入力。shape micro は seed 17 の合成 resident trellis を使う。',
        '実 checkpoint の step event と継続投入は別に測定する。個別 projection の after は融合を含まない。',
        '各 run は warmup 20 回・測定 20 回の p50、5 run を AB / BA で交互に測定した。',
        '表は 5 個の p50 の中央値。単位は µs。GEMM と linear の前には 128 MiB を書いて cache を置換する。',
        '微小 kernel の event 区間には host / launch の影響もあるため、stage の和を全 linear 時間に足し直さない。',
        'roofline は trellis bytes / 636 GB/s の weight-only 下限で、activation、partial、回転、launch を含まない。', '',
        '| M | N × K / bit | calls | rotate before / after | GEMM before / after | out before / after | linear before / after | GEMM / roof before / after |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in shape_rows:
        b,a=r['baseline'],r['final']
        columns=[f'{b[stage]["median"]:.2f} / {a[stage]["median"]:.2f}' for stage in ['rotate','gemm','out','linear']]
        lines.append(f'| {r["m"]} | {r["n"]} × {r["k"]} / {r["bits"]} | {r["calls_per_step"]} | '+' | '.join(columns)+f' | {b["gemm_roofline_ratio"]:.2f} / {a["gemm_roofline_ratio"]:.2f} |')
    lines += ['', '## launch 統合', '',
        'QKV と gate/up は SUH が異なるので、量子化した同じ入力を使い回していない。各 projection の演算と丸めを保ったまま grid をまとめた。',
        'linear は論理 169 個のまま、1〜255 行の対象 step では物理 launch が 507→291、実 dispatch が 169→97 になる。',
        '次表は既存の個別 projection 列と最終の統合列を交互に測定した値。表示は 5 run p50 の中央値 ± 標本標準偏差。', '',
        '| M | 対象 / bit | before µs | after µs |', '| ---: | --- | ---: | ---: |']
    groups = report['group_profile']
    for m,n,bits in sorted({(d['m'],d['n'],d['bits']) for d in groups}):
        pair={d['variant']:stats(d['us']) for d in groups if (d['m'],d['n'],d['bits'])==(m,n,bits)}
        b,a=pair[0],pair[9]
        lines.append(f'| {m} | {"QKV" if n==3072 else "gate/up"} / {bits} | {b["median"]:.2f} ± {b["stdev"]:.2f} | {a["median"]:.2f} ± {a["stdev"]:.2f} |')
    lines += ['', '## 実 step の残り', '',
        '既存の semantic HIP event を executor の一時 hook から使い、8 回目の decode-only step を 3 run 捕捉した。',
        '計測用変更は `plamo_linear_timing.patch` だけに保存し、通常 build から除去した。scheduler と各演算は変更していない。',
        'event 区間には計測 overhead がある。stepwall も同じ診断 run からの値であり、clean throughput は継続投入表だけを使う。',
        'event 合計と stepwall の差を使って厳密な内訳にはできない。', '',
        '| c | variant | 診断 run の step p50 ms | linear event ms | RMSNorm event ms | attention event ms | event 合計 ms |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in [1,64]:
        for variant in ['baseline','final']:
            p=profile_summary[str(c)][variant];e=p['events_us']
            lines.append(f'| {c} | {variant} | {p["stepwall_median_ms"]["median"]:.3f} | {e["LINEAR_EXL3_INT8"]["median"]/1000:.3f} | {e["RMS_NORM"]["median"]/1000:.3f} | {e["PAGED_ATTENTION"]["median"]/1000:.3f} | {p["event_total_us"]["median"]/1000:.3f} |')
    lines += ['', '同じ診断 run の固定長・同一段落の decode-only 窓は次のとおり。server の継続投入とは workload と計測範囲が異なる。',
        '診断付きの補助値であり、4,000 tok/s の server 目標を達成したという判定には使わない。', '',
        '| c | baseline decode tok/s | final decode tok/s |', '| ---: | ---: | ---: |']
    for c in [1,64]:
        b=stats([r['diagnostic_decode']['decode_tokens_per_s'] for r in profiles if r['concurrency']==c and r['variant']=='baseline'])
        a=stats([r['diagnostic_decode']['decode_tokens_per_s'] for r in profiles if r['concurrency']==c and r['variant']=='final'])
        lines.append(f'| {c} | {b["median"]:.2f} ± {b["stdev"]:.2f} | {a["median"]:.2f} ± {a["stdev"]:.2f} |')
    total_bytes=report['weight_only_bytes_per_step']
    lines += ['', f'linear の trellis は 1 step あたり {total_bytes/1e9:.4f} GB。1 回ずつ読む weight-only 下限は {total_bytes/636e6:.3f} ms。',
        '以前の 2.8〜3.2 ms は embedding を含む checkpoint 全体からの概算であり、ここでの linear 限定の byte 数とは異なる。',
        '測定で得た最高値は探索した候補での到達値であり、今後の kernel も含む厳密な上限ではない。', '']
    for c,goal in [(64,4000),(4,700)]:
        value=throughput[str(c)]['final']['mean']
        lines.append(f'c={c} の目標 {goal:,} tok/s は未達。平均との差は {goal-value:.2f} tok/s、さらに {100*(goal/value-1):.2f}% のスループット改善が必要。')
    lines += ['', 'closed-loop は新規 prefill を含むため、decode-only の改善率から全体の速度を換算できない。',
        '256 行以上の dense / flagged-row 経路、64 行を超える head、残る linear の復号と partial traffic は今回の採用経路で改善していない。',
        'RMSNorm、attention、その他の演算と待ち時間も残るが、今回それらの最適化や stall の因果調査は行っていない。', '',
        '## 品質', '',
        '各方向 18 文・875 位置を CPU BF16 と比較し、WikiText 521 位置を別集計した。各列を同じ batch 行数に複製する既存手順であり、混在文の teacher forcing ではない。',
        'c=1 / 64 の全 80 dump は直前の採用済み実装と byte 一致した。今回の対応位置 KL 差・最大 logits 差・top-1 低下は全て 0。', '',
        '| batch | 方向 | KL | top-1 | CPU BF16 との最大 logits 差 |', '| ---: | --- | ---: | ---: | ---: |']
    for batch in ['1','64']:
        for direction in ['ja-en','en-ja']:
            q=quality[batch]['summary'][direction]
            lines.append(f'| {batch} | {direction} | {q["mean_kl"]:.8f} | {q["top1_agreement"]*100:.4f}% | {q["max_logit_diff"]:.6f} |')
    lines += ['', 'c=64 の元の基準（KL 0.017130 / 0.017396）との差は、既採用 split-K 4 の累積差である。今回の変更では増減していない。',
        '元の基準に対する paired per-position KL と文単位 bootstrap 95% CI、および incumbent との差を JSON に保存する。', '',
        '## cold と batch 形状', '',
        '固定行数の kernel 比較は 1 / 4 / 16 / 17 / 32 / 33 / 48 / 49 / 64 / 70 / 128 / 192 / 255 行で BF16 出力と padding が一致した。',
        'HTTP の cold 64 件を各 5 回起動し直し、全 10 組の run 間出力差を比較した。',
        '以前の 5/64 は固定の再現数ではなく、今回の同条件 baseline も異なる件数になった。過去の 5 件以下を保証したという主張はしない。', '',
        '| variant | run 間差の平均 /64 | 中央値 /64 | 最大 /64 |', '| --- | ---: | ---: | ---: |']
    for name in ['baseline','final']:
        d=cold[name];lines.append(f'| {name} | {d["pairwise_mean"]:.2f} | {d["pairwise_median"]:.2f} | {d["pairwise_max"]} |')
    lines += ['', 'c=1 の繰り返しと baseline / final 間の出力は一致した。固定形状の数値差は増えておらず、cold の差も同条件 baseline と比較する。', '',
        '## 採否と再現', '',
        '採用: QKV / gate-up の独立回転を保つ launch 統合、M に応じた tile、5 bit の先読み省略、出力回転の列方向分割。split-K と BF16 中間値は維持した。',
        '探索の初期 JSON は 20 回平均 × 5 run、採用の比較は上記の p50 × 5 run と区別する。', '',
        '関連 kernel / unit と PLaMo BF16 / EXL3 / synthetic / template / startup server テストを実行した。結果の詳細は JSON に保存する。`--all` は実行していない。',
        f'詳細データ: [{args.output.name}.json]({args.output.name}.json)。']
    lines += ['', '### 不採用と保留', '',
        'INT8 への事前復号は、既存の WMMA fragment 順に展開した cache を使って試した。',
        '次表の cache は探索した MT / split-K の中の最速値で、baseline は production incumbent。split 数の変更も含む探索値であり、採用の精度検証を通した値ではない。', '',
        '| M | 対象 | baseline GEMM µs | 最速 cache GEMM µs | MT / split |',
        '| ---: | --- | ---: | ---: | --- |']
    exploratory=report['exploration_mean20_us']
    for shape,name in [((1,2560,10240,4),'down'),((64,2560,10240,4),'down'),
                       ((64,10240,2560,4),'gate/up'),((64,107520,2560,8),'lm_head')]:
        before=next(d for d in exploratory['stages'] if (d['m'],d['n'],d['k'],d['bits'])==shape and d['stage']=='gemm')
        choices=[d for d in exploratory['sweep'] if (d['m'],d['n'],d['k'],d['bits'])==shape and d['variant']==2]
        best=min(choices,key=lambda d:statistics.median(d['us']))
        lines.append(f'| {shape[0]} | {name} | {statistics.median(before["us"]):.2f} | {statistics.median(best["us"]):.2f} | {best["mt"]} / {best["splits"]} |')
    lines += ['', '4 bit 全体の展開は weight bytes が倍になり、この探索では遅くなったため不採用。',
        '8 bit head の cache は単体では改善したが、追加 cache は約 275 MB。削減は約 0.15 ms / decode step にとどまり、',
        'step 全体で約 1% の改善余地に相当する。head cache 単独の closed-loop / 拡張品質評価は行っておらず、採用せず保留した。',
        '採用コードには復号済み weight cache を残していない。', '',
        'direct INT8 dot4 も試した。split-K を incumbent に固定した比較では、M=1 の K/V や M=64 の head が遅くなった。',
        'split を増やすと勝つ小形状もあるが加算順が変わり、今回の融合に重ねた品質・速度上の優位を確認していない。WMMA を維持した。', '',
        '| M | 対象 | baseline µs | dot4 µs | split |', '| ---: | --- | ---: | ---: | ---: |']
    for shape,sp,name in [((1,512,2560,4),2,'K/V'),((64,107520,2560,8),1,'lm_head')]:
        before=next(d for d in exploratory['stages'] if (d['m'],d['n'],d['k'],d['bits'])==shape and d['stage']=='gemm')
        after=next(d for d in exploratory['extra'] if (d['m'],d['n'],d['k'],d['bits'])==shape and d['stage']=='gemm' and d['splits']==sp)
        lines.append(f'| {shape[0]} | {name} | {statistics.median(before["us"]):.2f} | {statistics.median(after["us"]):.2f} | {sp} |')
    lines += ['', 'split-K は 1 / 2 / 4 / 5 / 8 / 10、row tile は 1 / 2 / 4、wave は 4 / 8 を探索した。',
        '8 wave と split の変更は形状により勝敗があり、統合後の全体速度と品質を根拠に採用できる優位を確定していない。',
        '最終経路は 4 wave と既存 split を使い、加算順を維持する。5 bit の先読み省略は register の live 範囲を縮めるために選択した。', '',
        '適用行数を拡大した途中の候補では、MT=2 のままの M=192 gate が約 +11%、M=128 / 192 head が約 +12% / +15% 遅くなった。',
        'その選択は外し、4 bit は 129 行以上を MT=4、head は 65 行以上を既存経路に戻した。生の比較は `wide_candidate_p50_us` にある。',
        'greedy の head+argmax 融合は未実装。head の採用変更は tile と出力回転までであり、sampling を変更したとは主張しない。', '',
        '### 再現手順', '',
        '指定 worktree の root で実行する。baseline は開始時点 `d3a8a7b` の保存済み binary、final3 は最終コードの保存済み binary。',
        '参照データの作成は [既存手順](plamo3_exl3.md#拡大品質評価と継続投入の測定方法) に従う。',
        '各 runner が GPU UUID、library path、flock を設定する。CPU の build は GPU lock を取得しない。', '',
        '```bash',
        'export PATH=/tmp/venv-plamo/bin:/tmp/venv-so/bin:$PATH',
        'python3 docs/research/20261112/plamo_linear_build.py',
        'python3 docs/research/20261112/plamo_linear_scan.py --selected --output build-gfx1201/plamo-linear/final-selected.jsonl',
        'python3 docs/research/20261112/plamo_linear_scan.py --groups --output build-gfx1201/plamo-linear/final-groups-selected.jsonl',
        'python3 docs/research/20261112/plamo_linear_scan.py --sweep --mean20 --output build-gfx1201/plamo-linear/sweep.jsonl',
        'python3 docs/research/20261112/plamo_linear_scan.py --extra --mean20 --output build-gfx1201/plamo-linear/extra.jsonl',
        'python3 docs/research/20261112/plamo_linear_quality.py --bench build-gfx1201/plamo-linear/shelf-bench.final3 --output build-gfx1201/plamo-linear/quality-final',
        'python3 docs/research/20261112/plamo_linear_compare.py --baseline build-gfx1201/plamo-linear/shelf-compute.baseline --candidate build-gfx1201/plamo-linear/shelf-compute.final3 --output build-gfx1201/plamo-linear/runs --tag paired --concurrency 64 1 4 16 128',
        'python3 docs/research/20261112/plamo_linear_runs.py --compute build-gfx1201/plamo-linear/shelf-compute.final3 --output build-gfx1201/plamo-linear/runs --tag final-repeat --concurrency 128 64 16 4 1',
        'python3 docs/research/20261112/plamo_linear_cold.py --baseline build-gfx1201/plamo-linear/shelf-compute.baseline --candidate build-gfx1201/plamo-linear/shelf-compute.final3 --output build-gfx1201/plamo-linear/cold-final',
        'python3 docs/research/20261112/plamo_linear_tests.py --output build-gfx1201/plamo-linear/server-tests-final',
        'python3 docs/research/20261112/plamo_linear_report.py',
        '```', '',
        '棄却した large-M の MT=2 候補は `plamo_linear_build.py --wide-mt2` で生成できる。',
        '`plamo_linear_scan.py --selected --rows 128 192` で比較し、その後、通常の build.py で最終 kernel を再生成する。', '',
        'step event の再取得は次のように行う。計測 hook は既存の executor timing だけを有効にする。', '',
        '```bash',
        'git apply docs/research/20261112/plamo_linear_timing.patch',
        'cmake --build build-gfx1201 --target shelf-compute -j 6',
        'cp build-gfx1201/shelf-compute build-gfx1201/plamo-linear/shelf-compute.profile-final',
        'git apply -R docs/research/20261112/plamo_linear_timing.patch',
        'cmake --build build-gfx1201 --target shelf-compute shelf-bench -j 6',
        'python3 docs/research/20261112/plamo_linear_profile.py --compute build-gfx1201/plamo-linear/shelf-compute.profile-final --output build-gfx1201/plamo-linear/profiles-final',
        '```', '',
        'kernel / unit は `focused_ctest_log` にテスト名と結果を保存した。server test は PLaMo 用環境を使い、構造化出力の全回帰は行っていない。', '',
        '### commit', '',
        '- `e8401e8` — 独立した projection をまとめる kernel と BF16 / padding 一致テスト。',
        '- `61bfe35` — 対象 checkpoint の flag、実行範囲と trace を保つ dispatch、selector、developer contract。',
        '- 測定 harness の commit は Git history を参照。',
        '- 計測記録の commit はこの文書を含む Git history を参照。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
