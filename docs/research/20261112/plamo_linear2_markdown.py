import json
import statistics
from pathlib import Path


def measured(record):
    if not record:
        return '未完了'
    return f"{record['mean']:.2f} ± {record['sample_sd']:.2f}" if record['sample_sd'] is not None else f"{record['mean']:.2f} (n=1)"


def main():
    out = Path(__file__).parent
    data = json.loads((out / 'plamo3_exl3_linear2.json').read_text())
    compiler = json.loads((out / 'plamo3_exl3_linear2_compiler.json').read_text())
    old = (out / 'plamo3_exl3_linear2.md').read_text()
    start = old.find('## 候補ごとの単体比較')
    ends = [old.find(h, start) for h in ('## large-M と compiler の候補', '## 品質と cold')]
    end = min(e for e in ends if e >= 0)
    prior_micro = old[start:end] if start >= 0 and end > start else ''
    prior_micro = prior_micro.replace('最終コードは RowScale を保持する。', 'この旧 wide 候補と、後述する LDS / epilogue 融合版は別実装である。')
    prior_micro = prior_micro.replace('GEMM+out の融合までを今回の採用範囲とする。', 'logits の公開 contract を変える argmax 融合は今回採用していない。')
    lines = ['# PLaMo 3 EXL3-INT8 linear の最適化 第2弾', '',
             '2026-11-12〜13（JST）、R9700 / gfx1201、EXL3-INT8 4.05 bpw、BF16 KV、greedy、投機なし。開始点は `3d72e00`。',
             '対象は QKV / O / gate-up / down / lm_head の linear。RMSNorm / attention / scheduler / stall の実装は変更していない。', '',
             '## 継続投入', '',
             '`plamo_closed_loop.py`、seed 17、32 入力、lengths=1/2/4/6、最大出力 320 token。最初の完了後 20 秒 warmup、90 秒窓。',
             'before / 最終候補を各 c で2回測定する。表示は平均 ± run 間標本 SD。2回目は逆順。30秒探索と診断付き run は主表に含めない。',
             '各 GPU 実行に UUID と flock を指定し、run ごとに解放する。全 run の値、完了数、窓の時刻は JSON に保存する。', '',
             '| c | 第1弾報告 tok/s | 同条件 before tok/s | 第2弾途中 final2 tok/s | 第2弾最終 tok/s | before 比 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    previous = {'1': 151.73, '4': 541.09, '16': 1609.39, '64': 2784.13, '128': 3151.92}
    for c in previous:
        b = data['closed_loop']['before'][c]['tok_s']
        mid = data['closed_loop']['after'][c]['tok_s']
        final = data['closed_loop']['final'][c]['tok_s']
        gain = f"{100*(final['mean']/b['mean']-1):+.2f}%" if final else '未完了'
        lines.append(f"| {c} | {previous[c]:.2f} | {measured(b)} | {measured(mid)} | {measured(final)} | {gain} |")
    lines += ['', '第1弾の報告値と今回の before 再測定を区別する。採用比較は今回の before。final2 は追加指示前の実装であり、最終値と混ぜない。', '',
              '## 行数帯と head', '',
              'head は OutputGather が選択した OUTPUT_ROWS だけを処理する。prefill 全行を投影する無駄はなかった。BF16 logits を丸めてから F32 に変換する順序を維持する。',
              '次表は30秒の診断窓。Python / C++ の monotonic clock を照合して起動と drain を除外する。診断 overhead があるため主速度に使わない。', '',
              '| c | phase | M 帯 | steps | step 平均 ms | 窓内 step 時間比 |', '| ---: | --- | --- | ---: | ---: | ---: |']
    for key, run in data['diagnostic'].items():
        if not key.startswith(('before-exact-', 'prefill-')):
            continue
        for band, item in sorted(run['bands'].items(), key=lambda x: int(x[0].split('-')[0])):
            lines.append(f"| {run['concurrency']} | {key.rsplit('-',1)[0]} | {band} | {item['steps']} | {item['step_ms']['mean']:.3f} | {100*item['wall_share']:.2f}% |")
    lines += ['', '## linear と残る時間', '',
              'LINEAR_EXL3_INT8 の外側 HIP event を合計し、内包する EXL3 stage を再加算しない。表示は捕捉 step の中央値。',
              'step の形状は実装間で1対1に対応しないため、M の中央値も併記する。event と stepwall の差を厳密な stall として扱わない。', '',
              '| c | phase | M 帯 | samples | M 中央値 | linear ms | head ms | RMSNorm ms | attention ms | 有効 TOPS / peak比 |',
              '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for key, values in data['semantic_band_summary'].items():
        if not key.startswith(('final2-', 'prefill-')) or '-c1-' in key:
            continue
        phase, cb = key.split('-c', 1)
        c, band = cb.split('-', 1)
        es = [e for e in data['semantic_events'] if e['file'].startswith('mix-'+phase+'-') and str(e['concurrency'])==c and e['band']==band and e['outputs']>0 and e['within_measurement_window']]
        med = lambda name: statistics.median([e['us'].get(name, 0) for e in es])/1000
        eff = compiler['band_effective'].get(key)
        rate = f"{eff['tops']['median']:.2f} / {eff['peak_percent']['median']:.2f}%" if eff else '—'
        lines.append(f"| {c} | {phase} | {band} | {len(es)} | {statistics.median(e['rows'] for e in es):.0f} | {med('LINEAR_EXL3_INT8'):.3f} | {statistics.median(e['head_us'] for e in es)/1000:.3f} | {med('RMS_NORM'):.3f} | {med('PAGED_ATTENTION'):.3f} | {rate} |")
    lines += ['', '演算量は `2*(M*2,327,838,720 + output_rows*275,251,200)`。この値を linear event 合計で割る。',
              '公称 INT8 dense 383 TOPS を使用する。回転、復号、再量子化、flag 補正、launch 区間を含む有効率であり、WMMA busy の実測ではない。',
              '公称 peak の出典、SGPR / VGPR / LDS / scratch / occupancy、ISA、compiler A/B は [コンパイラ解析](plamo3_exl3_linear2_compiler.md)を参照。', '',
              '## 採用した実装', '',
              '- 65 行以上の 4 bit は trellis を直接使う LDS GEMM、5 bit は一時展開した INT8 の LDS GEMM。QKV と gate/up の group を維持し、出力回転と BF16 丸めを融合する。',
              '- 128 K ごとの整数内積と FP32 scale の累積順を保つ。swizzle 付き入力 LDS を二重 buffer にし、128 bit の load で WMMA 2回分を運ぶ。MT=4 は B の次 group の先読みを省く。',
              '- 256 行以上では RowScale の追加丸めと flagged-row の重複 GEMM を省く。この帯の旧 dense 出力とは一致しないため、拡張品質評価を別途実施する。',
              '- M=1 の 4 bit QKV / O の入力回転融合、M<=16 / N>=10240 / 全 split=1 の GEMM+out 融合、65〜255 行 head の MT4 を維持する。',
              '- 他モデルと明示的な dense / AB override は既存 selector を使う。新しい経路は既存 workspace を再利用し、永続 weight buffer を追加しない。', '',
              '実 checkpoint の body は96 groupのうち84 groupが2 launch、12 groupが3 launchとなる。head が17行以上の mixed step は合計207 launch。',
              'M=1 は230 launch、M=4/16 は266 launch、M=64 は291 launch。169 logical linear は維持する。',
              '単体 micro の rotate / out の約10 µsには launch / event 区間の影響がある。169×20 µsを除去可能な固定費とは扱わない。', '', prior_micro.rstrip(), '',
              '## large-M と compiler の候補', '',
              'MT=1/2/4/8、LDS swizzle、B pipeline、launch_bounds、waves_per_eu、scheduler barrier、非一時的 load を比較した。',
              'compiler flag は candidate の PLaMo TU に限定し、expert scheduler、schedule metric bias、memory clause、unroll-count、max-ILP を試した。全体 flag は変更しない。',
              '旧 MT4 の occupancy=8 指定は VGPR spill を発生させた。新しい MT4 は VGPR206 / LDS16,896B / scratch0 / waves7。単純に occupancy を下げるだけでは改善しない。',
              'expert scheduler の large-M 候補は bit 不一致、max-ILP は compiler crash、unroll-count=1/2 と swizzle 除去は速度悪化により不採用。',
              '単体の一部で非一時的 load が大きく改善しても、継続投入で同じ短縮を確認できなかった。主指標で反復改善を確定していない候補は採用しない。',
              '全候補の before/after・run 間分散・bit 比較はコンパイラ解析の表と JSON に保存する。', '',
              '## 品質と cold', '',
              '各方向18文・875位置、別集計の WikiText 521位置。c=1/64 の全 dump の SHA256 と対応位置 KL 差を確認する。', '',
              '| batch | 方向 | positions | KL | top-1 | incumbent と bit一致 |', '| ---: | --- | ---: | ---: | ---: | --- |']
    quality = data['quality'].get('quality-prefill', {})
    for c, q in quality.items():
        for direction, m in q['summary'].items():
            lines.append(f"| {c} | {direction} | {m['positions']} | {m['mean_kl']:.9f} | {100*m['top1_agreement']:.4f}% | {q['all_logits_bit_equal']} |")
    large = data['quality'].get('quality-prefill-320')
    lines += ['', 'M=320 は一時 hook で最後の319番目の行だけを採点する。head は1行とし、body は320行を通す。',
              '旧 flagged-row の先頭行だけを採点して RowScale の差を隠す評価を避ける。両方向875位置と対応位置 KL 差を別集計する。', '',
              '| M320 variant | 方向 | positions | KL | top-1 |', '| --- | --- | ---: | ---: | ---: |']
    if large:
        for variant, q in large['variants'].items():
            for direction, m in q['summary'].items():
                lines.append(f"| {variant} | {direction} | {m['positions']} | {m['mean_kl']:.9f} | {100*m['top1_agreement']:.4f}% |")
    lines += ['', 'cold は server を起動し直す c=64 を各5回、c=1 を各2回。全10組の出力差を比較する。', '',
              '| variant | run 間差 平均 /64 | 中央値 /64 | 最大 /64 |', '| --- | ---: | ---: | ---: | ---: |']
    cold = data.get('cold-prefill')
    if cold:
        for variant in ('baseline', 'final'):
            m = cold[variant]
            lines.append(f"| {variant} | {m['pairwise_mean']:.2f} | {m['pairwise_median']:.1f} | {m['pairwise_max']} |")
    lines += ['', '固定 M の bit 一致と、HTTP 到着時刻で batch 形状が変わる cold の出力差を区別する。', '',
              '## 目標と残り', '']
    for c, goal in [('64', 4000), ('4', 700)]:
        final = data['closed_loop']['final'][c]['tok_s']
        if final:
            value = final['mean']
            lines.append(f"c={c}: {value:.2f} tok/s。目標{goal}は{'到達' if value>=goal else '未達'}。差は{goal-value:.2f} tok/s、必要な追加改善は{100*(goal/value-1):.2f}%。")
    lines += ['', '残る線形処理の帯別時間は上表のとおり。gate/up と down の寄与は JSON の projection_us に残す。',
              'decode-only が目標を超えることと、prefill を混ぜた継続投入の到達を区別する。RMSNorm / attention はこの作業では変更しない。', '',
              '## 検証と再現', '',
              '関連10 ctestとPLaMo serverの5テストを対象とする。結果と exit code は JSON。skip は合格として扱わない。`--all` / rocprofv3 は実行しない。',
              '計測 hook は `plamo_linear2_timing.patch`、拡張品質 hook は `plamo_linear2_large_quality.patch` に保存し、通常コードと build から除去する。push / PR は作成しない。',
              'before-b/c16 は窓後の結果パイプ待ちで lock 解放が遅延したため除外し、同条件の before-c/c16 に置換した。詳細は protocol_exclusions。', '',
              '```bash', 'export PATH=/tmp/venv-plamo/bin:/tmp/venv-so/bin:$PATH',
              'python3 docs/research/20261112/plamo_linear_runs.py --compute build-gfx1201/plamo-linear2/shelf-compute.prefill --output build-gfx1201/plamo-linear2/runs --tag final-a',
              'python3 docs/research/20261112/plamo_linear2_compiler.py --variant baseline',
              'python3 docs/research/20261112/plamo_linear2_prefill_build.py --variant mt4-no-pipe --define PLAMO_MT=4 --define PLAMO_PIPE_B=0',
              'python3 docs/research/20261112/plamo_linear_quality.py --bench build-gfx1201/plamo-linear2/shelf-bench.prefill --output build-gfx1201/plamo-linear2/quality-prefill',
              'python3 docs/research/20261112/plamo_linear2_report.py --require-complete', 'python3 docs/research/20261112/plamo_linear2_compiler_report.py',
              'python3 docs/research/20261112/plamo_linear2_markdown.py', '```', '',
              '通常 build の binary を保存してから各 runner に渡す。GPU UUID / flock / library path は runner 内で設定する。',
              '詳細: [全結果 JSON](plamo3_exl3_linear2.json)、[compiler / ISA JSON](plamo3_exl3_linear2_compiler.json)。', '',
              '## commit', '', '実装、計測 harness、結果を小さく分けて commit する。最終 hash は Git 履歴と完了報告を参照。']
    (out / 'plamo3_exl3_linear2.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
