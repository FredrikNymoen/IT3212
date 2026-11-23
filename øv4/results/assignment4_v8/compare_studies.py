from pathlib import Path
import csv, statistics
root=Path(__file__).resolve().parent

def read(directory):
    rows=[r for f in sorted(directory.glob('seed_results*.csv'))
          for r in csv.DictReader(f.open(encoding='utf-8'))]
    assert len(rows)==40, f'{directory.name} incomplete: {len(rows)} rows'
    return rows

old,new=read(root.parent/'assignment4_v7'),read(root)
lines=['# v7 and v8 comparison',
       'Both studies use five training seeds, 100,000 training interactions per agent, '
       'and 200 primary evaluation episodes per trained agent per condition. '
       'Cells summarize trained agents, not independent pooled episodes. '
       'V8 changes multiple observation details together; differences do not identify '
       'the causal effect of an individual fix. Both selected Q position_bins=4 and PPO lr=0.001.',
       '| Setting | Agent | v7 success, mean ± SD | v8 success, mean ± SD | Change (percentage points) | v7 mean score | v8 mean score |',
       '|---|---|---|---|---|---|---|']
for condition in 'ABCD':
    for algorithm in ('Q','PPO'):
        a=[r for r in old if r['condition']==condition and r['algorithm']==algorithm]
        b=[r for r in new if r['condition']==condition and r['algorithm']==algorithm]
        av=[100*float(r['success_rate']) for r in a]
        bv=[100*float(r['success_rate']) for r in b]
        score_a=statistics.mean(float(r['mean_score']) for r in a)
        score_b=statistics.mean(float(r['mean_score']) for r in b)
        lines.append(f'| {condition} | {algorithm} | {statistics.mean(av):.1f}% ± {statistics.stdev(av):.1f}% | '
                     f'{statistics.mean(bv):.1f}% ± {statistics.stdev(bv):.1f}% | '
                     f'{statistics.mean(bv)-statistics.mean(av):+.1f} | {score_a:.0f} | {score_b:.0f} |')
lines += ['', 'Sources: ../assignment4_v7/seed_results_*.csv and seed_results_*.csv. '
          'Per-run configurations and raw episode logs are in the respective study directories. '
          'D is frozen transfer from B and has no additional training.']
text='\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:])+'\n'
(root/'v7_vs_v8.md').write_text(text,encoding='utf-8')
print(text)
