"""Replay paired B episode seeds with more sticky-action noise."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from improve_ppo_session import evaluate, finish_jobs
from mspacman import SETTINGS
from score_diagnostics import digest, read_rows, write_json


def stress(candidate, root, start):
    original = SETTINGS['B']
    SETTINGS['B'] = dict(original, sticky=.4)
    try:
        return evaluate(candidate, root, 'stress', start, 100)
    finally:
        SETTINGS['B'] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--champions', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    source, root = args.champions.resolve(), args.results.resolve()
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    comparison = json.loads((source/'comparison.json').read_text(encoding='utf-8'))
    candidates = comparison['candidates']
    for r in candidates.values():
        assert digest(Path(r['model'])) == manifest['checkpoint_sha256'][r['model']]
    root.mkdir(parents=True)
    start = manifest['eval_start']
    write_json(root/'manifest.json', dict(source=str(source), source_hash=digest(source/'comparison.json'),
        models=manifest['checkpoint_sha256'], code_sha256={**manifest['code_sha256'],
        Path(__file__).name:digest(Path(__file__))}, condition='B', original_sticky=.25,
        stress_sticky=.4, start=start, count=300))
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = []
        for i in range(3):
            for arm, row in candidates.items():
                entry = {k:row[k] for k in ('seed','model','temperature')}
                entry.update(name=f'{arm}_{i+1}', arm=arm)
                jobs.append(pool.submit(stress, entry, root, start+i*100))
        rows = finish_jobs(jobs, 'Sticky stress')
    means = {}
    for arm, candidate in candidates.items():
        subset = [r for r in rows if r['arm']==arm]
        baseline_rows = [e for r in comparison['runs'] if r['arm']==arm
                         for e in read_rows(source/'test'/r['name']/f"seed_{r['seed']}"/'test_B_episodes.csv')
                         if start<=int(e['seed'])<start+300]
        assert len(baseline_rows)==300 and len(subset)==3
        means[arm] = {'sticky_0.25': dict(
            success_rate=mean(e['outcome']=='success' for e in baseline_rows),
            mean_score=mean(float(e['raw_score']) for e in baseline_rows)),
            'sticky_0.4':{k:mean(r[k] for r in subset) for k in ('success_rate','mean_score')}}
    write_json(root/'comparison.json', dict(means=means, runs=sorted(rows,key=lambda r:r['name'])))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
