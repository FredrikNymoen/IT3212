"""Independently verify the strongest reliability and score checkpoints."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from improve_ppo_session import evaluate, finish_jobs
from score_diagnostics import digest, write_json


START = 991000
CHUNKS = 6
COUNT = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    frontier, root = args.frontier.resolve(), args.results.resolve()
    manifest = json.loads((frontier/'manifest.json').read_text(encoding='utf-8'))
    report = json.loads((frontier/'comparison.json').read_text(encoding='utf-8'))
    reliability = max(report['runs'], key=lambda r:(r['success_rate'],r['mean_score']))
    score = max(report['runs'], key=lambda r:(r['mean_score'],r['success_rate']))
    candidates = {'reliability': reliability, 'score': score}
    for r in candidates.values():
        assert digest(Path(r['model'])) == manifest['model_sha256'][r['model']]
    root.mkdir(parents=True)
    write_json(root/'manifest.json', dict(source=str(frontier), candidates=candidates,
        checkpoint_sha256={r['model']:manifest['model_sha256'][r['model']] for r in candidates.values()},
        code_sha256={**manifest['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        setting='B', protocol=manifest['protocol'], eval_start=START,
        episodes_per_checkpoint=CHUNKS*COUNT))
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = []
        for i in range(CHUNKS):
            for arm, r in candidates.items():
                entry = {k:r[k] for k in ('seed','model','temperature')}
                entry.update(name=f'{arm}_{i+1}', arm=arm)
                jobs.append(pool.submit(evaluate, entry, root, 'test', START+i*COUNT, COUNT))
        results = finish_jobs(jobs, 'Large confirmation')
    means = {}
    for arm in candidates:
        rows = [r for r in results if r['arm']==arm]
        count = sum(r['episodes'] for r in rows)
        means[arm] = dict(episodes=count, success_rate=mean(r['success_rate'] for r in rows),
                          mean_score=mean(r['mean_score'] for r in rows),
                          successes=sum(round(r['success_rate']*r['episodes']) for r in rows))
        ids = set()
        for r in rows:
            file = root/'test'/r['name']/f"seed_{r['seed']}"/'test_B_episodes.csv'
            with file.open(encoding='utf-8') as stream:
                ids.update(int(e['seed']) for e in csv.DictReader(stream))
        assert count == CHUNKS*COUNT and ids == set(range(START,START+count))
    write_json(root/'comparison.json', dict(candidates=candidates, means=means,
                                              runs=sorted(results, key=lambda r:r['name'])))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
