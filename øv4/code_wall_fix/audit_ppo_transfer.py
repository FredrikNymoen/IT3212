"""Evaluate the B-selected checkpoints on unseen maze D without further training."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from score_diagnostics import NavigationTask, TemperaturePolicy, digest, runner, write_json


EVAL_START = 999700
EPISODES = 100


def evaluate(candidate, out, count):
    runner.MsPacManTask = NavigationTask
    model = Path(candidate['model'])
    agent = TemperaturePolicy(runner.load_model(model), candidate['temperature'])
    path = out/candidate['arm']/f"seed_{candidate['seed']}"
    path.mkdir(parents=True)
    result = runner.evaluate(agent, 'D', count, seed_start=EVAL_START, log_dir=path,
                             label='transfer', trace_count=1)
    result.update(candidate)
    write_json(path/'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    sweep, out = args.sweep.resolve(), args.results.resolve()
    setup = json.loads((sweep/'manifest.json').read_text(encoding='utf-8'))
    choice = json.loads((sweep/'selection.json').read_text(encoding='utf-8'))['selected_temperature']
    main = Path(setup['source'])
    source = json.loads((main/'manifest.json').read_text(encoding='utf-8'))
    base = [row for row in source['candidates'] if row['name']=='trained_at_0.5']
    candidates = [dict(arm='prior', seed=r['seed'], model=r['model'], temperature=.5)
                  for r in base]
    for temp in sorted({.5, choice}):
        candidates.extend(dict(arm=f'improved_{temp:g}', seed=r['seed'], model=r['model'],
                               temperature=temp) for r in setup['checkpoints'])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'manifest.json', dict(protocol=runner.PROTOCOL, condition='D',
        source_study=str(sweep), candidates=candidates, episodes=EPISODES, seed_start=EVAL_START,
        model_sha256={row['model']: digest(Path(row['model'])) for row in candidates},
        code_sha256={**source['code_sha256'], Path(__file__).name: digest(Path(__file__))}))
    rows = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        for job in as_completed([pool.submit(evaluate, r, out, EPISODES) for r in candidates]):
            row = job.result()
            rows.append(row)
            print(f'D {row["arm"]}, seed {row["seed"]}: {row["success_rate"]:.1%} '
                  f'and {row["mean_score"]:.0f} points', flush=True)
    means = {arm: {k:mean(r[k] for r in rows if r['arm']==arm)
                   for k in ('success_rate','mean_score','failure_rate','timeout_rate')}
             for arm in sorted({r['arm'] for r in rows})}
    write_json(out/'comparison.json', dict(means=means, runs=sorted(rows, key=lambda r:(r['arm'],r['seed']))))
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
