"""Evaluate the final B candidates on a fresh, fixed set of D episodes."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from audit_ppo_transfer import NavigationTask, TemperaturePolicy, digest, runner, write_json


SEED_START = 997500
EPISODES = 150
ARMS = ('reliable_0.2', 'cold_30k', 'coordinate_dropout')


def evaluate(candidate, out):
    runner.MsPacManTask = NavigationTask
    agent = TemperaturePolicy(runner.load_model(Path(candidate['model'])), candidate['temperature'])
    path = out/candidate['name']/f"seed_{candidate['seed']}"
    path.mkdir(parents=True)
    result = runner.evaluate(agent, 'D', EPISODES, seed_start=SEED_START, log_dir=path,
                             label='test', trace_count=1)
    result.update(candidate)
    write_json(path/'test.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    frontier, out = args.frontier.resolve(), args.results.resolve()
    manifest = json.loads((frontier/'manifest.json').read_text(encoding='utf-8'))
    here = Path(__file__).resolve().parent
    for name, checksum in manifest['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source study.')
    candidates = [row for row in manifest['candidates'] if row['name'] in ARMS]
    for candidate in candidates:
        model = Path(candidate['model'])
        if digest(model) != manifest['model_sha256'][str(model)]:
            parser.error(f'Checkpoint changed: {model}')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'manifest.json', dict(
        code_sha256={**manifest['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        model_sha256={r['model']:manifest['model_sha256'][r['model']] for r in candidates},
        source=str(frontier), setting='D', seed_start=SEED_START, episodes=EPISODES,
        candidates=candidates, protocol=manifest['protocol']))
    rows = []
    with ProcessPoolExecutor(max_workers=9) as pool:
        for job in as_completed([pool.submit(evaluate, entry, out) for entry in candidates]):
            row = job.result()
            rows.append(row)
            print(f'D {row["name"]}, seed {row["seed"]}: {row["success_rate"]:.1%} '
                  f'and {row["mean_score"]:.0f} points', flush=True)
    means = {arm:{metric:mean(row[metric] for row in rows if row['name']==arm)
                  for metric in ('success_rate','mean_score')} for arm in ARMS}
    write_json(out/'comparison.json', dict(means=means,
                                          runs=sorted(rows, key=lambda r:(r['name'],r['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
