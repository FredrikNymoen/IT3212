"""Evaluate the frozen B champions with sticky actions disabled in Setting A."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from score_diagnostics import NavigationTask, TemperaturePolicy, digest, runner, write_json


def evaluate(name, candidate, out, start, block):
    runner.MsPacManTask = NavigationTask
    agent = TemperaturePolicy(runner.load_model(Path(candidate['model'])), candidate['temperature'])
    folder = out/name/f'block_{block}'
    folder.mkdir(parents=True)
    result = runner.evaluate(agent, 'A', 100, seed_start=start, log_dir=folder, label='test')
    result.update(name=name, block=block, seed_start=start)
    write_json(folder/'test.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--champions', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    champions, out = args.champions.resolve(), args.results.resolve()
    manifest = json.loads((champions/'manifest.json').read_text(encoding='utf-8'))
    candidates = json.loads((champions/'comparison.json').read_text(encoding='utf-8'))['candidates']
    for r in candidates.values():
        assert digest(Path(r['model'])) == manifest['checkpoint_sha256'][r['model']]
    out.mkdir(parents=True)
    start = manifest['eval_start']
    write_json(out/'manifest.json', dict(source=str(champions), seed_start=start, episodes=300,
        checkpoint_sha256=manifest['checkpoint_sha256'],
        code_sha256={**manifest['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        setting='A', sticky=0.))
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for job in as_completed([pool.submit(evaluate, name, candidate, out, start+100*i, i)
                                 for name, candidate in candidates.items() for i in range(3)]):
            rows.append(job.result())
    means = {name:{metric:mean(r[metric] for r in rows if r['name']==name)
                   for metric in ('success_rate','mean_score')} for name in candidates}
    write_json(out/'comparison.json', dict(means=means, runs=sorted(rows,key=lambda r:(r['name'],r['block']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
