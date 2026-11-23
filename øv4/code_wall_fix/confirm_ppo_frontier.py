"""Compare the strongest fixed candidates on one fresh B evaluation set."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from improve_ppo_session import evaluate, finish_jobs
from score_diagnostics import digest, summarize, write_json


TEST_START = 996300
EPISODES = 300
ARMS = ('reliable_0.2', 'reliable_0.15', 'cold_30k', 'coordinate_dropout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    results, root = args.results_root.resolve(), args.results.resolve()
    source = results/'assignment4_ppo_hour_temperature_B'
    config = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    cold = results/'assignment4_ppo_hour_cold_continuation_B'/'training'/'budget_30000'
    dropout = results/'assignment4_ppo_hour_dropout'/'coordinate_dropout'
    candidates = []
    for row in config['checkpoints']:
        seed = row['seed']
        for arm, model, temperature in zip(
                ARMS,
                (Path(row['model']), Path(row['model']), cold/f'seed_{seed}'/'model.npz',
                 dropout/f'seed_{seed}'/'model.npz'),
                (.2, .15, .2, .2)):
            candidates.append(dict(name=arm, seed=seed, temperature=temperature, model=str(model)))
    existing = json.loads((Path(config['source'])/'manifest.json').read_text(encoding='utf-8'))
    here = Path(__file__).resolve().parent
    for name, checksum in existing['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source experiment.')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        code_sha256={**existing['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        model_sha256={row['model']:digest(Path(row['model'])) for row in candidates},
        candidates=candidates, protocol=existing['protocol'], setting='B',
        test_start=TEST_START, test_episodes=EPISODES))
    with ProcessPoolExecutor(max_workers=9) as pool:
        tests = finish_jobs([pool.submit(evaluate, candidate, root, 'test', TEST_START, EPISODES)
                            for candidate in candidates], 'Frontier')
    means = {arm:summarize([row for row in tests if row['name']==arm]) for arm in ARMS}
    write_json(root/'comparison.json', dict(means=means, runs=sorted(tests, key=lambda r:(r['name'],r['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
