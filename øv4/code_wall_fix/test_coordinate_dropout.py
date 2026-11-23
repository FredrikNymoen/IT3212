"""Test B training with absolute-coordinate dropout and evaluate frozen transfer to D."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import improve_ppo_session as search
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, np, runner, write_json


BUDGET = 60_000
EPISODES = 100
TEST_START = {'B': 998200, 'D': 998100}
ARMS = ('starting_checkpoint', 'standard_training', 'coordinate_dropout')


class CoordinateDropoutTask(NavigationTask):
    def reset(self, seed, *args, **kwargs):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 67]))
        self.hide_coordinates = rng.random() < .5
        return super().reset(seed, *args, **kwargs)

    def features(self, ram):
        features = super().features(ram)
        if self.hide_coordinates:
            features[:2] = .5
        return features


def run(arm, seed, start, control, temperature, root):
    folder = root/arm/f'seed_{seed}'
    folder.mkdir(parents=True)
    if arm == 'coordinate_dropout':
        search.NavigationTask = CoordinateDropoutTask
        try:
            model = search.fit(start, seed, folder, BUDGET, 2e-5, temperature, .08, 512)
        finally:
            search.NavigationTask = NavigationTask
        print(f'Coordinate dropout seed {seed}: {BUDGET:,} training interactions complete', flush=True)
    else:
        model = start if arm == 'starting_checkpoint' else control
    runner.MsPacManTask = NavigationTask
    agent = TemperaturePolicy(runner.load_model(model), temperature)
    results = []
    for condition, first in TEST_START.items():
        result = runner.evaluate(agent, condition, EPISODES, seed_start=first, log_dir=folder,
                                 label='test', trace_count=1)
        result.update(seed=seed, arm=arm, condition=condition, model=str(model),
                      temperature=temperature, episodes=EPISODES, seed_start=first)
        write_json(folder/f'test_{condition}.json', result)
        results.append(result)
        print(f'{arm}, {condition}, seed {seed}: {result["success_rate"]:.1%} '
              f'and {result["mean_score"]:.0f} points', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuation', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    continuation, root = args.continuation.resolve(), args.results.resolve()
    previous = json.loads((continuation/'manifest.json').read_text(encoding='utf-8'))
    here = Path(__file__).resolve().parent
    for name, checksum in previous['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source experiment.')
    starts = {r['seed']:Path(r['model']) for r in previous['checkpoints']}
    controls = {s:continuation/'training'/'budget_60000'/f'seed_{s}'/'model.npz' for s in starts}
    if not all(path.is_file() for path in controls.values()):
        parser.error('The matched 60k controls must be saved before running this comparison.')
    temperature = previous['temperature']
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        code_sha256={**previous['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        inputs_sha256={str(p):digest(p) for p in (*starts.values(), *controls.values())},
        source=str(continuation), protocol=runner.PROTOCOL, train_condition='B',
        budget=BUDGET, seeds=sorted(starts), temperature=temperature,
        dropout_probability=.5, masked_coordinate=.5, episodes=EPISODES, test_seed_start=TEST_START))
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(run, arm, seed, starts[seed], controls[seed], temperature, root)
                for arm in ('coordinate_dropout','standard_training','starting_checkpoint') for seed in starts]
        for job in as_completed(jobs):
            rows.extend(job.result())
    means = {}
    for condition in TEST_START:
        means[condition] = {arm: {metric:mean(row[metric] for row in rows
                                             if row['arm']==arm and row['condition']==condition)
                                 for metric in ('success_rate','mean_score')}
                            for arm in ARMS}
    write_json(root/'comparison.json', dict(means=means,
                runs=sorted(rows, key=lambda r:(r['condition'],r['arm'],r['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
