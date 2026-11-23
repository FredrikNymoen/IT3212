"""Train with horizontally mirrored B episodes and measure frozen transfer."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import improve_ppo_session as search
from mspacman import SCREEN_DX
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, np, runner, write_json


EPISODES = 100
BUDGET = 60_000
TEST_START = {'B': 998400, 'D': 998300}
ARMS = ('starting_checkpoint', 'standard_training', 'horizontal_mirror')


class MirroredNavigationTask(NavigationTask):
    def __init__(self, protocol, setting):
        super().__init__(protocol, setting)
        opposite = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'UPLEFT': 'UPRIGHT',
                    'UPRIGHT': 'UPLEFT', 'DOWNLEFT': 'DOWNRIGHT', 'DOWNRIGHT': 'DOWNLEFT'}
        self.action_map = np.array([self.actions.index(opposite.get(a, a)) for a in self.actions])
        assert np.array_equal(self.action_map[self.action_map], np.arange(self.n_actions))

    def reset(self, seed, *args, **kwargs):
        random = np.random.default_rng(np.random.SeedSequence([int(seed), 67]))
        self.mirrored = random.random() < .5
        return super().reset(seed, *args, **kwargs)

    def step(self, action):
        return super().step(int(self.action_map[action]) if self.mirrored else action)

    def features(self, ram):
        obs = super().features(ram)
        if self.mirrored:
            # Convert the RAM x coordinate into the screen's sprite-centre
            # coordinate, reflect about the centre of the image, and convert back.
            obs[0] = (self.walls.shape[1]-1-2*(SCREEN_DX+4.5))/160-obs[0]
            obs[[2, 14, 26, 28]] *= -1
            obs[[12, 13, 19, 20, 23, 24]] = obs[[13, 12, 20, 19, 24, 23]]
            obs[30:] = obs[30:][self.action_map]
        return obs


def run(arm, seed, start, control, temperature, root):
    folder = root/arm/f'seed_{seed}'
    folder.mkdir(parents=True)
    if arm == 'horizontal_mirror':
        search.NavigationTask = MirroredNavigationTask
        try:
            model = search.fit(start, seed, folder, BUDGET, 2e-5, temperature, .08, 512)
        finally:
            search.NavigationTask = NavigationTask
        print(f'Mirror seed {seed}: {BUDGET:,} training interactions complete', flush=True)
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
    source, root = args.continuation.resolve(), args.results.resolve()
    config = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    here = Path(__file__).resolve().parent
    for name, checksum in config['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source experiment.')
    starts = {r['seed']:Path(r['model']) for r in config['checkpoints']}
    controls = {s:source/'training'/'budget_60000'/f'seed_{s}'/'model.npz' for s in starts}
    if not all(path.is_file() for path in controls.values()):
        parser.error('The matched 60k controls must be saved first.')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        code_sha256={**config['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        inputs_sha256={str(p):digest(p) for p in (*starts.values(), *controls.values())},
        source=str(source), protocol=runner.PROTOCOL, train_condition='B',
        budget=BUDGET, seeds=sorted(starts), temperature=config['temperature'],
        mirror_probability=.5, episodes=EPISODES, test_seed_start=TEST_START))
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(run, arm, seed, starts[seed], controls[seed], config['temperature'], root)
                for arm in ('horizontal_mirror','standard_training','starting_checkpoint') for seed in starts]
        for job in as_completed(jobs):
            rows.extend(job.result())
    means = {condition: {arm: {metric:mean(row[metric] for row in rows
                                            if row['arm']==arm and row['condition']==condition)
                                for metric in ('success_rate','mean_score')}
                         for arm in ARMS} for condition in TEST_START}
    write_json(root/'comparison.json', dict(means=means,
                    runs=sorted(rows, key=lambda r:(r['condition'],r['arm'],r['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
