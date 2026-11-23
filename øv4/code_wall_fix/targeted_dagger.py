"""Compare uniform and targeted teacher corrections for the same PPO students."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from imitate_reference import OriginalPPO, SEEDS, clone, np, runner
from compare_navigation import NavigationTask
from dagger_reference import collect
from extend_dagger_ppo import ENTROPY, LEARNING_RATE, digest, write_json


ROLLOUT_START = 980000
TEST_START = 990000
PPO_BUDGET = 60_000


def is_priority(observations):
    """Use late score or a nearby closing ghost during the low timer window."""
    late = observations[:, 7] >= .75
    danger = ((observations[:, 4] <= 40/160) & (observations[:, 6] <= .20)
              & (observations[:, 29] > 0))
    return late | danger


def collect_shared(seed, model, root, episodes, quota):
    student = runner.load_model(model)
    begin = ROLLOUT_START + 100*SEEDS.index(seed)
    x, y, episode_ids, _ = collect(student, range(begin, begin+episodes), beta=0.)
    validation = np.flatnonzero(episode_ids >= begin+episodes-4)
    available = np.flatnonzero(episode_ids < begin+episodes-4)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    quota = min(quota, len(available))
    priority = is_priority(x)
    uniform = rng.choice(available, quota, replace=False)
    candidates = available[priority[available]]
    preferred = rng.choice(candidates, min(int(.7*quota), len(candidates)), replace=False)
    remaining = np.setdiff1d(available, preferred)
    targeted = np.concatenate((preferred, rng.choice(remaining, quota-len(preferred), replace=False)))
    assert len(np.unique(uniform)) == len(np.unique(targeted)) == quota
    np.savez_compressed(root/f'labels_{seed}.npz', observations=x, actions=y,
                        episode_ids=episode_ids, validation=validation, priority=priority,
                        uniform=uniform, targeted=targeted)
    print(f'Seed {seed}: {len(x):,} states labeled, {quota:,} corrections per arm. '
          f'Priority share {priority[uniform].mean():.1%} uniform, '
          f'{priority[targeted].mean():.1%} targeted.', flush=True)


def train_arm(seed, arm, model, root, budget, epochs):
    target = root/arm/f'seed_{seed}'
    target.mkdir(parents=True)
    student = runner.load_model(model)
    student.rng = np.random.default_rng(seed)
    with np.load(root/f'labels_{seed}.npz', allow_pickle=False) as data:
        history = clone(student, data['observations'], data['actions'],
                        data[arm], data['validation'], epochs)
    write_json(target/'cloning_history.json', history)
    runner.save_model(student, target/'corrected.npz')

    class WarmStartPPO(OriginalPPO):
        def __init__(self, n_actions, n_features, train_seed, **config):
            super().__init__(n_actions, n_features, train_seed, hidden=student.p['W1'].shape[1],
                             **config)
            self.p = {name: value.copy() for name, value in student.p.items()}

    runner.MsPacManTask = NavigationTask
    runner.PPOAgent = WarmStartPPO
    try:
        trained = runner.train('PPO', 'B', seed, budget, lr=LEARNING_RATE,
                               entropy=ENTROPY, log_dir=target)
        assert trained.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
    runner.save_model(trained, target/'model.npz')
    print(f'{arm}, seed {seed}: {budget:,} PPO interactions complete', flush=True)
    return seed, arm


def evaluate(seed, arm, model, root, count):
    runner.MsPacManTask = NavigationTask
    target = root/arm/f'seed_{seed}'
    target.mkdir(parents=True, exist_ok=True)
    result = runner.evaluate(runner.load_model(model), 'B', count, seed_start=TEST_START,
                             log_dir=target, label='test', trace_count=1)
    with (target/'test_B_episodes.csv').open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    reached = sum(float(row['raw_score']) >= 1500 for row in rows)
    success = sum(row['outcome'] == 'success' for row in rows)
    result.update(seed=seed, arm=arm, episodes=count, reached_1500=reached,
                  conversion_after_1500=success/reached if reached else None)
    write_json(target/'test.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--source', type=Path, default=here.parent/'results'/'assignment4_dagger_budget_B')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, root = args.source.resolve(), args.results.resolve()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    rollout_count, quota, epochs, budget, test_count = ((5, 128, 1, 256, 2) if args.smoke
                                                        else (20, 8192, 12, PPO_BUDGET, 200))
    models = {seed: source/'budget_60000'/f'seed_{seed}'/'model.npz' for seed in seeds}
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['protocol'] != runner.PROTOCOL or manifest['setting'] != 'B'
            or not all(model.is_file() for model in models.values())):
        parser.error('The source must be the matching Setting B budget study with 60k checkpoints.')
    for name, checksum in manifest['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source study.')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        source=str(source), source_sha256={str(p.relative_to(source)): digest(p)
                                          for p in (source/'manifest.json', *models.values())},
        code_sha256={name: digest(here/name)
                     for name in ('targeted_dagger.py', *manifest['code_sha256'])},
        protocol=runner.PROTOCOL, setting='B', seeds=seeds, smoke=args.smoke,
        rollout_seed_start=ROLLOUT_START, rollout_episodes=rollout_count, validation_episodes=4,
        max_corrections_per_arm=quota, cloning_epochs=epochs, preferred_priority_fraction=.7,
        priority_rule='score >= 1500 or ghost <= 40 px, closing, global timer <= .20',
        fine_tuning_interactions=budget, learning_rate=LEARNING_RATE, entropy=ENTROPY,
        eval_seed_start=TEST_START, eval_episodes=test_count))
    arms = ('baseline', 'uniform', 'targeted')
    with ProcessPoolExecutor(max_workers=2*len(seeds)) as pool:
        collection = [pool.submit(collect_shared, seed, models[seed], root, rollout_count, quota)
                      for seed in seeds]
        tests = [pool.submit(evaluate, seed, 'baseline', models[seed], root, test_count)
                 for seed in seeds]
        training = []
        for seed, job in zip(seeds, collection):
            job.result()
            training.extend(pool.submit(train_arm, seed, arm, models[seed], root, budget, epochs)
                            for arm in arms[1:])
        for job in as_completed(training):
            seed, arm = job.result()
            tests.append(pool.submit(evaluate, seed, arm, root/arm/f'seed_{seed}'/'model.npz',
                                     root, test_count))
        results = []
        for job in as_completed(tests):
            result = job.result()
            results.append(result)
            print(f'{result["arm"]}, seed {result["seed"]}: {result["success_rate"]:.1%} success',
                  flush=True)
    metrics = ('success_rate', 'failure_rate', 'timeout_rate', 'mean_score', 'median_score',
               'reached_1500', 'conversion_after_1500')
    means = {}
    for arm in arms:
        group = [row for row in results if row['arm'] == arm]
        means[arm] = {}
        for metric in metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            means[arm][metric] = mean(values) if values else None
    write_json(root/'comparison.json', dict(arm_means=means,
                                           runs=sorted(results, key=lambda row: (row['arm'], row['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
