"""Select a longer PPO budget on validation episodes, then test it in Setting B."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from imitate_reference import OriginalPPO, SEEDS, runner
from compare_navigation import NavigationTask


BASELINE = 30_000
VALIDATION_START = 960000
TEST_START = 970000
LEARNING_RATE = 1e-4
ENTROPY = .003


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def evaluate(model, path, seed, budget, stage, count):
    runner.MsPacManTask = NavigationTask
    agent = runner.load_model(model)
    start = VALIDATION_START if stage == 'validation' else TEST_START
    metrics = runner.evaluate(agent, 'B', count, seed_start=start, log_dir=path,
                              label=stage, trace_count=int(stage == 'test'))
    metrics.update(seed=seed, budget=budget, episodes=count, model=str(model))
    write_json(path/f'{stage}.json', metrics)
    return metrics


def validate(seed, budget, source, output, count):
    path = output/f'budget_{budget}'/f'seed_{seed}'
    path.mkdir(parents=True)
    original = source/f'seed_{seed}'
    if budget == BASELINE:
        model = original/'dagger_ppo.npz'
    else:
        start = runner.load_model(original/'dagger.npz')

        class WarmStartPPO(OriginalPPO):
            def __init__(self, n_actions, n_features, train_seed, **config):
                super().__init__(n_actions, n_features, train_seed,
                                 hidden=start.p['W1'].shape[1], **config)
                self.p = {name: value.copy() for name, value in start.p.items()}

        runner.MsPacManTask = NavigationTask
        runner.PPOAgent = WarmStartPPO
        try:
            agent = runner.train('PPO', 'B', seed, budget, lr=LEARNING_RATE,
                                 entropy=ENTROPY, log_dir=path)
            assert agent.environment_interactions == budget
        finally:
            runner.PPOAgent = OriginalPPO
        model = path/'model.npz'
        runner.save_model(agent, model)
        print(f'{budget:,} interactions, seed {seed}: checkpoint saved', flush=True)
    return evaluate(model, path, seed, budget, 'validation', count)


def select(validation, budgets):
    scores = {}
    for budget in budgets:
        group = [row for row in validation if row['budget'] == budget]
        scores[budget] = {metric: mean(row[metric] for row in group)
                          for metric in ('success_rate', 'mean_score')}
    best = max(budgets, key=lambda budget: (scores[budget]['success_rate'],
                                           scores[budget]['mean_score'], -budget))
    return best, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parent.parent/'results'/'assignment4_dagger_B'
    parser.add_argument('--source', type=Path, default=default_source)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.results.resolve()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    budgets = (BASELINE, 256, 384) if args.smoke else (BASELINE, 60_000, 100_000)
    val_count, test_count = (2, 2) if args.smoke else (100, 200)
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if (original['protocol'] != runner.PROTOCOL or original['fine_tuning_interactions'] != BASELINE
            or original['setting'] != 'B' or original['fine_tuning_lr'] != LEARNING_RATE
            or original['fine_tuning_entropy'] != ENTROPY):
        parser.error('Source protocol and 30k fine-tuning settings must match.')
    script_dir = Path(__file__).resolve().parent
    for name, source_hash in original['code_sha256'].items():
        if digest(script_dir/name) != source_hash:
            parser.error(f'Current {name} differs from the original DAgger experiment.')
    output.mkdir(parents=True, exist_ok=False)
    source_files = ['manifest.json']+[f'seed_{seed}/{file}' for seed in seeds
                                     for file in ('dagger.npz', 'dagger_ppo.npz')]
    write_json(output/'manifest.json', dict(
        source=str(source), source_sha256={name: digest(source/name) for name in source_files},
        code_sha256={name: digest(script_dir/name)
                     for name in ('extend_dagger_ppo.py', *original['code_sha256'])},
        protocol=runner.PROTOCOL, setting='B', seeds=seeds, budgets=budgets, smoke=args.smoke,
        learning_rate=LEARNING_RATE, entropy=ENTROPY,
        validation_seed_start=VALIDATION_START, validation_episodes=val_count,
        test_seed_start=TEST_START, test_episodes=test_count,
        selection='One budget across seeds: mean validation success, mean score, then shorter budget'))

    validation, tests = [], []
    with ProcessPoolExecutor(max_workers=min(6, len(budgets)*len(seeds))) as pool:
        jobs = [pool.submit(validate, seed, budget, source, output, val_count)
                for budget in budgets for seed in seeds]
        for job in as_completed(jobs):
            row = job.result()
            validation.append(row)
            print(f'Validation {row["budget"]:,}, seed {row["seed"]}: {row["success_rate"]:.1%}',
                  flush=True)
        chosen, scores = select(validation, budgets)
        write_json(output/'selection.json', dict(selected_budget=chosen, validation_means=scores,
                                                validation_runs=validation))
        print(f'Validation selected {chosen:,} interactions', flush=True)
        test_budgets = (BASELINE,) if chosen == BASELINE else (BASELINE, chosen)
        jobs = [pool.submit(evaluate, Path(row['model']),
                            output/f'budget_{row["budget"]}'/f'seed_{row["seed"]}',
                            row['seed'], row['budget'], 'test', test_count)
                for row in validation if row['budget'] in test_budgets]
        for job in as_completed(jobs):
            row = job.result()
            tests.append(row)
            print(f'Test {row["budget"]:,}, seed {row["seed"]}: {row["success_rate"]:.1%}',
                  flush=True)
    test_means = {budget: {metric: mean(row[metric] for row in tests if row['budget']==budget)
                           for metric in ('success_rate', 'mean_score', 'median_score')}
                  for budget in test_budgets}
    paired = []
    for seed in seeds:
        by_budget = {row['budget']: row for row in tests if row['seed'] == seed}
        paired.append(dict(seed=seed, **{key: by_budget[chosen][key]-by_budget[BASELINE][key]
                                        for key in ('success_rate', 'mean_score')}))
    result = dict(selected_budget=chosen, validation_means=scores, test_means=test_means,
                  paired_deltas=paired, test_runs=sorted(tests, key=lambda r: (r['budget'], r['seed'])))
    write_json(output/'comparison.json', result)
    print(json.dumps(test_means, indent=2))


if __name__ == '__main__':
    main()
