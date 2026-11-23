"""Compare DAgger and PPO with the prior imitation run on fresh Setting B seeds."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from imitate_reference import OriginalPPO, SEEDS, clone, np, runner
from compare_navigation import NavigationTask
from reference_policy import MOVES, Reference


EVAL_START = 950000
ROLLOUT_START = 940000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def collect(student, episode_seeds, beta):
    task = NavigationTask(runner.PROTOCOL, 'B')
    observations, actions, seeds = [], [], []
    teacher_steps = 0
    try:
        for seed in episode_seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            student.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            obs = task.reset(seed)
            expert = Reference(task)
            expert.rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
            while True:
                label = expert.choose()
                observations.append(obs)
                actions.append(label)
                seeds.append(seed)
                use_teacher = rng.random() < beta
                action = label if use_teacher else student.act(obs)
                teacher_steps += use_teacher
                # Continuation bias tracks the action actually sent to the ROM.
                name = task.actions[action]
                expert.last_action = MOVES.index(name) if name in MOVES else None
                lives, timer = task.lives, task.frightened
                obs, _, done, info = task.step(action)
                expert.observe(info, lives, timer)
                if done:
                    break
    finally:
        task.close()
    return np.stack(observations), np.array(actions), np.array(seeds), teacher_steps


def fit_student(seed, source, directory, rounds, rollouts, epochs, budget):
    path = directory/f'seed_{seed}'
    path.mkdir()
    student = runner.load_model(source/f'seed_{seed}'/'cloned.npz')
    student.rng = np.random.default_rng(seed)
    with np.load(source/'demonstrations.npz', allow_pickle=False) as archive:
        x = archive['observations']
        y = archive['actions']
        validation = archive['episode_ids'] >= archive['validation_start'].item()
    round_summaries = []
    for index, beta in enumerate(rounds):
        first = ROLLOUT_START + 100*SEEDS.index(seed) + index*rollouts
        features, labels, ids, teacher_steps = collect(student, range(first, first+rollouts), beta)
        novel_validation = ids >= first+rollouts-2
        np.savez_compressed(path/f'round_{index+1}.npz', observations=features, actions=labels,
                            episode_ids=ids, validation=novel_validation)
        x = np.concatenate((x, features))
        y = np.concatenate((y, labels))
        validation = np.concatenate((validation, novel_validation))
        student.opt = {name: runner.Adam(params.shape) for name, params in student.p.items()}
        history = clone(student, x, y, np.flatnonzero(~validation), np.flatnonzero(validation), epochs)
        round_summaries.append(dict(round=index+1, teacher_mixing_probability=beta,
                                    collected=len(ids), teacher_steps=teacher_steps, history=history))
        print(f'Seed {seed}, DAgger round {index+1}: {len(ids):,} labeled decisions', flush=True)
    write_json(path/'dagger_history.json', round_summaries)
    runner.save_model(student, path/'dagger.npz')

    class WarmStartPPO(OriginalPPO):
        def __init__(self, n_actions, n_features, train_seed, **config):
            super().__init__(n_actions, n_features, train_seed, hidden=student.p['W1'].shape[1],
                             **config)
            self.p = {name: param.copy() for name, param in student.p.items()}

    runner.MsPacManTask = NavigationTask
    runner.PPOAgent = WarmStartPPO
    try:
        tuned = runner.train('PPO', 'B', seed, budget, lr=1e-4, entropy=.003, log_dir=path)
        assert tuned.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
    runner.save_model(tuned, path/'dagger_ppo.npz')
    return seed


def evaluate(seed, label, source, directory, count):
    path = directory/f'seed_{seed}'
    if label == 'prior_ppo':
        # Create the directory independently of the training worker.
        path = directory/f'prior_seed_{seed}'
        path.mkdir()
        model = source/f'seed_{seed}'/'fine_tuned.npz'
    else:
        model = path/f'{label}.npz'
    runner.MsPacManTask = NavigationTask
    agent = runner.load_model(model)
    result = runner.evaluate(agent, 'B', count, seed_start=EVAL_START,
                             log_dir=path, label=label, trace_count=1)
    result.update(seed=seed, stage=label)
    write_json(path/f'{label}_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent/
                        'results'/'assignment4_imitation_B')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, directory = args.source.resolve(), args.results.resolve()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    rounds = (.25,) if args.smoke else (.25, .10)
    rollouts, epochs, budget, eval_count = (3, 1, 256, 2) if args.smoke else (12, 12, 30_000, 200)
    files = [source/'demonstrations.npz', source/'manifest.json']
    for seed in seeds:
        files.extend(source/f'seed_{seed}'/name for name in ('cloned.npz', 'fine_tuned.npz'))
    if any(not file.is_file() for file in files):
        parser.error('The source must contain the original demonstrations and model checkpoints.')
    directory.mkdir(parents=True, exist_ok=False)
    script_dir = Path(__file__).resolve().parent
    scripts = ['dagger_reference.py', 'imitate_reference.py', 'reference_policy.py',
               'compare_navigation.py', 'part2_run.py', 'mspacman.py']
    write_json(directory/'manifest.json', dict(
        source=str(source), source_sha256={str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                                         for f in files},
        code_sha256={name: hashlib.sha256((script_dir/name).read_bytes()).hexdigest()
                     for name in scripts},
        protocol=runner.PROTOCOL, protocol_id=runner.PROTOCOL_ID, setting='B',
        seeds=seeds, sampled=True, eval_seed_start=EVAL_START, eval_episodes=eval_count,
        rollout_seed_start=ROLLOUT_START, teacher_mixing=rounds, rollouts_per_round=rollouts,
        validation_rollouts_per_round=2, cloning_epochs_per_round=epochs,
        fine_tuning_interactions=budget, fine_tuning_lr=1e-4, fine_tuning_entropy=.003))
    results = []
    with ProcessPoolExecutor(max_workers=2*len(seeds)) as pool:
        training = [pool.submit(fit_student, seed, source, directory, rounds, rollouts, epochs, budget)
                    for seed in seeds]
        evaluations = [pool.submit(evaluate, seed, 'prior_ppo', source, directory, eval_count)
                       for seed in seeds]
        for job in as_completed(training):
            seed = job.result()
            for label in ('dagger', 'dagger_ppo'):
                evaluations.append(pool.submit(evaluate, seed, label, source, directory, eval_count))
        for job in as_completed(evaluations):
            result = job.result()
            results.append(result)
            print(f'{result["stage"]}, seed {result["seed"]}: '
                  f'{result["success_rate"]:.1%} success, score {result["mean_score"]:.0f}', flush=True)
    stages = ('prior_ppo', 'dagger', 'dagger_ppo')
    scores = {stage: {metric: mean(r[metric] for r in results if r['stage']==stage)
                      for metric in ('success_rate', 'mean_score', 'median_score')}
              for stage in stages}
    write_json(directory/'comparison.json', dict(stage_means=scores,
                                                 runs=sorted(results, key=lambda r: (r['stage'], r['seed']))))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
