"""Behavior cloning and PPO fine-tuning on stochastic Setting B."""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

# Small matrix operations and multiple emulator workers benefit from one BLAS
# thread per process.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np

import part2_run as runner
from compare_navigation import NavigationTask
from reference_policy import Reference, evaluate as evaluate_reference


DEMO_START = 920000
EVAL_START = 930000
SEEDS = (42, 7, 2026)
OriginalPPO = runner.PPOAgent


def demonstrate(seeds):
    task = NavigationTask(runner.PROTOCOL, 'B')
    observations, actions, episode_ids = [], [], []
    successes = 0
    try:
        for seed in seeds:
            obs = task.reset(seed)
            expert = Reference(task)
            expert.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            while True:
                action = expert.choose()
                observations.append(obs)
                actions.append(action)
                episode_ids.append(seed)
                lives, timer = task.lives, task.frightened
                obs, _, done, info = task.step(action)
                expert.observe(info, lives, timer)
                if done:
                    successes += info['outcome'] == 'success'
                    break
    finally:
        task.close()
    return np.stack(observations), np.array(actions), np.array(episode_ids), successes


def clone(agent, x, y, train_ids, validation_ids, epochs):
    history = []
    best_loss = np.inf
    best_parameters = None
    for epoch in range(epochs):
        ordering = agent.rng.permutation(train_ids)
        for start in range(0, len(ordering), 256):
            ids = ordering[start:start+256]
            inputs = x[ids]
            probs = runner.softmax(agent.forward(inputs)[2])
            probs[np.arange(len(ids)), y[ids]] -= 1.
            grads = agent.backward(inputs, probs/len(ids), np.zeros(len(ids)))
            for name, grad in grads.items():
                agent.opt[name].step(agent.p[name], grad, .001)
        probs = runner.softmax(agent.forward(x[validation_ids])[2])
        labels = y[validation_ids]
        loss = float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean())
        accuracy = float(np.mean(probs.argmax(axis=1)==labels))
        history.append(dict(epoch=epoch+1, loss=loss, accuracy=accuracy))
        if loss < best_loss:
            best_loss = loss
            best_parameters = {name: value.copy() for name, value in agent.p.items()}
    agent.p = best_parameters
    return history


def experiment(seed, directory, budget, episodes, epochs):
    runner.MsPacManTask = NavigationTask
    with np.load(directory/'demonstrations.npz', allow_pickle=False) as data:
        x, y, episode_ids = data['observations'], data['actions'], data['episode_ids']
        validation = episode_ids >= data['validation_start'].item()
    path = directory/f'seed_{seed}'
    path.mkdir()
    agent = OriginalPPO(9, x.shape[1], seed, hidden=128)
    history = clone(agent, x, y, np.flatnonzero(~validation), np.flatnonzero(validation), epochs)
    (path/'cloning_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    runner.save_model(agent, path/'cloned.npz')
    print(f'Seed {seed}: cloning finished, validation accuracy '
          f'{min(history, key=lambda row: row["loss"])["accuracy"]:.1%}', flush=True)
    summaries = [runner.evaluate(agent, 'B', episodes, seed_start=EVAL_START,
                                 log_dir=path, label='cloned', trace_count=1)]
    print(f'Seed {seed}: cloned success {summaries[0]["success_rate"]:.1%}', flush=True)

    class WarmStartPPO(OriginalPPO):
        def __init__(self, n_actions, n_features, train_seed, **config):
            super().__init__(n_actions, n_features, train_seed, hidden=128, **config)
            self.p = {name: value.copy() for name, value in agent.p.items()}

    # Use the unchanged on-policy rollout/update loop and fresh optimizer state.
    runner.PPOAgent = WarmStartPPO
    try:
        tuned = runner.train('PPO', 'B', seed, budget, lr=1e-4, entropy=.003, log_dir=path)
        assert tuned.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
    runner.save_model(tuned, path/'fine_tuned.npz')
    summaries.append(runner.evaluate(tuned, 'B', episodes, seed_start=EVAL_START,
                                     log_dir=path, label='fine_tuned', trace_count=1))
    result = dict(seed=seed, fine_tuning_interactions=budget,
                  best_cloning_epoch=min(history, key=lambda row: row['loss']),
                  cloned=summaries[0], fine_tuned=summaries[1])
    (path/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    demo_episodes, validation_episodes, epochs = (3, 1, 1) if args.smoke else (64, 8, 24)
    budget, eval_episodes = (256, 2) if args.smoke else (30_000, 200)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    directory = args.results.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    manifest = dict(protocol=runner.PROTOCOL, protocol_id=runner.PROTOCOL_ID,
                    setting='B', policy='sampled', seeds=seeds, eval_seed_start=EVAL_START,
                    demo_seed_start=DEMO_START, demo_episodes=demo_episodes,
                    validation_episodes=validation_episodes, clone_epochs=epochs,
                    fine_tuning_interactions=budget, eval_episodes=eval_episodes,
                    hidden=128, cloning_lr=.001, fine_tuning_lr=1e-4, fine_tuning_entropy=.003,
                    code_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ('imitate_reference.py', 'compare_navigation.py',
                                              'reference_policy.py', 'mspacman.py', 'part2_run.py')})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Collecting reference demonstrations', flush=True)
    demo_seeds = list(range(DEMO_START, DEMO_START+demo_episodes))
    with ProcessPoolExecutor(max_workers=len(seeds)+3) as pool:
        groups = list(pool.map(demonstrate, [demo_seeds[i::len(seeds)] for i in range(len(seeds))]))
        obs, actions, ids = (np.concatenate(parts) for parts in zip(*(g[:3] for g in groups)))
        np.savez_compressed(directory/'demonstrations.npz', observations=obs, actions=actions,
                            episode_ids=ids, validation_start=DEMO_START+demo_episodes-validation_episodes)
        print(f'Demonstrations saved: {len(obs)} decisions, '
              f'{sum(g[3] for g in groups)}/{demo_episodes} successful episodes', flush=True)
        # Record reference outcomes on the same untouched seeds for direct comparison.
        jobs = [pool.submit(experiment, seed, directory, budget, eval_episodes, epochs) for seed in seeds]
        teacher_jobs = [
            pool.submit(evaluate_reference, min(40, eval_episodes-offset), EVAL_START+offset)
            for offset in range(0, eval_episodes, 40)
        ]
        results = []
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(f'Seed {result["seed"]}: fine-tuned success '
                  f'{result["fine_tuned"]["success_rate"]:.1%}', flush=True)
        teacher_rows = [row for job in teacher_jobs for row in job.result()]
    with (directory/'reference_episodes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=teacher_rows[0].keys())
        writer.writeheader()
        writer.writerows(teacher_rows)
    teacher_successes = sum(row['outcome']=='success' for row in teacher_rows)
    sums = {stage: {metric: mean(r[stage][metric] for r in results)
                    for metric in ('success_rate', 'mean_score', 'median_score')}
            for stage in ('cloned', 'fine_tuned')}
    summary = dict(reference_successes=teacher_successes,
                   reference_success_rate=teacher_successes/eval_episodes,
                   evaluation_episodes_per_policy=eval_episodes, stage_means=sums,
                   runs=sorted(results, key=lambda r: r['seed']))
    (directory/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(sums, indent=2), flush=True)


if __name__ == '__main__':
    main()
