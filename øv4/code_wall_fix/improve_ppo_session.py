"""Run a time-bounded PPO improvement study using two validation sets and a final test."""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from score_diagnostics import (ENTROPY, NavigationTask, OriginalPPO, TemperaturePolicy,
                               digest, np, read_rows, runner, summarize, write_json)
from train_cool_ppo import verify_temperature_gradient


SCREEN_START = 997000
CONFIRM_START = 998000
TEST_START = 999000
TRIALS = (
    (30_000, 3e-5, .5, 0., 512),
    (30_000, 1e-4, .5, .04, 512),
    (60_000, 3e-5, .5, .04, 512),
    (30_000, 3e-5, .4, .025, 512),
    (60_000, 3e-5, .6, .025, 512),
    (60_000, 5e-5, .5, .04, 1024),
    (90_000, 2e-5, .5, .025, 1024),
    (30_000, 5e-5, .5, .10, 512),
)


def fit(anchor, seed, output, budget, lr, temperature, kl_weight, rollout):
    parent = runner.load_model(anchor)

    class WarmPPO(OriginalPPO):
        def __init__(self, n_actions, n_features, training_seed, **config):
            super().__init__(n_actions, n_features, training_seed,
                             hidden=parent.p['W1'].shape[1], **config)
            self.p = {name: value.copy() for name, value in parent.p.items()}

        def forward(self, obs):
            h1, h2, logits, value = super().forward(obs)
            return h1, h2, logits/temperature, value

        def backward(self, obs, dlogits, dvalue):
            if kl_weight:
                current = runner.softmax(self.forward(obs)[2])
                reference = runner.softmax(parent.forward(obs)[2]/temperature)
                dlogits = dlogits + kl_weight*(current-reference)/len(obs)
            return super().backward(obs, dlogits/temperature, dvalue)

    runner.MsPacManTask = NavigationTask
    runner.PPOAgent = WarmPPO
    try:
        agent = runner.train('PPO', 'B', seed, budget, lr=lr, entropy=ENTROPY,
                             rollout=rollout, log_dir=output)
        assert agent.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
    model = output/'model.npz'
    runner.save_model(agent, model)
    return model


def evaluate(candidate, root, label, start, episodes):
    runner.MsPacManTask = NavigationTask
    agent = TemperaturePolicy(runner.load_model(Path(candidate['model'])), candidate['temperature'])
    folder = root/label/candidate['name']/f"seed_{candidate['seed']}"
    folder.mkdir(parents=True)
    result = runner.evaluate(agent, 'B', episodes, seed_start=start,
                             log_dir=folder, label=label, trace_count=int(label=='test'))
    rows = read_rows(folder/f'{label}_B_episodes.csv')
    reached = sum(float(row['raw_score']) >= 1500 for row in rows)
    result.update(**candidate, episodes=episodes, episode_seed_start=start,
                  reached_1500=reached, conversion_after_1500=(
                      sum(row['outcome']=='success' for row in rows)/reached if reached else None))
    write_json(folder/'metrics.json', result)
    return result


def train_trial(root, candidate, parent, trial, episodes):
    folder = root/'training'/candidate/f"seed_{parent['seed']}"
    folder.mkdir(parents=True)
    budget, lr, temperature, kl_weight, rollout = trial
    model = fit(Path(parent['model']), parent['seed'], folder, budget, lr,
                temperature, kl_weight, rollout)
    entry = dict(name=candidate, seed=parent['seed'], temperature=.5, model=str(model),
                 parent=parent['model'], parent_sha256=digest(Path(parent['model'])),
                 training=dict(budget=budget, lr=lr, rollout=rollout, kl_weight=kl_weight,
                               temperature=temperature))
    return evaluate(entry, root, 'screen', SCREEN_START, episodes)


def rank(rows, seed):
    return sorted((r for r in rows if r['seed']==seed),
                  key=lambda r: (r['success_rate'], r['mean_score']), reverse=True)


def compact(row):
    return {k: row[k] for k in ('name','seed','temperature','model')}


def finish_jobs(jobs, label):
    results = []
    for job in as_completed(jobs):
        row = job.result()
        results.append(row)
        print(f'{label} {row["name"]}, seed {row["seed"]}: '
              f'{row["success_rate"]:.1%}, score {row["mean_score"]:.0f}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--deadline', type=float, help='absolute Unix timestamp in seconds')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error('--minutes must be positive.')
    deadline = args.deadline or time.time()+60*args.minutes
    verify_temperature_gradient()
    root = args.results.resolve()
    base = (here.parent/'results'/'assignment4_cool_ppo_B').resolve()
    source_manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    for name, checksum in source_manifest['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the completed source experiment.')
    for name, checksum in source_manifest['inputs_sha256'].items():
        if digest(Path(name)) != checksum:
            parser.error(f'Input checkpoint changed: {name}')
    source = json.loads((base/'comparison.json').read_text(encoding='utf-8'))
    seeds = [42] if args.smoke else sorted({r['seed'] for r in source['runs']})
    screen_count, confirm_count, test_count = (2, 2, 2) if args.smoke else (40, 80, 200)
    trials = ((256, 3e-5, .5, .04, 512),) if args.smoke else TRIALS
    initial = [dict(name=r['arm'], seed=r['seed'], temperature=.5, model=r['model'])
               for r in source['runs'] if r['seed'] in seeds]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        source=str(base), deadline_epoch=deadline, protocol=runner.PROTOCOL, setting='B',
        code_sha256={name: digest(here/name) for name in ('improve_ppo_session.py', *source_manifest['code_sha256'])},
        initial_model_sha256={r['model']: digest(Path(r['model'])) for r in initial},
        candidates=initial, budget_trials=trials, screen_start=SCREEN_START,
        confirm_start=CONFIRM_START, test_start=TEST_START, episodes=[screen_count,confirm_count,test_count],
        seeds=seeds, smoke=args.smoke))
    with ProcessPoolExecutor(max_workers=3 if args.smoke else 9) as pool:
        rows = finish_jobs([pool.submit(evaluate, r, root, 'screen', SCREEN_START, screen_count)
                            for r in initial], 'Screen')
        write_json(root/'progress.json', dict(stage='training', trial=0, screen=rows))
        longest = 150.
        # Keep enough time for independent confirmation and the 200-episode test.
        reserve = 600.
        completed_trials = 0
        trial_index = 0
        tested_temperatures = set()
        while args.smoke or time.time()+reserve+max(2.2*longest, 210.)<deadline:
            started = time.monotonic()
            parents = [rank(rows, seed)[0] for seed in seeds]
            if trial_index % 3 == 2 and not args.smoke:
                jobs = []
                for parent in parents:
                    for temperature in (.35, .65):
                        key = (parent['model'], temperature)
                        if key in tested_temperatures:
                            continue
                        tested_temperatures.add(key)
                        candidate = dict(compact(parent), name=f'temp_{trial_index}_{temperature:g}',
                                         temperature=temperature)
                        jobs.append(pool.submit(evaluate, candidate, root, 'screen', SCREEN_START, screen_count))
                rows.extend(finish_jobs(jobs, 'Temperature screen'))
            else:
                trial = trials[completed_trials % len(trials)]
                name = f'trial_{completed_trials+1:02d}'
                print(f'{name}: train {trial[0]:,}, lr={trial[1]:g}, temperature={trial[2]:g}, '
                      f'anchor weight={trial[3]:g}, rollout={trial[4]}', flush=True)
                rows.extend(finish_jobs([pool.submit(train_trial, root, name, parent, trial, screen_count)
                                         for parent in parents], 'Screen'))
                completed_trials += 1
            longest = max(longest, time.monotonic()-started)
            trial_index += 1
            write_json(root/'progress.json', dict(stage='training', trial=completed_trials, screen=rows))
            if args.smoke:
                break
        print(f'Confirming finalists, {max(0,(deadline-time.time())/60):.1f} minutes remain', flush=True)
        shortlist = []
        for seed in seeds:
            ordered = rank(rows, seed)
            finalists = ordered[:2]
            baseline = next(r for r in ordered if r['name']=='trained_at_0.5')
            if all(row['name']!=baseline['name'] for row in finalists):
                finalists.append(baseline)
            shortlist.extend(finalists)
        confirmation = finish_jobs([pool.submit(evaluate, compact(r), root, 'confirm',
                                                 CONFIRM_START, confirm_count)
                                    for r in shortlist], 'Confirmation')
        selected = [rank(confirmation, seed)[0] for seed in seeds]
        baseline = [next(r for r in confirmation if r['seed']==seed and r['name']=='trained_at_0.5')
                    for seed in seeds]
        write_json(root/'progress.json', dict(stage='test', trial=completed_trials, screen=rows,
                                             confirmation=confirmation, selected=selected))
        plans = [dict(compact(r), original_name=r['name'], name=arm)
                 for arm, group in [('baseline', baseline), ('selected', selected)] for r in group]
        test = finish_jobs([pool.submit(evaluate, r, root, 'test', TEST_START, test_count)
                            for r in plans], 'Final test')
    result = dict(completed_trials=completed_trials, screen=rows, confirmation=confirmation,
                  selected=selected, test=test, test_means={
                      arm: summarize([r for r in test if r['name']==arm])
                      for arm in ('baseline','selected')})
    write_json(root/'comparison.json', result)
    write_json(root/'progress.json', dict(stage='complete', trial=completed_trials,
                                         results=str(root/'comparison.json')))
    print(json.dumps(result['test_means'], indent=2), flush=True)


if __name__ == '__main__':
    main()
