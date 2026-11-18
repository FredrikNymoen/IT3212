"""Tune only the REINFORCE settings visible in the user's notebook cell.

Run from the repository root: .venv/Scripts/python.exe lecture/tune_reinforce.py
The notebook and its learning/evaluation implementation are left unchanged.
Selection uses stochastic validation win rate; final evaluation uses unseen games.
This bounded search does not establish a global optimum.
"""

import concurrent.futures
import itertools
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/lecture/reinforce_tuning'
VISIBLE = {
    'actor_lr', 'critic_lr', 'gamma', 'temperature', 'entropy_coef',
    'reinforce_baseline', 'batch_episodes',
}
BASE = dict(actor_lr=.01, critic_lr=.03, gamma=1., temperature=1.,
            entropy_coef=.0001, reinforce_baseline=True, batch_episodes=64)


def initialize():
    notebook = json.loads((ROOT / 'lecture/RL-policy-based.ipynb').read_text(encoding='utf-8'))
    source = ''.join(notebook['cells'][0]['source']).split('\ndef compare_agents(')[0]
    exec(compile(source, 'RL-policy-based.ipynb:helpers', 'exec'), globals())


def run_trial(job):
    assert set(job['settings']) <= VISIBLE
    agent = build_agent('reinforce', episodes=job['episodes'], seed=job.get('seed', 42),
                        **job['settings']).train()
    results = [agent.evaluate(job['evaluation_episodes'], seed=job['evaluation_seed'],
                              deterministic=greedy) for greedy in (False, True)]
    return {**job, 'stochastic': results[0], 'greedy': results[1]}


def save(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def batch(pool, stage, jobs):
    rows = []
    futures = [pool.submit(run_trial, job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
        row = future.result()
        rows.append(row)
        save(stage + '.json', rows)
        print(json.dumps(dict(stage=stage, completed=len(rows), total=len(jobs),
                              id=row['id'], episodes=row['episodes'], seed=row.get('seed', 42),
                              stochastic=row['stochastic']['win_rate'],
                              greedy=row['greedy']['win_rate'], settings=row['settings'])), flush=True)
    return rows


def ranked(rows):
    return sorted(rows, key=lambda row: (row['stochastic']['win_rate'],
                                         row['stochastic']['mean_reward']), reverse=True)


def main():
    initialize()
    started = time.perf_counter()
    candidates = []
    for actor_lr, batch_episodes in itertools.product((.005, .01, .03, .06), (16, 32, 64, 128)):
        candidates.append(dict(BASE, actor_lr=actor_lr, batch_episodes=batch_episodes))
    # A reproducible sample broadens the search without an expensive full grid.
    rng = np.random.default_rng(3212)
    choices = dict(actor_lr=(.005, .01, .03, .06, .1), critic_lr=(.01, .03, .1),
                   gamma=(.95, .99, 1.), temperature=(.5, .7, 1., 1.3),
                   entropy_coef=(0., .0001, .001), reinforce_baseline=(False, True),
                   batch_episodes=(16, 32, 64, 128))
    while len(candidates) < 40:
        candidate = {key: values[int(rng.integers(len(values)))] for key, values in choices.items()}
        if candidate not in candidates:
            candidates.append(candidate)
    save('protocol.json', dict(algorithm='reinforce', objective='stochastic validation win_rate',
         screening_episodes=40_000, refinement_episodes=300_000,
         validation_seed=200_042, validation_episodes_screening=10_000,
         validation_episodes_refinement=30_000, final_seed=900_042,
         final_evaluation_episodes=100_000, training_seeds=(42, 7, 2026),
         fixed_settings='All nonvisible parameters retain their notebook defaults',
         candidates=candidates, python=platform.python_version(), numpy=np.__version__,
         gymnasium=gym.__version__, notebook_helper_source=source_snapshot()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        screening = batch(pool, 'screening', [dict(id=f'candidate_{i:02}', settings=settings,
            episodes=40_000, evaluation_episodes=10_000, evaluation_seed=200_042)
            for i, settings in enumerate(candidates)])
        finalists = ranked(screening)[:6]
        refinement = batch(pool, 'refinement', [dict(id=row['id'], settings=row['settings'],
            episodes=300_000, evaluation_episodes=30_000, evaluation_seed=200_042) for row in finalists])
        winner = ranked(refinement)[0]
        save('selected.json', winner)
        print('LOCKED WINNER ' + json.dumps(winner['settings']), flush=True)
        # Settings are locked before these final games and repeat-training checks.
        controls = [dict(id='original_screenshot', settings=BASE, episodes=100),
                    dict(id='current_notebook', settings=dict(BASE, batch_episodes=32), episodes=100_000),
                    dict(id='same_budget_control', settings=dict(BASE, batch_episodes=32), episodes=300_000),
                    dict(id='selected', settings=winner['settings'], episodes=300_000)]
        jobs = [dict(**control, seed=seed, evaluation_episodes=100_000, evaluation_seed=900_042)
                for control in controls for seed in (42, 7, 2026)]
        final = batch(pool, 'final', jobs)
    summary = {}
    for control in controls:
        subset = [row for row in final if row['id'] == control['id']]
        summary[control['id']] = dict(episodes=control['episodes'], settings=control['settings'])
        for mode in ('stochastic', 'greedy'):
            rates = [row[mode]['win_rate'] for row in subset]
            summary[control['id']][mode] = dict(mean_win_rate=float(np.mean(rates)),
                min_win_rate=min(rates), max_win_rate=max(rates),
                seed_win_rates={str(row['seed']): row[mode]['win_rate'] for row in subset},
                mean_reward=float(np.mean([row[mode]['mean_reward'] for row in subset])))
    save('summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)
    print('Elapsed seconds: ' + str(time.perf_counter() - started), flush=True)


def source_snapshot():
    notebook = json.loads((ROOT / 'lecture/RL-policy-based.ipynb').read_text(encoding='utf-8'))
    return ''.join(notebook['cells'][0]['source']).split('\ndef compare_agents(')[0]


if __name__ == '__main__':
    main()
