"""Short development-only check of saved policies; no training or report edits.

Run from the repository root:
python notebooks/assignment4/pilot/policy_check.py
Uses 20 new development seeds, not the earlier held-out evaluation seeds.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from mspacman_env import MsPacmanTask, TaskConfig

OUTPUT = ROOT / 'results/assignment4/quick_pilot/policy_check'
SEEDS = range(650000, 650020)


def run_job(job):
    algorithm, training_seed, mode = job
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        path = ROOT / 'results/assignment4/part2_controlled_v2/final' / f'B_{algorithm}_{training_seed}/agent.pkl'
        with path.open('rb') as handle:
            agent = pickle.load(handle)
        before = agent.fingerprint()
        rows = []
        env = MsPacmanTask(TaskConfig('B'))
        started = perf_counter()
        try:
            for seed in SEEDS:
                obs, _ = env.reset(seed=seed)
                agent.rng = np.random.default_rng(seed+2000000)
                counts = np.zeros(5, dtype=int)
                stationary = 0
                for step in range(1000):
                    previous = env.ram[[10,16]].copy()
                    action = agent.action(obs, explore=int(mode == 'sample'))
                    counts[action] += 1
                    obs, _, term, trunc, info = env.step(action)
                    stationary += int(np.array_equal(previous, env.ram[[10,16]]))
                    if term or trunc:
                        rows.append(dict(algorithm=algorithm, training_seed=training_seed,
                            mode=mode, development_seed=seed, score=info['score'],
                            success=int(info['success']), length=info['episode_steps'],
                            stationary_fraction=stationary/info['episode_steps'],
                            actions=counts.tolist()))
                        break
        finally:
            env.close()
        assert before == agent.fingerprint()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        file = OUTPUT / f'{algorithm}_{training_seed}_{mode}.json'
        file.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        return dict(algorithm=algorithm, training_seed=training_seed, mode=mode,
            mean_score=float(np.mean([r['score'] for r in rows])),
            success_rate=float(np.mean([r['success'] for r in rows])),
            stationary_fraction=float(np.mean([r['stationary_fraction'] for r in rows])),
            evaluation_episodes=len(rows), seconds=perf_counter()-started)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = list(HERE.glob('*.py')) + [ROOT/'reports/assignment4/Sections/01_predictions.tex']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    protocol = dict(development_only=True, training_interactions=0, setting='B',
        development_seeds=list(SEEDS), training_seeds=[42,7,2026],
        original_source_hashes=hashes, main_report_unchanged=True,
        interpretation='Exploratory diagnosis after prior final results; not an independent test.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs = [(algorithm, seed, mode) for seed in (42,7,2026)
            for algorithm, mode in [('Q-learning','greedy'),('PPO','greedy'),('PPO','sample')]]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run_job, job) for job in jobs]):
            row=future.result()
            rows.append(row)
            print(json.dumps(row),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUTPUT/'summary.csv',index=False)
    print(frame.groupby(['algorithm','mode'])[['mean_score','success_rate','stationary_fraction']].mean().to_string(),flush=True)
    for name,expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name


if __name__=='__main__':
    main()
