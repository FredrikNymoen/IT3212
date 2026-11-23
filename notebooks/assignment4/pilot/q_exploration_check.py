"""Development-only epsilon evaluation of frozen original Q tables."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from mspacman_env import MsPacmanTask,TaskConfig
OUTPUT=ROOT/'results/assignment4/quick_pilot/q_exploration'


def job(args):
    seed,epsilon=args
    with threadpool_limits(limits=1):
        with (ROOT/f'results/assignment4/part2_controlled_v2/final/B_Q-learning_{seed}/agent.pkl').open('rb') as f:
            agent=pickle.load(f)
        fingerprint=agent.fingerprint()
        env=MsPacmanTask(TaskConfig('B'));rows=[]
        try:
            for dev_seed in range(650000,650020):
                obs,_=env.reset(seed=dev_seed)
                agent.rng=np.random.default_rng(dev_seed+2000000)
                for step in range(1000):
                    action=agent.action(obs,explore=epsilon)
                    obs,_,term,trunc,info=env.step(action)
                    if term or trunc:
                        rows.append(dict(training_seed=seed,epsilon=epsilon,development_seed=dev_seed,
                            score=info['score'],success=int(info['success']),length=info['episode_steps']))
                        break
        finally:env.close()
        assert fingerprint==agent.fingerprint()
        OUTPUT.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(OUTPUT/f'Q_{seed}_{epsilon}.csv',index=False)
        return dict(training_seed=seed,epsilon=epsilon,mean_score=np.mean([r['score'] for r in rows]),
                    success_rate=np.mean([r['success'] for r in rows]),evaluation_episodes=20)


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'protocol.json').write_text(json.dumps(dict(development_only=True,training_interactions=0,
        training_seeds=[42,7],development_seeds=list(range(650000,650020)),epsilons=[.05,.1],
        note='Different evaluation policies, reported separately from original greedy Q. No table updates.'),indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,(s,e)) for s in (42,7) for e in (.05,.1)]):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUTPUT/'summary.csv',index=False)
    print(frame.groupby('epsilon')[['mean_score','success_rate']].mean().to_string(),flush=True)
