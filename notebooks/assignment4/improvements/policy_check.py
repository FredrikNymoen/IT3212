"""Frozen v3 policy execution diagnostic on A/B development only; never D."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp
from sb3_numpy import FrozenActor


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return evaluate(job)


def evaluate(job):
    source=exp.ROOT/'results/assignment4/part2_current_v3/final'/f'{job["setting"]}_{job["algorithm"]}_{job["seed"]}'
    q=job['algorithm']=='Q-learning'
    file=source/f'checkpoint_262144{".pkl" if q else ".zip"}'
    checksum=hashlib.sha256(file.read_bytes()).hexdigest()
    if q:
        with file.open('rb') as handle:model=pickle.load(handle)
        before=model.fingerprint()
    else:model=FrozenActor(PPO.load(file,device='cpu'))
    directory=exp.ROOT/'results/assignment4/execution_policy_v1'
    directory.mkdir(parents=True,exist_ok=True)
    name=f'{job["setting"]}_{job["algorithm"]}_{job["seed"]}_{job["parameter"]}'
    rows=[];env=exp.ImprovedTask(job['setting'])
    try:
        for seed in range(830000,830030):
            obs,_=env.reset(seed=seed);rng=np.random.default_rng(seed+4000000)
            if q:model.rng=rng
            for _ in range(1000):
                if q:action=model.action(model.state(obs),job['parameter'])
                else:
                    probabilities=model.probabilities(obs[:45])
                    if job['parameter']==0:action=int(probabilities.argmax())
                    else:
                        p=probabilities.astype(float)**(1/job['parameter']);p/=p.sum()
                        action=int(rng.choice(5,p=p))
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:
                    rows.append(dict(evaluation_seed=seed,success=int(info['success']),score=info['score'],length=info['episode_steps']))
                    break
    finally:env.close()
    if q:assert model.fingerprint()==before
    assert hashlib.sha256(file.read_bytes()).hexdigest()==checksum
    frame=pd.DataFrame(rows);assert len(frame)==30
    frame.to_csv(directory/f'{name}.csv',index=False)
    result=dict(**job,success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        model_sha256=checksum,training_interactions=0,episodes=30)
    exp.write(directory/f'{name}.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    output=exp.ROOT/'results/assignment4/execution_policy_v1';output.mkdir(parents=True,exist_ok=True)
    sources=exp.source_hashes();sources[str(Path(__file__).relative_to(exp.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    exp.write(output/'protocol.json',dict(source_hashes=sources,development_seeds=list(range(830000,830030)),
        purpose='Exploratory frozen execution-policy diagnostic after v3 result inspection. A/B only, no training or D.',
        parameters='Q epsilon 0/.01/.03; PPO argmax or categorical temperature .5/1. Models retain v3 training.'))
    jobs=[dict(setting=setting,algorithm=algorithm,seed=seed,parameter=parameter)
        for setting in 'AB' for algorithm,parameters in [('Q-learning',[0,.01,.03]),('PPO',[0,.5,1])]
        for seed in (42,7) for parameter in parameters]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(output/'summary.csv',index=False)
