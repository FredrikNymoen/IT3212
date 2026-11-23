"""Frozen-policy development confirmation after selecting an exploratory run.

This does not retrain, change the report, or use the original final-test seeds.
Use a new output label for a new choice: completed records are never overwritten.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from navigation_features import RichTask
from mspacman_env import MsPacmanTask,TaskConfig
from navigation_q import NavigationQ
from sb3_numpy import FrozenActor


def main(args):
    directory=ROOT/'results/assignment4/further_exploration/confirmation'/args.label
    if directory.exists():raise FileExistsError('Choose a new label; existing evidence is protected.')
    directory.mkdir(parents=True)
    path=Path(args.model).resolve()
    if args.kind=='sb3':
        model=PPO.load(path,device='cpu')
        agent=FrozenActor(model)
    elif args.kind=='random':agent=None
    else:
        with path.open('rb') as f:agent=pickle.load(f)
    memory_hash=lambda:hashlib.sha256(pickle.dumps((agent.table,getattr(agent,'other',None)))).hexdigest()
    memory_before=memory_hash() if hasattr(agent,'table') else None
    source_before=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    protocol=vars(args)|dict(model_sha256=source_before,development_only=True,learning_disabled=True,
        task=dict(goal=2000,horizon=1000,setting='B'),
        seeds=list(range(args.seed_start,args.seed_start+args.episodes)))
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    env=MsPacmanTask(TaskConfig('B')) if args.kind.startswith('old') else RichTask()
    rows=[];rng=np.random.default_rng(998877);actor_error=0.;actor_checks=0
    try:
        for seed in protocol['seeds']:
            obs,_=env.reset(seed=seed);torch.manual_seed(seed+2000000)
            policy_rng=np.random.default_rng(seed+2000000)
            if hasattr(agent,'rng'):agent.rng=np.random.default_rng(seed+2000000)
            count=np.zeros(5,dtype=int);stationary=0;unseen=0
            while True:
                if args.kind=='sb3':
                    if len(rows)==0:
                        with torch.no_grad():
                            actual=model.policy.get_distribution(torch.as_tensor(obs[None,:])).distribution.logits.cpu().numpy()[0]
                        logits=agent.logits(obs)
                        logp=logits-np.log(np.exp(logits-logits.max()).sum())-logits.max()
                        error=float(np.max(np.abs(actual-logp)))
                        actor_error=max(actor_error,error);actor_checks+=1
                        assert error<1e-5,'NumPy/Torch actor mismatch'
                        if args.mode=='greedy':assert int(actual.argmax())==int(logits.argmax())
                    action=agent.action(obs,policy_rng,deterministic=args.mode=='greedy')
                elif args.kind=='q':
                    state=agent.state(obs)
                    key=state[1] if args.canonical else state
                    unseen+=int(key not in agent.table)
                    action=agent.action(state,epsilon=args.epsilon)
                elif args.kind=='random':action=int(rng.integers(5))
                else:action=agent.action(obs,explore=int(args.mode=='sample'))
                base=env if args.kind.startswith('old') else env.env
                previous=base.ram[[10,16]].copy()
                count[action]+=1
                obs,_,term,trunc,info=env.step(action)
                stationary+=int(np.array_equal(previous,base.ram[[10,16]]))
                if term or trunc:
                    row=dict(seed=seed,score=info['score'],success=int(info['success']),length=info['episode_steps'],
                        reason=info['reason'],stationary_fraction=stationary/info['episode_steps'],
                        unseen_fraction=unseen/info['episode_steps'],actions=count.tolist())
                    rows.append(row);break
    finally:env.close()
    if path.is_file():assert hashlib.sha256(path.read_bytes()).hexdigest()==source_before
    if memory_before is not None:assert memory_hash()==memory_before
    pd.DataFrame(rows).to_csv(directory/'episodes.csv',index=False)
    summary=dict(label=args.label,mean_score=np.mean([r['score'] for r in rows]),
        success_rate=np.mean([r['success'] for r in rows]),successes=sum(r['success'] for r in rows),
        episodes=len(rows),stationary_fraction=np.mean([r['stationary_fraction'] for r in rows]),
        unseen_fraction=np.mean([r['unseen_fraction'] for r in rows]),learning_disabled_verified=True,
        actor_logp_max_error=actor_error if args.kind=='sb3' else None,actor_checks=actor_checks)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--kind',choices=('oldq','oldppo','q','sb3','random'),required=True)
    parser.add_argument('--mode',choices=('greedy','sample'),default='greedy')
    parser.add_argument('--canonical',action='store_true')
    parser.add_argument('--seed-start',type=int,default=680000)
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--epsilon',type=float,default=0.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(args)
