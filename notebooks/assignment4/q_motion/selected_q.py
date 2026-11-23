"""Load the locked experimental Q candidate with its compatible Gymnasium env."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from motion_q import MotionQ,TabularQ,ImprovedTask
from tunnel_features import TunnelTask


def load_selected(condition='A',seed=42):
    if condition not in ('A','B','D'):raise ValueError('Use A, B or frozen B-to-D transfer.')
    registry=json.loads((ROOT/'results/assignment4/q_representation_confirmation_v1/locked.json').read_text())
    recorded_condition='A' if condition=='A' else 'D'
    job=next(row for row in registry['jobs'] if row['condition']==recorded_condition and row['seed']==seed and row['role']=='candidate')
    file=ROOT/job['model'];assert hashlib.sha256(file.read_bytes()).hexdigest()==job['model_sha256']
    with file.open('rb') as handle:agent=pickle.load(handle)
    assert type(agent) is (MotionQ if job['representation']=='motion' else TabularQ)
    assert agent.update_calls==job['checkpoint']
    env=(TunnelTask if job['representation']=='tunnels' else ImprovedTask)(condition)
    return agent,env,job


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['A','B','D'],default='A')
    parser.add_argument('--seed',type=int,choices=[42,7],default=42);args=parser.parse_args()
    agent,env,job=load_selected(args.condition,args.seed);before=agent.fingerprint()
    episode_seed=970000;agent.rng=np.random.default_rng(episode_seed+3000000)
    try:
        observation,_=env.reset(seed=episode_seed)
        for _ in range(1000):
            action=agent.action(agent.state(observation))
            observation,reward,terminated,truncated,info=env.step(action)
            if terminated or truncated:
                print(json.dumps(dict(condition=args.condition,training_seed=args.seed,representation=job['representation'],
                    checkpoint=job['checkpoint'],score=info['score'],success=info['success'],length=info['episode_steps'])),flush=True)
                break
    finally:env.close()
    assert agent.fingerprint()==before
