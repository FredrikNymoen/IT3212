"""Load the development-selected frozen B agents and use Gymnasium reset/step.

Q remains ordinary tabular Q-learning (earlier checkpoint). PPO is the stable
motion-aware package. Loading uses the model hashes locked before confirmation.
This is not an A-trained agent or a new controlled A--D experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from agents import ImprovedTask,FrozenMotion


class SelectedAgent:
    def __init__(self,algorithm,seed=42):
        if algorithm not in ('Q-learning','PPO'):raise ValueError(algorithm)
        path=ROOT/'results/assignment4'/('q_checkpoint_confirmation_v1/locked.json' if algorithm=='Q-learning'
            else 'improvement_confirmation_v1/locked_PPO.json')
        registry=json.loads(path.read_text())
        job=next(j for j in registry['jobs'] if j['seed']==seed and not j['baseline'])
        self.path=ROOT/job['model'];self.expected=job['sha256'];self.algorithm=algorithm
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==self.expected
        if algorithm=='Q-learning':
            with self.path.open('rb') as handle:self.model=pickle.load(handle)
            assert self.model.update_calls==job['checkpoint']
            self.before=self.model.fingerprint()
        else:
            self.model=PPO.load(self.path,device='cpu');self.actor=FrozenMotion(self.model)
            self.before=hashlib.sha256(b''.join(w.tobytes() for w in self.actor.w.values())).hexdigest()

    def reset_rng(self,episode_seed):
        if self.algorithm=='Q-learning':self.model.rng=np.random.default_rng(episode_seed+3000000)

    def action(self,observation):
        if self.algorithm=='Q-learning':return self.model.action(self.model.state(observation))
        return int(self.actor.probabilities(observation).argmax())

    def verify_frozen(self):
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==self.expected
        if self.algorithm=='Q-learning':assert self.model.fingerprint()==self.before
        else:assert hashlib.sha256(b''.join(w.tobytes() for w in self.actor.w.values())).hexdigest()==self.before


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--algorithm',choices=['Q-learning','PPO'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,7],default=42);parser.add_argument('--episodes',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    agent=SelectedAgent(args.algorithm,args.seed);env=ImprovedTask('B')
    try:
        for index in range(args.episodes):
            seed=870000+index;obs,_=env.reset(seed=seed);agent.reset_rng(seed)
            for _ in range(1000):
                obs,reward,terminated,truncated,info=env.step(agent.action(obs))
                if terminated or truncated:
                    print(json.dumps(dict(algorithm=args.algorithm,episode_seed=seed,score=info['score'],
                        success=info['success'],length=info['episode_steps'])),flush=True)
                    break
    finally:env.close()
    agent.verify_frozen()
