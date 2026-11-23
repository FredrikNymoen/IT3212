"""Independent Gymnasium-compatible PPO reference, development data only."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from mspacman_env import MsPacmanTask,TaskConfig


class ScaleReward(gym.RewardWrapper):
    def reward(self,reward):return reward/100


def model_hash(model):
    return hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes()
        for value in model.policy.state_dict().values())).hexdigest()


def evaluate(model,mode,directory,step,env_factory):
    before=model_hash(model)
    rng=torch.get_rng_state()
    numpy_rng=np.random.get_state()
    training=model.policy.training
    env=env_factory();rows=[]
    from sb3_numpy import FrozenActor
    actor=FrozenActor(model)
    fast=bool(getattr(model,'_pilot_fast_evaluation',False))
    try:
        for seed in range(670000,670020):
            obs,_=env.reset(seed=seed)
            torch.manual_seed(seed+2000000)
            policy_rng=np.random.default_rng(seed+2000000)
            for i in range(1000):
                if fast:
                    action=actor.action(obs,policy_rng,deterministic=(mode=='greedy'))
                else:action,_=model.predict(obs,deterministic=(mode=='greedy'))
                obs,_,term,trunc,info=env.step(int(action))
                if term or trunc:
                    rows.append(dict(development_seed=seed,score=info['score'],
                        success=int(info['success']),length=info['episode_steps'],reason=info['reason']))
                    break
    finally:
        env.close();torch.set_rng_state(rng);np.random.set_state(numpy_rng)
        model.policy.set_training_mode(training)
    assert before==model_hash(model)
    pd.DataFrame(rows).to_csv(directory/f'development_{step}_{mode}.csv',index=False)
    return dict(interactions=step,mode=mode,mean_score=np.mean([r['score'] for r in rows]),
        success_rate=np.mean([r['success'] for r in rows]),episodes=len(rows))


def main(args):
    label=f'reference_{args.observation}' if args.policy=='mlp' else f'shared_{args.observation}'
    suffix=f'_{args.tag}' if args.tag else ''
    directory=ROOT/'results/assignment4/further_exploration'/f'{label}_{args.seed}_{args.budget}{suffix}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.csv').exists():
        print('Completed run already exists:',directory,flush=True);return
    if (directory/'protocol.json').exists():raise FileExistsError('Incomplete run exists; preserve it and use a new --tag.')
    def factory():
        if args.observation=='base':return MsPacmanTask(TaskConfig('B'))
        from navigation_features import RichTask
        return RichTask()
    protocol=dict(algorithm='Stable Baselines3 PPO',version=metadata.version('stable-baselines3'),
        setting='B',observation=args.observation,policy=args.policy,training_seed=args.seed,training_budget=args.budget,
        evaluation_backend='NumPy categorical' if args.fast_eval else 'SB3 Torch categorical',
        goal=2000,horizon=1000,development_seeds=list(range(670000,670020)),num_envs=4,
        learning_rate=.0003,gamma=.995,rollout_per_environment=512,batch_size=256,epochs=4,
        entropy=.01,target_kl=.02,policy_net=[64,64],value_net=[64,64],reward_scale=.01,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Exploratory reference configuration, not an otherwise-identical implementation comparison.')
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for source in (Path(__file__),Path(__file__).with_name('navigation_features.py'),Path(__file__).with_name('maze_geometry.json'),Path(__file__).with_name('shared_policy.py'),Path(__file__).with_name('sb3_numpy.py')):
        (directory/source.name).write_bytes(source.read_bytes())
    envs=DummyVecEnv([lambda i=i:Monitor(ScaleReward(factory()),str(directory/f'training_{i}'),
                                      info_keywords=('score','success','reason')) for i in range(4)])
    if args.policy=='shared':
        from shared_policy import SharedActionPolicy
        policy=SharedActionPolicy
    else:policy='MlpPolicy'
    model=PPO(policy,envs,learning_rate=.0003,n_steps=512,batch_size=256,n_epochs=4,
        gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.01,target_kl=.02,seed=args.seed,
        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])),device='cpu',verbose=0)
    model._pilot_fast_evaluation=args.fast_eval
    rows=[];training_seconds=0.
    try:
        for stop in range(65536,args.budget+1,65536):
            started=perf_counter()
            model.learn(total_timesteps=65536,reset_num_timesteps=False)
            training_seconds+=perf_counter()-started
            assert model.num_timesteps==stop
            model.save(directory/f'checkpoint_{stop}')
            for mode in ('greedy','sample'):
                row=evaluate(model,mode,directory,stop,factory)
                row['training_seconds']=training_seconds
                rows.append(row);print(json.dumps(row),flush=True)
            pd.DataFrame(rows).to_csv(directory/'progress.csv',index=False)
        pd.DataFrame(rows).to_csv(directory/'summary.csv',index=False)
    finally:envs.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget',type=int,default=262144)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--observation',choices=('base','rich'),default='base')
    parser.add_argument('--policy',choices=('mlp','shared'),default='mlp')
    parser.add_argument('--fast-eval',action='store_true')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.budget%65536:raise ValueError('Budget must be a multiple of 65536.')
    if args.policy=='shared' and args.observation!='rich':raise ValueError('Shared actor requires rich observations.')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(args)
