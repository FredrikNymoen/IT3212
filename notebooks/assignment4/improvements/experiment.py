"""Bounded development search. Never evaluates D or edits the registered report."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from agents import ImprovedTask,ImprovedQ,MotionPolicy,FrozenMotion
sys.path.insert(0,str(HERE.parent/'pilot'))
from sb3_numpy import FrozenActor

BUDGET=262144
DEV_SEEDS=tuple(range(810000,810050))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def source_hashes():
    paths=[HERE/'agents.py',Path(__file__),HERE.parent/'current/environment.py',
        HERE.parent/'mspacman_env.py',*[HERE.parent/'pilot'/p for p in
        ('tabular_q.py','shared_policy.py','sb3_numpy.py','navigation_features.py','maze_geometry.json')],
        ROOT/'reports/assignment4/Sections/01_predictions.tex']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class RewardScale(gym.RewardWrapper):
    def reward(self,value):return value/100


class EntropySchedule(BaseCallback):
    def _on_step(self):
        self.model.ent_coef=.002+.018*max(0,1-self.num_timesteps/BUDGET)
        return True


def evaluate(model,algorithm,setting,directory,baseline=False):
    if algorithm=='Q-learning':
        before=model.fingerprint();rng=model.rng;actor=None
    else:
        before=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()
        actor=FrozenActor(model) if baseline else FrozenMotion(model)
    env=ImprovedTask(setting);records=[];start=perf_counter()
    try:
        for episode,seed in enumerate(DEV_SEEDS):
            obs,_=env.reset(seed=seed)
            if algorithm=='Q-learning':model.rng=np.random.default_rng(seed+3000000)
            for step in range(1000):
                used=obs[:45] if baseline else obs
                if actor is None:action=model.action(model.state(used))
                else:
                    probability=actor.probabilities(used)
                    if episode==0 and step<25:
                        with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(used[None])).distribution.probs.numpy()[0]
                        assert np.allclose(actual,probability,atol=1e-5) and actual.argmax()==probability.argmax()
                    action=int(probability.argmax())
                obs,reward,term,trunc,info=env.step(action)
                if term or trunc:
                    records.append(dict(seed=seed,success=int(info['success']),score=info['score'],length=info['episode_steps'],reason=info['reason']))
                    break
    finally:env.close()
    if algorithm=='Q-learning':
        model.rng=rng;assert model.fingerprint()==before
    else:assert hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()==before
    frame=pd.DataFrame(records);assert len(frame)==len(DEV_SEEDS)
    frame.to_csv(directory/'development.csv',index=False)
    return dict(success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        mean_length=float(frame.length.mean()),evaluation_seconds=perf_counter()-start,episodes=len(frame))


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory'])
    if (directory/'metrics.json').exists():return json.loads((directory/'metrics.json').read_text())
    if directory.exists():raise FileExistsError(f'Partial run preserved: {directory}')
    directory.mkdir(parents=True);write(directory/'job.json',job)
    algorithm=job['algorithm'];seed=job['seed'];setting=job['setting'];variant=job['variant']
    if variant=='baseline':
        source=ROOT/'results/assignment4/part2_current_v3/final'/f'{setting}_{algorithm}_{seed}'
        file=source/f'checkpoint_262144{".pkl" if algorithm=="Q-learning" else ".zip"}'
        checksum=hashlib.sha256(file.read_bytes()).hexdigest()
        if algorithm=='Q-learning':
            with file.open('rb') as handle:model=pickle.load(handle)
        else:model=PPO.load(file,device='cpu')
        result=dict(**job,**evaluate(model,algorithm,setting,directory,baseline=True),training_seconds=0,training_interactions=0,source_model=str(file.relative_to(ROOT)),model_sha256=checksum)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==checksum
    else:
        start=perf_counter()
        if algorithm=='Q-learning':
            model=ImprovedQ(seed,variant);env=ImprovedTask(setting)
            obs,_=env.reset(seed=seed*1000);episode=0;records=[]
            try:
                for step in range(BUDGET):
                    epsilon=max(.02,1-.98*step/(BUDGET*.4))
                    state=model.state(obs);action=model.action(state,epsilon)
                    nxt,reward,term,trunc,info=env.step(action)
                    model.update(state,action,reward,model.state(nxt),term or trunc);obs=nxt
                    if term or trunc:
                        records.append(dict(interactions=step+1,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                        episode+=1;obs,_=env.reset(seed=seed*1000+episode)
                assert model.update_calls==BUDGET and not hasattr(model,'other') and not hasattr(model,'buffer')
            finally:env.close()
            pd.DataFrame(records).to_csv(directory/'training.csv',index=False)
            file=directory/'model.pkl'
            with file.open('wb') as handle:pickle.dump(model,handle)
        else:
            envs=DummyVecEnv([lambda i=i:Monitor(RewardScale(ImprovedTask(setting)),str(directory/f'training_{i}'),
                info_keywords=('score','success','reason')) for i in range(4)])
            scheduled=variant=='stable'
            rate=(lambda progress:.00005+.00025*max(0,progress)) if scheduled else .0003
            model=PPO(MotionPolicy,envs,learning_rate=rate,n_steps=512,batch_size=256,
                n_epochs=6 if scheduled else 4,gamma=.995,gae_lambda=.97 if scheduled else .95,
                clip_range=.15 if scheduled else .2,ent_coef=.02 if scheduled else .01,
                target_kl=.015 if scheduled else .02,seed=seed,device='cpu',verbose=0)
            try:model.learn(BUDGET,callback=EntropySchedule() if scheduled else None)
            finally:envs.close()
            assert model.num_timesteps==BUDGET
            file=directory/'model.zip';model.save(file)
            write(directory/'optimizer.json',{k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)})
        seconds=perf_counter()-start
        checksum=hashlib.sha256(file.read_bytes()).hexdigest()
        result=dict(**job,**evaluate(model,algorithm,setting,directory),training_seconds=seconds,
            training_interactions=BUDGET,model_sha256=checksum)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==checksum
    write(directory/'metrics.json',result)
    return result


def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    protocol=dict(source_hashes=source_hashes(),budget=BUDGET,development_seeds=list(DEV_SEEDS),
        scope='Post-v3-inspection development revision, not a new independent final test or Part 3 mechanism study.',
        reward='Native score increments A/B, minus 0.1 per decision in C; no other shaping.',
        q_variants=['credit','context'],ppo_variants=['motion','stable'],
        selection='B-development success then mean raw score at the fixed final training budget; never D.',
        preserves='Ordinary single-table Q, exactly one update per real Gymnasium transition; original predictions/report/results unchanged.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Changed sources: use a new run ID'
    else:
        write(path,protocol)
        for name in protocol['source_hashes']:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    if args.stage=='screen':
        settings='B';seeds=[42];variants={'Q-learning':['credit','context'],'PPO':['motion','stable']}
    else:
        selected=json.loads((output/'selected.json').read_text())
        settings=args.settings;seeds=[42,7];variants={algorithm:[variant] for algorithm,variant in selected.items()}
    jobs=[]
    for setting in settings:
        assert setting in 'ABC','D is never used for development selection'
        for algorithm,choices in variants.items():
            for seed in seeds:
                for variant in ['baseline',*choices]:
                    jobs.append(dict(algorithm=algorithm,variant=variant,seed=seed,setting=setting,
                        directory=str(output/f'{setting}_{algorithm}_{variant}_{seed}')))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    frame=pd.DataFrame(results);frame.to_csv(output/f'{args.stage}_{settings}_summary.csv',index=False)
    if args.stage=='screen':
        selected={}
        for algorithm in variants:
            candidate=frame[(frame.algorithm==algorithm)&(frame.variant!='baseline')].sort_values(['success_rate','mean_score'],ascending=False).iloc[0]
            selected[algorithm]=candidate.variant
        write(output/'selected.json',selected);print('SELECTED '+json.dumps(selected),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm'])
    parser.add_argument('--run-id',default='improvements_v1');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--settings',default='B')
    main(parser.parse_args())
