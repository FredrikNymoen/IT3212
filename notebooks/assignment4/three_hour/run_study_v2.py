"""Versioned bounded training/evaluation. New run IDs after any source change."""
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
from candidates import StudyTask,StudyPolicy,FrozenStudy,make_q,ppo_settings,Q_VARIANTS,PPO_VARIANTS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def sources():
    paths=[Path(__file__),HERE/'candidates.py',HERE.parent/'improvements/agents.py',
        HERE.parent/'q_motion/motion_q.py',HERE.parent/'current/environment.py',HERE.parent/'mspacman_env.py',
        *[HERE.parent/'pilot'/name for name in ('tabular_q.py','navigation_features.py','maze_geometry.json')],
        ROOT/'reports/assignment4/Sections/01_predictions.tex',ROOT/'IT3212 - Assignment 4.pdf']
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}

def fingerprint(model,algorithm):
    if algorithm=='Q-learning':return model.fingerprint()
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()

def native_score(ram):
    bcd=lambda v:int(v)//16*10+int(v)%16
    return bcd(ram[122])*10000+bcd(ram[121])*100+bcd(ram[120])


def evaluate(model,algorithm,setting,seeds,directory,name,trace=False):
    before=fingerprint(model,algorithm);rng=model.rng if algorithm=='Q-learning' else None
    actor=FrozenStudy(model) if algorithm=='PPO' else None
    env=StudyTask(setting);rows=[];unknown=0;decisions=0;error=0;start=perf_counter()
    traces={key:[] for key in ('ram','observation','next_observation','values','action','episode','step','reward','raw_reward','score_after')}
    try:
        assert abs(env.unwrapped.ale.getFloat('repeat_action_probability')-(0 if setting=='A' else .25))<1e-8
        for index,seed in enumerate(seeds):
            obs,_=env.reset(seed=int(seed));initial=native_score(env.env.env.ram)
            if algorithm=='Q-learning':model.rng=np.random.default_rng(int(seed)+3000000)
            for step in range(1000):
                if actor is None:
                    state=model.state(obs);values=model.values(state);unknown+=state[1] not in model.table
                    action=model.action(state)
                else:
                    values=actor.probabilities(obs);action=int(values.argmax())
                    if index==0 and step<20:
                        with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(obs[None])).distribution.probs.numpy()[0]
                        error=max(error,float(np.max(np.abs(actual-values))))
                        assert np.allclose(actual,values,atol=1e-5) and actual.argmax()==action
                if trace:
                    for key,value in [('ram',env.env.env.ram.copy()),('observation',obs.copy()),('values',values.copy()),
                                      ('action',action),('episode',index),('step',step)]:traces[key].append(value)
                nxt,reward,term,trunc,info=env.step(action);decisions+=1
                assert native_score(env.env.env.ram)-initial==info['score']
                assert np.isclose(reward,info['raw_reward']-(.1 if setting=='C' else 0))
                if trace:
                    for key,value in [('next_observation',nxt.copy()),('reward',reward),('raw_reward',info['raw_reward']),
                                      ('score_after',info['score'])]:traces[key].append(value)
                obs=nxt
                if term or trunc:
                    rows.append(dict(evaluation_seed=int(seed),success=int(info['success']),score=info['score'],
                        task_return=info['task_return'],length=info['episode_steps'],reason=info['reason']));break
    finally:
        env.close()
        if algorithm=='Q-learning':model.rng=rng
    seconds=perf_counter()-start
    assert fingerprint(model,algorithm)==before
    frame=pd.DataFrame(rows);assert len(frame)==len(seeds)
    assert np.array_equal(frame.success.astype(bool),frame.score>=2000) and frame.length.between(1,1000).all()
    frame.to_csv(directory/f'{name}.csv',index=False)
    if trace:
        np.savez_compressed(directory/f'{name}_traces.npz',**{key:np.asarray(values) for key,values in traces.items()})
    outcome_length=lambda success:float(frame.loc[frame.success==success,'length'].mean()) if (frame.success==success).any() else None
    return dict(episodes=len(frame),success_rate=float(frame.success.mean()),failure_rate=float(1-frame.success.mean()),
        mean_score=float(frame.score.mean()),mean_return=float(frame.task_return.mean()),mean_length=float(frame.length.mean()),
        success_length=outcome_length(1),failure_length=outcome_length(0),evaluation_seconds=seconds,
        evaluation_interactions=decisions,unseen_state_fraction=unknown/decisions if algorithm=='Q-learning' else None,
        native_score_verified_every_step=True,frozen_parameters_verified=True,torch_numpy_max_error=error,
        policy='Greedy Q with episode-seeded uniform ties; PPO argmax.',further_training_interactions=0)


class RewardScale(gym.RewardWrapper):
    def reward(self,reward):return reward/100


class Checkpoints(BaseCallback):
    def __init__(self,budget,points,save):
        super().__init__();self.budget=budget;self.points=set(points);self.save=save
    def _on_rollout_start(self):
        if self.num_timesteps in self.points:self.save(self.num_timesteps)
    def _on_step(self):
        self.model.ent_coef=.002+.018*max(0,1-self.num_timesteps/self.budget)
        return True


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return train(job)

def train(job):
    for path,expected in job['source_hashes'].items():assert sha(ROOT/path)==expected,path
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(f'Partial job preserved: {directory}')
    directory.mkdir(parents=True);write(directory/'job.json',job)
    algorithm=job['algorithm'];variant=job['variant'];budget=job['budget'];seed=job['seed'];setting=job['setting']
    curves=[];training_seconds=0.;last_clock=perf_counter()
    def checkpoint(point):
        nonlocal training_seconds,last_clock
        training_seconds+=perf_counter()-last_clock
        suffix='.pkl' if algorithm=='Q-learning' else '.zip'
        path=directory/f'checkpoint_{point}{suffix}'
        if algorithm=='Q-learning':
            assert model.update_calls==point
            with path.open('wb') as handle:pickle.dump(model,handle)
        else:
            assert model.num_timesteps==point
            model.save(path)
        metrics=evaluate(model,algorithm,setting,job['development_seeds'],directory,f'development_{point}')
        record=dict(interactions=point,training_seconds=training_seconds,model=str(path.relative_to(ROOT)),
            model_sha256=sha(path),**metrics)
        curves.append(record);pd.DataFrame(curves).to_csv(directory/'curves.csv',index=False)
        print(json.dumps(dict(event='checkpoint',algorithm=algorithm,variant=variant,setting=setting,seed=seed,
            interactions=point,success_rate=metrics['success_rate'],mean_score=metrics['mean_score'])),flush=True)
        last_clock=perf_counter()
    if algorithm=='Q-learning':
        model=make_q(variant,seed);env=StudyTask(setting);obs,_=env.reset(seed=seed*1000);episode=0;rows=[]
        try:
            if 0 in job['checkpoints']:checkpoint(0)
            for step in range(budget):
                state=model.state(obs);epsilon=max(.05,1-.95*step/(budget*.8))
                action=model.action(state,epsilon);nxt,reward,term,trunc,info=env.step(action)
                model.update(state,action,reward,model.state(nxt),term or trunc);obs=nxt
                if term or trunc:
                    rows.append(dict(interactions=step+1,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;obs,_=env.reset(seed=seed*1000+episode)
                if step+1 in job['checkpoints']:checkpoint(step+1)
            assert model.update_calls==budget and not hasattr(model,'other') and not hasattr(model,'buffer')
        finally:env.close()
        pd.DataFrame(rows).to_csv(directory/'training.csv',index=False)
    else:
        settings=ppo_settings(variant);start_rate=settings.pop('lr_start');end_rate=settings.pop('lr_end')
        settings.pop('entropy_start');settings.pop('entropy_end')
        envs=DummyVecEnv([lambda i=i:Monitor(RewardScale(StudyTask(setting)),str(directory/f'training_{i}'),
            info_keywords=('score','success','reason','episode_steps')) for i in range(4)])
        model=PPO(StudyPolicy,envs,learning_rate=lambda progress:end_rate+(start_rate-end_rate)*max(0,progress),
            ent_coef=.02,n_steps=512,batch_size=256,seed=seed,device='cpu',verbose=0,
            policy_kwargs=dict(actor_variant=variant),**settings)
        try:
            model.learn(budget,callback=Checkpoints(budget,[point for point in job['checkpoints'] if point<budget],checkpoint))
            checkpoint(budget)
        finally:envs.close()
    result=dict(job,training_interactions=budget,training_seconds=training_seconds,checkpoints=curves,
        ordinary_q=True if algorithm=='Q-learning' else None)
    write(done,result);return result


def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    registry=sources();points=sorted(set([0,*[int(x) for x in args.checkpoints.split(',')],args.budget]))
    assert all(0<=point<=args.budget and point%2048==0 for point in points)
    assert set(args.settings)<=set('ABC'),'D cannot enter training/development.'
    jobs=[]
    for setting in args.settings:
        for algorithm,variants in [('Q-learning',args.q_variants.split(',')),('PPO',args.ppo_variants.split(','))]:
            for variant in variants:
                if not variant:continue
                assert variant in (Q_VARIANTS if algorithm=='Q-learning' else PPO_VARIANTS)
                for seed in [int(x) for x in args.seeds.split(',')]:
                    jobs.append(dict(algorithm=algorithm,variant=variant,setting=setting,seed=seed,budget=args.budget,
                        checkpoints=points,development_seeds=list(range(args.dev_start,args.dev_start+args.episodes)),
                        directory=str(output/f'{setting}_{algorithm}_{variant}_{seed}'),source_hashes=registry))
    protocol=dict(jobs=jobs,source_hashes=registry,scope='Post-inspection development; no D selection; ordinary Q only.',
        reward='Unchanged native score A/B; C subtracts .1 per decision; internal reward /100.',
        selection='Equal A/B mean development success across available training seeds, then mean game score, earlier common checkpoint.')
    lock=output/'protocol.json'
    if lock.exists():assert json.loads(lock.read_text())==protocol,'Sources/configuration changed: use a new run ID.'
    else:
        write(lock,protocol)
        for name in registry:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result)
            print('DONE '+json.dumps({key:result[key] for key in ('algorithm','variant','setting','seed','training_seconds')}),flush=True)
    comparison=pd.concat([pd.read_csv(Path(job['directory'])/'curves.csv').assign(algorithm=job['algorithm'],
        variant=job['variant'],setting=job['setting'],seed=job['seed']) for job in jobs],ignore_index=True)
    comparison.to_csv(output/'comparison.csv',index=False)
    write(output/'completed.json',dict(jobs=len(jobs),training_interactions=sum(job['budget'] for job in jobs)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--budget',type=int,default=262144);parser.add_argument('--settings',default='AB')
    parser.add_argument('--seeds',default='42');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--dev-start',type=int,default=1110000)
    parser.add_argument('--checkpoints',default='65536,131072,196608,262144')
    parser.add_argument('--q-variants',default=','.join(Q_VARIANTS));parser.add_argument('--ppo-variants',default=','.join(PPO_VARIANTS))
    main(parser.parse_args())
