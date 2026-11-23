"""Prespecified five-seed one-factor Part 3 study. No model/config selection.

Baseline and modified arms are ALL trained afresh on the same CPU/software.
45 trained models, fixed 262144 interactions; 200 frozen episodes per model.
The historical Part 2 models/results remain unchanged. Use a new run ID after
any source change. This is mechanism investigation, not a new tuning search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ASSIGNMENT=HERE.parent
ROOT=ASSIGNMENT.parents[1]
sys.path[:0]=[str(ASSIGNMENT),str(ASSIGNMENT/'three_hour')]
import q_learning as q
import ppo as p
from fast_evaluation_v2 import FastStudyTask

SEEDS=[17,29,101,314,2718]
ARMS={
 'Q_base':dict(algorithm='Q-learning',alpha=.15,gamma=.995,epsilon_fraction=.8),
 'Q_fast_epsilon':dict(algorithm='Q-learning',alpha=.15,gamma=.995,epsilon_fraction=.2),
 'Q_gamma99':dict(algorithm='Q-learning',alpha=.15,gamma=.99,epsilon_fraction=.8),
 'Q_alpha05':dict(algorithm='Q-learning',alpha=.05,gamma=.995,epsilon_fraction=.8),
 'PPO_base':dict(algorithm='PPO',gamma=.995,entropy_start=.02,entropy_end=.002,lr_start=.0003,lr_end=.00005,epochs=6),
 'PPO_gamma99':dict(algorithm='PPO',gamma=.99,entropy_start=.02,entropy_end=.002,lr_start=.0003,lr_end=.00005,epochs=6),
 'PPO_entropy0':dict(algorithm='PPO',gamma=.995,entropy_start=0.,entropy_end=0.,lr_start=.0003,lr_end=.00005,epochs=6),
 'PPO_lr1e4':dict(algorithm='PPO',gamma=.995,entropy_start=.02,entropy_end=.002,lr_start=.0001,lr_end=.00005,epochs=6),
 'PPO_epochs2':dict(algorithm='PPO',gamma=.995,entropy_start=.02,entropy_end=.002,lr_start=.0003,lr_end=.00005,epochs=2),
}


def make_env(setting):
    return FastStudyTask(setting)


def evaluate(agent,algorithm,seeds,output,name,trace=False,mode='primary'):
    """Use the shared Gymnasium evaluator; fast wrapper changes computation only."""
    saved=q.make_environment
    q.make_environment=make_env
    try:
        return q.evaluate(agent,algorithm,'B',seeds,output,name,trace=trace)
    finally:
        q.make_environment=saved


class Callback(p.Checkpoints):
    def __init__(self,budget,points,save,zero_entropy):
        super().__init__(budget,points,save);self.zero_entropy=zero_entropy
    def _on_step(self):
        if self.zero_entropy:
            self.model.ent_coef=0.;return True
        return super()._on_step()


def train(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return _train(job)


def _train(job):
    q.verify_sources(job['sources'])
    output=Path(job['directory'])
    if (output/'completed.json').exists():
        result=json.loads((output/'completed.json').read_text())
        assert result['job']==job and q.sha(output/result['model'])==result['model_sha256']
        return result
    output.mkdir(parents=True,exist_ok=False);q.write_json(output/'job.json',job)
    cfg=job['config'];algorithm=cfg['algorithm'];seed=job['seed'];budget=job['budget']
    points=q.checkpoint_points(budget);curves=[];seconds=0.
    def checkpoint(point):
        nonlocal seconds,clock
        seconds+=perf_counter()-clock
        file=output/f'checkpoint_{point}{".npz" if algorithm=="Q-learning" else ".zip"}'
        if algorithm=='Q-learning':
            assert model.update_calls==point;model.save(file);agent=model
            extra=dict(table_states=len(model.table),epsilon=max(.05,1-.95*point/(cfg['epsilon_fraction']*budget)))
        else:
            assert model.num_timesteps==point;model.save(file);agent=p.FrozenPolicy(model)
            logs=model.logger.name_to_value
            extra={key:float(logs['train/'+key]) if 'train/'+key in logs else None
                   for key in ['approx_kl','clip_fraction','entropy_loss','explained_variance','value_loss']}
            extra['sb3_n_updates']=int(model._n_updates)
        metrics=evaluate(agent,algorithm,job['development_seeds'],output,f'development_{point}')
        curves.append(dict(interactions=point,training_seconds=seconds,model=file.name,model_sha256=q.sha(file),**extra,**metrics))
        q.write_json(output/'curves.json',curves)
        print(json.dumps(dict(event='checkpoint',arm=job['arm'],seed=seed,interactions=point)),flush=True)
        clock=perf_counter()
    if algorithm=='Q-learning':
        model=q.TabularQ(seed,alpha=cfg['alpha'],gamma=cfg['gamma']);env=make_env('B');training=[]
        try:
            obs,_=env.reset(seed=seed*1000);episode=0;clock=perf_counter();checkpoint(0)
            for step in range(budget):
                state=model.state(obs);epsilon=max(.05,1-.95*step/(cfg['epsilon_fraction']*budget))
                action=model.action(state,epsilon);nxt,reward,term,trunc,info=env.step(action)
                model.update(state,action,reward,model.state(nxt),term or trunc);obs=nxt
                if term or trunc:
                    training.append(dict(interactions=step+1,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;obs,_=env.reset(seed=seed*1000+episode)
                if step+1 in points:checkpoint(step+1)
            assert model.update_calls==budget
        finally:env.close()
        q.write_csv(output/'training.csv',training)
    else:
        envs=DummyVecEnv([lambda i=i:Monitor(p.RewardScale(make_env('B')),str(output/f'training_{i}'),
                         info_keywords=('score','success','reason','episode_steps')) for i in range(4)])
        try:
            start,end=cfg['lr_start'],cfg['lr_end']
            model=p.build_model(envs,seed,gamma=cfg['gamma'],ent_coef=cfg['entropy_start'],n_epochs=cfg['epochs'],
                                learning_rate=lambda progress:end+(start-end)*max(0,progress))
            clock=perf_counter()
            model.learn(budget,callback=Callback(budget,[x for x in points if x<budget],checkpoint,cfg['entropy_start']==0))
            checkpoint(budget)
        finally:envs.close()
    final=curves[-1]
    result=dict(job=job,model=final['model'],model_sha256=final['model_sha256'],training_interactions=budget,
                training_seconds=seconds,update_calls=model.update_calls if algorithm=='Q-learning' else None,
                selection='Final full-budget checkpoint fixed before training; no development/test checkpoint selection.')
    q.write_json(output/'completed.json',result);return result


def final_evaluation(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        q.verify_sources(job['sources']);output=Path(job['evaluation_directory'])
        if (output/'metrics.json').exists():
            old=json.loads((output/'metrics.json').read_text());assert old['job']==job;return old
        output.mkdir(parents=True,exist_ok=False);q.write_json(output/'job.json',job)
        file=Path(job['directory'])/job['model'];assert q.sha(file)==job['model_sha256']
        algorithm=job['config']['algorithm']
        agent=q.TabularQ.load(file) if algorithm=='Q-learning' else p.FrozenPolicy.load(file)
        count=agent.update_calls if algorithm=='Q-learning' else agent.model.num_timesteps
        assert count==job['budget']
        metrics=evaluate(agent,algorithm,job['evaluation_seeds'],output,'evaluation',trace=True)
        assert q.sha(file)==job['model_sha256']
        result=dict(job=job,**metrics);q.write_json(output/'metrics.json',result);return result


def run(args):
    output=ROOT/'results/assignment4'/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    sources=q.source_hashes();sources[Path(__file__).relative_to(ROOT).as_posix()]=q.sha(__file__)
    jobs=[]
    for arm,cfg in ARMS.items():
        for seed in (SEEDS if not args.smoke else [17]):
            jobs.append(dict(arm=arm,config=cfg,seed=seed,budget=262144 if not args.smoke else 2048,
                             development_seeds=list(range(1480000,1480010)) if not args.smoke else [1480000],
                             directory=str(output/'training'/f'{arm}_{seed}'),sources=sources))
    protocol=dict(jobs=jobs,training_seeds=SEEDS if not args.smoke else [17],condition='B',workers=args.workers,
        evaluation_seeds=list(range(1500000,1500200)),sources=sources,
        hardware=dict(cpu=platform.processor(),os=platform.platform(),python=platform.python_version(),
                      packages={n:version(n) for n in ['numpy','torch','gymnasium','ale-py','stable-baselines3']}),
        selection='No configuration selection: report all one-factor arms, five paired seeds; fixed full-budget checkpoint.',
        clarification='All baselines and modifications retrained on same host/software as Part2. Historical models preserved.',
        stopping='Every model consumes its fixed budget. No tuning on final evaluation; diagnostic modes reported separately.',
        smoke=args.smoke)
    lock=output/'protocol.json'
    if lock.exists():assert json.loads(lock.read_text())==protocol,'Sources/configuration changed; choose a new run ID.'
    else:
        q.write_json(lock,protocol)
        for name in sources:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    if args.stage in ['train','run']:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(train,job) for job in jobs]):
                row=future.result();print('TRAINED '+row['job']['arm']+' '+str(row['job']['seed']),flush=True)
        q.write_json(output/'training_completed.json',dict(models=len(jobs),training_interactions=sum(j['budget'] for j in jobs)))
    if args.stage in ['evaluate','run'] and not args.smoke:
        assert (output/'training_completed.json').exists()
        evaluation_jobs=[]
        for job in jobs:
            saved=json.loads((Path(job['directory'])/'completed.json').read_text());assert saved['job']==job
            evaluation_jobs.append(dict(job,model=saved['model'],model_sha256=saved['model_sha256'],
                evaluation_directory=str(output/'evaluation'/f'{job["arm"]}_{job["seed"]}'),evaluation_seeds=protocol['evaluation_seeds']))
        final_lock=output/'final_lock.json'
        if final_lock.exists():assert json.loads(final_lock.read_text())==evaluation_jobs
        else:q.write_json(final_lock,evaluation_jobs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(final_evaluation,job) for job in evaluation_jobs]):
                row=future.result();print('EVALUATED '+row['job']['arm']+' '+str(row['job']['seed']),flush=True)
        q.write_json(output/'completed.json',dict(models=len(jobs),episodes=200*len(jobs),training_interactions=sum(j['budget'] for j in jobs)))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=['train','evaluate','run']);cli.add_argument('--run-id',required=True)
    cli.add_argument('--workers',type=int,default=6);cli.add_argument('--smoke',action='store_true')
    run(cli.parse_args())
