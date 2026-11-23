"""Prespecified diagnostic: entropy learning effect versus execution-mode effect.

Freeze all five PPO_base and five PPO_entropy0 final models. Same 50 fresh
seeds under argmax and sampled actions. No learning, model selection, extra
budget or reinterpretation of these secondary modes as 200-episode primary tests.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'three_hour')]
import q_learning as q
import ppo as p
from fast_evaluation_v2 import FastStudyTask


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return evaluate(job)


def evaluate(job):
    q.verify_sources(job['sources']);directory=Path(job['directory'])
    if (directory/'metrics.json').exists():
        old=json.loads((directory/'metrics.json').read_text());assert old['job']==job;return old
    directory.mkdir(parents=True,exist_ok=False);q.write_json(directory/'job.json',job)
    file=Path(job['model']);assert q.sha(file)==job['model_sha256']
    actor=p.FrozenPolicy.load(file);assert actor.model.num_timesteps==262144
    before=actor.fingerprint();env=FastStudyTask('B');rows=[];trace={k:[] for k in ['episode','step','action','probabilities','raw_reward','ram']}
    clock=perf_counter()
    try:
        for episode,seed in enumerate(job['evaluation_seeds']):
            obs,_=env.reset(seed=seed);rng=np.random.default_rng(seed+3000000)
            entropy=0.;disagree=0
            for step in range(1000):
                probabilities=actor.values_for_observation(obs)
                distribution=probabilities.astype(float);distribution/=distribution.sum()
                action=int(probabilities.argmax()) if job['mode']=='argmax' else int(rng.choice(5,p=distribution))
                entropy-=float(np.sum(distribution*np.log(distribution+1e-30)))
                disagree+=action!=int(probabilities.argmax())
                ram=env.unwrapped.ale.getRAM().copy()
                obs,reward,term,trunc,info=env.step(action)
                for key,value in [('episode',episode),('step',step),('action',action),('probabilities',probabilities),('raw_reward',info['raw_reward']),('ram',ram)]:trace[key].append(value)
                if term or trunc:
                    rows.append(dict(evaluation_seed=seed,success=int(info['success']),length=step+1,score=info['score'],
                                     mean_action_entropy=entropy/(step+1),sampled_disagreement=disagree/(step+1)))
                    break
    finally:env.close()
    assert actor.fingerprint()==before and q.sha(file)==job['model_sha256'] and len(rows)==50
    metrics=dict(job=job,episodes=50,success_rate=float(np.mean([r['success'] for r in rows])),
                 mean_score=float(np.mean([r['score'] for r in rows])),evaluation_seconds=perf_counter()-clock,
                 frozen_parameters_verified=True,training_interactions=0)
    q.write_csv(directory/'episodes.csv',rows)
    np.savez_compressed(directory/'traces.npz',**{key:np.asarray(value) for key,value in trace.items()})
    q.write_json(directory/'metrics.json',metrics);return metrics


def main(args):
    output=q.ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    training=q.ROOT/'results/assignment4'/args.study
    protocol_file=training/'protocol.json';source_protocol=json.loads(protocol_file.read_text())
    sources=q.source_hashes();sources[Path(__file__).relative_to(q.ROOT).as_posix()]=q.sha(__file__)
    plan=dict(arms=['PPO_base','PPO_entropy0'],training_seeds=[17,29,101,314,2718],modes=['argmax','sampled'],
        evaluation_seeds=list(range(1700000,1700050)),sources=sources,study_protocol_sha256=q.sha(protocol_file),
        question='Does an entropy-arm advantage persist under sampled execution, or depend on argmax concentrating trajectories?',
        predictions='A persistent paired effect across modes supports a learned-distribution explanation; a reversed/removed gap supports execution-mode mediation. Neither is proof of universal superiority.',
        scope='Secondary bounded diagnostic; primary per-model evaluation remains200 episodes. No learning, extra training budget or configuration selection.')
    path=output/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:q.write_json(path,plan)
    if args.command=='plan':print('Diagnostic plan locked before mechanism test outcomes.');return
    assert (training/'training_completed.json').exists()
    jobs=[]
    for arm in plan['arms']:
        for seed in plan['training_seeds']:
            source=training/'training'/f'{arm}_{seed}';saved=json.loads((source/'completed.json').read_text())
            assert saved['job']['budget']==262144
            for mode in plan['modes']:
                jobs.append(dict(arm=arm,seed=seed,mode=mode,model=str(source/saved['model']),model_sha256=saved['model_sha256'],
                                 sources=sources,evaluation_seeds=plan['evaluation_seeds'],directory=str(output/f'{arm}_{seed}_{mode}')))
    lock=output/'locked.json'
    if lock.exists():assert json.loads(lock.read_text())==jobs
    else:q.write_json(lock,jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();print(json.dumps({key:result['job'][key] for key in ['arm','seed','mode']}),flush=True)
    q.write_json(output/'completed.json',dict(cells=20,episodes=1000,training_interactions=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('command',choices=['plan','run'])
    cli.add_argument('--run-id',default='completion_diagnostic_v1');cli.add_argument('--study',default='completion_part3_v1')
    cli.add_argument('--workers',type=int,default=3);main(cli.parse_args())
