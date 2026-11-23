"""Run the exact frozen model identified in the complete updated report."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
import run_study_v2 as study
from candidates import StudyTask,FrozenStudy
from additions import LowBitsTask

def main(args):
    torch.set_num_threads(1)
    root=study.ROOT/'results/assignment4'/args.final_run
    if not ((root/'completed.json').exists() or (root/'interrupted.json').exists()):
        raise RuntimeError('Verify the locked final evaluation before loading a reported model.')
    registry=json.loads((root/'locked.json').read_text())
    choices=[job for job in registry['jobs'] if job['condition']==args.setting and job['algorithm']==args.algorithm]
    seed=args.seed if args.seed is not None else min(job['seed'] for job in choices)
    job=next(job for job in choices if job['seed']==seed)
    assert (Path(job['directory'])/'metrics.json').exists(),'This model/condition has no complete reported evaluation.'
    path=study.ROOT/job['model'];assert study.sha(path)==job['model_sha256']
    for source,expected in registry['source_hashes'].items():assert study.sha(study.ROOT/source)==expected,source
    if args.algorithm=='Q-learning':
        with path.open('rb') as handle:model=pickle.load(handle)
        actor=None
    else:model=PPO.load(path,device='cpu');actor=FrozenStudy(model)
    before=study.fingerprint(model,args.algorithm)
    env=(LowBitsTask if job['variant']=='powerclock' else StudyTask)(args.setting)
    print(json.dumps({key:job[key] for key in ('algorithm','variant','seed','condition','checkpoint_interactions','model','model_sha256')}),flush=True)
    try:
        for index in range(args.episodes):
            episode_seed=1200000+index;obs,_=env.reset(seed=episode_seed)
            if actor is None:model.rng=np.random.default_rng(episode_seed+3000000)
            for _ in range(1000):
                action=model.action(model.state(obs)) if actor is None else int(actor.probabilities(obs).argmax())
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:
                    print(json.dumps(dict(evaluation_seed=episode_seed,success=info['success'],score=info['score'],
                        length=info['episode_steps'],reason=info['reason'])),flush=True);break
    finally:env.close()
    assert study.fingerprint(model,args.algorithm)==before and study.sha(path)==job['model_sha256']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--algorithm',choices=['Q-learning','PPO'],required=True)
    parser.add_argument('--setting',choices=list('ABCD'),default='B');parser.add_argument('--seed',type=int)
    parser.add_argument('--episodes',type=int,default=1);parser.add_argument('--final-run',default='three_hour_20261122/final_v1')
    main(parser.parse_args())
