"""Reproduce the selected main configurations and all controlled mechanism arms.

Fresh run ID required. This consumes hours; --plan only prints commands.
It never overwrites the canonical report or uses final outcomes for selection.
Development search provenance is supplied separately; selected configurations
are fixed before this reproduction. Historical forced-map D is not invoked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import os

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import q_learning as q

def worker(job):
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    script=HERE.parent/('q_learning.py' if job['algorithm']=='Q-learning' else 'ppo.py')
    directory=Path(job['directory'])
    command=[sys.executable,str(script),'train','--setting',job['condition'],'--seed',str(job['seed']),'--output',str(directory)]
    subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(script),'evaluate','--selection',str(directory/'selected.json'),
                    '--setting',job['condition'],'--output',str(directory/'final')],cwd=ROOT,check=True)
    return job

def occlusion(job):
    import torch
    from threadpoolctl import threadpool_limits
    import ppo as p
    from observation_shift import ObservationShift
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=False)
        path=Path(job['selection']);selected=json.loads(path.read_text());q.verify_sources(selected['source_hashes'])
        model=path.parent/selected['model'];assert q.sha(model)==selected['model_sha256']
        agent=q.TabularQ.load(model) if job['algorithm']=='Q-learning' else p.FrozenPolicy.load(model)
        q.write_json(directory/'locked.json',dict(job,selected=selected,ROI=[16,24,48,56],evaluation_seeds=list(range(1600000,1600200))))
        previous=q.make_environment;q.make_environment=lambda _:ObservationShift(job['occluded'])
        try:metrics=q.evaluate(agent,job['algorithm'],'B',range(1600000,1600200),directory,'evaluation',trace=True)
        finally:q.make_environment=previous
        q.write_json(directory/'metrics.json',dict(metrics,job=job,training_interactions=0))
    return job

def launch(script,*args):subprocess.run([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,check=True)

def main(args):
    prefix=args.run_id;directory=(ROOT/'results/assignment4'/prefix).resolve()
    assert directory.is_relative_to(ROOT/'results/assignment4') and '..' not in Path(prefix).parts
    if args.plan:
        print(json.dumps(dict(selected_main_models=30,main_interactions=30*262144,mechanism_models=45,
            mechanism_interactions=45*262144,main_primary_episodes=6000,clarified_D_episodes=4000,
            mechanism_primary_episodes=9000,secondary_diagnostic_episodes=1000,
            training_roots=[17,29,101,314,2718],stage=args.stage,output=str(directory),
            note='Fixed selected configurations; historical search not rerun. Reproduction uses the original registered seeds and is not a new independent test.'),indent=2));return
    directory.mkdir(parents=True,exist_ok=False)
    q.write_json(directory/'launch.json',dict(arguments=vars(args),sources=q.source_hashes(),reproducer_sha256=q.sha(Path(__file__)),
        scope='Fixed source-selected configurations; original seed lists. All arms reported; no final-test promotion. Raw source search logs accompany canonical submission.'))
    if args.stage in ['main','all']:
        jobs=[dict(algorithm=alg,condition=c,seed=s,directory=str(directory/'main'/f'{c}_{alg}_{s}'))
              for c in 'ABC' for alg in ['Q-learning','PPO'] for s in [17,29,101,314,2718]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(worker,j) for j in jobs]):print('MAIN '+json.dumps(f.result()),flush=True)
        d=[]
        for j in jobs:
            if j['condition']=='B':
                for masked in [False,True]:d.append(dict(algorithm=j['algorithm'],seed=j['seed'],occluded=masked,
                    selection=str(Path(j['directory'])/'selected.json'),directory=str(directory/'robustness'/f'{j["algorithm"]}_{j["seed"]}_{masked}')))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(occlusion,j) for j in d]):print('D '+json.dumps(f.result()),flush=True)
    if args.stage in ['mechanisms','all']:
        launch('study.py','run','--run-id',prefix+'/mechanisms','--workers',args.workers)
        launch('diagnostic.py','run','--run-id',prefix+'/diagnostic','--study',prefix+'/mechanisms','--workers',args.workers)
    q.write_json(directory/'completed.json',dict(stage=args.stage,report_overwritten=False))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--run-id',required=True)
    cli.add_argument('--stage',choices=['main','mechanisms','all'],default='all');cli.add_argument('--workers',type=int,default=6)
    cli.add_argument('--plan',action='store_true');main(cli.parse_args())
