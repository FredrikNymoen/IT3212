"""Preplanned source-only selection/confirmation of fixed operating policies.

Keep this n=1 policy check separate from the five-seed training benchmark.
D inherits the chosen B agent; no target-map outcomes select a policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import run_study_v2 as study
from candidates import StudyTask,TabularQ
from additions import LowBitsTask

ROOT=study.ROOT;BASE=ROOT/'results/assignment4/three_hour_20261122'
OUT=BASE/'operating_v1';DEV=list(range(1140000,1140050));FINAL=list(range(1190000,1190200))

def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory=Path(job['directory']);done=directory/'metrics.json'
        if done.exists():return json.loads(done.read_text())
        directory.mkdir(parents=True,exist_ok=False)
        path=ROOT/job['model'];assert study.sha(path)==job['model_sha256']
        study.StudyTask=LowBitsTask if job['variant']=='powerclock' else StudyTask
        if job['algorithm']=='Q-learning':
            with path.open('rb') as handle:model=pickle.load(handle)
            assert type(model).update is TabularQ.update and model.update_calls==job['checkpoint_interactions']
        else:model=PPO.load(path,device='cpu')
        seeds=DEV if job['phase']=='development' else FINAL
        result=dict(**job,**study.evaluate(model,job['algorithm'],job['condition'],seeds,directory,
            'development' if job['phase']=='development' else 'evaluation',trace=job['phase']=='final'))
        assert study.sha(path)==job['model_sha256'];study.write(done,result);return result

def run_jobs(jobs,workers):
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps({key:result[key] for key in ('phase','condition','algorithm','seed','success_rate','mean_score')}),flush=True)
    return results

def main(args):
    OUT.mkdir(parents=True,exist_ok=True)
    main_dir=BASE/'main_v1';protocol=json.loads((main_dir/'protocol.json').read_text())
    hashes=dict(protocol['source_hashes']);hashes[str(Path(__file__).relative_to(ROOT))]=study.sha(__file__)
    plan=dict(main_protocol_sha256=study.sha(main_dir/'protocol.json'),source_hashes=hashes,
        candidates='Only already selected gamma995/stable families; source seeds42/7 plus all five fresh main seeds for A/B, main seeds for C. Each model first uses its own predeclared source checkpoint rule.',
        development_seeds=DEV,final_seeds=FINAL,
        selection='Re-evaluate every candidate on the same50 source-condition development episodes; choose success, then game score, earlier checkpoint, lower seed. One fixed policy per algorithm/condition. D reuses selected B model unchanged.',
        scope='n=1 fixed-policy confirmation, not training reliability; all five training seeds stay in the primary table. No primary-final or D scores enter this selection.',
        timing='Plan locked during main training before any current-round D evaluation.')
    target=OUT/'plan.json'
    if target.exists():assert json.loads(target.read_text())==plan,'Preserve the pre-evaluation plan.'
    else:study.write(target,plan)
    if args.plan_only:print('Fixed-policy selection plan locked; no episodes evaluated.',flush=True);return
    assert (main_dir/'completed.json').exists()
    for path,expected in hashes.items():assert study.sha(ROOT/path)==expected,path
    source_jobs=list(protocol['jobs'])
    for seed,stage in [(42,'screen_v1'),(7,'replica_v2')]:
        for setting in 'AB':
            for algorithm,variant in protocol['selected_variants'].items():
                folder=BASE/stage/f'{setting}_{algorithm}_{variant}_{seed}'
                source_jobs.append(json.loads((folder/'job.json').read_text()))
    jobs=[]
    for source in source_jobs:
        metrics=json.loads((Path(source['directory'])/'metrics.json').read_text())
        points=[row for row in metrics['checkpoints'] if row['interactions']>0]
        row=sorted(points,key=lambda x:(-x['success_rate'],-x['mean_score'],x['interactions']))[0]
        jobs.append(dict(algorithm=source['algorithm'],variant=source['variant'],seed=source['seed'],
            condition=source['setting'],training_setting=source['setting'],training_interactions=source['budget'],
            checkpoint_interactions=row['interactions'],model=row['model'],model_sha256=row['model_sha256'],
            phase='development',directory=str(OUT/f'dev_{source["setting"]}_{source["algorithm"]}_{source["seed"]}')))
    assert len(jobs)==38
    development_lock=dict(jobs=jobs,source_hashes=hashes,evaluation_seeds=DEV,plan_sha256=study.sha(target))
    dev_path=OUT/'development_locked.json'
    if dev_path.exists():assert json.loads(dev_path.read_text())==development_lock
    else:study.write(dev_path,development_lock)
    development=run_jobs(jobs,args.workers);frame=pd.DataFrame(development);frame.to_csv(OUT/'development_comparison.csv',index=False)
    final_jobs=[]
    for setting in 'ABC':
        for algorithm in ['Q-learning','PPO']:
            chosen=frame[(frame.condition==setting)&(frame.algorithm==algorithm)].sort_values(
                ['success_rate','mean_score','checkpoint_interactions','seed'],ascending=[False,False,True,True]).iloc[0]
            base={key:chosen[key] for key in ['algorithm','variant','seed','condition','training_setting','training_interactions',
                'checkpoint_interactions','model','model_sha256']}
            for key in ['seed','training_interactions','checkpoint_interactions']:base[key]=int(base[key])
            for condition in [setting]+(['D'] if setting=='B' else []):
                final_jobs.append(dict(base,condition=condition,phase='final',
                    directory=str(OUT/f'final_{condition}_{algorithm}')))
    lock=dict(jobs=final_jobs,source_hashes=hashes,evaluation_seeds=FINAL,
        selection='Preplanned source-only fixed-agent selection; D inherits B; n=1 selected trained policy per condition.',
        development_sha256=study.sha(OUT/'development_comparison.csv'),plan_sha256=study.sha(target))
    final_lock=OUT/'locked.json'
    if final_lock.exists():assert json.loads(final_lock.read_text())==lock
    else:study.write(final_lock,lock)
    final=run_jobs(final_jobs,args.workers);pd.DataFrame(final).to_csv(OUT/'comparison.csv',index=False)
    study.write(OUT/'completed.json',dict(conditions=8,episodes=1600,source_only_selection=True,native_scores_verified=True,
        frozen_parameters_verified=True,all_primary_training_seeds_retained=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');parser.add_argument('--workers',type=int,default=8)
    main(parser.parse_args())
