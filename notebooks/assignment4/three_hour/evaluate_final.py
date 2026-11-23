"""Lock source-selected checkpoints, then evaluate all A--D models frozen."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_study_v2 as study
from candidates import StudyTask,TabularQ,ppo_settings
from additions import LowBitsTask,AlignmentQ

SEEDS=tuple(range(1150000,1150200))

def work(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for path,expected in job['source_hashes'].items():assert study.sha(study.ROOT/path)==expected,path
        directory=Path(job['directory']);done=directory/'metrics.json'
        if done.exists():return json.loads(done.read_text())
        if directory.exists():raise FileExistsError(f'Partial evaluation preserved: {directory}')
        directory.mkdir(parents=True);study.write(directory/'job.json',job)
        file=study.ROOT/job['model'];assert study.sha(file)==job['model_sha256']
        study.StudyTask=LowBitsTask if job['variant']=='powerclock' else StudyTask
        if job['algorithm']=='Q-learning':
            with file.open('rb') as handle:model=pickle.load(handle)
            assert type(model).update is TabularQ.update and model.update_calls==job['checkpoint_interactions']
            assert model.alpha==.15 and model.gamma==({'gamma995':.995,'gamma999':.999}.get(job['variant'],.99))
            assert not hasattr(model,'other') and not hasattr(model,'buffer')
        else:
            model=PPO.load(file,device='cpu')
            assert model.num_timesteps==job['checkpoint_interactions']
            assert model.policy.actor_variant==job['variant']
        result=dict(**job,**study.evaluate(model,job['algorithm'],job['condition'],SEEDS,directory,'evaluation',trace=True))
        assert study.sha(file)==job['model_sha256']
        study.write(done,result);return result


def main(args):
    training=study.ROOT/'results/assignment4'/args.training_run
    assert (training/'completed.json').exists(),'Complete every registered training job before final locking.'
    protocol=json.loads((training/'protocol.json').read_text());hashes=protocol['source_hashes']
    for path,expected in hashes.items():assert study.sha(study.ROOT/path)==expected,path
    curves=pd.read_csv(training/'comparison.csv')
    assert set(curves.setting)==set('ABC') and set(curves.algorithm)=={'Q-learning','PPO'}
    chosen={}
    for algorithm in ['Q-learning','PPO']:
        source=curves[(curves.algorithm==algorithm)&(curves.interactions>0)]
        chosen[algorithm]={}
        for seed in sorted(source.seed.unique()):
            for setting in 'ABC':
                if args.stopping_rule=='per_model':used=source[(source.seed==seed)&(source.setting==setting)]
                elif args.stopping_rule=='per_seed':used=source[(source.seed==seed)&source.setting.isin(['A','B'])]
                else:used=source[source.setting.isin(['A','B'])]
                grouped=used.groupby('interactions')[['success_rate','mean_score']].mean().reset_index()
                selected=grouped.sort_values(['success_rate','mean_score','interactions'],ascending=[False,False,True]).iloc[0]
                chosen[algorithm][f'{int(seed)}_{setting}']=int(selected.interactions)
    output=study.ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    hashes=dict(hashes);hashes[str(Path(__file__).relative_to(study.ROOT))]=study.sha(__file__)
    jobs=[]
    for job in protocol['jobs']:
        source=Path(job['directory']);metrics=json.loads((source/'metrics.json').read_text())
        point=chosen[job['algorithm']][f'{job["seed"]}_{job["setting"]}']
        row=next(row for row in metrics['checkpoints'] if row['interactions']==point)
        assert study.sha(study.ROOT/row['model'])==row['model_sha256']
        for condition in [job['setting']]+(['D'] if job['setting']=='B' else []):
            jobs.append(dict(algorithm=job['algorithm'],variant=job['variant'],seed=job['seed'],condition=condition,
                training_setting=job['setting'],training_interactions=job['budget'],checkpoint_interactions=point,
                training_seconds=metrics['training_seconds'] if condition!='D' else 0.,
                model=row['model'],model_sha256=row['model_sha256'],source_hashes=hashes,
                directory=str(output/f'{condition}_{job["algorithm"]}_{job["seed"]}')))
    lock=dict(training_run=args.training_run,source_hashes=hashes,jobs=jobs,evaluation_seeds=list(SEEDS),
        checkpoints=chosen,stopping_rule=args.stopping_rule,
        selection='Per-model: one checkpoint per trained A/B/C model using only its source-condition development success, then game score, then earlier point. The identical rule/checkpoint allowance applies to both algorithms and every condition; variants/hyperparameters are locked from A/B before main training. D reuses B checkpoint. Every run consumes the full common cap; no held-out final or D outcome selects anything.',
        development_export_sha256=study.sha(training/'comparison.csv'),
        scope='Further post-inspection development; previously inspected transfer map is not a new independent domain.')
    path=output/'locked.json'
    if path.exists():assert json.loads(path.read_text())==lock,'Final lock changed; preserve it.'
    else:study.write(path,lock)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work,job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps({key:result[key] for key in ('algorithm','variant','seed','condition','success_rate','mean_score','evaluation_seconds')}),flush=True)
    pd.DataFrame([{key:value for key,value in result.items() if key!='source_hashes'} for result in results]).to_csv(output/'comparison.csv',index=False)
    study.write(output/'completed.json',dict(conditions=len(results),episodes=sum(result['episodes'] for result in results),
        native_scores_verified=True,frozen_parameters_verified=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--stopping-rule',choices=['per_model','per_seed','common'],default='per_model')
    main(parser.parse_args())
