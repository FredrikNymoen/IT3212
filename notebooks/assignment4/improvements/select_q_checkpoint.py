"""Development-selected Q stopping point; locked confirmation, no new training.

Selection uses the old *development* curves, never v3 final episode outcomes.
The complete original training run consumed 262,144 decisions. The selected
checkpoint contains only the updates through its recorded earlier step.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return evaluate(job)


def evaluate(job):
    directory=Path(job['directory']);file=exp.ROOT/job['model']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==job['sha256']
    if (directory/'metrics.json').exists():return json.loads((directory/'metrics.json').read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    with file.open('rb') as handle:model=pickle.load(handle)
    assert model.update_calls==job['checkpoint'] and not hasattr(model,'other') and not hasattr(model,'buffer')
    exp.DEV_SEEDS=tuple(range(860000,860200))
    result=dict(**job,**exp.evaluate(model,'Q-learning','B',directory))
    (directory/'development.csv').replace(directory/'evaluation.csv')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==job['sha256']
    exp.write(directory/'metrics.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    output=exp.ROOT/'results/assignment4/q_checkpoint_confirmation_v1';output.mkdir(parents=True,exist_ok=True)
    frames=[];data_hashes={}
    for seed in (42,7):
        path=exp.ROOT/f'results/assignment4/part2_current_v3/final/B_Q-learning_{seed}/curves.csv'
        frame=pd.read_csv(path);frame['seed']=seed;frames.append(frame)
        data_hashes[str(path.relative_to(exp.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    curve=pd.concat(frames);curve=curve[curve.interactions>0]
    grouped=curve.groupby('interactions')[['success_rate','mean_score']].mean().reset_index()
    selected=int(grouped.sort_values(['success_rate','mean_score','interactions'],ascending=[False,False,True]).iloc[0].interactions)
    jobs=[]
    for seed in (42,7):
        for baseline,checkpoint in [(True,262144),(False,selected)]:
            source=exp.ROOT/f'results/assignment4/part2_current_v3/final/B_Q-learning_{seed}/checkpoint_{checkpoint}.pkl'
            jobs.append(dict(seed=seed,baseline=baseline,checkpoint=checkpoint,model=str(source.relative_to(exp.ROOT)),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                directory=str(output/f'{"baseline" if baseline else "selected"}_{seed}')))
    sources=exp.source_hashes();sources[str(Path(__file__).relative_to(exp.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(source_hashes=sources,development_curve_hashes=data_hashes,selected_checkpoint=selected,
        training_run_interactions=262144,checkpoint_updates=selected,evaluation_seeds=list(range(860000,860200)),jobs=jobs,
        selection='Mean 20-episode development success across both training seeds, then mean game score, then earliest checkpoint.',
        history='Post-v3 development stopping-rule revision after failed Q packages; cumulative tuning was unequal. No new training, no D, and no final outcomes select the checkpoint.')
    path=output/'locked.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:exp.write(path,protocol)
    grouped.to_csv(output/'development_selection.csv',index=False)
    print(f'LOCKED Q CHECKPOINT {selected}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(output/'comparison.csv',index=False)
