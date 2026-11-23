"""Prelock a paired ordinary-Q gamma confirmation; never selects a model."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import run_study_v2 as study
from candidates import TabularQ

SEEDS=list(range(1170000,1170200))

def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        path=study.ROOT/job['model'];assert study.sha(path)==job['model_sha256']
        directory=Path(job['directory']);done=directory/'metrics.json'
        if done.exists():return json.loads(done.read_text())
        directory.mkdir(parents=True,exist_ok=False)
        with path.open('rb') as handle:model=pickle.load(handle)
        assert type(model) is TabularQ and model.gamma==job['gamma'] and model.alpha==.15 and model.update_calls==262144
        result=dict(**job,**study.evaluate(model,'Q-learning',job['condition'],SEEDS,directory,'evaluation',trace=False))
        assert study.sha(path)==job['model_sha256']
        study.write(done,result);return result

def main(args):
    output=study.ROOT/'results/assignment4/three_hour_20261122/q_gamma_confirmation_v1'
    output.mkdir(parents=True,exist_ok=True);jobs=[]
    hashes=study.sources();hashes[str(Path(__file__).relative_to(study.ROOT))]=study.sha(__file__)
    for seed in [42,7]:
        for setting in ['A','B','D']:
            train='B' if setting=='D' else setting
            for baseline in [True,False]:
                if baseline:directory=study.ROOT/f'results/assignment4/part2_current_v3/final/{train}_Q-learning_{seed}'
                else:
                    stage='screen_v1' if seed==42 else 'replica_v2'
                    directory=study.ROOT/f'results/assignment4/three_hour_20261122/{stage}/{train}_Q-learning_gamma995_{seed}'
                path=directory/'checkpoint_262144.pkl';assert path.exists()
                jobs.append(dict(condition=setting,seed=seed,baseline=baseline,gamma=.99 if baseline else .995,
                    model=str(path.relative_to(study.ROOT)),model_sha256=study.sha(path),checkpoint_interactions=262144,
                    directory=str(output/f'{setting}_{"original" if baseline else "gamma995"}_{seed}')))
    lock=dict(jobs=jobs,evaluation_seeds=SEEDS,source_hashes=hashes,
        purpose='Paired frozen confirmation at the same 262144 budget and seeds42/7, differing only in Q gamma. No configuration/checkpoint selection from these A/B/D outcomes. This two-seed check is not the five-seed Part3 mechanism experiment.',
        timing='Locked while the independent fresh-seed main cohort trains, before any current-round D evaluation.')
    target=output/'locked.json'
    if target.exists():assert json.loads(target.read_text())==lock
    else:study.write(target,lock)
    if args.lock_only:print('Paired Q confirmation locked; no episodes evaluated.',flush=True);return
    for path,expected in hashes.items():assert study.sha(study.ROOT/path)==expected,path
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps({key:result[key] for key in ('condition','seed','baseline','success_rate','mean_score')}),flush=True)
    pd.DataFrame(results).to_csv(output/'comparison.csv',index=False)
    study.write(output/'completed.json',dict(episodes=sum(x['episodes'] for x in results),paired_models=12,
        native_scores_verified=True,frozen_parameters_verified=True,no_selection_from_confirmation=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-only',action='store_true');parser.add_argument('--workers',type=int,default=8)
    main(parser.parse_args())
