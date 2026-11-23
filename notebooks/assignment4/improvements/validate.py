"""Lock selected B models, then confirm on 200 unused episodes per model.

Does not select configurations, train, evaluate D or overwrite the v3 report.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys

import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp

SEEDS=tuple(range(860000,860200))


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory']);file=exp.ROOT/job['model']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==job['sha256']
    if (directory/'metrics.json').exists():return json.loads((directory/'metrics.json').read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    if job['algorithm']=='PPO':model=PPO.load(file,device='cpu')
    else:
        with file.open('rb') as handle:model=pickle.load(handle)
        assert model.update_calls==exp.BUDGET
    # Reuse the frozen evaluator with this explicit, separately locked seed list.
    # Its on-disk source and training sources are never edited.
    exp.DEV_SEEDS=SEEDS
    result=dict(**job,**exp.evaluate(model,job['algorithm'],'B',directory,baseline=job['baseline']))
    assert hashlib.sha256(file.read_bytes()).hexdigest()==job['sha256']
    (directory/'development.csv').replace(directory/'evaluation.csv')
    exp.write(directory/'metrics.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--algorithm',choices=['Q-learning','PPO'],required=True)
    parser.add_argument('--q-alpha',type=float,default=.3);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();output=exp.ROOT/'results/assignment4/improvement_confirmation_v1'
    output.mkdir(parents=True,exist_ok=True);jobs=[]
    for seed in (42,7):
        for baseline in (True,False):
            if baseline:
                source=exp.ROOT/'results/assignment4/part2_current_v3/final'/f'B_{args.algorithm}_{seed}'/f'checkpoint_262144{".zip" if args.algorithm=="PPO" else ".pkl"}'
            elif args.algorithm=='PPO':source=exp.ROOT/f'results/assignment4/improvements_v1/B_PPO_stable_{seed}/model.zip'
            else:source=exp.ROOT/f'results/assignment4/q_alpha_followup_v1/B_alpha_{args.q_alpha}_{seed}/model.pkl'
            assert source.is_file(),f'Complete both training seeds before locking: {source}'
            jobs.append(dict(algorithm=args.algorithm,seed=seed,baseline=baseline,
                directory=str(output/f'{args.algorithm}_{"baseline" if baseline else "revised"}_{seed}'),
                model=str(source.relative_to(exp.ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    hashes=exp.source_hashes();hashes[str(Path(__file__).relative_to(exp.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.algorithm=='Q-learning':
        path=HERE/'fast_q.py';hashes[str(path.relative_to(exp.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=dict(source_hashes=hashes,evaluation_seeds=list(SEEDS),jobs=jobs,
        policy='Greedy Q with seeded uniform ties; PPO highest probability; no updates.',
        history='Locked B-only confirmation after v3 inspection and development selection. Two training seeds, not full A-D or Part 3. Never tune on these results.')
    path=output/f'locked_{args.algorithm}.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Locked confirmation mismatch'
    else:exp.write(path,protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(output/f'{args.algorithm}_comparison.csv',index=False)
