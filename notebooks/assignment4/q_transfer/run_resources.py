"""Additional B-only task-resource candidate; no changes to registered drivers."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys

import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_study as study
from resources_q import ResourceQ


def worker(job):
    # Extend the explicit class registry in this worker process only.
    study.FACTORIES['resources']=ResourceQ
    return study.worker(job)


def main(args):
    output=study.ROOT/'results/assignment4/q_resources_v1';output.mkdir(parents=True,exist_ok=True)
    original=json.loads((study.ROOT/'results/assignment4/q_transfer_v1/protocol.json').read_text())
    hashes=dict(original['source_hashes'])
    for path in (Path(__file__),HERE/'resources_q.py'):hashes[str(path.relative_to(study.ROOT))]=study.study.sha(path)
    for name,expected in hashes.items():assert study.study.sha(study.ROOT/name)==expected,name
    protocol=dict(source_hashes=hashes,training_seeds=[42,7],budget=study.BUDGET,checkpoints=list(study.STOPS),
        development_seeds=list(study.DEV_SEEDS),alpha=.15,gamma=.99,epsilon='Original 1 to .05 over 80%',
        change='Keep original key and append last-life bit and score>=1500 bit from the existing common observation.',
        history='Additional B-only resource-information hypothesis before fresh D lock. This two-variable package is not a one-factor causal study. No D data select it.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        study.study.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((study.ROOT/name).read_bytes())
    jobs=[dict(variant='resources',seed=seed,baseline_hashes=original['baseline_model_hashes'][str(seed)],
        directory=str(output/f'B_resources_{seed}')) for seed in (42,7)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    rows=[]
    for result in results:
        for row in result['evaluations']:rows.append(dict(variant='resources',seed=result['seed'],**row))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    print(frame.groupby('checkpoint')[['success_rate','mean_score']].mean().to_string(),flush=True)
    study.study.write(output/'completed.json',dict(new_training_runs=2,development_episodes=200,D_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);main(parser.parse_args())
