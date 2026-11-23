"""Separate B-only power-route information candidate before the fresh D lock."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_study as study
from power_route_q import PowerRouteQ


def worker(job):
    study.FACTORIES['power_route']=PowerRouteQ
    return study.worker(job)


def main(args):
    output=study.ROOT/'results/assignment4/q_power_route_v1';output.mkdir(parents=True,exist_ok=True)
    original=json.loads((study.ROOT/'results/assignment4/q_transfer_v1/protocol.json').read_text())
    hashes=dict(original['source_hashes'])
    for path in (Path(__file__),HERE/'power_route_q.py'):hashes[str(path.relative_to(study.ROOT))]=study.study.sha(path)
    for name,expected in hashes.items():assert study.study.sha(study.ROOT/name)==expected,name
    protocol=dict(source_hashes=hashes,training_seeds=[42,7],budget=study.BUDGET,checkpoints=list(study.STOPS),
        development_seeds=list(study.DEV_SEEDS),alpha=.15,gamma=.99,epsilon='Original 1 to .05 over 80%',
        change='Append a rotated best-power-route mask from the existing observation only when original danger is nonzero and power proxy is zero.',
        history='Additional B-only observation-retention hypothesis specified before new D lock. No new reward, mask, planning, teacher or D selection.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        study.study.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((study.ROOT/name).read_bytes())
    jobs=[dict(variant='power_route',seed=seed,baseline_hashes=original['baseline_model_hashes'][str(seed)],
        directory=str(output/f'B_power_route_{seed}')) for seed in (42,7)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    rows=[]
    for result in results:
        for row in result['evaluations']:rows.append(dict(variant='power_route',seed=result['seed'],**row))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    print(frame.groupby('checkpoint')[['success_rate','mean_score']].mean().to_string(),flush=True)
    study.study.write(output/'completed.json',dict(new_training_runs=2,development_episodes=200,D_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);main(parser.parse_args())
