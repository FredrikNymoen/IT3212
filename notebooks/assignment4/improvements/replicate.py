"""Replicate one declared development candidate without retraining rejected ones."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--algorithm',choices=['Q-learning','PPO'],default='PPO')
    parser.add_argument('--variant',default='stable');parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--setting',choices=['A','B','C'],default='B')
    args=parser.parse_args()
    assert (args.algorithm=='PPO' and args.variant in ['motion','stable']) or (args.algorithm=='Q-learning' and args.variant in ['credit','context'])
    output=exp.ROOT/'results/assignment4/improvements_v1'
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['source_hashes']==exp.source_hashes(),'Registered training sources changed'
    exp.write(output/'replication_runner.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Replication of a declared candidate on a second seed or setting; all evaluations remain development.'))
    job=dict(algorithm=args.algorithm,variant=args.variant,seed=args.seed,setting=args.setting,
        directory=str(output/f'{args.setting}_{args.algorithm}_{args.variant}_{args.seed}'))
    print(json.dumps(exp.worker(job)),flush=True)
