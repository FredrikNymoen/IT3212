"""Second Q development screen after v1 state/credit packages failed.

Keeps original state, gamma and epsilon schedule. Only the update size changes.
One ordinary update per real transition. No replay/Double Q/eligibility traces.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp
from tabular_q import TabularQ


class RefinedQ(TabularQ):
    def __init__(self,seed,variant):
        super().__init__(seed,alpha=.05 if variant=='smooth' else .25,gamma=.99)
        self.variant=variant;self.visits={}

    def update(self,state,action,reward,next_state,terminal):
        if self.variant=='decay':
            anchor,key=state
            index=0 if action==0 else 1+(int(np.flatnonzero(self.ACTIONS==action)[0])-anchor)%4
            pair=(key,index);self.visits[pair]=self.visits.get(pair,0)+1
            self.alpha=max(.02,.25/np.sqrt(1+self.visits[pair]/10))
        super().update(state,action,reward,next_state,terminal)


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory'])
    if (directory/'metrics.json').exists():return json.loads((directory/'metrics.json').read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);exp.write(directory/'job.json',job)
    model=RefinedQ(job['seed'],job['variant']);env=exp.ImprovedTask(job['setting'])
    episode=0;obs,_=env.reset(seed=job['seed']*1000);records=[];start=perf_counter()
    try:
        for step in range(exp.BUDGET):
            state=model.state(obs);epsilon=max(.05,1-.95*step/(exp.BUDGET*.8))
            action=model.action(state,epsilon)
            nxt,reward,term,trunc,info=env.step(action)
            model.update(state,action,reward,model.state(nxt),term or trunc);obs=nxt
            if term or trunc:
                records.append(dict(interactions=step+1,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                episode+=1;obs,_=env.reset(seed=job['seed']*1000+episode)
        assert model.update_calls==exp.BUDGET and not hasattr(model,'other') and not hasattr(model,'buffer')
    finally:env.close()
    seconds=perf_counter()-start
    pd.DataFrame(records).to_csv(directory/'training.csv',index=False)
    file=directory/'model.pkl'
    with file.open('wb') as handle:pickle.dump(model,handle)
    checksum=hashlib.sha256(file.read_bytes()).hexdigest()
    metrics=dict(**job,**exp.evaluate(model,'Q-learning',job['setting'],directory),
        training_seconds=seconds,training_interactions=exp.BUDGET,model_sha256=checksum)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==checksum
    exp.write(directory/'metrics.json',metrics);return metrics


def main(args):
    output=exp.ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    hashes=exp.source_hashes();hashes[str(Path(__file__).relative_to(exp.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(source_hashes=hashes,budget=exp.BUDGET,development_seeds=list(exp.DEV_SEEDS),
        gamma=.99,epsilon='Original 1 to .05 over 80%',variants=['smooth','decay'],
        history='Additional Q-only development after both matched-v1 Q packages failed; unequal cumulative tuning effort is disclosed. No D/final selection.')
    file=output/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:
        exp.write(file,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((exp.ROOT/name).read_bytes())
    variants=['smooth','decay'] if args.variant=='all' else [args.variant]
    jobs=[dict(algorithm='Q-learning',setting=args.setting,seed=args.seed,variant=variant,
        directory=str(output/f'{args.setting}_{variant}_{args.seed}')) for variant in variants]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):print(json.dumps(future.result()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='q_refinement_v1')
    parser.add_argument('--variant',choices=['all','smooth','decay'],default='all')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--setting',choices=['A','B','C'],default='B')
    parser.add_argument('--workers',type=int,default=1);main(parser.parse_args())
