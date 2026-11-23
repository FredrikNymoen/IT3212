"""Ordinary Q alpha follow-up: original state/gamma/exploration, larger updates."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment as exp
from tabular_q import TabularQ


def run(args):
    torch.set_num_threads(1)
    output=exp.ROOT/'results/assignment4/q_alpha_followup_v1'
    output.mkdir(parents=True,exist_ok=True)
    hashes=exp.source_hashes();hashes[str(Path(__file__).relative_to(exp.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(source_hashes=hashes,budget=exp.BUDGET,development_seeds=list(exp.DEV_SEEDS),
        gamma=.99,epsilon='Original 1 to .05 over 80%',state='Unchanged original compact state',
        history='Additional Q-only development after lower-alpha and state/credit packages failed; cumulative tuning is unequal. No D or final selection.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        exp.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((exp.ROOT/name).read_bytes())
    directory=output/f'{args.setting}_alpha_{args.alpha}_{args.seed}'
    if (directory/'metrics.json').exists():print((directory/'metrics.json').read_text());return
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir();exp.write(directory/'job.json',vars(args))
    model=TabularQ(args.seed,alpha=args.alpha,gamma=.99);env=exp.ImprovedTask(args.setting)
    episode=0;obs,_=env.reset(seed=args.seed*1000);records=[];start=perf_counter()
    try:
        for step in range(exp.BUDGET):
            state=model.state(obs);epsilon=max(.05,1-.95*step/(exp.BUDGET*.8))
            action=model.action(state,epsilon)
            nxt,reward,term,trunc,info=env.step(action)
            model.update(state,action,reward,model.state(nxt),term or trunc);obs=nxt
            if term or trunc:
                records.append(dict(interactions=step+1,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                episode+=1;obs,_=env.reset(seed=args.seed*1000+episode)
        assert model.update_calls==exp.BUDGET and not hasattr(model,'other') and not hasattr(model,'buffer')
    finally:env.close()
    seconds=perf_counter()-start
    pd.DataFrame(records).to_csv(directory/'training.csv',index=False)
    file=directory/'model.pkl'
    with file.open('wb') as handle:pickle.dump(model,handle)
    checksum=hashlib.sha256(file.read_bytes()).hexdigest()
    metrics=dict(setting=args.setting,algorithm='Q-learning',alpha=args.alpha,seed=args.seed,
        **exp.evaluate(model,'Q-learning',args.setting,directory),training_seconds=seconds,
        training_interactions=exp.BUDGET,model_sha256=checksum)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==checksum
    exp.write(directory/'metrics.json',metrics);print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--alpha',type=float,default=.3)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--setting',choices=['A','B','C'],default='B')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args)
