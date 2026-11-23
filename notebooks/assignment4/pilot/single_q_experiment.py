"""Part 1-aligned Q pilot: one table and one update per real interaction.

Run from repository root. The historical Double Q files/results are preserved.
The corrected common Gymnasium observation and task are retained. Final models
are evaluated on a newly reserved seed set after the fixed training budget;
development curves never select another checkpoint for the final evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from navigation_features import RichTask
from tabular_q import TabularQ

DEVELOPMENT=tuple(range(730000,730020))
FINAL=tuple(range(740000,740200))


def evaluate(agent,seeds,directory,name):
    before=agent.fingerprint()
    training_rng=agent.rng
    env=RichTask();rows=[]
    try:
        for seed in seeds:
            obs,_=env.reset(seed=seed)
            agent.rng=np.random.default_rng(seed+2000000)
            while True:
                obs,_,term,trunc,info=env.step(agent.action(agent.state(obs)))
                if term or trunc:
                    rows.append(dict(seed=seed,score=info['score'],success=int(info['success']),
                        length=info['episode_steps'],reason=info['reason']))
                    break
    finally:env.close();agent.rng=training_rng
    assert agent.fingerprint()==before,'Evaluation changed the learned Q model'
    frame=pd.DataFrame(rows)
    assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all()
    frame.to_csv(directory/f'{name}.csv',index=False)
    return dict(episodes=len(rows),successes=int(frame.success.sum()),
        success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        mean_success_length=float(frame[frame.success==1].length.mean()),
        mean_failure_length=float(frame[frame.success==0].length.mean()),
        learning_disabled_verified=True)


def main(args):
    directory=ROOT/'results/assignment4/single_q_v1'/f'B_single_q_{args.seed}_{args.budget}'
    if (directory/'summary.json').exists():
        print('Completed ordinary-Q run already exists:',directory,flush=True);return
    if directory.exists():raise FileExistsError('Preserve incomplete evidence; use a different output version.')
    directory.mkdir(parents=True)
    sources=[Path(__file__),Path(__file__).with_name('tabular_q.py'),
        Path(__file__).with_name('navigation_features.py'),Path(__file__).with_name('maze_geometry.json'),
        HERE/'mspacman_env.py',ROOT/'reports/assignment4/Sections/01_predictions.tex']
    protocol=dict(algorithm='Ordinary single-table tabular Q-learning',training_seed=args.seed,
        training_interactions=args.budget,alpha=.15,gamma=.99,reward_scale=.01,
        updates_per_interaction=1,replay=False,double_q=False,eligibility_traces=False,
        epsilon='Linear 1 to .05 over first 80% of budget; greedy evaluation with seeded uniform ties',
        setting='B',goal=2000,horizon=1000,frameskip=4,sticky_probability=.25,
        observation='Common corrected 45-value observation, compact food-relative Q bins',
        development_seeds=list(DEVELOPMENT),final_evaluation_seeds=list(FINAL),
        final_checkpoint='Fixed full budget; no selection using final episode seeds',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitation='B-only rerun; two training seeds; inherited power signal is capped, not a verified precise countdown.')
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=directory/'sources';snapshot.mkdir()
    for source in sources:(snapshot/source.name).write_bytes(source.read_bytes())
    agent=TabularQ(args.seed)
    env=RichTask();episode=0;steps=0;training_seconds=0.;curves=[];episodes=[]
    obs,_=env.reset(seed=args.seed*1000)
    try:
        for stop in range(65536,args.budget+1,65536):
            started=perf_counter()
            while steps<stop:
                state=agent.state(obs)
                epsilon=max(.05,1-.95*steps/(args.budget*.8))
                action=agent.action(state,epsilon)
                next_obs,reward,term,trunc,info=env.step(action)
                agent.update(state,action,reward,agent.state(next_obs),term or trunc)
                steps+=1;obs=next_obs
                if term or trunc:
                    episodes.append(dict(interactions=steps,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;obs,_=env.reset(seed=args.seed*1000+episode)
            training_seconds+=perf_counter()-started
            assert agent.update_calls==steps,'Must have exactly one update per environment transition'
            assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
            with (directory/f'checkpoint_{stop}.pkl').open('wb') as handle:pickle.dump(agent,handle)
            curve=evaluate(agent,DEVELOPMENT,directory,f'development_{stop}')
            curve.update(interactions=stop,states=len(agent.table),training_seconds=training_seconds,
                update_calls=agent.update_calls)
            curves.append(curve);print(json.dumps(curve),flush=True)
            pd.DataFrame(curves).to_csv(directory/'progress.csv',index=False)
            pd.DataFrame(episodes).to_csv(directory/'training.csv',index=False)
        path=directory/f'checkpoint_{args.budget}.pkl'
        fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
        final_protocol=dict(model_sha256=fingerprint,seeds=list(FINAL),episodes=200,
            training_interactions=args.budget,training_updates=agent.update_calls,
            model_frozen_before_final_evaluation=True)
        (directory/'final_protocol.json').write_text(json.dumps(final_protocol,indent=2))
        result=evaluate(agent,FINAL,directory,'final_episodes')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==fingerprint
        result.update(algorithm=protocol['algorithm'],training_seed=args.seed,
            training_interactions=args.budget,training_updates=agent.update_calls,
            states=len(agent.table),training_seconds=training_seconds,model_sha256=fingerprint)
        (directory/'summary.json').write_text(json.dumps(result,indent=2))
        print('FINAL '+json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--budget',type=int,default=262144)
    args=parser.parse_args()
    if args.budget<=0 or args.budget%65536:raise ValueError('Budget must be a positive multiple of 65536.')
    with threadpool_limits(limits=1):main(args)
