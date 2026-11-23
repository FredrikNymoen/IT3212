"""Matched A/B pilot: original Q state versus one additional closing bit.

Run from repository root. D is never used in the development stage.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from motion_q import MotionQ,TabularQ,ImprovedTask,closing_bit

BUDGET=262144
CHECKPOINTS=(196608,262144)
TRAIN_SEEDS=(42,7)
DEVELOPMENT_SEEDS=tuple(range(890000,890050))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    assignment=HERE.parent
    paths=[Path(__file__),HERE/'motion_q.py',assignment/'improvements/agents.py',
        assignment/'current/environment.py',assignment/'mspacman_env.py',
        *[assignment/'pilot'/name for name in ('tabular_q.py','navigation_features.py','maze_geometry.json')],
        ROOT/'IT3212 - Assignment 4.pdf',ROOT/'reports/assignment4/Sections/01_predictions.tex']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def evaluate(agent,setting,seeds,directory,name):
    before=agent.fingerprint();rng=agent.rng
    env=ImprovedTask(setting);rows=[];decisions=0;unknown=0;flagged=0;keys=set();start=perf_counter()
    try:
        for episode_seed in seeds:
            obs,_=env.reset(seed=episode_seed);agent.rng=np.random.default_rng(episode_seed+3000000)
            for _ in range(1000):
                state=agent.state(obs);unknown+=state[1] not in agent.table
                keys.add(state[1]);flagged+=closing_bit(obs);decisions+=1
                action=agent.action(state)
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:
                    rows.append(dict(evaluation_seed=episode_seed,success=int(info['success']),
                        score=info['score'],length=info['episode_steps'],reason=info['reason']))
                    break
    finally:env.close();agent.rng=rng
    assert agent.fingerprint()==before
    frame=pd.DataFrame(rows)
    assert len(frame)==len(seeds) and np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
    frame.to_csv(directory/f'{name}.csv',index=False)
    return dict(success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        mean_length=float(frame.length.mean()),episodes=len(frame),evaluation_seconds=perf_counter()-start,
        evaluation_decisions=decisions,unseen_state_fraction=unknown/decisions,
        closing_flag_fraction=flagged/decisions,unique_evaluation_states=len(keys),table_states=len(agent.table),
        frozen_parameters_verified=True)


def train(job,directory):
    agent=MotionQ(job['seed']);env=ImprovedTask(job['setting'])
    obs,_=env.reset(seed=job['seed']*1000);episode=0;steps=0;records=[];seconds=0.;training_closing=0
    try:
        for stop in CHECKPOINTS:
            start=perf_counter()
            while steps<stop:
                state=agent.state(obs);training_closing+=closing_bit(obs)
                epsilon=max(.05,1-.95*steps/(BUDGET*.8))
                action=agent.action(state,epsilon)
                nxt,reward,term,trunc,info=env.step(action)
                agent.update(state,action,reward,agent.state(nxt),term or trunc)
                steps+=1;obs=nxt
                if term or trunc:
                    records.append(dict(interactions=steps,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;obs,_=env.reset(seed=job['seed']*1000+episode)
            seconds+=perf_counter()-start
            assert agent.update_calls==stop and agent.alpha==.15 and agent.gamma==.99
            assert type(agent) is MotionQ and not hasattr(agent,'other') and not hasattr(agent,'buffer')
            with (directory/f'checkpoint_{stop}.pkl').open('wb') as handle:pickle.dump(agent,handle)
    finally:env.close()
    pd.DataFrame(records).to_csv(directory/'training.csv',index=False)
    return dict(training_interactions=steps,training_seconds=seconds,training_closing_fraction=training_closing/steps)


def worker(job):
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(f'Partial run preserved: {directory}')
    directory.mkdir(parents=True);write(directory/'job.json',job)
    if job['variant']=='motion':training=train(job,directory)
    else:training=dict(training_interactions=BUDGET,training_seconds=None,reused_original_training=True)
    rows=[]
    for stop in CHECKPOINTS:
        file=(directory/f'checkpoint_{stop}.pkl' if job['variant']=='motion' else
            ROOT/f'results/assignment4/part2_current_v3/final/{job["setting"]}_Q-learning_{job["seed"]}/checkpoint_{stop}.pkl')
        checksum=sha(file)
        if job['variant']=='baseline':assert checksum==job['baseline_hashes'][str(stop)]
        with file.open('rb') as handle:agent=pickle.load(handle)
        assert agent.update_calls==stop and agent.alpha==.15 and agent.gamma==.99
        assert type(agent) is (MotionQ if job['variant']=='motion' else TabularQ)
        rows.append(dict(checkpoint=stop,model=str(file.relative_to(ROOT)),model_sha256=checksum,
            **evaluate(agent,job['setting'],DEVELOPMENT_SEEDS,directory,f'development_{stop}')))
        assert sha(file)==checksum
    result=dict(**job,**training,evaluations=rows)
    write(done,result);return result


def summarize(output):
    rows=[]
    for setting in 'AB':
        for variant in ('baseline','motion'):
            for seed in TRAIN_SEEDS:
                result=json.loads((output/f'{setting}_{variant}_{seed}/metrics.json').read_text())
                for metric in result['evaluations']:
                    rows.append(dict(setting=setting,variant=variant,seed=seed,**metric))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    means=frame.groupby(['setting','checkpoint','variant'])[['success_rate','mean_score','table_states','unseen_state_fraction']].mean().reset_index()
    means.to_csv(output/'means.csv',index=False)
    print(means.to_string(index=False),flush=True)
    # No automatic promotion: this is a two-seed, 50-episode development pilot.
    write(output/'completed.json',dict(training_runs=4,reused_baseline_runs=4,development_episodes=800,
        heldout_D_used=False,primary_checkpoint=BUDGET,secondary_checkpoint=196608,
        scope='One-factor A/B development pilot, not five-seed Part 3 evidence or a new full report.'))


def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    baseline_hashes={}
    for setting in 'AB':
        for seed in TRAIN_SEEDS:
            baseline_hashes[f'{setting}_{seed}']={str(stop):sha(ROOT/f'results/assignment4/part2_current_v3/final/{setting}_Q-learning_{seed}/checkpoint_{stop}.pkl') for stop in CHECKPOINTS}
    old=json.loads((ROOT/'results/assignment4/part2_current_v3/protocol.json').read_text())
    for name,expected in old['source_hashes'].items():assert sha(ROOT/name)==expected,name
    protocol=dict(source_hashes=sources(),baseline_model_hashes=baseline_hashes,
        settings=['A','B'],training_seeds=list(TRAIN_SEEDS),training_budget=BUDGET,
        checkpoints=list(CHECKPOINTS),primary_checkpoint=BUDGET,secondary_checkpoint=196608,
        development_seeds=list(DEVELOPMENT_SEEDS),alpha=.15,gamma=.99,
        epsilon='1 to .05 over first 80% of 262144, then .05',
        sole_change='One appended bit: unpowered nearest ghost within 48 pixels and observed relative closing speed > 0.8 pixels per decision.',
        policy='Greedy Q with seeded uniform ties, read-only lookup, no updates.',
        history='New one-factor development study after prior multi-factor Q failures and v3 final inspection. No D selection, no old final episodes used for tuning.',
        limitations=['Two seeds and 50 development episodes per checkpoint; not Part 3 five-seed evidence.',
            'Power remains a capped proxy; no precise-timer claim.','Nearest ghost uses inherited Manhattan proximity; walls can affect relevance.'])
    file=output/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Sources/config changed: use a new run ID'
    else:
        write(file,protocol)
        for name in protocol['source_hashes']:
            destination=output/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/name).read_bytes())
    jobs=[dict(setting=setting,variant=variant,seed=seed,
        baseline_hashes=baseline_hashes[f'{setting}_{seed}'],directory=str(output/f'{setting}_{variant}_{seed}'))
        for setting in 'AB' for seed in TRAIN_SEEDS for variant in ('baseline','motion')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result()
            print(json.dumps(dict(setting=result['setting'],variant=result['variant'],seed=result['seed'],
                evaluations=result['evaluations'])),flush=True)
    summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='q_motion_v1');main(parser.parse_args())
