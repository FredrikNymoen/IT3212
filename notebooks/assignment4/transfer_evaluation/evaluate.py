"""Lock all ten original B models and evaluate D with explicit censoring.

No checkpoint/model selection and no training. Success and known failure are
disjoint from map-exit censoring. Bounds for the original task are [S/N,(S+C)/N].
The prefix-return statistic stops at the last verified in-map observation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
from boundary_env import BoundaryTask
import run_study_v2 as study
from candidates import FrozenStudy,TabularQ

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SEEDS=list(range(1250000,1250200))

def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return evaluate(job)

def evaluate(job):
    for name,expected in job['source_hashes'].items():assert study.sha(ROOT/name)==expected,name
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(f'Preserved partial case: {directory}')
    directory.mkdir(parents=True);study.write(directory/'job.json',job)
    path=ROOT/job['model'];assert study.sha(path)==job['model_sha256']
    algorithm=job['algorithm']
    if algorithm=='Q-learning':
        with path.open('rb') as handle:model=pickle.load(handle)
        assert type(model).update is TabularQ.update and model.update_calls==job['checkpoint_interactions']
        assert model.alpha==.15 and model.gamma==.995
        actor=None;saved_rng=model.rng
    else:
        model=PPO.load(path,device='cpu');assert model.num_timesteps==job['checkpoint_interactions']
        actor=FrozenStudy(model)
    before=study.fingerprint(model,algorithm);env=BoundaryTask('D');rows=[];unknown=0;decisions=0;parity=0.
    trace={name:[] for name in ['ram','ram_after','observation','values','action','episode','step',
        'raw_reward','reward','score_after','terminated','truncated','censored']}
    start=perf_counter()
    try:
        assert abs(env.unwrapped.ale.getFloat('repeat_action_probability')-.25)<1e-8
        for index,seed in enumerate(SEEDS):
            obs,_=env.reset(seed=seed);initial=study.native_score(env.unwrapped.ale.getRAM())
            if actor is None:model.rng=np.random.default_rng(seed+3000000)
            for step in range(1000):
                ram=env.unwrapped.ale.getRAM().copy();assert int(ram[0])==1
                if actor is None:
                    state=model.state(obs);values=model.values(state);action=model.action(state)
                    unknown+=int(state[1] not in model.table)
                else:
                    values=actor.probabilities(obs);action=int(values.argmax())
                    if index==0 and step<10:
                        with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(obs[None])).distribution.probs.numpy()[0]
                        assert np.allclose(actual,values,atol=1e-5) and actual.argmax()==action
                        parity=max(parity,float(abs(actual-values).max()))
                nxt,reward,term,trunc,info=env.step(action);after=env.unwrapped.ale.getRAM().copy()
                assert study.native_score(after)-initial==info['score']
                assert reward==info['raw_reward']
                for key,value in [('ram',ram),('ram_after',after),('observation',obs.copy()),('values',values.copy()),
                    ('action',action),('episode',index),('step',step),('raw_reward',info['raw_reward']),('reward',reward),
                    ('score_after',info['score']),('terminated',term),('truncated',trunc),('censored',info['censored'])]:
                    trace[key].append(value)
                decisions+=1;obs=nxt
                if term or trunc:
                    rows.append(dict(evaluation_seed=seed,success=int(info['success']),known_failure=int(info['known_failure']),
                        censored=int(info['censored']),map_exit=int(info['map_exit']),length=info['episode_steps'],
                        native_score=info['score'],observed_return=info['observed_prefix_return'],
                        observed_score=info['observed_prefix_score'],reason=info['reason']))
                    break
            else:raise AssertionError('Unstopped episode')
    finally:
        env.close()
        if actor is None:model.rng=saved_rng
    seconds=perf_counter()-start
    assert study.fingerprint(model,algorithm)==before and study.sha(path)==job['model_sha256']
    frame=pd.DataFrame(rows);assert len(frame)==200 and (frame[['success','known_failure','censored']].sum(axis=1)==1).all()
    length=lambda key:float(frame.loc[frame[key].eq(1),'length'].mean()) if frame[key].any() else None
    result=dict(job,episodes=200,observed_success_rate=float(frame.success.mean()),
        success_lower_bound=float(frame.success.mean()),success_upper_bound=float((frame.success+frame.censored).mean()),
        known_failure_rate=float(frame.known_failure.mean()),censored_rate=float(frame.censored.mean()),
        map_exits=int(frame.map_exit.sum()),mean_return=float(frame.observed_return.mean()),mean_score=float(frame.observed_score.mean()),
        success_length=length('success'),failure_length=length('known_failure'),censored_length=length('censored'),
        evaluation_seconds=seconds,evaluation_interactions=decisions,training_seconds=0.,further_training_interactions=0,
        frozen_parameters_verified=True,native_scores_verified_every_step=True,no_actions_after_map_exit=True,
        unseen_state_fraction=unknown/decisions if actor is None else None,torch_numpy_max_error=parity)
    frame.to_csv(directory/'evaluation.csv',index=False)
    np.savez_compressed(directory/'traces.npz',**{name:np.asarray(values) for name,values in trace.items()})
    study.write(done,result);return result

def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    original=ROOT/'results/assignment4/three_hour_20261122/final_v1/locked.json'
    old=json.loads(original.read_text());hashes=dict(old['source_hashes'])
    for name in ['boundary_env.py','evaluate.py']:
        file=HERE/name;hashes[str(file.relative_to(ROOT))]=study.sha(file)
    engineering=ROOT/'results/assignment4/transfer_boundary_v1/engineering.json'
    check=json.loads(engineering.read_text())
    for name,expected in check['sources'].items():assert study.sha(ROOT/name)==expected
    jobs=[]
    for oldjob in old['jobs']:
        if oldjob['condition']!='B':continue
        jobs.append(dict(algorithm=oldjob['algorithm'],seed=oldjob['seed'],condition='D',model=oldjob['model'],
            model_sha256=oldjob['model_sha256'],checkpoint_interactions=oldjob['checkpoint_interactions'],
            original_training_interactions=oldjob['training_interactions'],source_hashes=hashes,
            directory=str(output/f'D_{oldjob["algorithm"]}_{oldjob["seed"]}')))
    assert len(jobs)==10 and {job['seed'] for job in jobs}=={17,29,101,314,2718}
    protocol=dict(jobs=jobs,evaluation_seeds=SEEDS,source_hashes=hashes,original_selection_lock_sha256=study.sha(original),
        engineering_sha256=study.sha(engineering),workers=args.workers,training_interactions=0,
        policy='Original greedy Q with seeded ties; PPO argmax. All ten original B models, no reselection.',
        boundary='Any map exit is external truncation/censoring, not agent failure; no action on another map. Crossing reward excluded from observed in-map prefix.',
        statistics='Observed success S/N; original-task success bound [S/N,(S+C)/N]; known failure F/N and censoring C/N separate. Across-agent mean/sample SD; no pooled training replications.',
        scope='Post-inspection protocol repair after the known native map-exit error. Fresh episode seeds do not create an independent new domain. Original A/B/C records remain unchanged.')
    locked=output/'protocol.json'
    if locked.exists():assert json.loads(locked.read_text())==protocol
    else:
        study.write(locked,protocol)
        for name,expected in hashes.items():
            assert study.sha(ROOT/name)==expected
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();records.append(result)
            print(json.dumps({key:result[key] for key in ['algorithm','seed','observed_success_rate','censored_rate','mean_return','evaluation_seconds']}),flush=True)
    pd.DataFrame([{key:value for key,value in row.items() if key!='source_hashes'} for row in records]).to_csv(output/'comparison.csv',index=False)
    study.write(output/'completed.json',dict(models=10,episodes=2000,training_interactions=0,
        original_models_and_checkpoints_preserved=True,censored_episodes=sum(row['map_exits'] for row in records)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='transfer_boundary_v1')
    parser.add_argument('--workers',type=int,default=8);main(parser.parse_args())
