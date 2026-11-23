"""Separate one-factor A/B experiment: repair feature-graph tunnel edges."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
# Spawned Windows workers inherit paths changed by the legacy modules. Load
# this study by absolute file path, never the ambiguous name "experiment".
spec=importlib.util.spec_from_file_location('q_motion_study',HERE/'experiment.py')
study=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=study
spec.loader.exec_module(study)
from motion_q import TabularQ
from tunnel_features import TunnelTask

ROOT=study.ROOT


def evaluate(agent,setting,directory,stop):
    before=agent.fingerprint();rng=agent.rng;rows=[];unknown=0;decisions=0;keys=set()
    env=TunnelTask(setting);start=perf_counter()
    try:
        for seed in study.DEVELOPMENT_SEEDS:
            obs,_=env.reset(seed=seed);agent.rng=np.random.default_rng(seed+3000000)
            for _ in range(1000):
                state=agent.state(obs);unknown+=state[1] not in agent.table;keys.add(state[1]);decisions+=1
                obs,_,term,trunc,info=env.step(agent.action(state))
                if term or trunc:
                    rows.append(dict(evaluation_seed=seed,success=int(info['success']),score=info['score'],
                        length=info['episode_steps'],reason=info['reason']));break
    finally:env.close();agent.rng=rng
    assert agent.fingerprint()==before
    frame=pd.DataFrame(rows)
    assert len(frame)==50 and np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
    frame.to_csv(directory/f'development_{stop}.csv',index=False)
    return dict(success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),mean_length=float(frame.length.mean()),
        episodes=len(frame),evaluation_seconds=perf_counter()-start,evaluation_decisions=decisions,
        unseen_state_fraction=unknown/decisions,unique_evaluation_states=len(keys),table_states=len(agent.table),frozen_parameters_verified=True)


def worker(job):
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);study.write(directory/'job.json',job)
    if job['variant']=='baseline':
        source=ROOT/f'results/assignment4/q_motion_v1/{job["setting"]}_baseline_{job["seed"]}'
        assert study.sha(source/'metrics.json')==job['reference_sha256']
        result=json.loads((source/'metrics.json').read_text())
        result.update(directory=str(directory),reused_reference=str(source.relative_to(ROOT)))
        for stop in study.CHECKPOINTS:shutil.copy2(source/f'development_{stop}.csv',directory/f'development_{stop}.csv')
        study.write(done,result);return result
    agent=TabularQ(job['seed'],alpha=.15,gamma=.99);env=TunnelTask(job['setting'])
    obs,_=env.reset(seed=job['seed']*1000);steps=0;episode=0;training=[];seconds=0.
    try:
        for stop in study.CHECKPOINTS:
            start=perf_counter()
            while steps<stop:
                state=agent.state(obs);epsilon=max(.05,1-.95*steps/(study.BUDGET*.8))
                action=agent.action(state,epsilon);nxt,reward,term,trunc,info=env.step(action)
                agent.update(state,action,reward,agent.state(nxt),term or trunc);steps+=1;obs=nxt
                if term or trunc:
                    training.append(dict(interactions=steps,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;obs,_=env.reset(seed=job['seed']*1000+episode)
            seconds+=perf_counter()-start
            assert type(agent) is TabularQ and agent.update_calls==stop
            assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
            with (directory/f'checkpoint_{stop}.pkl').open('wb') as handle:pickle.dump(agent,handle)
    finally:env.close()
    pd.DataFrame(training).to_csv(directory/'training.csv',index=False)
    evaluations=[]
    for stop in study.CHECKPOINTS:
        file=directory/f'checkpoint_{stop}.pkl';checksum=study.sha(file)
        with file.open('rb') as handle:model=pickle.load(handle)
        evaluations.append(dict(checkpoint=stop,model=str(file.relative_to(ROOT)),model_sha256=checksum,
            **evaluate(model,job['setting'],directory,stop)))
        assert study.sha(file)==checksum
    result=dict(**job,training_interactions=study.BUDGET,training_seconds=seconds,evaluations=evaluations)
    study.write(done,result);return result


def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'results/assignment4/q_motion_v1'
    assert (previous/'completed.json').exists(),'Complete the motion/control study first'
    original=json.loads((previous/'protocol.json').read_text())
    for name,expected in original['source_hashes'].items():assert study.sha(ROOT/name)==expected,name
    hashes=study.sources()
    for path in (Path(__file__),HERE/'tunnel_features.py'):hashes[str(path.relative_to(ROOT))]=study.sha(path)
    references={f'{setting}_{seed}':study.sha(previous/f'{setting}_baseline_{seed}/metrics.json') for setting in 'AB' for seed in study.TRAIN_SEEDS}
    protocol=dict(source_hashes=hashes,reference_record_hashes=references,training_budget=study.BUDGET,
        training_seeds=list(study.TRAIN_SEEDS),development_seeds=list(study.DEVELOPMENT_SEEDS),
        checkpoints=list(study.CHECKPOINTS),primary_checkpoint=study.BUDGET,
        alpha=.15,gamma=.99,epsilon='Original 1 to .05 over 80% of 262144',
        sole_change='Connect paired left/right boundary degree-one nodes in the fixed feature graph. Original Q key/update unchanged; no closing bit.',
        graph_units='Unweighted graph hops, including tunnel edge; not exact movement times.',
        evidence='41 recorded native training-map B wrap transitions at rows 4/8; no D evidence used.',
        history='Additional separate Q development study after missing tunnel links were discovered. Baseline development episodes are reused from q_motion_v1; no final/D selection.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        study.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    jobs=[dict(setting=setting,variant=variant,seed=seed,reference_sha256=references[f'{setting}_{seed}'],
        directory=str(output/f'{setting}_{variant}_{seed}')) for setting in 'AB' for seed in study.TRAIN_SEEDS for variant in ('baseline','tunnels')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();print(json.dumps(dict(setting=result['setting'],variant=result['variant'],seed=result['seed'],evaluations=result['evaluations'])),flush=True)
    rows=[]
    for job in jobs:
        result=json.loads((Path(job['directory'])/'metrics.json').read_text())
        for metric in result['evaluations']:rows.append(dict(setting=job['setting'],variant=job['variant'],seed=job['seed'],**metric))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    means=frame.groupby(['setting','checkpoint','variant'])[['success_rate','mean_score','table_states','unseen_state_fraction']].mean().reset_index()
    means.to_csv(output/'means.csv',index=False);print(means.to_string(index=False),flush=True)
    study.write(output/'completed.json',dict(new_training_runs=4,new_development_episodes=400,reused_reference_episodes=400,D_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='q_tunnels_v2');main(parser.parse_args())
