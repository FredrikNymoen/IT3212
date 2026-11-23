"""Select from A/B development, lock, then confirm A and frozen B-to-D.

No D outcomes are inspected before representation/checkpoint selection.
Confirmation never changes the selected model or the registered v3 report.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('q_motion_confirm_study',HERE/'experiment.py')
study=importlib.util.module_from_spec(spec);sys.modules[spec.name]=study;spec.loader.exec_module(study)
from motion_q import MotionQ,TabularQ,ImprovedTask
from tunnel_features import TunnelTask
ROOT=study.ROOT
SEEDS=tuple(range(960000,960200))


def worker(job):
    with threadpool_limits(limits=1):return evaluate(job)


def evaluate(job):
    file=ROOT/job['model'];assert study.sha(file)==job['model_sha256']
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);study.write(directory/'job.json',job)
    with file.open('rb') as handle:agent=pickle.load(handle)
    assert type(agent) is (MotionQ if job['representation']=='motion' else TabularQ)
    assert agent.update_calls==job['checkpoint'] and agent.alpha==.15 and agent.gamma==.99
    assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
    before=agent.fingerprint();rng=agent.rng
    env=(TunnelTask if job['representation']=='tunnels' else ImprovedTask)(job['condition'])
    rows=[];unknown=0;decisions=0;start=perf_counter()
    try:
        for episode_seed in SEEDS:
            obs,_=env.reset(seed=episode_seed);agent.rng=np.random.default_rng(episode_seed+3000000)
            for _ in range(1000):
                state=agent.state(obs);unknown+=state[1] not in agent.table;decisions+=1
                obs,_,term,trunc,info=env.step(agent.action(state))
                if term or trunc:
                    assert info['maze']==int(job['condition']=='D')
                    rows.append(dict(evaluation_seed=episode_seed,success=int(info['success']),score=info['score'],
                        length=info['episode_steps'],reason=info['reason'],maze=info['maze']));break
    finally:env.close();agent.rng=rng
    assert agent.fingerprint()==before and study.sha(file)==job['model_sha256']
    frame=pd.DataFrame(rows)
    assert len(frame)==200 and np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
    frame.to_csv(directory/'evaluation.csv',index=False)
    result=dict(**job,episodes=200,success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        mean_length=float(frame.length.mean()),evaluation_seconds=perf_counter()-start,
        unseen_state_fraction=unknown/decisions,frozen_parameters_verified=True,further_training_interactions=0)
    study.write(done,result);return result


def main(args):
    motion=ROOT/'results/assignment4/q_motion_v1';tunnels=ROOT/'results/assignment4/q_tunnels_v2'
    assert (motion/'completed.json').exists() and (tunnels/'completed.json').exists()
    sources=study.sources()
    for directory in (motion,tunnels):
        for name,expected in json.loads((directory/'protocol.json').read_text())['source_hashes'].items():
            assert study.sha(ROOT/name)==expected,name
            sources[name]=expected
    sources[str(Path(__file__).relative_to(ROOT))]=study.sha(__file__)
    frames=[];development_hashes={}
    for directory in (motion,tunnels):
        path=directory/'comparison.csv';frames.append(pd.read_csv(path))
        development_hashes[str(path.relative_to(ROOT))]=study.sha(path)
    frame=pd.concat(frames).drop_duplicates(['setting','variant','seed','checkpoint'])
    counts=frame.groupby(['variant','checkpoint']).size();assert (counts==4).all()
    ranking=frame.groupby(['variant','checkpoint'])[['success_rate','mean_score']].mean().reset_index()
    ranking=ranking.sort_values(['success_rate','mean_score','checkpoint','variant'],ascending=[False,False,True,True])
    selected=ranking.iloc[0];variant=str(selected.variant);stop=int(selected.checkpoint)
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for condition in ('A','D'):
        training_setting='A' if condition=='A' else 'B'
        for seed in study.TRAIN_SEEDS:
            roles=[('baseline','baseline',stop),('candidate',variant,stop)]
            if condition=='D' and stop!=study.BUDGET:roles.append(('full_budget_reference','baseline',study.BUDGET))
            for role,representation,checkpoint in roles:
                record=frame[(frame.setting==training_setting)&(frame.variant==representation)&(frame.seed==seed)&(frame.checkpoint==checkpoint)].iloc[0]
                jobs.append(dict(condition=condition,training_setting=training_setting,seed=seed,role=role,
                    representation=representation,checkpoint=checkpoint,model=record.model,model_sha256=record.model_sha256,
                    directory=str(output/f'{condition}_{role}_{seed}')))
    protocol=dict(source_hashes=sources,development_data_hashes=development_hashes,
        selection='Equal-weight mean success across A/B and seeds42/7, then mean game score, then earlier checkpoint, then variant name.',
        selected_representation=variant,selected_checkpoint=stop,evaluation_seeds=list(SEEDS),jobs=jobs,
        history='Development choice after prior inspected results. Select once before new confirmation; never reselect from these A/D outcomes.',
        training='A confirmation uses A-trained agents. D uses the exact frozen B model, zero additional training.',
        policy='Greedy Q, seeded uniform ties, unchanged learned table and model files.',
        limitations='Two training seeds; not a new complete A-D report or five-seed Part3 evidence.')
    file=output/'locked.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Locked source/model/selection mismatch'
    else:
        study.write(file,protocol)
        for name in sources:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    ranking.to_csv(output/'development_selection.csv',index=False)
    print(f'LOCKED {variant}, checkpoint {stop}; A/D confirmation has not selected this configuration.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    result=pd.DataFrame(results);result.to_csv(output/'comparison.csv',index=False)
    means=result.groupby(['condition','role'])[['success_rate','mean_score','mean_length']].mean().reset_index()
    means.to_csv(output/'means.csv',index=False);print(means.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='q_representation_confirmation_v1');main(parser.parse_args())
