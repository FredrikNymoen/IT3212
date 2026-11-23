"""Choose the strongest new B candidate, then lock and test frozen D once."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter,sleep

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_study as study
from transfer_q import FACTORIES,ImprovedTask,TabularQ
from resources_q import ResourceQ
from power_route_q import PowerRouteQ
FACTORIES['resources']=ResourceQ
FACTORIES['power_route']=PowerRouteQ
ROOT=study.ROOT
SEEDS=tuple(range(1060000,1060200))
LOADED_SOURCE_HASHES={str(path):study.study.sha(path) for path in
    [Path(__file__),HERE/'run_study.py',HERE/'transfer_q.py',HERE/'resources_q.py',HERE/'power_route_q.py']}


def native_score(ram):
    bcd=lambda value:int(value)//16*10+int(value)%16
    return bcd(ram[122])*10000+bcd(ram[121])*100+bcd(ram[120])


def worker(job):
    with threadpool_limits(limits=1):return evaluate(job)


def evaluate(job):
    path=ROOT/job['model'];assert study.study.sha(path)==job['model_sha256']
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);study.study.write(directory/'job.json',job)
    with path.open('rb') as handle:agent=pickle.load(handle)
    assert type(agent) is FACTORIES[job['variant']] and type(agent).update is TabularQ.update
    assert agent.update_calls==job['checkpoint'] and agent.alpha==.15 and agent.gamma==.99
    before=agent.fingerprint();rng=agent.rng;env=ImprovedTask('D');rows=[];unknown=0;decisions=0;start=perf_counter()
    try:
        for seed in SEEDS:
            obs,_=env.reset(seed=seed);agent.rng=np.random.default_rng(seed+3000000)
            initial=native_score(env.env.env.ram)
            for _ in range(1000):
                state=agent.state(obs);unknown+=state[1] not in agent.table;decisions+=1
                obs,_,term,trunc,info=env.step(agent.action(state))
                score=native_score(env.env.env.ram)-initial
                assert score==info['score'] and info['maze']==1
                if term or trunc:
                    rows.append(dict(evaluation_seed=seed,success=int(info['success']),score=score,
                        length=info['episode_steps'],reason=info['reason']));break
    finally:env.close();agent.rng=rng
    assert agent.fingerprint()==before and study.study.sha(path)==job['model_sha256']
    frame=pd.DataFrame(rows);assert len(frame)==200 and np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all()
    frame.to_csv(directory/'evaluation.csv',index=False)
    result=dict(**job,episodes=200,success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
        mean_length=float(frame.length.mean()),unseen_state_fraction=unknown/decisions,
        evaluation_seconds=perf_counter()-start,further_training_interactions=0,native_score_verified_every_step=True,
        frozen_parameters_verified=True)
    study.study.write(done,result);return result


def main(args):
    development=ROOT/'results/assignment4/q_transfer_v1'
    assert (development/'completed.json').exists()
    protocol=json.loads((development/'protocol.json').read_text())
    for name,expected in protocol['source_hashes'].items():assert study.study.sha(ROOT/name)==expected,name
    resource_output=ROOT/'results/assignment4/q_resources_v1'
    # This additional B-only hypothesis was specified before this D lock/run.
    deadline=perf_counter()+3600
    while not (resource_output/'completed.json').exists():
        if perf_counter()>deadline:raise TimeoutError('Complete the registered B resource study first.')
        sleep(5)
    resource_protocol=json.loads((resource_output/'protocol.json').read_text())
    for name,expected in resource_protocol['source_hashes'].items():assert study.study.sha(ROOT/name)==expected,name
    route_output=ROOT/'results/assignment4/q_power_route_v1'
    while not (route_output/'completed.json').exists():
        if perf_counter()>deadline:raise TimeoutError('Complete the registered B power-route study first.')
        sleep(5)
    route_protocol=json.loads((route_output/'protocol.json').read_text())
    for name,expected in route_protocol['source_hashes'].items():assert study.study.sha(ROOT/name)==expected,name
    for name,expected in LOADED_SOURCE_HASHES.items():
        assert study.study.sha(name)==expected,'Loaded confirmation source changed while waiting; restart before evaluating.'
    frame=pd.concat([pd.read_csv(development/'comparison.csv'),pd.read_csv(resource_output/'comparison.csv'),
        pd.read_csv(route_output/'comparison.csv')],ignore_index=True)
    ranking=frame[frame.variant!='baseline'].groupby(['variant','checkpoint'])[['success_rate','mean_score']].mean().reset_index()
    best=ranking.sort_values(['success_rate','mean_score','checkpoint','variant'],ascending=[False,False,True,True]).iloc[0]
    variant=str(best.variant);checkpoint=int(best.checkpoint)
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for seed in (42,7):
        for role,kind,stop in [('candidate',variant,checkpoint),('reference_196608','baseline',196608),('reference_262144','baseline',262144)]:
            record=frame[(frame.variant==kind)&(frame.seed==seed)&(frame.checkpoint==stop)].iloc[0]
            jobs.append(dict(seed=seed,role=role,variant=kind,checkpoint=stop,model=record.model,
                model_sha256=record.model_sha256,directory=str(output/f'{role}_{seed}')))
    hashes=dict(protocol['source_hashes']);hashes[str(Path(__file__).relative_to(ROOT))]=study.study.sha(__file__)
    hashes.update(resource_protocol['source_hashes'])
    hashes.update(route_protocol['source_hashes'])
    locked=dict(source_hashes=hashes,development_sha256=study.study.sha(development/'comparison.csv'),
        resource_development_sha256=study.study.sha(resource_output/'comparison.csv'),
        power_route_development_sha256=study.study.sha(route_output/'comparison.csv'),
        selected_variant=variant,selected_checkpoint=checkpoint,evaluation_seeds=list(SEEDS),jobs=jobs,
        selection='Strongest NEW variant by mean B-development success, then score, earlier checkpoint and name; baseline retained as controls even when stronger in B.',
        history='Earlier D map outcomes have been inspected. These are fresh conditional episodes, not a new independent domain. No new D data select or change these models.',
        training='All models are B-trained; frozen D evaluation adds zero training. Native score is checked every step.')
    file=output/'locked.json'
    if file.exists():assert json.loads(file.read_text())==locked
    else:
        study.study.write(file,locked)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    print(f'LOCKED {variant} at {checkpoint}; no D outcome selects this candidate.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    result=pd.DataFrame(results);result.to_csv(output/'comparison.csv',index=False)
    means=result.groupby('role')[['success_rate','mean_score','mean_length']].mean()
    means.to_csv(output/'means.csv');print(means.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='q_transfer_confirmation_v2');main(parser.parse_args())
