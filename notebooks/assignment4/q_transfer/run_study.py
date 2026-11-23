"""B-only Q development before a separately locked transfer confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from transfer_q import FACTORIES,TabularQ,ImprovedTask
spec=importlib.util.spec_from_file_location('q_transfer_evaluation',HERE.parent/'q_motion/experiment.py')
study=importlib.util.module_from_spec(spec);sys.modules[spec.name]=study;spec.loader.exec_module(study)

STOPS=(196608,262144)
BUDGET=262144
DEV_SEEDS=tuple(range(1000000,1000050))


def worker(job):
    with threadpool_limits(limits=1):return work(job)


def work(job):
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);study.write(directory/'job.json',job)
    training_seconds=None
    if job['variant']!='baseline':
        agent=FACTORIES[job['variant']](job['seed'],alpha=.15,gamma=.99)
        env=ImprovedTask('B');obs,_=env.reset(seed=job['seed']*1000)
        episode=0;steps=0;records=[];training_seconds=0.
        try:
            for stop in STOPS:
                start=perf_counter()
                while steps<stop:
                    state=agent.state(obs);epsilon=max(.05,1-.95*steps/(BUDGET*.8))
                    action=agent.action(state,epsilon)
                    nxt,reward,term,trunc,info=env.step(action)
                    agent.update(state,action,reward,agent.state(nxt),term or trunc);steps+=1;obs=nxt
                    if term or trunc:
                        records.append(dict(interactions=steps,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                        episode+=1;obs,_=env.reset(seed=job['seed']*1000+episode)
                training_seconds+=perf_counter()-start
                assert agent.update_calls==stop and agent.alpha==.15 and agent.gamma==.99
                assert type(agent).update is TabularQ.update
                assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
                with (directory/f'checkpoint_{stop}.pkl').open('wb') as handle:pickle.dump(agent,handle)
        finally:env.close()
        pd.DataFrame(records).to_csv(directory/'training.csv',index=False)
    evaluations=[]
    for stop in STOPS:
        path=(ROOT/f'results/assignment4/part2_current_v3/final/B_Q-learning_{job["seed"]}/checkpoint_{stop}.pkl'
            if job['variant']=='baseline' else directory/f'checkpoint_{stop}.pkl')
        checksum=study.sha(path)
        if job['variant']=='baseline':assert checksum==job['baseline_hashes'][str(stop)]
        with path.open('rb') as handle:agent=pickle.load(handle)
        assert agent.update_calls==stop and type(agent) is FACTORIES[job['variant']]
        evaluations.append(dict(checkpoint=stop,model=str(path.relative_to(ROOT)),model_sha256=checksum,
            **study.evaluate(agent,'B',DEV_SEEDS,directory,f'development_{stop}')))
        assert study.sha(path)==checksum
    result=dict(**job,training_seconds=training_seconds,training_interactions=BUDGET,
        reused_training=job['variant']=='baseline',evaluations=evaluations)
    study.write(done,result);return result


def main(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    hashes=study.sources()
    for path in (Path(__file__),HERE/'transfer_q.py'):hashes[str(path.relative_to(ROOT))]=study.sha(path)
    old=json.loads((ROOT/'results/assignment4/part2_current_v3/protocol.json').read_text())
    for name,expected in old['source_hashes'].items():assert study.sha(ROOT/name)==expected,name
    baseline={str(seed):{str(stop):study.sha(ROOT/f'results/assignment4/part2_current_v3/final/B_Q-learning_{seed}/checkpoint_{stop}.pkl')
        for stop in STOPS} for seed in (42,7)}
    protocol=dict(source_hashes=hashes,baseline_model_hashes=baseline,training_seeds=[42,7],budget=BUDGET,
        checkpoints=list(STOPS),primary_checkpoint=BUDGET,development_seeds=list(DEV_SEEDS),
        variants=['baseline','threat','power'],alpha=.15,gamma=.99,epsilon='Original 1 to .05 over 80% of 262144',
        changes={'threat':'Replace only directional binary danger with levels at .13/.36.',
            'power':'Replace only binary inherited capped power proxy with zero/low/high split at 16/63; decoder unchanged.'},
        selection='Mean B-development success across seeds42/7, then mean raw score, then earlier checkpoint, then variant name.',
        history='Additional development after old D outcomes were inspected. Those episodes are verification/history, not a reusable holdout. No D episode selects these candidates.',
        constraints='Ordinary single-table one-step Q, native score reward, no replay/traces/masks; original predictions and main report preserved.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        study.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    jobs=[dict(variant=variant,seed=seed,baseline_hashes=baseline[str(seed)],
        directory=str(output/f'B_{variant}_{seed}')) for seed in (42,7) for variant in ('baseline','threat','power')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();print(json.dumps(dict(variant=result['variant'],seed=result['seed'],evaluations=result['evaluations'])),flush=True)
    rows=[]
    for job in jobs:
        result=json.loads((Path(job['directory'])/'metrics.json').read_text())
        for row in result['evaluations']:rows.append(dict(variant=job['variant'],seed=job['seed'],**row))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    means=frame.groupby(['variant','checkpoint'])[['success_rate','mean_score','table_states','unseen_state_fraction']].mean().reset_index()
    means.to_csv(output/'means.csv',index=False);print(means.to_string(index=False),flush=True)
    selected=means.sort_values(['success_rate','mean_score','checkpoint','variant'],ascending=[False,False,True,True]).iloc[0]
    study.write(output/'selected.json',dict(variant=str(selected.variant),checkpoint=int(selected.checkpoint),development_only=True))
    study.write(output/'completed.json',dict(new_training_runs=4,new_development_episodes=600,settings=['B'],D_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='q_transfer_v1');main(parser.parse_args())
