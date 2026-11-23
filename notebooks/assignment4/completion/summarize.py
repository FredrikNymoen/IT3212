"""Recompute complete-study summaries from episode rows, with provenance.

Never rank/select agents from these evaluation outcomes. Statistical units are
five trained models, not pooled episodes. Raw operational times remain intact;
known user-suspension intervals are recorded separately in derived wall costs.
"""
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'synthesis'))
from part2_evidence import local_path,sha

BASE=ROOT/'results/assignment4'
OUT=BASE/'completion_analysis_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def suspension(directory,kind):
    file=BASE/'completion_pause_state.json'
    if not file.exists():return 0.
    state=read(file);duration=datetime.fromisoformat(state['resumed_at_utc']).timestamp()-datetime.fromisoformat(state['paused_at_utc']).timestamp()
    directory=Path(directory)
    if kind=='training':
        rows=[r for r in state.get('phase_ledger',[]) if r['job']==directory.name and r['inferred_phase']=='training_after_checkpoint']
        return duration if rows else 0.
    start=datetime.fromisoformat(state['paused_at_utc']).timestamp();end=datetime.fromisoformat(state['resumed_at_utc']).timestamp()
    job=directory/'job.json';done=directory/'metrics.json'
    return duration if job.exists() and done.exists() and job.stat().st_mtime<start and done.stat().st_mtime>end else 0.


def case_record(job,directory,study,condition,arm,train_seconds,model,model_sha,budget,checkpoint,seeds,trace_name):
    directory=local_path(directory);frame=pd.read_csv(directory/'evaluation.csv');metrics=read(directory/'metrics.json')
    assert len(frame)==200 and frame.evaluation_seed.tolist()==seeds
    assert sha(local_path(model))==model_sha
    assert metrics['frozen_parameters_verified']
    assert frame.success.isin([0,1]).all() and frame.length.between(1,1000).all()
    failure=frame.get('known_failure',1-frame.success);censored=frame.get('censored',pd.Series(0,index=frame.index))
    assert (frame.success+failure+censored==1).all() and not censored.any()
    assert np.array_equal(frame.success.to_numpy().astype(bool),frame.score.to_numpy()>=2000)
    assert np.allclose(frame.task_return,frame.score-(.1*frame.length if condition=='C' else 0))
    mean=lambda series:float(series.mean()) if len(series) else None
    training_folder=local_path(job['directory']) if study=='M3' else None
    t_pause=suspension(training_folder,'training') if training_folder else 0.
    e_pause=suspension(directory,'evaluation') if study=='D' else 0.
    assert train_seconds+1e-6>=t_pause and metrics['evaluation_seconds']+1e-6>=e_pause
    result=dict(study=study,condition=condition,arm=arm,algorithm=job.get('algorithm',job.get('config',{}).get('algorithm')),
        seed=job.get('seed'),episodes=200,success_rate=float(frame.success.mean()),failure_rate=float(failure.mean()),
        mean_return=float(frame.task_return.mean()),mean_score=float(frame.score.mean()),
        success_length=mean(frame.loc[frame.success.eq(1),'length']),failure_length=mean(frame.loc[failure.eq(1),'length']),
        training_interactions=budget,checkpoint_interactions=checkpoint,
        training_seconds_raw=train_seconds,training_pause_seconds=t_pause,training_seconds=train_seconds-t_pause,
        evaluation_seconds_raw=metrics['evaluation_seconds'],evaluation_pause_seconds=e_pause,
        evaluation_seconds=metrics['evaluation_seconds']-e_pause,evaluation_interactions=int(frame.length.sum()),
        model=str(local_path(model).relative_to(ROOT)),model_sha256=model_sha,
        evaluation_csv=str((directory/'evaluation.csv').relative_to(ROOT)),evaluation_csv_sha256=sha(directory/'evaluation.csv'),
        trace=str((directory/trace_name).relative_to(ROOT)))
    return result


def aggregate(records,keys):
    frame=pd.DataFrame(records);result=[]
    fields=['success_rate','failure_rate','mean_return','mean_score','success_length','failure_length','training_seconds','evaluation_seconds']
    for identity,group in frame.groupby(keys,sort=False):
        if not isinstance(identity,tuple):identity=(identity,)
        assert len(group)==5 and set(group.seed)=={17,29,101,314,2718}
        row=dict(zip(keys,identity));row.update(trained_models=5,episodes=1000)
        for field in fields:
            values=pd.to_numeric(group[field],errors='coerce').dropna()
            row[field]=float(values.mean()) if len(values) else None
            row[field+'_sd']=float(values.std(ddof=1)) if len(values)>1 else None
        result.append(row)
    return result


def main():
    m3=BASE/'completion_part3_v1';d=BASE/'clarified_D_v1';diag=BASE/'completion_diagnostic_v1'
    for directory in [m3,d,diag]:assert (directory/'completed.json').exists(),f'Incomplete study: {directory.name}'
    original=BASE/'three_hour_20261122/final_v1';old_lock=read(original/'locked.json')
    d_lock=read(d/'protocol.json');m_lock=read(m3/'protocol.json');m_final=read(m3/'final_lock.json')
    records=[]
    for job in old_lock['jobs']:
        if job['condition'] not in 'ABC':continue
        records.append(case_record(job,job['directory'],'P2',job['condition'],job['algorithm'],job['training_seconds'],
            job['model'],job['model_sha256'],job['training_interactions'],job['checkpoint_interactions'],old_lock['evaluation_seeds'],'evaluation_traces.npz'))
    for job in d_lock['jobs']:
        assert d_lock['training_interactions']==0
        partner=next(x for x in old_lock['jobs'] if x['condition']=='B' and x['algorithm']==job['algorithm'] and x['seed']==job['seed'])
        assert partner['model_sha256']==job['model_sha256']
        records.append(case_record(job,job['directory'],'D','D' if job['occluded'] else 'D_clean',job['algorithm'],0.,
            job['model'],job['model_sha256'],0,job['checkpoint_interactions'],job['evaluation_seeds'],'evaluation_traces.npz'))
    for job in m_final:
        completed=read(local_path(job['directory'])/'completed.json');assert completed['training_interactions']==262144
        if job['config']['algorithm']=='Q-learning':assert completed['update_calls']==262144
        records.append(case_record(job,job['evaluation_directory'],'M3','B',job['arm'],completed['training_seconds'],
            str(local_path(job['directory'])/job['model']),job['model_sha256'],262144,262144,job['evaluation_seeds'],'evaluation_traces.npz'))
    assert len(records)==95
    frame=pd.DataFrame(records)
    groups=aggregate(records,['study','condition','arm','algorithm'])
    mechanism=frame[frame.study.eq('M3')];deltas=[]
    for arm in mechanism.arm.unique():
        if arm.endswith('_base'):continue
        algorithm='Q-learning' if arm.startswith('Q_') else 'PPO'
        baseline='Q_base' if algorithm=='Q-learning' else 'PPO_base'
        a=mechanism[mechanism.arm.eq(arm)].set_index('seed');b=mechanism[mechanism.arm.eq(baseline)].set_index('seed')
        delta=100*(a.success_rate-b.success_rate)
        deltas.append(dict(arm=arm,baseline=baseline,mean_pp=float(delta.mean()),sd_pp=float(delta.std(ddof=1)),
            per_seed={str(seed):float(value) for seed,value in delta.items()},seeds_improved=int((delta>0).sum()),seeds_worsened=int((delta<0).sum())))
    diagnostic=[]
    for job in read(diag/'locked.json'):
        directory=local_path(job['directory']);r=read(directory/'metrics.json');episodes=pd.read_csv(directory/'episodes.csv')
        assert r['frozen_parameters_verified'] and len(episodes)==50 and episodes.evaluation_seed.tolist()==job['evaluation_seeds']
        assert sha(local_path(job['model']))==job['model_sha256']
        assert np.isclose(episodes.success.mean(),r['success_rate'])
        diagnostic.append(dict(arm=job['arm'],seed=job['seed'],mode=job['mode'],success_rate=r['success_rate'],mean_score=r['mean_score'],
            mean_action_entropy=float(episodes.mean_action_entropy.mean()),sampled_disagreement=float(episodes.sampled_disagreement.mean()),
            episodes=50,model_sha256=job['model_sha256']))
    probes=read(BASE/'completion_prediction_probes_v1/answers.json')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [original/'locked.json',d/'protocol.json',m3/'protocol.json',m3/'final_lock.json',diag/'plan.json',diag/'locked.json',
        BASE/'completion_inspection_v4/inspection.json',BASE/'completion_inspection_v4/annotations.json',BASE/'completion_prediction_probes_v1/answers.json',Path(__file__)]}
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/'agents.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(groups=groups,mechanism_deltas=deltas,diagnostic=diagnostic,inputs=inputs,
        pause_policy='Raw seconds preserved. Logged suspension removed only for the identified training/evaluation phase, using checkpoint/CSV publication timestamps. These remain shared-host wall times, not isolated CPU speed estimates.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(trained_case_records=95,groups=len(groups),M3_comparisons=deltas),indent=2))


if __name__=='__main__':main()
