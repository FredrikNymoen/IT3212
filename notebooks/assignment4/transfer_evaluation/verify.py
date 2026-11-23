"""Independent aggregate/trace checks and direct native replay of D outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
from boundary_env import BoundaryTask
import run_study_v2 as study

ROOT=Path(__file__).resolve().parents[3]
gym.register_envs(ale_py)

def native_replay(job,row,z):
    mask=z['episode']==int(row.episode_index)
    arrays={name:z[name][mask] for name in ['ram','ram_after','action','raw_reward','score_after']}
    env=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,repeat_action_probability=.25,mode=0)
    try:
        env.reset(seed=int(row.evaluation_seed));env.unwrapped.ale.setRAM(0,1)
        initial=study.native_score(env.unwrapped.ale.getRAM());score=0.
        for index,action in enumerate(arrays['action']):
            before=env.unwrapped.ale.getRAM()
            assert np.array_equal(before,arrays['ram'][index]) and int(before[0])==1
            ram,reward,term,trunc,_=env.step(int(action));score+=reward
            assert np.array_equal(ram,arrays['ram_after'][index]) and reward==arrays['raw_reward'][index]
            assert study.native_score(ram)-initial==score==arrays['score_after'][index]
            if int(ram[0])!=1:
                assert row.censored and row.map_exit and index+1==len(arrays['action'])
            if index+1<len(arrays['action']):assert not term and not trunc and score<2000
        assert len(arrays['action'])==row.length and score==row.native_score
    finally:env.close()
    return dict(algorithm=job['algorithm'],training_seed=job['seed'],evaluation_seed=int(row.evaluation_seed),
        success=bool(row.success),censored=bool(row.censored),length=int(row.length),native_RAM_rewards_reproduced=True)

def main(args):
    output=ROOT/'results/assignment4'/args.run_id
    assert (output/'completed.json').exists()
    protocol=json.loads((output/'protocol.json').read_text())
    for name,expected in protocol['source_hashes'].items():assert study.sha(ROOT/name)==expected,name
    source=ROOT/'results/assignment4/three_hour_20261122/final_v1/locked.json'
    assert study.sha(source)==protocol['original_selection_lock_sha256']
    old=json.loads(source.read_text());records=[];replays=[];total=0
    for job in protocol['jobs']:
        directory=Path(job['directory']);metrics=json.loads((directory/'metrics.json').read_text())
        frame=pd.read_csv(directory/'evaluation.csv');assert len(frame)==200
        assert frame.evaluation_seed.tolist()==protocol['evaluation_seeds']
        assert frame[['success','known_failure','censored']].isin([0,1]).all().all()
        assert (frame[['success','known_failure','censored']].sum(axis=1)==1).all()
        assert (frame.map_exit<=frame.censored).all() and frame.length.between(1,1000).all()
        assert (frame.loc[frame.success.eq(1),'observed_score']>=2000).all()
        assert (frame.loc[frame.known_failure.eq(1),'observed_score']<2000).all()
        assert np.allclose(frame.observed_score,frame.observed_return)
        for column,key in [('success','observed_success_rate'),('known_failure','known_failure_rate'),
            ('censored','censored_rate'),('observed_return','mean_return')]:assert np.isclose(frame[column].mean(),metrics[key])
        assert np.isclose(metrics['success_upper_bound'],(frame.success+frame.censored).mean())
        assert metrics['training_seconds']==metrics['further_training_interactions']==0
        assert metrics['frozen_parameters_verified'] and metrics['no_actions_after_map_exit']
        partner=next(row for row in old['jobs'] if row['condition']=='B' and row['algorithm']==job['algorithm'] and row['seed']==job['seed'])
        assert job['model']==partner['model'] and job['model_sha256']==partner['model_sha256']==study.sha(ROOT/job['model'])
        with np.load(directory/'traces.npz') as z:
            n=int(frame.length.sum());total+=n
            assert z['ram'].shape==z['ram_after'].shape==(n,128)
            assert z['observation'].shape==(n,61) and z['values'].shape==(n,5)
            assert (z['ram'][:,0]==1).all()
            assert np.array_equal(np.bincount(z['episode'],minlength=200),frame.length)
            assert np.allclose(np.bincount(z['episode'],weights=z['raw_reward'],minlength=200),frame.native_score)
            assert np.array_equal(z['reward'],z['raw_reward'])
            bcd=lambda v:(v.astype(int)//16)*10+v.astype(int)%16
            scores=lambda ram:bcd(ram[:,122])*10000+bcd(ram[:,121])*100+bcd(ram[:,120])
            assert np.array_equal(scores(z['ram'])+z['raw_reward'],scores(z['ram_after']))
            assert np.array_equal(scores(z['ram_after']),z['score_after'])
            assert np.allclose(z['values'][np.arange(n),z['action']],z['values'].max(axis=1))
            if job['algorithm']=='PPO':assert np.array_equal(z['action'],z['values'].argmax(axis=1))
            end=np.cumsum(frame.length.to_numpy())-1
            assert np.array_equal(z['censored'][end],frame.censored.astype(bool))
            assert not np.any(z['censored'][np.setdiff1d(np.arange(n),end)])
            assert np.array_equal((z['ram_after'][end,0]!=1),frame.map_exit.astype(bool))
            expected=frame.native_score.to_numpy()-np.where(frame.map_exit,z['raw_reward'][end],0)
            assert np.allclose(frame.observed_return,expected)
            selected=frame.reset_index(names='episode_index')
            # A median outcome per trained model, plus every map exit.
            for outcome in ['success','known_failure']:
                group=selected[selected[outcome].eq(1)]
                if len(group):
                    index=(group.length-group.length.median()).abs().idxmin()
                    replays.append(native_replay(job,selected.loc[index],z))
            for _,row in selected[selected.censored.eq(1)].iterrows():replays.append(native_replay(job,row,z))
        records.append({key:value for key,value in metrics.items() if key!='source_hashes'})
    table=pd.DataFrame(records);assert len(table)==10 and table.seed.nunique()==5
    table.to_csv(output/'verified_comparison.csv',index=False)
    summary=[]
    for algorithm in ['Q-learning','PPO']:
        group=table[table.algorithm.eq(algorithm)]
        values=dict(algorithm=algorithm,trained_models=5,episodes=1000)
        for field in ['observed_success_rate','success_upper_bound','known_failure_rate','censored_rate',
            'mean_return','success_length','failure_length','evaluation_seconds']:
            values[field+'_mean']=float(group[field].mean())
            values[field+'_sd']=float(group[field].std(ddof=1))
        values['map_exits']=int(group.map_exits.sum());summary.append(values)
    study.write(output/'summary.json',dict(algorithms=summary))
    study.write(output/'verification.json',dict(episodes=2000,native_decisions=total,
        original_B_models_and_source_hashes_verified=True,all_traces_native_scores_rewards_actions_verified=True,
        no_action_after_map_exit=True,censoring_separate_from_failure=True,representative_and_all_censored_replays=replays,
        verifier_sha256=study.sha(Path(__file__)),further_training_interactions=0))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='transfer_boundary_v1');main(parser.parse_args())
