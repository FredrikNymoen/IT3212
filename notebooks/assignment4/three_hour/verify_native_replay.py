"""Replay representative final actions directly in native Gymnasium, no task wrappers."""
import argparse
import hashlib
import json
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
gym.register_envs(ale_py)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def native_score(ram):
    bcd=lambda v:int(v)//16*10+int(v)%16
    return bcd(ram[122])*10000+bcd(ram[121])*100+bcd(ram[120])

def main(args):
    directory=ROOT/'results/assignment4'/args.final_run
    lock=json.loads((directory/'locked.json').read_text());cases=[]
    assert (directory/'completed.json').exists() or (directory/'interrupted.json').exists()
    for condition in 'ABCD':
        for algorithm in ['Q-learning','PPO']:
            records=[]
            for job in lock['jobs']:
                if job['condition']!=condition or job['algorithm']!=algorithm:continue
                if not (Path(job['directory'])/'metrics.json').exists():
                    assert condition=='D' and algorithm=='Q-learning' and job['seed']==314
                    continue
                frame=pd.read_csv(Path(job['directory'])/'evaluation.csv')
                frame['job_directory']=job['directory'];frame['model']=job['model'];frame['model_sha256']=job['model_sha256'];frame['training_seed']=job['seed']
                frame['episode_index']=np.arange(len(frame));records.append(frame)
            population=pd.concat(records,ignore_index=True)
            for outcome in [1,0]:
                subset=population[population.success==outcome].copy()
                if subset.empty:continue
                subset['median_distance']=abs(subset.length-subset.length.median())
                selected=subset.sort_values(['median_distance','training_seed','evaluation_seed']).iloc[0]
                assert sha(ROOT/selected.model)==selected.model_sha256
                with np.load(Path(selected.job_directory)/'evaluation_traces.npz') as z:
                    mask=z['episode']==int(selected.episode_index)
                    rams=z['ram'][mask];actions=z['action'][mask];raw_rewards=z['raw_reward'][mask]
                    task_rewards=z['reward'][mask];scores=z['score_after'][mask]
                env=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,
                    repeat_action_probability=0.0 if condition=='A' else .25,mode=0)
                try:
                    env.reset(seed=int(selected.evaluation_seed));env.unwrapped.ale.setRAM(0,int(condition=='D'))
                    initial=native_score(env.unwrapped.ale.getRAM());earned=0.
                    for step,action in enumerate(actions):
                        assert np.array_equal(env.unwrapped.ale.getRAM(),rams[step])
                        ram,reward,term,trunc,info=env.step(int(action));earned+=reward
                        assert reward==raw_rewards[step]
                        assert np.isclose(task_rewards[step],reward-(.1 if condition=='C' else 0))
                        assert native_score(ram)-initial==earned==scores[step]
                        assert int(ram[0])==int(condition=='D')
                        if step+1<len(actions):assert not term and not trunc and earned<2000
                    assert earned==selected.score and len(actions)==selected.length
                    assert int(earned>=2000)==int(selected.success)
                finally:env.close()
                case=dict(condition=condition,algorithm=algorithm,training_seed=int(selected.training_seed),
                    evaluation_seed=int(selected.evaluation_seed),success=bool(selected.success),score=float(earned),
                    length=len(actions),all_native_ram_actions_rewards_reproduced=True,
                    selection='Closest to the conditional median length within this setting/algorithm/outcome.')
                cases.append(case);print(json.dumps(case),flush=True)
    result=dict(cases=cases,episodes_replayed=len(cases),native_decisions=sum(x['length'] for x in cases),
        independent_direct_gymnasium_replay=True,source_sha256=sha(__file__))
    (directory/'native_replay_validation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final-run',required=True);main(parser.parse_args())
