"""Independent trace accounting and wrapper-free native replay of final evidence."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'synthesis'))
from part2_evidence import sha,local_path
gym.register_envs(ale_py)

def read(p):return json.loads(Path(p).read_text())
def scores(ram):
    a=ram.astype(np.int32)
    b=lambda x:(x//16)*10+x%16
    return b(a[:,122])*10000+b(a[:,121])*100+b(a[:,120])

def native(record,frame,z):
    """One median recorded outcome, directly in ALE, bypassing custom wrappers."""
    index=int((frame.length-frame.length.median()).abs().idxmin())
    use=z['episode']==index
    seed=int(frame.iloc[index].evaluation_seed)
    env=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,
                 repeat_action_probability=0 if record['condition']=='A' else .25,mode=0)
    count=0
    try:
        env.reset(seed=seed)
        for i,(before,action,reward) in enumerate(zip(z['ram'][use],z['action'][use],z['raw_reward'][use])):
            assert np.array_equal(env.unwrapped.ale.getRAM(),before)
            obs,r,term,trunc,_=env.step(int(action))
            assert r==reward
            if 'ram_after' in z.files:assert np.array_equal(obs,z['ram_after'][use][i])
            else:assert scores(obs[None,:])[0]==z['score_after'][use][i]
            count+=1
    finally:env.close()
    return dict(study=record['study'],arm=record['arm'],condition=record['condition'],seed=int(record['seed']),
                evaluation_seed=seed,decisions=count,native_RAM_rewards_identical=True)

def main():
    base=ROOT/'results/assignment4'; analysis=base/'completion_analysis_v1'
    frame=pd.read_csv(analysis/'agents.csv'); checks=[];replays=[];total=0
    for record in frame.to_dict('records'):
        csv=ROOT/record['evaluation_csv']; episodes=pd.read_csv(csv)
        assert sha(csv)==record['evaluation_csv_sha256']
        assert sha(ROOT/record['model'])==record['model_sha256']
        with np.load(ROOT/record['trace']) as z:
            n=int(episodes.length.sum()); total+=n
            assert z['ram'].shape==(n,128)
            if 'ram_after' in z.files:assert z['ram_after'].shape==(n,128)
            assert z['observation'].shape==(n,61) and z['values'].shape==(n,5)
            assert np.isfinite(z['observation']).all() and np.isfinite(z['values']).all()
            assert (z['ram'][:,0]==0).all()
            if 'censored' in z.files:assert not z['censored'].any()
            lengths=episodes.length.to_numpy(); ends=np.cumsum(lengths)-1
            assert np.array_equal(np.bincount(z['episode'],minlength=200),lengths)
            assert np.array_equal(z['step'],np.concatenate([np.arange(x) for x in lengths]))
            assert np.allclose(np.bincount(z['episode'],weights=z['raw_reward'],minlength=200),episodes.score)
            assert np.array_equal(z['score_after']-scores(z['ram']),z['raw_reward'])
            if 'ram_after' in z.files:assert np.array_equal(scores(z['ram_after']),z['score_after'])
            same=z['episode'][1:]==z['episode'][:-1]
            assert np.array_equal(scores(z['ram'])[1:][same],z['score_after'][:-1][same])
            assert np.array_equal(z['score_after'][ends],episodes.score)
            assert np.allclose(z['reward'],z['raw_reward']-(.1 if record['condition']=='C' else 0))
            assert np.allclose(z['values'][np.arange(n),z['action']],z['values'].max(axis=1))
            if 'terminated' in z.files:assert np.array_equal(z['terminated']|z['truncated'],np.isin(np.arange(n),ends))
            if record['algorithm']=='PPO':
                assert np.array_equal(z['action'],z['values'].argmax(axis=1))
                assert (z['values']>=0).all() and np.allclose(z['values'].sum(axis=1),1,atol=1e-5)
            if record['seed']==17:
                replays.append(native(record,episodes,z))
        checks.append(dict(trace=record['trace'],sha256=sha(ROOT/record['trace']),episodes=200,decisions=n))
        print(json.dumps(dict(verified=record['study'],arm=record['arm'],condition=record['condition'],seed=record['seed'])),flush=True)
    diag=base/'completion_diagnostic_v1'; diagsteps=0
    for job in read(diag/'locked.json'):
        directory=local_path(job['directory']); episodes=pd.read_csv(directory/'episodes.csv')
        assert sha(local_path(job['model']))==job['model_sha256']
        with np.load(directory/'traces.npz') as z:
            n=int(episodes.length.sum());diagsteps+=n
            assert np.array_equal(np.bincount(z['episode'],minlength=50),episodes.length)
            assert np.allclose(np.bincount(z['episode'],weights=z['raw_reward'],minlength=50),episodes.score)
            assert np.allclose(z['probabilities'].sum(axis=1),1,atol=1e-5)
            if job['mode']=='argmax':assert np.array_equal(z['action'],z['probabilities'].argmax(axis=1))
            a=scores(z['ram']); same=z['episode'][1:]==z['episode'][:-1]
            assert np.array_equal((a[1:]-a[:-1])[same],z['raw_reward'][:-1][same])
        checks.append(dict(trace=str((directory/'traces.npz').relative_to(ROOT)),sha256=sha(directory/'traces.npz'),episodes=50,decisions=n))
    for run in ['completion_part3_v1','clarified_D_v1','completion_diagnostic_v1']:
        protocol=read(base/run/('plan.json' if run=='completion_diagnostic_v1' else 'protocol.json'))
        for name,digest in protocol.get('sources',protocol.get('source_hashes',{})).items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'reports/assignment4/Sections/01_predictions.tex')=='17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a'
    result=dict(primary_episodes=19000,primary_decisions=total,secondary_episodes=1000,secondary_decisions=diagsteps,
                trace_checks=checks,native_replays=replays,verifier_sha256=sha(Path(__file__)))
    (analysis/'records_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['trace_checks','native_replays']}))

if __name__=='__main__':main()
