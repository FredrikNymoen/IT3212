"""Independently recount old results and replay them against native score RAM."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from motion_q import MotionQ,TabularQ,ImprovedTask
from tunnel_features import TunnelTask

ROOT=Path(__file__).resolve().parents[3]


def score_from_ram(ram):
    bcd=lambda value:(int(value)//16)*10+int(value)%16
    assert all(int(ram[i])%16<10 and int(ram[i])//16<10 for i in (120,121,122))
    return bcd(ram[122])*10000+bcd(ram[121])*100+bcd(ram[120])


def main():
    output=ROOT/'results/assignment4/q_representation_confirmation_v1'
    registry=json.loads((output/'locked.json').read_text())
    for name,expected in registry['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    records=[];counted=0;replays=[]
    for job in registry['jobs']:
        frame=pd.read_csv(Path(job['directory'])/'evaluation.csv')
        assert frame.evaluation_seed.tolist()==registry['evaluation_seeds']
        assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
        count=int(frame.success.sum());counted+=len(frame)
        metrics=json.loads((Path(job['directory'])/'metrics.json').read_text())
        assert np.isclose(count/len(frame),metrics['success_rate'])
        records.append(dict(condition=job['condition'],role=job['role'],seed=job['seed'],successes=count,episodes=len(frame)))
        path=ROOT/job['model'];assert hashlib.sha256(path.read_bytes()).hexdigest()==job['model_sha256']
        with path.open('rb') as handle:agent=pickle.load(handle)
        before=agent.fingerprint()
        for success in (0,1):
            subset=frame[frame.success==success].sort_values(['length','evaluation_seed'])
            if not len(subset):continue
            row=subset.iloc[len(subset)//2];seed=int(row.evaluation_seed)
            env=(TunnelTask if job['representation']=='tunnels' else ImprovedTask)(job['condition'])
            obs,_=env.reset(seed=seed);agent.rng=np.random.default_rng(seed+3000000)
            initial=score_from_ram(env.env.env.ram)
            try:
                for step in range(1000):
                    obs,reward,term,trunc,info=env.step(agent.action(agent.state(obs)))
                    native=score_from_ram(env.env.env.ram)-initial
                    assert native==info['score']
                    if term or trunc:
                        assert info['score']==row.score and info['episode_steps']==row.length
                        assert int(info['success'])==success
                        replays.append(dict(condition=job['condition'],role=job['role'],seed=job['seed'],
                            episode_seed=seed,score=native,length=step+1,success=success));break
            finally:env.close()
            assert agent.fingerprint()==before
        assert hashlib.sha256(path.read_bytes()).hexdigest()==job['model_sha256']
    assert counted==2000
    (output/'independent_native_replay.json').write_text(json.dumps(dict(recounted_episodes=counted,
        per_model_counts=records,native_replays=replays,all_steps_match_native_BCD_score=True),indent=2))
    print(pd.DataFrame(records).to_string(index=False),flush=True)
    print(f'PASS: independently recounted {counted} episodes and replayed {len(replays)} outcome cases; every step matches native score RAM.',flush=True)


if __name__=='__main__':main()
