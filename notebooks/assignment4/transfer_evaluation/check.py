"""Verify unchanged pre-boundary Gymnasium behaviour and the known exit."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from boundary_env import BoundaryTask,FastStudyTask
import run_study_v2 as study

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/transfer_boundary_v1'

def main():
    checks=[]
    for setting in 'ABCD':
        old=FastStudyTask(setting);new=BoundaryTask(setting);steps=0
        try:
            a,_=old.reset(seed=391);b,_=new.reset(seed=391);assert np.array_equal(a,b)
            rng=np.random.default_rng(72)
            for index in range(400):
                action=int(rng.integers(5));a,ra,ta,xa,ia=old.step(action);b,rb,tb,xb,ib=new.step(action)
                assert np.array_equal(a,b) and (ra,ta,xa)==(rb,tb,xb)
                assert np.array_equal(old.unwrapped.ale.getRAM(),new.unwrapped.ale.getRAM())
                for field in ['score','task_return','reason','lives','episode_steps']:assert ia[field]==ib[field]
                steps+=1
                if ta or xa:a,_=old.reset(seed=500+index);b,_=new.reset(seed=500+index)
        finally:old.close();new.close()
        checks.append(dict(setting=setting,identical_steps=steps))
    original=ROOT/'results/assignment4/three_hour_20261122/final_v1'
    job=json.loads((original/'D_Q-learning_314/job.json').read_text())
    assert study.sha(ROOT/job['model'])==job['model_sha256']
    with (ROOT/job['model']).open('rb') as handle:model=pickle.load(handle)
    before=model.fingerprint();old=FastStudyTask('D');new=BoundaryTask('D')
    try:
        seed=1150094;a,_=old.reset(seed=seed);b,_=new.reset(seed=seed)
        model.rng=np.random.default_rng(seed+3000000)
        for step in range(1000):
            action=model.action(model.state(a))
            b,rb,tb,xb,ib=new.step(action)
            try:a,ra,ta,xa,ia=old.step(action)
            except RuntimeError:
                assert step+1==975 and not tb and xb and ib['censored']
                assert ib['reason']=='map_exit' and not ib['success'] and not ib['known_failure']
                assert ib['score']==ib['observed_prefix_score']==1760 and ib['lives']==2
                assert np.array_equal(old.unwrapped.ale.getRAM(),new.unwrapped.ale.getRAM())
                assert new.observation_space.contains(b)
                try:new.step(0)
                except RuntimeError:pass
                else:raise AssertionError('An action was allowed after leaving the held-out map')
                checks.append(dict(known_exit_seed=seed,step=975,score=1760,censored_not_failure=True));break
            assert np.array_equal(a,b) and (ra,ta,xa)==(rb,tb,xb)
        else:raise AssertionError('Known exit not reproduced')
        assert model.fingerprint()==before and study.sha(ROOT/job['model'])==job['model_sha256']
    finally:old.close();new.close()
    study.write(BASE/'engineering.json',dict(checks=checks,sources={str(path.relative_to(ROOT)):study.sha(path)
        for path in [Path(__file__),Path(__file__).with_name('boundary_env.py')]}))
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
