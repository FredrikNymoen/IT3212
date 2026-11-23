"""Bitwise equivalence of original/faster outer wrappers, including task limits."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from candidates import StudyTask
from additions import LowBitsTask
from fast_evaluation import FastStudyTask,FastLowBitsTask

ROOT=Path(__file__).resolve().parents[3]

def main():
    checks=[]
    for setting in 'ABCD':
        old=StudyTask(setting);new=FastStudyTask(setting);count=0
        try:
            a,_=old.reset(seed=255);b,_=new.reset(seed=255);assert np.array_equal(a,b)
            rng=np.random.default_rng(179)
            for index in range(400):
                action=int(rng.integers(5));a,ra,ta,xa,ia=old.step(action);b,rb,tb,xb,ib=new.step(action)
                assert np.array_equal(a,b) and (ra,ta,xa)==(rb,tb,xb)
                assert np.array_equal(old.env.env.ram,new.env.env.ram) and new.observation_space.contains(b)
                for field in ['score','raw_reward','success','episode_steps','reason','task_return','maze','emulator_frames','lives']:
                    assert ia[field]==ib[field],field
                count+=1
                if ta or xa:a,_=old.reset(seed=256+index);b,_=new.reset(seed=256+index)
            old.env.env.steps=999;new.env.env.steps=999
            a,ra,ta,xa,ia=old.step(0);b,rb,tb,xb,ib=new.step(0)
            assert np.array_equal(a,b) and (ra,ta,xa)==(rb,tb,xb) and ta and ia['reason']==ib['reason']=='time_limit'
        finally:old.close();new.close()
        checks.append(dict(setting=setting,native_steps=count+1,bitwise_observations=True,identical_native_ram_rewards_flags_info=True))
    old=LowBitsTask('B');new=FastLowBitsTask('B')
    try:
        a,_=old.reset(seed=257);b,_=new.reset(seed=257);assert np.array_equal(a,b)
        for index in range(200):
            a,ra,ta,xa,ia=old.step(index%5);b,rb,tb,xb,ib=new.step(index%5)
            assert np.array_equal(a,b) and (ra,ta,xa)==(rb,tb,xb)
    finally:old.close();new.close()
    result=dict(checks=checks,low_bit_proxy_steps=200,only_discarded_computation_removed=True,
        sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),Path(__file__).with_name('fast_evaluation.py')]})
    output=ROOT/'results/assignment4/three_hour_20261122/engineering/fast_evaluation.json'
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
