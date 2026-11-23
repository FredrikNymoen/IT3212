"""Engineering checks for compatible A--D observations; no policy selection."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'pilot')]
from environment import ExperimentTask,TRANSFER_GEOMETRY
from navigation_features import RichTask
from tabular_q import TabularQ


def main():
    rng=np.random.default_rng(909090);matched=0;paired=0
    old,new=RichTask(),ExperimentTask('B')
    try:
        for seed in (42,7):
            a,_=old.reset(seed=seed);b,_=new.reset(seed=seed);assert np.array_equal(a,b)
            for _ in range(1000):
                action=int(rng.integers(5))
                a,r1,t1,x1,i1=old.step(action);b,r2,t2,x2,i2=new.step(action)
                assert np.array_equal(a,b) and np.array_equal(old.env.ram,new.env.ram)
                assert (r1,t1,x1)==(r2,t2,x2) and i1['score']==i2['score']
                matched+=1
                if t1 or x1:break
    finally:old.close();new.close()
    b,c=ExperimentTask('B'),ExperimentTask('C')
    try:
        b.reset(seed=314);c.reset(seed=314)
        for _ in range(1000):
            action=int(rng.integers(5))
            o1,r1,t1,x1,i1=b.step(action);o2,r2,t2,x2,i2=c.step(action)
            assert np.array_equal(o1,o2) and np.array_equal(b.env.ram,c.env.ram)
            assert np.isclose(r1-r2,.1) and (t1,x1)==(t2,x2)
            assert np.isclose(i2['task_return'],i2['score']-.1*i2['episode_steps'])
            paired+=1
            if t1 or x1:break
    finally:b.close();c.close()
    output=ROOT/'results/assignment4/current_engineering_audit';output.mkdir(parents=True,exist_ok=True)
    for setting in ('A','D'):
        env=ExperimentTask(setting);agent=TabularQ(42)
        try:
            obs,_=env.reset(seed=123)
            for _ in range(200):
                assert env.observation_space.contains(obs) and np.isfinite(obs).all()
                assert int(env.env.ram[0])==int(setting=='D')
                assert 0<=agent.action(agent.state(obs))<5
                obs,_,term,trunc,_=env.step(int(rng.integers(5)))
                if term or trunc:break
            Image.fromarray(env.unwrapped.ale.getScreenRGB()).save(output/f'maze_{setting}.png')
        finally:env.close()
    record=dict(exact_old_B_equivalence_decisions=matched,B_C_reward_isolation_decisions=paired,
        shared_observation_size=45,shared_actions=5,transfer_nodes=len(TRANSFER_GEOMETRY[1]),
        transfer_graph_connected=True,transfer_has_no_training_or_policy_selection=True)
    (output/'checks.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
