"""Numerical and native checks before registering bounded candidates."""
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from candidates import StudyTask,StudyPolicy,FrozenStudy,make_q,Q_VARIANTS,PPO_VARIANTS,TabularQ,ImprovedTask

def main():
    torch.set_num_threads(1);transitions=0;updates=0
    for setting in 'ABC':
        old=ImprovedTask(setting);new=StudyTask(setting)
        try:
            a,_=old.reset(seed=121);b,_=new.reset(seed=121)
            assert np.array_equal(a,b[:49]) and new.observation_space.contains(b)
            rng=np.random.default_rng(155)
            for _ in range(300):
                action=int(rng.integers(5));a,ra,ta,xa,ia=old.step(action);b,rb,tb,xb,ib=new.step(action)
                assert np.array_equal(a,b[:49]) and (ra,ta,xa)==(rb,tb,xb)
                assert np.array_equal(old.env.env.ram,new.env.env.ram) and new.observation_space.contains(b)
                active=b[57:61].astype(bool)
                ram=new.env.env.ram
                deltas=np.column_stack((ram[6:10].astype(float)-float(ram[10]),ram[12:16].astype(float)-float(ram[16])))/[160,210]
                assert np.allclose(b[49:57].reshape(4,2)[active],np.clip(deltas,-1,1)[active])
                assert np.all(b[49:57].reshape(4,2)[~active]==0)
                transitions+=1
                if ta or xa:a,_=old.reset(seed=121+transitions);b,_=new.reset(seed=121+transitions)
        finally:old.close();new.close()
    env=StudyTask('B');obs,_=env.reset(seed=121)
    try:
        for variant in Q_VARIANTS:
            model=make_q(variant,42);assert type(model).update is TabularQ.update
            for n in range(100):
                state=model.state(obs);action=int(n%5);nxt,reward,term,trunc,_=env.step(action)
                target_state=model.state(nxt);before=model.values(state)[action]
                target=reward/100+(0 if term or trunc else model.gamma*model.values(target_state).max())
                expected=before+model.alpha*(target-before)
                model.update(state,action,reward,target_state,term or trunc)
                assert np.isclose(model.values(state)[action],expected,atol=1e-6)
                obs=nxt;updates+=1
                if term or trunc:obs,_=env.reset(seed=122+n)
            assert model.update_calls==100 and not hasattr(model,'other') and not hasattr(model,'buffer')
            before=model.fingerprint();model.action(model.state(obs));assert model.fingerprint()==before
        for variant in PPO_VARIANTS:
            model=PPO(StudyPolicy,env,n_steps=8,batch_size=8,policy_kwargs=dict(actor_variant=variant),seed=42)
            actor=FrozenStudy(model)
            for index in range(20):
                with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(obs[None])).distribution.probs.numpy()[0]
                expected=actor.probabilities(obs)
                assert np.allclose(actual,expected,atol=1e-6) and actual.argmax()==expected.argmax()
                obs,_,term,trunc,_=env.step(index%5)
                if term or trunc:obs,_=env.reset(seed=131+index)
            model.learn(16)
            target=HERE.parents[2]/'results/assignment4/three_hour_20261122/engineering'/f'{variant}.zip'
            target.parent.mkdir(parents=True,exist_ok=True);model.save(target)
            loaded=PPO.load(target)
            assert loaded.policy.actor_variant==variant
            assert np.allclose(FrozenStudy(loaded).probabilities(obs),FrozenStudy(model).probabilities(obs))
    finally:env.close()
    print(f'PASS: {transitions} identical native transitions, {updates} ordinary Q updates, four PPO actors and save/load.',flush=True)

if __name__=='__main__':main()
