"""Engineering checks before learning; no tuning or held-out-map inspection."""
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium.utils.env_checker import check_env

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from agents import ImprovedTask,ImprovedQ,MotionPolicy,FrozenMotion
from environment import ExperimentTask


def main():
    torch.set_num_threads(1)
    checked=0
    for setting in 'ABC':
        old=ExperimentTask(setting);new=ImprovedTask(setting)
        try:
            a,_=old.reset(seed=123);b,_=new.reset(seed=123)
            assert np.array_equal(a,b[:45]) and new.observation_space.contains(b)
            rng=np.random.default_rng(71)
            for step in range(300):
                action=int(rng.integers(5))
                a,ra,ta,xa,ia=old.step(action);b,rb,tb,xb,ib=new.step(action)
                assert np.array_equal(a,b[:45]) and (ra,ta,xa)==(rb,tb,xb)
                assert ia['score']==ib['score'] and new.observation_space.contains(b)
                assert np.isclose(b[47],ib['score']/2000)
                assert np.isclose(b[48],ib['lives']/5)
                checked+=1
                if ta or xa:break
        finally:old.close();new.close()
    env=ImprovedTask('B')
    try:
        # The frozen underlying wrapper has no recreatable constructor spec.
        # All explicitly created environments are closed in finally blocks.
        check_env(env,skip_render_check=True,skip_close_check=True)
        obs,_=env.reset(seed=123)
        for variant in ('credit','context'):
            agent=ImprovedQ(42,variant);state=agent.state(obs)
            target_obs=obs.copy();target_obs[47]=.75
            nxt=agent.state(target_obs)
            agent.table[nxt[1]]=np.array([1.,2.,3.,4.,5.],dtype=np.float32)
            old_values=agent.values(state).copy();old_value=old_values[0]
            expected=old_value+agent.alpha*(.1+agent.gamma*5-old_value)
            agent.update(state,0,10,nxt,False)
            assert np.isclose(agent.values(state)[0],expected)
            previous=agent.values(state)[0]
            agent.update(state,0,10,nxt,True)
            assert np.isclose(agent.values(state)[0],previous+agent.alpha*(.1-previous))
            assert agent.update_calls==2 and not hasattr(agent,'other') and not hasattr(agent,'buffer')
            before=agent.fingerprint();agent.action(state);assert agent.fingerprint()==before
        model=PPO(MotionPolicy,env,n_steps=8,batch_size=8,seed=42,device='cpu')
        actor=FrozenMotion(model)
        rng=np.random.default_rng(12)
        for _ in range(30):
            with torch.no_grad():actual=model.policy.get_distribution(torch.as_tensor(obs[None])).distribution.probs.numpy()[0]
            assert np.allclose(actual,actor.probabilities(obs),atol=1e-6)
            obs,_,term,trunc,_=env.step(int(rng.integers(5)))
            if term or trunc:obs,_=env.reset(seed=123)
    finally:env.close()
    print(f'PASS: {checked} unchanged native transitions/rewards; Gymnasium API; ordinary Q targets; frozen PPO inference.')


if __name__=='__main__':main()
