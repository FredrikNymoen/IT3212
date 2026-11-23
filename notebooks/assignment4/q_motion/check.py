"""Validate closing direction and unchanged ordinary-Q learning numerically."""
import numpy as np
from motion_q import MotionQ,TabularQ,ImprovedTask,closing_bit


def main():
    obs=np.zeros(49,dtype=np.float32)
    obs[2]=24/160;obs[4]=24/320;obs[45]=-2/8
    assert closing_bit(obs)==1                         # Ghost approaches.
    obs[45]=2/8;assert closing_bit(obs)==0              # Ghost recedes.
    obs[45]=-2/8;obs[41]=-4/8
    assert closing_bit(obs)==0                        # Player escapes faster.
    obs[41]=0;obs[12]=1;assert closing_bit(obs)==0      # Powered proxy.
    obs[12]=0;obs[4]=80/320;assert closing_bit(obs)==0   # Far encounter.
    obs[4]=24/320
    # Rotation of both physical deltas and velocities leaves closing invariant.
    obs[45]=-2/8;obs[46]=0
    rotated=obs.copy();rotated[2:4]=[0,24/210];rotated[45:47]=[0,-2/8]
    assert closing_bit(obs)==closing_bit(rotated)==1
    env=ImprovedTask('B');rng=np.random.default_rng(123)
    try:
        obs,_=env.reset(seed=123)
        old=TabularQ(42);new=MotionQ(42)
        assert (new.alpha,new.gamma)==(.15,.99)
        for _ in range(300):
            state=new.state(obs);base=old.state(obs)
            assert state[0]==base[0] and state[1][:-1]==base[1]
            action=int(rng.integers(5));nxt,reward,term,trunc,_=env.step(action)
            # Use identical numeric estimates at two distinct test states.
            target=(0,('numeric-target',));new.table[target[1]]=np.array([1,2,3,4,5],dtype=np.float32)
            previous=float(new.values(state)[action])
            expected=previous+.15*(reward/100+(.99*5 if not(term or trunc) else 0)-previous)
            new.update(state,action,reward,target,term or trunc)
            assert np.isclose(new.values(state)[action],expected,atol=1e-6)
            before=new.fingerprint();new.action(state);assert new.fingerprint()==before
            obs=nxt
            if term or trunc:obs,_=env.reset(seed=123)
        assert new.update_calls==300 and not hasattr(new,'other') and not hasattr(new,'buffer')
    finally:env.close()
    print('PASS: approach/recede/escape/power/far/rotation; original key prefix; 300 ordinary Bellman updates; read-only action selection.')


if __name__=='__main__':main()
