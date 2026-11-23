"""Check both Q changes retain original targets and all other key components."""
import numpy as np
from transfer_q import TabularQ,ThreatQ,PowerQ,ImprovedTask


def main():
    assert ThreatQ.update is PowerQ.update is TabularQ.update
    obs=np.zeros(49,dtype=np.float32);obs[15:19]=1;obs[5]=.1;obs[7]=.1
    base=TabularQ(42);threat=ThreatQ(42);power=PowerQ(42)
    old=base.state(obs);new=threat.state(obs)
    assert new[0]==old[0] and new[1][:2]==old[1][:2] and new[1][3:]==old[1][3:]
    obs[23:27]=[.1,.2,.4,.8]
    anchor,key=threat.state(obs)
    expected=np.roll(np.array([0,1,2,2])[[0,1,3,2]],-anchor)
    assert key[2]==sum(int(v)*3**i for i,v in enumerate(expected))
    for value,level in [(0,0),(8/63,1),(16/63,1),(17/63,2),(1,2)]:
        obs[12]=value;old=base.state(obs);new=power.state(obs)
        assert new[0]==old[0] and new[1][:3]==old[1][:3] and new[1][4:]==old[1][4:]
        assert new[1][3]==level
    env=ImprovedTask('B');rng=np.random.default_rng(19)
    try:
        obs,_=env.reset(seed=123)
        for _ in range(100):
            for agent in (threat,power):
                state=agent.state(obs);target=(0,('test-target',));agent.table[target[1]]=np.arange(1,6,dtype=np.float32)
                old=float(agent.values(state)[0]);expected=old+.15*(.1+.99*5-old)
                agent.update(state,0,10,target,False)
                assert np.isclose(agent.values(state)[0],expected)
                old=float(agent.values(state)[0]);agent.update(state,0,10,target,True)
                assert np.isclose(agent.values(state)[0],old+.15*(.1-old))
            obs,_,term,trunc,_=env.step(int(rng.integers(5)))
            if term or trunc:obs,_=env.reset(seed=123)
    finally:env.close()
    print('PASS: only the declared key component changes; original one-step Q update; terminal/nonterminal targets.')


if __name__=='__main__':main()
