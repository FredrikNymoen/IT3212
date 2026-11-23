"""A genuine Q-table shared across rotations of the local navigation state.

The reference direction is a currently shortest food route. This only rotates
the coordinates in which the full five-action table is learned; it does not
choose the behavior action or initialize any action favorably. Noop, reverse
and wall-facing actions can all be selected by learned Q-values/exploration.
"""
import numpy as np
from navigation_q import NavigationQ


class CanonicalQ(NavigationQ):
    # Cyclic clockwise direction index -> native action number.
    ACTIONS=np.array([1,2,4,3])

    def state(self,obs):
        reorder=[0,1,3,2]
        available=obs[15:19][reorder]>.5
        costs=obs[19:23][reorder].copy();costs[~available]=2
        anchor=int(np.argmin(costs))
        directions=np.array([[0,-1],[1,0],[0,1],[-1,0]],dtype=float)
        forward=directions[anchor];right=np.array([-forward[1],forward[0]])
        bits=lambda values:sum(int(v)<<i for i,v in enumerate(np.roll(values,-anchor)))
        best=(costs==costs.min())&available
        danger=(obs[23:27][reorder]>.36)&available
        offset=np.array([obs[31:33]@right,-obs[31:33]@forward])
        motion=obs[41:43]
        movement=4 if np.linalg.norm(motion)<.01 else (int(np.argmax(directions@motion))-anchor)%4
        key=(bits(available),bits(best),bits(danger),int(obs[12]>0),movement,
             *np.digitize(offset,[-.2,.2]).tolist())
        return (anchor,key)

    def canonical_values(self,state):return self.table.get(state[1],np.zeros(5,dtype=np.float32))

    def values(self,state):
        canonical=self.canonical_values(state)
        result=np.empty(5,dtype=np.float32);result[0]=canonical[0]
        result[self.ACTIONS[(np.arange(4)+state[0])%4]]=canonical[1:]
        return result

    def update(self,state,action,reward,nxt,terminal):
        assert self.trace==0,'Only one-step Q-learning is supported here.'
        ca=0 if action==0 else 1+(int(np.flatnonzero(self.ACTIONS==action)[0])-state[0])%4
        values=self.table.setdefault(state[1],np.zeros(5,dtype=np.float32))
        target=reward/100+self.gamma*(not terminal)*float(self.canonical_values(nxt).max())
        values[ca]+=self.alpha*(target-values[ca])
        self.counts[(state[1],ca)]=self.counts.get((state[1],ca),0)+1
