"""Double Q table in a food-relative coordinate frame, learned from replay.

Near food, use its observed offset to orient the coordinates; farther away use
the observed shortest food route. Rotating a state never prescribes an action:
all five values start equally and the world-action mapping is bijective.
Unlike the earlier canonical pilot, this avoids an arbitrary reference direction
when every adjacent graph edge ties because the closest pellet is at this node.
"""
import numpy as np
from replay_double_q import ReplayDoubleQ


class GoalReplayQ(ReplayDoubleQ):
    ACTIONS=np.array([1,2,4,3])
    DIRECTIONS=np.array([[0.,-1.],[1.,0.],[0.,1.],[-1.,0.]])

    def state(self,obs):
        reorder=[0,1,3,2]
        available=obs[15:19][reorder]>.5
        costs=obs[19:23][reorder].copy();costs[~available]=2
        distance=float(obs[7]*320)
        delta=obs[5:7]*np.array([160/8,210/12])
        projection=self.DIRECTIONS@delta
        near=distance<=12 and np.any((projection>0)&available)
        if near:
            projection[~available]=-100
            anchor=int(np.argmax(projection))
            best=(projection==projection[anchor])&available
        else:
            anchor=int(np.argmin(costs))
            best=(costs==costs.min())&available
        bits=lambda values:sum(int(v)<<i for i,v in enumerate(np.roll(values,-anchor)))
        danger=(obs[23:27][reorder]>.36)&available
        motion=obs[41:43]
        moving=4 if np.linalg.norm(motion)<.01 else (int(np.argmax(self.DIRECTIONS@motion))-anchor)%4
        key=(bits(available),bits(best),bits(danger),int(obs[12]>0),
            int(np.searchsorted([3.,12.],distance)),moving,int(np.max(np.abs(obs[31:33]))<.2))
        return anchor,key

    def values(self,state):
        canonical=super().values(state[1])
        world=np.empty(5,dtype=np.float32);world[0]=canonical[0]
        world[self.ACTIONS[(np.arange(4)+state[0])%4]]=canonical[1:]
        return world

    def update(self,state,action,reward,nxt,terminal):
        ca=0 if action==0 else 1+(int(np.flatnonzero(self.ACTIONS==action)[0])-state[0])%4
        super().update(state[1],ca,reward,nxt[1],terminal)
