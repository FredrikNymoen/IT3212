"""Ordinary single-table Q-learning for the original Part 1 assumptions.

One observed Gymnasium transition produces exactly one Bellman update. There is
no second estimate, replay buffer, eligibility trace or neural Q function. The
compact state uses the same observed geometry/motion information available to
PPO. A food-relative coordinate system shares entries across rotated local
situations; it never forces an action or gives an action a preferred initial Q.
"""
import hashlib
import pickle

import numpy as np


class TabularQ:
    ACTIONS=np.array([1,2,4,3])  # Clockwise UP, RIGHT, DOWN, LEFT -> native ID.
    DIRECTIONS=np.array([[0.,-1.],[1.,0.],[0.,1.],[-1.,0.]])
    representation='single_goal'

    def __init__(self,seed,alpha=.15,gamma=.99):
        self.rng=np.random.default_rng(seed)
        self.alpha=float(alpha)
        self.gamma=float(gamma)
        self.table={}
        self.update_calls=0

    def state(self,obs):
        """Compact bins of position context, nearby ghosts, food and power status.

        All inputs are from the common 45-value observation: corridor flags,
        food offsets/path directions, ghost proximity, coarse power status,
        motion and position relative to the nearest maze node. Magnitudes and
        detailed coordinates are deliberately coarsened for the table.
        """
        order=[0,1,3,2]
        available=obs[15:19][order]>.5
        costs=obs[19:23][order].copy()
        costs[~available]=2
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
        danger=(obs[23:27][order]>.36)&available
        motion=obs[41:43]
        moving=4 if np.linalg.norm(motion)<.01 else (int(np.argmax(self.DIRECTIONS@motion))-anchor)%4
        key=(bits(available),bits(best),bits(danger),int(obs[12]>0),
            int(np.searchsorted([3.,12.],distance)),moving,int(np.max(np.abs(obs[31:33]))<.2))
        return anchor,key

    def values(self,state):
        anchor,key=state
        canonical=self.table.get(key,np.zeros(5,dtype=np.float32))
        world=np.empty(5,dtype=np.float32)
        world[0]=canonical[0]
        world[self.ACTIONS[(np.arange(4)+anchor)%4]]=canonical[1:]
        return world

    def action(self,state,epsilon=0.):
        if self.rng.random()<epsilon:return int(self.rng.integers(5))
        values=self.values(state)
        return int(self.rng.choice(np.flatnonzero(values==values.max())))

    def update(self,state,action,reward,next_state,terminal):
        anchor,key=state
        action_index=0 if action==0 else 1+(int(np.flatnonzero(self.ACTIONS==action)[0])-anchor)%4
        values=self.table.setdefault(key,np.zeros(5,dtype=np.float32))
        next_values=self.table.get(next_state[1],np.zeros(5,dtype=np.float32))
        # Exactly the ordinary off-policy Q-learning target, in scaled points.
        target=float(reward)/100
        if not terminal:target+=self.gamma*float(next_values.max())
        values[action_index]+=self.alpha*(target-float(values[action_index]))
        self.update_calls+=1

    def fingerprint(self):
        return hashlib.sha256(pickle.dumps((self.table,self.update_calls))).hexdigest()
