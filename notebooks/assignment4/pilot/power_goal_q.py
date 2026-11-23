"""Food-relative Q state retaining bonus-target information already given to PPO.

The earlier compressed Q states discard distant power-pill routes and distant
ghost routes. This adds a coarse context feature: direction toward power pills
when unpowered, or toward ghosts when powered. This is an observation only;
all action values are still learned from unmodified observed game rewards.
"""
import numpy as np
from goal_replay_q import GoalReplayQ


class PowerGoalQ(GoalReplayQ):
    def state(self,obs):
        anchor,key=super().state(obs)
        order=[0,1,3,2]
        available=obs[15:19][order]>.5
        if obs[12]>0:
            direction=obs[23:27][order].copy()
            direction[~available]=-1
            best=(direction==direction[available].max())&available if obs[4]<1 else np.zeros(4,dtype=bool)
        else:
            direction=obs[27:31][order].copy()
            direction[~available]=2
            best=(direction==direction[available].min())&available
        target=sum(int(v)<<i for i,v in enumerate(np.roll(best,-anchor)))
        return anchor,(*key,target)
