"""Ordinary Q retaining two observed task-resource bits omitted by the base key."""
from transfer_q import TabularQ


class ResourceQ(TabularQ):
    representation='original_plus_last_life_and_near_goal'

    def state(self,obs):
        anchor,key=super().state(obs)
        lives=int(round(float(obs[48])*5))
        return anchor,(*key,int(lives<=1),int(obs[47]>=.75))
