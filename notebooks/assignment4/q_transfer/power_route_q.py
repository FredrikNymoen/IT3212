"""Ordinary Q retaining an already observed power-pellet route in danger."""
import numpy as np
from transfer_q import TabularQ


class PowerRouteQ(TabularQ):
    representation='original_plus_power_route_when_threatened'

    def state(self,obs):
        anchor,key=super().state(obs);mask=0
        if key[2]!=0 and key[3]==0:
            order=[0,1,3,2]
            available=obs[15:19][order]>.5
            costs=obs[27:31][order].copy();costs[~available]=2
            best=(costs==costs.min())&available
            mask=sum(int(value)<<i for i,value in enumerate(np.roll(best,-anchor)))
        return anchor,(*key,mask)
