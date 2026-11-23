"""Separate ordinary-Q representation candidates selected on A/B only."""
from pathlib import Path
import sys
import numpy as np

ASSIGNMENT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ASSIGNMENT/'pilot'),str(ASSIGNMENT/'improvements')]
from tabular_q import TabularQ
from agents import ImprovedTask


class ThreatQ(TabularQ):
    """Only replace binary directional danger with far/medium/near levels."""
    representation='directional_threat_three_levels'

    def state(self,obs):
        anchor,key=super().state(obs)
        order=[0,1,3,2]
        available=obs[15:19][order]>.5
        levels=np.searchsorted([.13,.36],obs[23:27][order])*available
        code=sum(int(value)*3**index for index,value in enumerate(np.roll(levels,-anchor)))
        return anchor,(*key[:2],code,*key[3:])


class PowerQ(TabularQ):
    """Only split the existing observed power proxy into zero/low/high.

    No RAM decoder change. These are proxy bins, not verified remaining seconds.
    """
    representation='power_proxy_three_levels'

    def state(self,obs):
        anchor,key=super().state(obs)
        level=0 if obs[12]<=0 else (1 if obs[12]<=16/63 else 2)
        return anchor,(*key[:3],level,*key[4:])


FACTORIES={'baseline':TabularQ,'threat':ThreatQ,'power':PowerQ}
