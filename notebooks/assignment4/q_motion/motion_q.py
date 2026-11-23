"""One-factor ordinary-Q variant: one conditional relative-closing bit.

No second table, replay, traces, action masks, changed rewards or look-ahead.
The 49-vector is the already audited common observed feature vector.
"""
from pathlib import Path
import sys

import numpy as np

ASSIGNMENT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ASSIGNMENT/'pilot'),str(ASSIGNMENT/'improvements')]
from tabular_q import TabularQ
from agents import ImprovedTask

NEAR_PIXELS=48.
CLOSING_PIXELS_PER_DECISION=.8


def closing_bit(obs):
    """Observed relative velocity, not a prediction of the next transition.

    Position deltas use pixel units; ghost/player velocities are both stored
    divided by 8. Flag only nearby, unpowered encounters. The inherited power
    signal remains a proxy; this experiment does not claim to repair its timer.
    """
    if obs[4]*320>NEAR_PIXELS or obs[12]>0:return 0
    delta=obs[2:4]*np.array([160.,210.])
    norm=float(np.linalg.norm(delta))
    if norm<1:return 0
    relative_velocity=(obs[45:47]-obs[41:43])*8
    closing=-float(delta@relative_velocity)/norm
    return int(closing>CLOSING_PIXELS_PER_DECISION)


class MotionQ(TabularQ):
    representation='original_plus_relative_closing_bit'

    def state(self,obs):
        anchor,key=super().state(obs)
        return anchor,(*key,closing_bit(obs))
