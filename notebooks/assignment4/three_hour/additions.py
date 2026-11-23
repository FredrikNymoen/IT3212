"""Separate fifth candidates; never modify the already registered screen."""
import numpy as np
from candidates import StudyTask,TabularQ


class AlignmentQ(TabularQ):
    """Retain signed food-relative corridor offset instead of one alignment bit."""
    representation='ordinary_q_relative_alignment_3x3'
    def state(self,obs):
        anchor,key=super().state(obs)
        forward=self.DIRECTIONS[anchor];side=np.array([-forward[1],forward[0]])
        offset=obs[31:33]
        code=int(np.searchsorted([-.2,.2],float(offset@forward)))*3+int(np.searchsorted([-.2,.2],float(offset@side)))
        return anchor,(*key[:-1],code)


class LowBitsTask(StudyTask):
    """Experimental low-six-bit power proxy, not a validated exact timer."""
    def extend(self,obs,info):
        result=super().extend(obs,info)
        result[12]=(int(self.env.env.ram[116])&63)/63
        return result
