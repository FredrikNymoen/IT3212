"""Cache unchanged geometry only when all previous observed velocities are zero."""
import numpy as np
from fast_evaluation import FastStudyTask as OriginalFastTask

class FastStudyTask(OriginalFastTask):
    def _key(self):
        ram=self.env.env.ram
        return (ram[6:11].tobytes()+ram[12:17].tobytes()+bytes(int(ram[i]) for i in [0,19,47,56])
            +ram[59:101].tobytes()+ram[116:118].tobytes())
    def reset(self,*,seed=None,options=None):
        self._last_key=None;self._last_observation=None;self.cache_hits=0
        obs,info=super().reset(seed=seed,options=options)
        self._last_key=self._key();self._last_observation=obs.copy()
        return obs,info
    def step(self,action):
        _,reward,term,trunc,info=self.env.env.step(int(action))
        key=self._key()
        if self._last_key==key and not np.any(self._last_observation[33:43]):
            obs=self._last_observation.copy()
            obs[14]=max(0,1-self.env.env.steps/1000)
            obs[47]=min(self.env.env.score/2000,1);obs[48]=info['lives']/5
            self.cache_hits+=1
        else:obs=self.extend(self.env.observation(),info)
        self._last_key=key;self._last_observation=obs.copy()
        return obs,reward,term,trunc,info

class FastLowBitsTask(FastStudyTask):
    def extend(self,obs,info):
        result=super().extend(obs,info);result[12]=(int(self.env.env.ram[116])&63)/63
        return result
