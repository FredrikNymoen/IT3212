"""Equivalent outer observations, avoiding discarded inner feature computation.

All game interaction remains native Gymnasium reset/step. No policy, reward,
termination, action mask or observation value is changed. Check equivalence
against the frozen wrapper before using this only for evaluation.
"""
import gymnasium as gym
import numpy as np
from candidates import StudyTask
from mspacman_env import MsPacmanTask


class RawTask(MsPacmanTask):
    def __init__(self,config,env):
        super().__init__(config=config,env=env)
        self.observation_space=gym.spaces.Box(0,255,shape=(128,),dtype=np.uint8)

    def reset(self,*,seed=None,options=None):
        _,info=self.env.reset(seed=seed,options=options)
        self.unwrapped.ale.setRAM(0,self.config.maze)
        self.ram=self.unwrapped.ale.getRAM().copy()
        self.steps,self.score,self.episode_return=0,0.,0.
        self.last_frame=int(info['episode_frame_number']);self.finished=False
        return self.ram.copy(),dict(info)

    def step(self,action):
        if self.finished:raise RuntimeError('Call reset() after a terminal episode.')
        ram,raw_reward,terminal,truncated,info=self.env.step(int(action))
        self.steps+=1;self.score+=float(raw_reward)
        success=self.score>=self.config.target_score
        timeout=self.steps>=self.config.horizon and not success
        if int(ram[0])!=self.config.maze:
            raise RuntimeError('Natural level transition crossed the map holdout. Revisit task design.')
        self.ram=ram.copy()
        reward=float(raw_reward)-(self.config.step_penalty if self.config.setting=='C' else 0)
        self.episode_return+=reward
        self.finished=bool(terminal or truncated or success or timeout)
        reason='success' if success else ('time_limit' if timeout else ('game_over' if terminal else 'external_limit'))
        info=dict(info,score=self.score,raw_reward=float(raw_reward),success=bool(success),
            episode_steps=self.steps,reason=reason,task_return=self.episode_return,maze=int(ram[0]))
        info['emulator_frames']=int(info['episode_frame_number'])-self.last_frame
        self.last_frame=int(info['episode_frame_number'])
        return self.ram.copy(),reward,bool(terminal or success or timeout),bool(truncated),info


class FastStudyTask(StudyTask):
    def __init__(self,setting='B'):
        super().__init__(setting)
        old=self.env.env
        self._unused_legacy_wrapper=old
        self.env.env=RawTask(old.config,old.env)

    def extend(self,obs,info):
        ram=self.env.env.ram
        dx=ram[6:10].astype(float)-float(ram[10]);dy=ram[12:16].astype(float)-float(ram[16])
        active=(np.arange(4)<=int(ram[19]))&bool(ram[47])
        distances=np.abs(dx)+np.abs(dy);distances[~active]=np.inf
        result=np.empty(61,dtype=np.float32);result[:45]=obs
        result[45:47]=obs[33:41].reshape(4,2)[distances.argmin()] if active.any() else 0
        result[47]=min(self.env.env.score/2000,1);result[48]=info['lives']/5
        offsets=result[49:57].reshape(4,2)
        offsets[:,0]=np.clip(dx/160,-1,1);offsets[:,1]=np.clip(dy/210,-1,1);offsets[~active]=0
        result[57:61]=active
        return result


class FastLowBitsTask(FastStudyTask):
    def extend(self,obs,info):
        result=super().extend(obs,info);result[12]=(int(self.env.env.ram[116])&63)/63
        return result
