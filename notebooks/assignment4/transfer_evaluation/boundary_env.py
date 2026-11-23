"""Evaluation-only Gymnasium wrapper: stop and censor any map exit.

Native transitions/rewards and all pre-boundary observations are unchanged.
No policy action is taken on another map. A map exit is not labelled failure.
The crossing action's reward is excluded from the observed in-map prefix;
its exact native reward/RAM are still recorded for independent verification.
"""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'three_hour'))
from fast_evaluation import RawTask
from fast_evaluation_v2 import FastStudyTask


class BoundaryRawTask(RawTask):
    def step(self,action):
        if self.finished:raise RuntimeError('Call reset() after a terminal episode.')
        ram,raw_reward,terminal,truncated,info=self.env.step(int(action))
        self.steps+=1;self.score+=float(raw_reward);self.ram=ram.copy()
        reward=float(raw_reward)-(self.config.step_penalty if self.config.setting=='C' else 0)
        self.episode_return+=reward
        map_exit=int(ram[0])!=self.config.maze
        success=not map_exit and self.score>=self.config.target_score
        timeout=self.steps>=self.config.horizon and not success
        censored=bool(map_exit or (truncated and not success))
        terminated=bool(not map_exit and (terminal or success or timeout))
        stopped=bool(censored or terminated)
        self.finished=stopped
        reason=('map_exit' if map_exit else 'success' if success else
            'time_limit' if timeout else 'game_over' if terminal else 'external_limit')
        info=dict(info,score=self.score,raw_reward=float(raw_reward),success=bool(success),
            episode_steps=self.steps,reason=reason,task_return=self.episode_return,
            maze=int(ram[0]),map_exit=bool(map_exit),censored=censored,
            known_failure=bool(stopped and not success and not censored),
            observed_prefix_score=self.score-(float(raw_reward) if map_exit else 0),
            observed_prefix_return=self.episode_return-(reward if map_exit else 0),
            terminal_observation_valid=not map_exit)
        info['emulator_frames']=int(info['episode_frame_number'])-self.last_frame
        self.last_frame=int(info['episode_frame_number'])
        return self.ram.copy(),reward,terminated,censored,info


class BoundaryTask(FastStudyTask):
    """Same 61-value actor input until censoring; never used for training."""
    def __init__(self,setting='D'):
        super().__init__(setting)
        previous=self.env.env
        self.env.env=BoundaryRawTask(previous.config,previous.env)

    def step(self,action):
        _,reward,term,trunc,info=self.env.env.step(int(action))
        if info['map_exit']:
            # An explicit terminal placeholder, never passed to the actor.
            obs=self._last_observation.copy()
            obs[14]=max(0,1-self.env.env.steps/1000)
            obs[47]=min(self.env.env.score/2000,1);obs[48]=info['lives']/5
        else:
            key=self._key()
            if self._last_key==key and not np.any(self._last_observation[33:43]):
                obs=self._last_observation.copy()
                obs[14]=max(0,1-self.env.env.steps/1000)
                obs[47]=min(self.env.env.score/2000,1);obs[48]=info['lives']/5
                self.cache_hits+=1
            else:obs=self.extend(self.env.observation(),info)
            self._last_key=key
        self._last_observation=obs.copy()
        return obs,reward,term,trunc,info
