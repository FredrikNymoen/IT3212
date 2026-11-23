"""Development variants; ordinary single-table Q and clipped on-policy PPO."""
import hashlib
from pathlib import Path
import pickle
import sys

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

ASSIGNMENT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ASSIGNMENT/'current'),str(ASSIGNMENT/'pilot')]
from environment import ExperimentTask
from tabular_q import TabularQ


class ImprovedTask(gym.Wrapper):
    """Same native task/reward, with observed nearest-ghost motion/score/lives.

    Adds four values to the shared 45-vector. No look-ahead or action masking.
    The inherited capped power signal is retained, not relabelled as a timer.
    """
    def __init__(self,setting='B'):
        super().__init__(ExperimentTask(setting))
        self.observation_space=gym.spaces.Box(-1,1,shape=(49,),dtype=np.float32)

    def extend(self,obs,info):
        ram=self.env.env.ram
        player=np.array([float(ram[10])-8,float(ram[16])+6])
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        active=np.arange(4)<=int(ram[19])
        if not ram[47]:active[:]=False
        distance=np.abs(ghosts-player).sum(axis=1);distance[~active]=np.inf
        motion=obs[33:41].reshape(4,2)[distance.argmin()] if active.any() else np.zeros(2)
        assert 'lives' in info, 'Native ALE life count missing'
        return np.r_[obs,motion,min(self.env.env.score/2000,1),info['lives']/5].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        obs,info=self.env.reset(seed=seed,options=options)
        return self.extend(obs,info),info

    def step(self,action):
        obs,reward,term,trunc,info=self.env.step(action)
        return self.extend(obs,info),reward,term,trunc,info


class ImprovedQ(TabularQ):
    """Exactly one ordinary Bellman update per real transition; no replay."""
    def __init__(self,seed,variant):
        super().__init__(seed,alpha=.1 if variant=='credit' else .15,gamma=.995)
        self.variant=variant

    def state(self,obs):
        anchor,key=super().state(obs)
        if self.variant=='credit':return anchor,key
        available=obs[15:19][[0,1,3,2]]>.5
        danger=np.searchsorted([.18,.60],obs[23:27][[0,1,3,2]])*available
        danger_code=sum(int(v)*3**i for i,v in enumerate(np.roll(danger,-anchor)))
        delta=obs[2:4]*np.array([160,210])
        closing=-float(delta@obs[45:47])/max(1.,float(np.linalg.norm(delta)))
        closing_bin=0 if obs[4]>.15 else int(np.searchsorted([-.1,.1],closing))+1
        return anchor,(*key[:2],danger_code,*key[3:],closing_bin,int(obs[47]>=.5))


class MotionScorer(nn.Module):
    latent_dim_pi=5
    latent_dim_vf=64

    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(23,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
        self.value_net=nn.Sequential(nn.Linear(49,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.register_buffer('directions',torch.tensor([[0.,0.],[0.,-1.],[1.,0.],[-1.,0.],[0.,1.]]))

    def actor_features(self,obs):
        batch=len(obs);d=self.directions;p=torch.stack((-d[:,1],d[:,0]),dim=1)
        ones=torch.ones_like(obs[:,:1])
        global_values=obs[:,[4,7,12,14,43,47,48]].unsqueeze(1).expand(-1,5,-1)
        noop=torch.zeros((batch,5),device=obs.device);noop[:,0]=1
        columns=[torch.cat((ones,obs[:,15:19]),dim=1),torch.cat((ones,obs[:,19:23]),dim=1),
            torch.cat((obs[:,23:27].max(dim=1,keepdim=True).values,obs[:,23:27]),dim=1),
            torch.cat((ones,obs[:,27:31]),dim=1),noop]
        for pair in (obs[:,41:43],obs[:,2:4],obs[:,5:7],obs[:,31:33],obs[:,45:47]):
            columns.extend([pair@d.T,pair@p.T])
        # Density observation order is UP, RIGHT, DOWN, LEFT; actions swap the last two.
        columns.append(torch.cat((torch.zeros_like(ones),obs[:,[8,9,11,10]]),dim=1))
        return torch.cat((global_values,torch.stack(columns,dim=2)),dim=2)

    def forward_actor(self,obs):return self.scorer(self.actor_features(obs)).squeeze(-1)
    def forward_critic(self,obs):return self.value_net(obs)
    def forward(self,obs):return self.forward_actor(obs),self.forward_critic(obs)


class MotionPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):self.mlp_extractor=MotionScorer().to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net=nn.Identity()
        nn.init.orthogonal_(self.mlp_extractor.scorer[-1].weight,.01)
        nn.init.zeros_(self.mlp_extractor.scorer[-1].bias)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class FrozenMotion:
    def __init__(self,model):
        self.w={k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}

    def probabilities(self,obs):
        d=np.array([[0,0],[0,-1],[1,0],[-1,0],[0,1]],dtype=np.float32)
        p=np.column_stack((-d[:,1],d[:,0]))
        columns=[np.r_[1,obs[15:19]],np.r_[1,obs[19:23]],np.r_[obs[23:27].max(),obs[23:27]],
            np.r_[1,obs[27:31]],np.array([1,0,0,0,0])]
        for pair in (obs[41:43],obs[2:4],obs[5:7],obs[31:33],obs[45:47]):columns.extend([d@pair,p@pair])
        columns.append(np.r_[0,obs[[8,9,11,10]]])
        x=np.column_stack((np.tile(obs[[4,7,12,14,43,47,48]],(5,1)),*columns)).astype(np.float32)
        for index in (0,2):x=np.tanh(x@self.w[f'mlp_extractor.scorer.{index}.weight'].T+self.w[f'mlp_extractor.scorer.{index}.bias'])
        logits=(x@self.w['mlp_extractor.scorer.4.weight'].T+self.w['mlp_extractor.scorer.4.bias']).ravel()
        prob=np.exp(logits-logits.max());return prob/prob.sum()
