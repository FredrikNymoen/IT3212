"""Fresh, bounded candidates. Native Gymnasium task, ordinary one-table Q/PPO.

Common 61-value observation: prior 49 plus all four ghost offsets and activity.
No teacher, action mask, reward shaping, replay, traces or Double Q learning.
"""
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'improvements'),str(HERE.parent/'q_motion'),str(HERE.parent/'pilot')]
from agents import ImprovedTask
from tabular_q import TabularQ
from motion_q import MotionQ


class StudyTask(ImprovedTask):
    def __init__(self,setting='B'):
        super().__init__(setting)
        self.observation_space=gym.spaces.Box(-1,1,shape=(61,),dtype=np.float32)

    def extend(self,obs,info):
        original=super().extend(obs,info)
        ram=self.env.env.ram
        player=np.array([float(ram[10])-8,float(ram[16])+6])
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        active=(np.arange(4)<=int(ram[19])) & bool(ram[47])
        delta=np.clip((ghosts-player)/[160,210],-1,1)
        delta[~active]=0
        return np.r_[original,delta.ravel(),active].astype(np.float32)


class BearingQ(TabularQ):
    """Replace one danger-mask component by coarse nearest-ghost geometry."""
    representation='ordinary_q_nearest_ghost_bearing'
    def state(self,obs):
        anchor,key=super().state(obs)
        distance=float(obs[4]*320)
        code=0
        if distance<=64:
            dx,dy=obs[2:4]*[160,210]
            sector=int(np.floor(np.arctan2(dx,-dy)/(np.pi/4)+.5))%8
            code=1+(sector-2*anchor)%8+8*int(distance>24)
        return anchor,(*key[:2],code,*key[3:])


Q_VARIANTS=('baseline','motion','gamma995','bearing')
PPO_VARIANTS=('stable','long','allghost','careful')

def make_q(variant,seed):
    if variant=='motion': return MotionQ(seed)
    if variant=='bearing': return BearingQ(seed)
    assert variant in ('baseline','gamma995')
    return TabularQ(seed,alpha=.15,gamma=.995 if variant=='gamma995' else .99)


class StudyScorer(nn.Module):
    latent_dim_pi=5
    latent_dim_vf=64
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.scorer=nn.Sequential(nn.Linear(43 if variant=='allghost' else 23,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
        self.value_net=nn.Sequential(nn.Linear(61 if variant=='allghost' else 49,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
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
        columns.append(torch.cat((torch.zeros_like(ones),obs[:,[8,9,11,10]]),dim=1))
        if self.variant=='allghost':
            for ghost in range(4):
                active=obs[:,57+ghost:58+ghost]
                position=obs[:,49+2*ghost:51+2*ghost]
                motion=obs[:,33+2*ghost:35+2*ghost]*active
                columns.extend([position@d.T,position@p.T,motion@d.T,motion@p.T,active.expand(-1,5)])
        return torch.cat((global_values,torch.stack(columns,dim=2)),dim=2)

    def forward_actor(self,obs):return self.scorer(self.actor_features(obs)).squeeze(-1)
    def forward_critic(self,obs):return self.value_net(obs if self.variant=='allghost' else obs[:,:49])
    def forward(self,obs):return self.forward_actor(obs),self.forward_critic(obs)


class StudyPolicy(ActorCriticPolicy):
    def __init__(self,*args,actor_variant='stable',**kwargs):
        self.actor_variant=actor_variant
        super().__init__(*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=StudyScorer(self.actor_variant).to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule);self.action_net=nn.Identity()
        nn.init.orthogonal_(self.mlp_extractor.scorer[-1].weight,.01)
        nn.init.zeros_(self.mlp_extractor.scorer[-1].bias)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class FrozenStudy:
    def __init__(self,model):
        self.variant=model.policy.actor_variant
        self.w={k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}
    def probabilities(self,obs):
        d=np.array([[0,0],[0,-1],[1,0],[-1,0],[0,1]],dtype=np.float32)
        p=np.column_stack((-d[:,1],d[:,0]))
        columns=[np.r_[1,obs[15:19]],np.r_[1,obs[19:23]],np.r_[obs[23:27].max(),obs[23:27]],
            np.r_[1,obs[27:31]],np.array([1,0,0,0,0])]
        for pair in (obs[41:43],obs[2:4],obs[5:7],obs[31:33],obs[45:47]):columns.extend([d@pair,p@pair])
        columns.append(np.r_[0,obs[[8,9,11,10]]])
        if self.variant=='allghost':
            for ghost in range(4):
                active=obs[57+ghost];position=obs[49+2*ghost:51+2*ghost];motion=obs[33+2*ghost:35+2*ghost]*active
                columns.extend([d@position,p@position,d@motion,p@motion,np.full(5,active)])
        x=np.column_stack((np.tile(obs[[4,7,12,14,43,47,48]],(5,1)),*columns)).astype(np.float32)
        for index in (0,2):x=np.tanh(x@self.w[f'mlp_extractor.scorer.{index}.weight'].T+self.w[f'mlp_extractor.scorer.{index}.bias'])
        logits=(x@self.w['mlp_extractor.scorer.4.weight'].T+self.w['mlp_extractor.scorer.4.bias']).ravel()
        prob=np.exp(logits-logits.max());return prob/prob.sum()


def ppo_settings(variant):
    assert variant in PPO_VARIANTS
    return dict(gamma=.999 if variant=='long' else .995,gae_lambda=.97,
        n_epochs=4 if variant=='careful' else 6,clip_range=.1 if variant=='careful' else .15,
        target_kl=.01 if variant=='careful' else .015,
        lr_start=.0002 if variant=='careful' else .0003,lr_end=.00005,
        entropy_start=.02,entropy_end=.002)
