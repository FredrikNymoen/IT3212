"""PPO policy with learned shared action scoring in a direction-relative frame.

Rotating an identical local situation shares the actor's weights. This is a
network inductive bias, not a scripted policy: all logits are learned from
on-policy returns and all five task actions remain possible. Critic sees the
complete corrected observation. No preferred-action targets or demonstrations.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class ActionScorer(nn.Module):
    latent_dim_pi=5
    latent_dim_vf=64

    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(18,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
        self.value_net=nn.Sequential(nn.Linear(45,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        # Native actions NOOP, UP, RIGHT, LEFT, DOWN.
        self.register_buffer('directions',torch.tensor([[0.,0.],[0.,-1.],[1.,0.],[-1.,0.],[0.,1.]]))

    def forward_actor(self,obs):
        batch=obs.shape[0];d=self.directions
        perpendicular=torch.stack((-d[:,1],d[:,0]),dim=1)
        project=lambda values,directions:values@directions.T
        global_values=obs[:,[4,7,12,14,43]].unsqueeze(1).expand(-1,5,-1)
        # Per-action state measurements; NOOP gets its own indicator.
        available=torch.cat((torch.ones_like(obs[:,:1]),obs[:,15:19]),dim=1)
        food=torch.cat((torch.ones_like(obs[:,:1]),obs[:,19:23]),dim=1)
        threat=torch.cat((obs[:,23:27].max(dim=1,keepdim=True).values,obs[:,23:27]),dim=1)
        power=torch.cat((torch.ones_like(obs[:,:1]),obs[:,27:31]),dim=1)
        is_noop=torch.zeros((batch,5),device=obs.device);is_noop[:,0]=1
        columns=[available,food,threat,power,is_noop,
            project(obs[:,41:43],d),project(obs[:,41:43],perpendicular),
            project(obs[:,2:4],d),project(obs[:,2:4],perpendicular),
            project(obs[:,5:7],d),project(obs[:,5:7],perpendicular),
            project(obs[:,31:33],d),project(obs[:,31:33],perpendicular)]
        features=torch.cat((global_values,torch.stack(columns,dim=2)),dim=2)
        return self.scorer(features).squeeze(-1)

    def forward_critic(self,obs):return self.value_net(obs)

    def forward(self,obs):return self.forward_actor(obs),self.forward_critic(obs)


class SharedActionPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):self.mlp_extractor=ActionScorer().to(self.device)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        # Preserve the five shared action scores directly as categorical logits.
        self.action_net=nn.Identity()
        nn.init.orthogonal_(self.mlp_extractor.scorer[-1].weight,.01)
        nn.init.zeros_(self.mlp_extractor.scorer[-1].bias)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
