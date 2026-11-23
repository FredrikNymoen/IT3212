"""A smaller representation for each algorithm; native task is unchanged."""
from torch import nn
from candidates import TabularQ,StudyPolicy,StudyScorer

class CompactQ(TabularQ):
    representation='ordinary_q_compact_motion_alignment'
    def state(self,obs):
        anchor,key=super().state(obs)
        return anchor,(*key[:5],int(key[5]==4))

class SmallPolicy(StudyPolicy):
    def _build_mlp_extractor(self):
        scorer=StudyScorer(self.actor_variant)
        scorer.scorer=nn.Sequential(nn.Linear(23,16),nn.Tanh(),nn.Linear(16,16),nn.Tanh(),nn.Linear(16,1))
        scorer.value_net=nn.Sequential(nn.Linear(49,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        scorer.latent_dim_vf=32
        self.mlp_extractor=scorer.to(self.device)
