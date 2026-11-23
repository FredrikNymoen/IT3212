"""Frozen NumPy actor inference, checked against the saved SB3 actor logits."""
import numpy as np


class FrozenActor:
    def __init__(self,model):
        self.weights={k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}
        self.shared='mlp_extractor.scorer.0.weight' in self.weights

    def logits(self,obs):
        w=self.weights
        if not self.shared:
            x=obs
            for index in (0,2):x=np.tanh(x@w[f'mlp_extractor.policy_net.{index}.weight'].T+w[f'mlp_extractor.policy_net.{index}.bias'])
            return x@w['action_net.weight'].T+w['action_net.bias']
        d=np.array([[0,0],[0,-1],[1,0],[-1,0],[0,1]],dtype=np.float32)
        p=np.column_stack((-d[:,1],d[:,0]))
        values=[np.r_[1,obs[15:19]],np.r_[1,obs[19:23]],np.r_[obs[23:27].max(),obs[23:27]],
            np.r_[1,obs[27:31]],np.array([1,0,0,0,0],dtype=np.float32)]
        for pair in (obs[41:43],obs[2:4],obs[5:7],obs[31:33]):values.extend([d@pair,p@pair])
        x=np.column_stack((np.tile(obs[[4,7,12,14,43]],(5,1)),*values)).astype(np.float32)
        for index in (0,2):x=np.tanh(x@w[f'mlp_extractor.scorer.{index}.weight'].T+w[f'mlp_extractor.scorer.{index}.bias'])
        return (x@w['mlp_extractor.scorer.4.weight'].T+w['mlp_extractor.scorer.4.bias']).ravel()

    def probabilities(self,obs):
        logits=self.logits(obs);p=np.exp(logits-logits.max());return p/p.sum()

    def action(self,obs,rng,deterministic=True):
        p=self.probabilities(obs)
        return int(np.argmax(p)) if deterministic else int(rng.choice(5,p=p.astype(float)/p.sum(dtype=float)))
