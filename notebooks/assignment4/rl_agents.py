"""Tabular off-policy Q-learning and on-policy clipped actor-critic PPO.

PPO follows the clipped objective in Schulman et al. (2017), arXiv:1707.06347.
The implementations share fixed score scaling (1/100); evaluation exports use
unscaled game points and task rewards. NumPy inference uses a frozen copy of
the current Torch network to avoid one-sample Torch overhead on the CPU.
"""
import copy
import hashlib
import pickle

import numpy as np
import torch
from torch import nn

from mspacman_env import coarse_state, FEATURES


class QLearning:
    def __init__(self, seed, learning_rate=.1, gamma=.99, **unused):
        self.rng = np.random.default_rng(seed)
        self.learning_rate, self.gamma = learning_rate, gamma
        self.table = {}

    def scores(self, obs):
        return self.table.get(coarse_state(obs), np.zeros(5, dtype=np.float32)).copy()

    def action(self, obs, explore=0):
        scores = self.scores(obs)
        if self.rng.random() < explore:
            return int(self.rng.integers(5))
        tied = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(tied))

    def update(self, obs, action, reward, next_obs, terminated):
        state = coarse_state(obs)
        values = self.table.setdefault(state, np.zeros(5, dtype=np.float32))
        target = reward/100 + self.gamma * (not terminated) * float(self.scores(next_obs).max())
        values[action] += self.learning_rate * (target-values[action])

    def fingerprint(self):
        return hashlib.sha256(pickle.dumps(self.table)).hexdigest()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 5), nn.Linear(64, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1)

    def forward(self, obs):
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class PPO:
    def __init__(self, seed, learning_rate=3e-4, gamma=.99, **unused):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.network = Network()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.refresh()

    def refresh(self):
        self.weights = {name: value.detach().cpu().numpy().copy()
                        for name, value in self.network.state_dict().items()}

    def distribution(self, obs):
        w = self.weights
        x = np.tanh(obs @ w['body.0.weight'].T + w['body.0.bias'])
        x = np.tanh(x @ w['body.2.weight'].T + w['body.2.bias'])
        logits = x @ w['actor.weight'].T + w['actor.bias']
        value = float((x @ w['critic.weight'].T + w['critic.bias']).item())
        probs = np.exp(logits - np.max(logits))
        return probs/probs.sum(), value

    def scores(self, obs):
        return self.distribution(obs)[0]

    def action(self, obs, explore=0):
        probs = self.scores(obs)
        return int(self.rng.choice(5, p=probs)) if explore else int(np.argmax(probs))

    def update(self, rollout):
        obs, actions, logp, values, rewards, next_values, terminated, done = map(np.asarray, zip(*rollout))
        advantage = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        for t in range(len(rewards)-1, -1, -1):
            delta = rewards[t]/100 + self.gamma*next_values[t]*(1-terminated[t])-values[t]
            last = delta + self.gamma*.95*(1-done[t])*last
            advantage[t] = last
        returns = advantage + values
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_logp = torch.tensor(logp, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = torch.tensor(advantage)
        rows = []
        for epoch in range(4):
            permutation = self.rng.permutation(len(rollout))
            for start in range(0, len(rollout), 128):
                idx = permutation[start:start+128]
                logits, predicted = self.network(obs[idx])
                distribution = torch.distributions.Categorical(logits=logits)
                ratio = (distribution.log_prob(actions[idx])-old_logp[idx]).exp()
                policy = -torch.minimum(ratio*advantage[idx], ratio.clamp(.8, 1.2)*advantage[idx]).mean()
                value = .5*(returns[idx]-predicted).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy + .5*value - .01*entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5)
                self.optimizer.step()
                rows.append((float(policy.detach()), float(value.detach()), float(entropy.detach())))
        self.refresh()
        return np.mean(rows, axis=0).tolist()

    def fingerprint(self):
        return hashlib.sha256(b''.join(value.tobytes() for value in self.weights.values())).hexdigest()


def make_agent(algorithm, seed, parameters):
    return (QLearning if algorithm == 'Q-learning' else PPO)(seed, **parameters)
