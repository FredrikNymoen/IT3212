"""Assignment 4: selected PPO actor/critic, training and frozen evaluation.

  python notebooks/assignment4/ppo.py train --setting B --seed 17 --output results/assignment4/ppo_B17_NEW
  python notebooks/assignment4/ppo.py evaluate --selection results/assignment4/ppo_B17_NEW/selected.json --setting D --output results/assignment4/ppo_D17_NEW

The actor, critic, reward scale, optimiser settings, entropy schedule and
training loop are defined here. Stable-Baselines3 performs PPO's on-policy
rollout collection, GAE and clipped surrogate/value updates. The Gymnasium
environment and common evaluation/logging helpers are shared with q_learning.py.
"""
from __future__ import annotations

import hashlib
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from q_learning import (HERE, require, make_environment, new_directory, training_config,
                        checkpoint_points, evaluate, lock_selection, evaluate_selection,
                        parser, sha, write_csv, write_json)

# Give saved SB3 policies a stable module path when this file is run as a script.
if __name__ == "__main__":
    sys.modules.setdefault("ppo", sys.modules[__name__])


class ActionScorer(nn.Module):
    """Shared 23-feature action scorer and separate 49-input value network.

    Each of the five available actions is expressed in relative coordinates
    and scored by the same two-layer network. No action is forced or masked.
    The selected policy uses the first 49 of the common 61 observed values.
    """
    latent_dim_pi = 5
    latent_dim_vf = 64

    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(23, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.value_net = nn.Sequential(nn.Linear(49, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.register_buffer("directions", torch.tensor([[0., 0.], [0., -1.], [1., 0.], [-1., 0.], [0., 1.]]))

    def actor_features(self, obs):
        direction = self.directions
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        ones = torch.ones_like(obs[:, :1])
        global_values = obs[:, [4, 7, 12, 14, 43, 47, 48]].unsqueeze(1).expand(-1, 5, -1)
        noop = torch.zeros((len(obs), 5), device=obs.device)
        noop[:, 0] = 1
        columns = [torch.cat((ones, obs[:, 15:19]), dim=1),
                   torch.cat((ones, obs[:, 19:23]), dim=1),
                   torch.cat((obs[:, 23:27].max(dim=1, keepdim=True).values, obs[:, 23:27]), dim=1),
                   torch.cat((ones, obs[:, 27:31]), dim=1), noop]
        for pair in (obs[:, 41:43], obs[:, 2:4], obs[:, 5:7], obs[:, 31:33], obs[:, 45:47]):
            columns.extend([pair @ direction.T, pair @ perpendicular.T])
        columns.append(torch.cat((torch.zeros_like(ones), obs[:, [8, 9, 11, 10]]), dim=1))
        return torch.cat((global_values, torch.stack(columns, dim=2)), dim=2)

    def forward_actor(self, obs):
        return self.scorer(self.actor_features(obs)).squeeze(-1)

    def forward_critic(self, obs):
        return self.value_net(obs[:, :49])

    def forward(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)


class SharedActionPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = ActionScorer().to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        nn.init.orthogonal_(self.mlp_extractor.scorer[-1].weight, .01)
        nn.init.zeros_(self.mlp_extractor.scorer[-1].bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


if __name__ == "__main__":
    ActionScorer.__module__ = "ppo"
    SharedActionPolicy.__module__ = "ppo"


class FrozenPolicy:
    """Read-only highest-probability evaluation, with learned-weight fingerprint."""
    def __init__(self, model):
        self.model = model

    def values_for_observation(self, obs):
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(torch.as_tensor(obs[None], device=self.model.device))
            return distribution.distribution.probs.cpu().numpy()[0].copy()

    def fingerprint(self):
        return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                      for value in self.model.policy.state_dict().values())).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(PPO.load(path, device="cpu"))


class RewardScale(gym.RewardWrapper):
    def reward(self, reward):
        return reward / 100


class Checkpoints(BaseCallback):
    def __init__(self, budget, points, save):
        super().__init__()
        self.budget, self.points, self.save = budget, set(points), save

    def _on_rollout_start(self):
        # Save/evaluate at rollout boundaries, after the previous PPO update.
        if self.num_timesteps in self.points:
            self.save(self.num_timesteps)

    def _on_step(self):
        self.model.ent_coef = .002 + .018 * max(0, 1 - self.num_timesteps / self.budget)
        return True


def build_model(envs, seed, **overrides):
    """Reported stable configuration; all settings stay fixed across A/B/C."""
    settings = dict(learning_rate=lambda progress: .00005 + (.0003 - .00005) * max(0, progress),
                    gamma=.995, gae_lambda=.97, ent_coef=.02, n_steps=512, batch_size=256,
                    n_epochs=6, clip_range=.15, target_kl=.015, seed=seed, device="cpu", verbose=0)
    settings.update(overrides)
    return PPO(SharedActionPolicy, envs, **settings)


def train(args):
    torch.set_num_threads(1)
    output = new_directory(args.output)
    sources = training_config(args, "PPO", output)
    envs = DummyVecEnv([lambda i=i: Monitor(RewardScale(make_environment(args.setting)), str(output / f"training_{i}"),
                           info_keywords=("score", "success", "reason", "episode_steps")) for i in range(4)])
    curves, seconds = [], 0.0
    try:
        model = build_model(envs, args.seed)
        clock = perf_counter()
        def checkpoint(point):
            nonlocal clock, seconds
            seconds += perf_counter() - clock
            require(model.num_timesteps == point, "Saved PPO transition count differs")
            path = output / f"checkpoint_{point}.zip"
            model.save(path)
            metrics = evaluate(FrozenPolicy(model), "PPO", args.setting,
                               range(1130000, 1130000 + args.development_episodes), output, f"development_{point}")
            curves.append(dict(interactions=point, model=path.name, model_sha256=sha(path), training_seconds=seconds, **metrics))
            write_csv(output / "curves.csv", curves)
            print(f"PPO {args.setting} seed {args.seed}: {point:,} interactions; development success {metrics['success_rate']:.1%}", flush=True)
            clock = perf_counter()
        # SB3 collects 4 environments x 512 steps = 2,048 real interactions.
        # It reuses only the current rollout for <=6 clipped-policy update epochs.
        points = checkpoint_points(args.budget)
        model.learn(args.budget, callback=Checkpoints(args.budget, [p for p in points if p < args.budget], checkpoint))
        checkpoint(args.budget)
        require(model.num_timesteps == args.budget, "PPO exceeded the declared interaction budget")
        selected = lock_selection(output, "PPO", args.setting, args.seed, args.budget, curves, sources)
        write_json(output / "completed.json", dict(training_interactions=model.num_timesteps, training_seconds=seconds,
                   sb3_n_updates=int(model._n_updates), selected=selected))
    finally:
        envs.close()


def check():
    torch.set_num_threads(1)
    sys.path.insert(0, str(HERE / "three_hour"))
    from candidates import StudyScorer, StudyPolicy
    env = make_environment("B")
    try:
        obs, _ = env.reset(seed=17000)
        rows = []
        for index in range(256):
            rows.append(obs.copy())
            obs, _, term, trunc, _ = env.step(index % 5)
            if term or trunc:
                obs, _ = env.reset(seed=17001 + index)
        inputs = torch.as_tensor(np.asarray(rows))
        torch.manual_seed(17)
        original = StudyScorer("stable")
        current = ActionScorer()
        current.load_state_dict(original.state_dict(), strict=True)
        with torch.no_grad():
            require(torch.equal(current.actor_features(inputs), original.actor_features(inputs)), "Actor inputs differ")
            require(torch.equal(current.forward_actor(inputs), original.forward_actor(inputs)), "Actor scores differ")
            require(torch.equal(current.forward_critic(inputs), original.forward_critic(inputs)), "Critic values differ")
        # Initialization must match too: same seed and complete SB3 policy build.
        torch.manual_seed(17)
        original_policy = StudyPolicy(env.observation_space, env.action_space, lambda _: .0003, actor_variant="stable")
        torch.manual_seed(17)
        policy = SharedActionPolicy(env.observation_space, env.action_space, lambda _: .0003)
        require(all(torch.equal(value, policy.state_dict()[key]) for key, value in original_policy.state_dict().items()),
                "Policy initialization differs from the reported architecture")
    finally:
        env.close()
    print("PPO parity passed: 256 native observations; identical features, actor/critic outputs and seeded policy initialization.")


if __name__ == "__main__":
    args = parser("PPO").parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_selection(args, "PPO", FrozenPolicy.load)
    else:
        check()
