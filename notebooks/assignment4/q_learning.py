"""Assignment 4: ordinary tabular Q-learning, training and frozen evaluation.

Run from the repository root (see --help for train/evaluate/check):
  python notebooks/assignment4/q_learning.py train --setting B --seed 17 --output results/assignment4/q_B17_NEW
  python notebooks/assignment4/q_learning.py evaluate --selection results/assignment4/q_B17_NEW/selected.json --setting D --output results/assignment4/q_D17_NEW

This file contains the actual Q algorithm and Gymnasium training loop. The
audited native environment is shared with ppo.py. Existing report checkpoints
and their original runtime sources are preserved, not rewritten or relabelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = "assignment4-readable-agents-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_directory(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "results/assignment4"), "Use an output under results/assignment4/")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hashes():
    names = ["q_learning.py", "ppo.py", "mspacman_env.py", "current/environment.py",
             "improvements/agents.py", "pilot/tabular_q.py", "pilot/navigation_features.py",
             "pilot/maze_geometry.json", "q_motion/motion_q.py", "three_hour/candidates.py",
             "three_hour/fast_evaluation.py", "three_hour/fast_evaluation_v2.py",
             "transfer_evaluation/boundary_env.py"]
    return {(HERE / name).relative_to(ROOT).as_posix(): sha(HERE / name) for name in names}


def verify_sources(expected):
    for name, digest in expected.items():
        require(sha(ROOT / name) == digest, f"Source changed since training: {name}")


def make_environment(setting):
    """Gymnasium reset -> (obs, info); step -> (obs, reward, term, trunc, info).

    Reuse the audited native game/61-value observation. D uses the established
    evaluation-only map-exit censoring wrapper and is never used for training.
    """
    require(setting in "ABCD" and len(setting) == 1, "Choose setting A/B/C/D")
    for folder in (HERE / "three_hour", HERE / "transfer_evaluation"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    if setting == "D":
        from boundary_env import BoundaryTask
        env = BoundaryTask("D")
    else:
        from candidates import StudyTask
        env = StudyTask(setting)
    require(isinstance(env, gym.Env) and env.observation_space.shape == (61,)
            and env.action_space.n == 5, "Incompatible Gymnasium environment")
    return env


class TabularQ:
    """One table, one Bellman update per newly observed transition.

    The food-relative discrete key and world-action mapping match the reported
    selected Q configuration. There is no replay, second Q table or action mask.
    """
    ACTIONS = np.array([1, 2, 4, 3])  # Clockwise UP, RIGHT, DOWN, LEFT -> ALE ID.
    DIRECTIONS = np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]])

    def __init__(self, seed, alpha=.15, gamma=.995):
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.table = {}
        self.update_calls = 0

    def state(self, obs):
        """Coarsen common observed food, corridors, threat, motion and power."""
        order = [0, 1, 3, 2]
        available = obs[15:19][order] > .5
        costs = obs[19:23][order].copy()
        costs[~available] = 2
        distance = float(obs[7] * 320)
        delta = obs[5:7] * np.array([160 / 8, 210 / 12])
        projection = self.DIRECTIONS @ delta
        near = distance <= 12 and np.any((projection > 0) & available)
        if near:
            projection[~available] = -100
            anchor = int(np.argmax(projection))
            best = (projection == projection[anchor]) & available
        else:
            anchor = int(np.argmin(costs))
            best = (costs == costs.min()) & available
        def bits(values):
            return sum(int(value) << i for i, value in enumerate(np.roll(values, -anchor)))
        danger = (obs[23:27][order] > .36) & available
        motion = obs[41:43]
        moving = 4 if np.linalg.norm(motion) < .01 else (int(np.argmax(self.DIRECTIONS @ motion)) - anchor) % 4
        key = (bits(available), bits(best), bits(danger), int(obs[12] > 0),
               int(np.searchsorted([3., 12.], distance)), moving, int(np.max(np.abs(obs[31:33])) < .2))
        return anchor, key

    def values(self, state):
        anchor, key = state
        canonical = self.table.get(key, np.zeros(5, dtype=np.float32))
        world = np.empty(5, dtype=np.float32)
        world[0] = canonical[0]
        world[self.ACTIONS[(np.arange(4) + anchor) % 4]] = canonical[1:]
        return world

    def values_for_observation(self, obs):
        return self.values(self.state(obs))

    def action(self, state, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(5))
        values = self.values(state)
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action, reward, next_state, terminal):
        anchor, key = state
        index = 0 if action == 0 else 1 + (int(np.flatnonzero(self.ACTIONS == action)[0]) - anchor) % 4
        values = self.table.setdefault(key, np.zeros(5, dtype=np.float32))
        # Q(s,a) <- Q(s,a) + alpha * [r/100 + gamma*max Q(s',.) - Q(s,a)]
        target = float(reward) / 100
        if not terminal:
            target += self.gamma * float(self.table.get(next_state[1], np.zeros(5, dtype=np.float32)).max())
        values[index] += self.alpha * (target - float(values[index]))
        self.update_calls += 1

    def fingerprint(self):
        digest = hashlib.sha256(str((self.alpha, self.gamma, self.update_calls)).encode())
        for key, values in sorted(self.table.items()):
            digest.update(np.asarray(key, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def save(self, path):
        keys = sorted(self.table)
        np.savez_compressed(path, protocol=PROTOCOL, keys=np.asarray(keys, dtype=np.int64).reshape(-1, 7),
                            values=np.asarray([self.table[key] for key in keys], dtype=np.float32).reshape(-1, 5),
                            alpha=self.alpha, gamma=self.gamma, update_calls=self.update_calls)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            require(str(data["protocol"]) == PROTOCOL, "Incompatible Q checkpoint")
            model = cls(0, float(data["alpha"]), float(data["gamma"]))
            model.table = {tuple(map(int, key)): value.copy() for key, value in zip(data["keys"], data["values"])}
            model.update_calls = int(data["update_calls"])
        return model


def evaluate(agent, algorithm, setting, seeds, output, name, trace=False):
    """Shared frozen evaluation; parameters never update, and Q ties are seeded."""
    before = agent.fingerprint()
    env, rows = make_environment(setting), []
    traces = {key: [] for key in ["ram", "ram_after", "observation", "values", "action", "episode", "step",
                                  "raw_reward", "reward", "score_after", "terminated", "truncated", "censored"]}
    start = perf_counter()
    try:
        for episode, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed) + 3000000)
            bcd = lambda v: int(v) // 16 * 10 + int(v) % 16
            native_score = lambda ram: bcd(ram[122]) * 10000 + bcd(ram[121]) * 100 + bcd(ram[120])
            initial = native_score(env.unwrapped.ale.getRAM())
            for step in range(1000):
                values = agent.values_for_observation(obs)
                if algorithm == "Q-learning":
                    # Match the recorded Q action() RNG sequence without changing training RNG.
                    rng.random()
                    action = int(rng.choice(np.flatnonzero(values == values.max())))
                else:
                    action = int(values.argmax())
                ram = env.unwrapped.ale.getRAM().copy()
                nxt, reward, terminated, truncated, info = env.step(action)
                after = env.unwrapped.ale.getRAM().copy()
                require(native_score(after) - initial == info["score"], "Native score disagreement")
                censored = bool(info.get("censored", False))
                if trace:
                    record = dict(ram=ram, ram_after=after, observation=obs.copy(), values=values.copy(),
                                  action=action, episode=episode, step=step, raw_reward=info["raw_reward"],
                                  reward=reward, score_after=info["score"], terminated=terminated,
                                  truncated=truncated, censored=censored)
                    for key, value in record.items():
                        traces[key].append(value)
                obs = nxt
                if terminated or truncated:
                    rows.append(dict(evaluation_seed=int(seed), success=int(info["success"]),
                                     known_failure=int(not info["success"] and not censored), censored=int(censored),
                                     score=info.get("observed_prefix_score", info["score"]),
                                     task_return=info.get("observed_prefix_return", info["task_return"]),
                                     length=info["episode_steps"], reason=info["reason"]))
                    break
            else:
                raise RuntimeError("Gymnasium episode did not stop at the task horizon")
    finally:
        env.close()
    seconds = perf_counter() - start
    require(agent.fingerprint() == before, "Evaluation modified learned parameters")
    require(len(rows) == len(seeds), "Incomplete evaluation")
    write_csv(output / f"{name}.csv", rows)
    if trace:
        np.savez_compressed(output / f"{name}_traces.npz", **{key: np.asarray(value) for key, value in traces.items()})
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    def length(key):
        values = [row["length"] for row in rows if row[key]]
        return float(np.mean(values)) if values else None
    return dict(episodes=len(rows), success_rate=mean("success"), failure_rate=mean("known_failure"),
                censored_rate=mean("censored"), success_upper_bound=mean("success") + mean("censored"),
                mean_score=mean("score"), mean_return=mean("task_return"),
                success_length=length("success"), failure_length=length("known_failure"),
                evaluation_interactions=sum(row["length"] for row in rows), evaluation_seconds=seconds,
                frozen_parameters_verified=True, further_training_interactions=0)


def checkpoint_points(budget):
    return sorted({0, budget, *range(65536, budget + 1, 65536)})


def lock_selection(output, algorithm, setting, seed, budget, curves, sources):
    selected = max((row for row in curves if row["interactions"] > 0),
                   key=lambda row: (row["success_rate"], row["mean_score"], -row["interactions"]))
    result = dict(protocol=PROTOCOL, algorithm=algorithm, training_setting=setting, training_seed=seed,
                  consumed_training_interactions=budget, checkpoint_interactions=selected["interactions"],
                  model=selected["model"], model_sha256=selected["model_sha256"], source_hashes=sources,
                  selection="Source-setting development success, then raw score, then earlier checkpoint; no final/D outcomes.")
    write_json(output / "selected.json", result)
    return result


def training_config(args, algorithm, output):
    require(args.budget > 0 and args.budget % 2048 == 0, "Use a positive multiple of the common 2,048-transition rollout")
    require(args.development_episodes > 0, "Development episodes must be positive")
    sources = source_hashes()
    write_json(output / "configuration.json", dict(protocol=PROTOCOL, algorithm=algorithm, arguments=vars(args),
               source_hashes=sources, python=platform.python_version(), hardware=platform.processor(), os=platform.platform(),
               packages={name: version(name) for name in ["numpy", "gymnasium", "ale-py", "torch", "stable-baselines3"]},
               scope="Readable selected-configuration rerun. Not a new tuning search or independent validation of inspected seeds."))
    for name in sources:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return sources


def train(args):
    output = new_directory(args.output)
    sources = training_config(args, "Q-learning", output)
    model, env = TabularQ(args.seed), make_environment(args.setting)
    curves, training_rows, seconds = [], [], 0.0
    points = checkpoint_points(args.budget)
    def checkpoint(point):
        path = output / f"checkpoint_{point}.npz"
        model.save(path)
        metrics = evaluate(model, "Q-learning", args.setting, range(1130000, 1130000 + args.development_episodes),
                           output, f"development_{point}")
        curves.append(dict(interactions=point, model=path.name, model_sha256=sha(path), training_seconds=seconds, **metrics))
        write_csv(output / "curves.csv", curves)
        print(f"Q {args.setting} seed {args.seed}: {point:,} interactions; development success {metrics['success_rate']:.1%}", flush=True)
    try:
        obs, _ = env.reset(seed=args.seed * 1000)
        episode = 0
        checkpoint(0)
        clock = perf_counter()
        for step in range(args.budget):
            state = model.state(obs)
            epsilon = max(.05, 1 - .95 * step / (.8 * args.budget))
            action = model.action(state, epsilon)
            nxt, reward, terminated, truncated, info = env.step(action)
            model.update(state, action, reward, model.state(nxt), terminated or truncated)
            obs = nxt
            if terminated or truncated:
                training_rows.append(dict(interactions=step + 1, score=info["score"],
                                          success=int(info["success"]), length=info["episode_steps"]))
                episode += 1
                obs, _ = env.reset(seed=args.seed * 1000 + episode)
            if step + 1 in points:
                seconds += perf_counter() - clock
                checkpoint(step + 1)
                clock = perf_counter()
        require(model.update_calls == args.budget, "One real transition must cause exactly one Q update")
    finally:
        env.close()
    write_csv(output / "training.csv", training_rows)
    selected = lock_selection(output, "Q-learning", args.setting, args.seed, args.budget, curves, sources)
    write_json(output / "completed.json", dict(training_interactions=args.budget, update_calls=model.update_calls,
               training_seconds=seconds, alpha=model.alpha, gamma=model.gamma, selected=selected))


def evaluate_selection(args, algorithm, loader):
    selection_path = Path(args.selection).resolve()
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selected["protocol"] == PROTOCOL and selected["algorithm"] == algorithm, "Incompatible selection")
    require((args.setting == "D" and selected["training_setting"] == "B")
            or args.setting == selected["training_setting"], "Evaluate on the training condition, or frozen B-to-D")
    require(args.episodes >= 200, "Primary evaluation requires at least 200 episodes per trained model")
    verify_sources(selected["source_hashes"])
    model_path = (selection_path.parent / selected["model"]).resolve()
    require(model_path.is_relative_to(selection_path.parent), "Model path outside the selected run")
    require(sha(model_path) == selected["model_sha256"], "Selected checkpoint changed")
    output = new_directory(args.output)
    write_json(output / "locked.json", dict(selected, condition=args.setting, episodes=args.episodes,
               policy="Greedy Q with seeded uniform ties; PPO argmax", seed_start=1250000 if args.setting == "D" else 1150000))
    model = loader(model_path)
    learned_interactions = (model.update_calls if algorithm == "Q-learning" else model.model.num_timesteps)
    require(learned_interactions == selected["checkpoint_interactions"], "Checkpoint training count disagrees with selection")
    start = 1250000 if args.setting == "D" else 1150000
    metrics = evaluate(model, algorithm, args.setting, range(start, start + args.episodes), output, "evaluation", trace=True)
    require(sha(model_path) == selected["model_sha256"], "Evaluation changed the saved model")
    write_json(output / "metrics.json", dict(metrics, selected=selected))
    print(json.dumps(metrics, indent=2))


def parser(algorithm):
    cli = argparse.ArgumentParser(description=__doc__ if algorithm == "Q-learning" else "PPO: selected actor-critic, training and frozen evaluation")
    sub = cli.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="Train one selected A/B/C configuration; use the same seed list for both algorithms")
    fit.add_argument("--setting", choices=list("ABC"), default="B")
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--budget", type=int, default=262144)
    fit.add_argument("--development-episodes", type=int, default=10)
    fit.add_argument("--output", required=True)
    test = sub.add_parser("evaluate", help="Evaluate a development-selected checkpoint without learning")
    test.add_argument("--selection", required=True)
    test.add_argument("--setting", choices=list("ABCD"), required=True)
    test.add_argument("--episodes", type=int, default=200)
    test.add_argument("--output", required=True)
    sub.add_parser("check", help="Check equivalence with the original implementation, without a full experiment")
    return cli


def check():
    # Verify real-transition keys, action RNG, values and updates against the
    # untouched implementation used by the report, not just an equation example.
    sys.path.insert(0, str(HERE / "pilot"))
    from tabular_q import TabularQ as OriginalQ
    current, original = TabularQ(17), OriginalQ(17, alpha=.15, gamma=.995)
    env = make_environment("B")
    try:
        obs, _ = env.reset(seed=17000)
        for index in range(1024):
            state = current.state(obs)
            require(state == original.state(obs), "State representation differs")
            action = current.action(state, .3)
            require(action == original.action(state, .3), "Seeded actions differ")
            nxt, reward, term, trunc, _ = env.step(action)
            next_state = current.state(nxt)
            for agent in (current, original):
                agent.update(state, action, reward, next_state, term or trunc)
            require(np.array_equal(current.values(state), original.values(state)), "Bellman update differs")
            obs = env.reset(seed=17001 + index)[0] if term or trunc else nxt
        require(current.update_calls == original.update_calls == 1024, "Incorrect update count")
        before = current.fingerprint()
        current.values((0, (-1,) * 7))
        require(current.fingerprint() == before, "Read-only lookup modified the table")
    finally:
        env.close()
    print("Q parity passed: 1,024 native transitions; identical keys, seeded actions and ordinary Bellman updates.")


if __name__ == "__main__":
    args = parser("Q-learning").parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_selection(args, "Q-learning", TabularQ.load)
    else:
        check()
