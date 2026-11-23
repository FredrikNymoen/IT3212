"""Train PPO on a mixture of normal episodes and saved late-game states."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import improve_ppo_session as search
import late_transfer as late
import refine_transfer as refine
import transfer_ppo as transfer
from cached_navigation import CachedNavigationTask
from score_diagnostics import TemperaturePolicy, digest, np, runner, write_json, write_rows


CHECKPOINT_FIELDS = ("walls", "pellets", "_maze_key", "graph", "_pellet_nodes",
                     "last_action", "previous", "previous_lives", "start_lives",
                     "lives", "score", "t", "frightened")
BRANCH_THRESHOLDS = (1150, 1500, 1750)
REPLAY_PROBABILITY = .45


@dataclass
class Snapshot:
    maze: int
    emulator: object
    task_fields: dict
    features: np.ndarray

    @classmethod
    def save(cls, task, features, include_rng=False):
        emulator = task.env.unwrapped.ale.cloneState(include_rng)
        fields = {name: getattr(task, name) for name in CHECKPOINT_FIELDS}
        return cls(task.s["maze"], emulator, fields, features.copy())

    def resume(self, task, seed):
        task.s = dict(task.s, maze=self.maze)
        task.env.reset(seed=int(seed))
        ale = task.env.unwrapped.ale
        ale.restoreState(self.emulator)
        for name, value in self.task_fields.items():
            setattr(task, name, value)
        task.cached_pellets = None
        task.layout_graphs[self.maze] = task._maze_key, task.graph, task._pellet_nodes
        ram = ale.getRAM()
        if int(ram[0]) != self.maze or int(ale.lives()) != self.task_fields["lives"]:
            raise RuntimeError("The emulator restored a different maze or life count.")
        return task.features(ram)


def next_maze(counts):
    return min(refine.WEIGHTS, key=lambda maze: counts[maze] / refine.WEIGHTS[maze])


def collect(parent, seed, budget):
    bank = {maze: [] for maze in refine.WEIGHTS}
    counts = dict.fromkeys(refine.WEIGHTS, 0)
    task = CachedNavigationTask(runner.PROTOCOL, "B")
    random = np.random.default_rng(seed)
    agent = TemperaturePolicy(runner.load_model(parent), .25)
    agent.rng = np.random.default_rng(seed + 31)
    try:
        while task.decisions < budget:
            maze = next_maze(counts)
            task.s = dict(task.s, maze=maze)
            start = task.decisions
            obs = task.reset(int(random.integers(1_000_000, 2**31 - 1)),
                             min(runner.START_DECISIONS, budget - start))
            saved = set()
            while task.decisions < budget:
                obs, _, done, _ = task.step(agent.act(obs))
                if done:
                    break
                crossed = {threshold for threshold in BRANCH_THRESHOLDS if task.score >= threshold}
                if crossed - saved and task.score < 1950:
                    bank[maze].append(Snapshot.save(task, obs))
                    saved.update(crossed)
            counts[maze] += task.decisions - start
    finally:
        task.close()
    if sum(counts.values()) != budget or min(map(len, bank.values())) < 3:
        raise RuntimeError(f"Unexpected replay collection: {counts}, {[len(s) for s in bank.values()]}")
    return bank, counts


def training_task(bank, exposure, arm):
    class ReplayTask(CachedNavigationTask):
        def __init__(self, protocol, setting):
            super().__init__(protocol, setting)
            self.layout_interactions = dict.fromkeys(refine.WEIGHTS, 0)

        def reset(self, seed, startup_steps=runner.START_DECISIONS, after_startup_step=None):
            if exposure:
                previous = exposure[-1]
                self.layout_interactions[previous["maze"]] += self.decisions - previous["interaction_start"]
            maze = next_maze(self.layout_interactions)
            self.s = dict(self.s, maze=maze)
            random = np.random.default_rng(np.random.SeedSequence([int(seed), 1407]))
            replay = bool(exposure) and random.random() < REPLAY_PROBABILITY
            snapshot = bank[maze][int(random.integers(len(bank[maze])))] if replay else None
            exposure.append(dict(seed=seed, maze=maze, interaction_start=self.decisions,
                                 startup_interactions=0 if replay else startup_steps,
                                 replay=replay, initial_score=snapshot.task_fields["score"] if replay else 0,
                                 initial_lives=snapshot.task_fields["lives"] if replay else None))
            if snapshot is not None:
                return snapshot.resume(self, seed)
            return super().reset(seed, startup_steps, after_startup_step)

        def step(self, action):
            previous_score, previous_lives = self.score, self.lives
            obs, reward, done, info = super().step(action)
            if arm == "replay_survival":
                lost = max(0, previous_lives - info["lives"])
                reward = late.reward_for("survival", previous_score, reward, info["raw_reward"],
                                         lost, self.p["success_score"])
            return obs, reward, done, info

    return ReplayTask


def fit(root, row, arm, total_budget, collection_budget, count):
    seed = row["seed"]
    parent = Path(row["model"])
    if digest(parent) != row["model_sha256"]:
        raise ValueError(f"Checkpoint changed: {parent}")
    folder = root / arm / f"seed_{seed}"
    folder.mkdir(parents=True)
    bank, collection = collect(parent, 49000 + seed, collection_budget)
    probe = CachedNavigationTask(runner.PROTOCOL, "B")
    try:
        for layout in bank:
            snap = bank[layout][0]
            np.testing.assert_array_equal(snap.resume(probe, 49100 + seed), snap.features)
    finally:
        probe.close()
    print(f"{arm} seed {seed} collected {[len(bank[m]) for m in bank]} states", flush=True)
    update_budget = total_budget - collection_budget
    exposure = []
    search.NavigationTask = training_task(bank, exposure, arm)
    try:
        model = search.fit(parent, 31000 + seed, folder, update_budget,
                           late.LR, late.TEMPERATURE, late.KL, transfer.ROLLOUT)
    finally:
        search.NavigationTask = CachedNavigationTask
    counts, replay_counts = [dict.fromkeys(refine.WEIGHTS, 0) for _ in range(2)]
    for index, row_exposure in enumerate(exposure):
        end = exposure[index + 1]["interaction_start"] if index + 1 < len(exposure) else update_budget
        interactions = end - row_exposure["interaction_start"]
        row_exposure["interactions"] = interactions
        maze = row_exposure["maze"]
        counts[maze] += interactions
        if row_exposure["replay"]:
            replay_counts[maze] += interactions
    assert sum(counts.values()) == update_budget and set(counts) == {0, 2, 3}
    write_rows(folder / "layout_exposure.csv", exposure)
    result = dict(arm=arm, seed=seed, model=str(model), model_sha256=digest(model),
                  parent=str(parent), parent_sha256=row["model_sha256"], feature_type="navigation",
                  interactions={maze: n + collection[maze] for maze, n in counts.items()},
                  update_interactions=counts, collection_interactions=collection,
                  replay_interactions=replay_counts, cached_states={m: len(bank[m]) for m in bank})
    print(f"{arm} seed {seed} update interactions {counts}, replay {replay_counts}", flush=True)
    return late.validate(root, result, count, late.VALIDATION_START)


def selfcheck():
    left = CachedNavigationTask(runner.PROTOCOL, "B")
    right = CachedNavigationTask(runner.PROTOCOL, "B")
    random = np.random.default_rng(1254)
    try:
        for maze in refine.WEIGHTS:
            left.s = dict(left.s, maze=maze)
            left.reset(327 + maze)
            for _ in range(60):
                obs, _, done, _ = left.step(int(random.integers(left.n_actions)))
                assert not done
            snap = Snapshot.save(left, obs, include_rng=True)
            np.testing.assert_array_equal(snap.resume(right, 527 + maze), obs)
            for _ in range(70):
                action = int(random.integers(left.n_actions))
                x, y = left.step(action), right.step(action)
                np.testing.assert_array_equal(x[0], y[0])
                assert x[1:] == y[1:]
                if x[2]:
                    break
        print("Replay restoration matched observations, scores, lives and outcomes on all training layouts.")
    finally:
        left.close()
        right.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--arms", nargs="+", choices=["replay", "replay_survival"], default=["replay"])
    parser.add_argument("--budget", type=int, default=240_000)
    parser.add_argument("--collect", type=int, default=35_000)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.parent is None or args.results is None or not 10_000 <= args.collect < args.budget - 1000:
        parser.error("Specify the parent and output with a collection budget inside the total budget.")
    parents = [r for r in json.loads(args.parent.read_text(encoding="utf-8"))["runs"]
               if r["arm"] == late.BASELINE]
    if len(parents) != 3 or any(set(r["interactions"]) != {"0", "2", "3"} for r in parents):
        parser.error("Expected three checkpoint parents trained on layouts 0, 2 and 3.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    referenced = ("replay_transfer.py", "late_transfer.py", "refine_transfer.py",
                  "transfer_ppo.py", "improve_ppo_session.py", "cached_navigation.py",
                  "score_diagnostics.py", "part2_run.py", "mspacman.py")
    write_json(root / "manifest.json", dict(
        parent_study=str(args.parent.resolve()), parents=parents, total_budget=args.budget,
        collection_budget=args.collect, episodes=args.episodes, arms=args.arms,
        replay_probability=REPLAY_PROBABILITY, thresholds=BRANCH_THRESHOLDS,
        training_layouts=[0, 2, 3], validation_seed=late.VALIDATION_START,
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in referenced}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fit, root, row, arm, args.budget, args.collect, args.episodes)
                for arm in args.arms for row in parents]
        rows = [job.result() for job in as_completed(jobs)]
    means = refine.comparison(rows, (0, 2, 3), ("success_rate", "mean_score"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=rows, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

