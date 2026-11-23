"""Train a late-game PPO policy while retaining the parent's opening decisions."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import late_transfer as late
import refine_transfer as refine
import replay_transfer as replay
from cached_navigation import CACHED_TASKS, CachedNavigationTask
from score_diagnostics import (TemperaturePolicy, digest, np, read_rows, runner,
                               write_json)
from transfer_ghost_ppo import GHOST_FEATURES
from transfer_power_ppo import POWER_FEATURES


SWITCH_SCORE = 1500
VALIDATION_START = 981600
ARMS = ("specialist", "specialist_ghost")
BASE_SNAPSHOT = replay.Snapshot


class PhasePolicy(TemperaturePolicy):
    """Keep the opening checkpoint until the raw score reaches SWITCH_SCORE."""

    def __init__(self, opening, student, temperature, switch_score=SWITCH_SCORE):
        super().__init__(student, temperature)
        self.opening = TemperaturePolicy(opening, temperature)
        self.opening_width = opening.p["W1"].shape[0]
        self.threshold = np.float32(switch_score / runner.PROTOCOL["success_score"])

    def policy_value(self, obs):
        if obs[7] < self.threshold:
            return self.opening.policy_value(obs[:self.opening_width])
        return super().policy_value(obs)


class GhostSnapshot(BASE_SNAPSHOT):
    @classmethod
    def save(cls, task, features, include_rng=False):
        snap = super().save(task, features, include_rng)
        snap.task_fields["available_power"] = snap.task_fields["available_power"].copy()
        return snap

    def resume(self, task, seed):
        fields = dict(self.task_fields, available_power=self.task_fields["available_power"].copy())
        return BASE_SNAPSHOT(self.maze, self.emulator, fields, self.features).resume(task, seed)


def make_policy(row, temperature):
    model = Path(row["model"])
    if digest(model) != row["model_sha256"]:
        raise ValueError(f"Checkpoint changed: {model}")
    student = runner.load_model(model)
    if "opening_model" not in row:
        return TemperaturePolicy(student, temperature)
    opening = Path(row["opening_model"])
    if digest(opening) != row["opening_sha256"]:
        raise ValueError(f"Opening policy changed: {opening}")
    return PhasePolicy(runner.load_model(opening), student, temperature, row["switch_score"])


def evaluate(root, row, layout, count, start, label="test", temperature=.2):
    directory = root / row["arm"] / f"seed_{row['seed']}" / f"{label}_maze_{layout}"
    directory.mkdir(parents=True, exist_ok=False)
    condition = "D" if layout == 1 else "B"
    base = CACHED_TASKS[row["feature_type"]]

    class EvaluationTask(base):
        def __init__(self, protocol, setting):
            super().__init__(protocol, setting)
            self.s = dict(self.s, maze=layout)

    runner.MsPacManTask = EvaluationTask
    try:
        result = runner.evaluate(make_policy(row, temperature), condition, count,
                                 seed_start=start, log_dir=directory, label="eval", trace_count=1)
    finally:
        runner.MsPacManTask = CachedNavigationTask
    records = read_rows(directory / f"eval_{condition}_episodes.csv")
    reached = sum(float(r["raw_score"]) >= SWITCH_SCORE for r in records)
    successes = sum(r["outcome"] == "success" for r in records)
    late_deaths = sum(float(r["raw_score"]) >= SWITCH_SCORE and r["outcome"] == "failure"
                      for r in records)
    result.update(arm=row["arm"], seed=row["seed"], layout=layout, episodes=count,
                  seed_start=start, reached_1500=reached / count,
                  conversion=successes / reached if reached else 0.,
                  late_life_loss=late_deaths / count)
    print(f"{label} {row['arm']} seed {row['seed']}, maze {layout}: "
          f"{result['success_rate']:.1%} success, {result['conversion']:.1%} conversion", flush=True)
    return result


def fit(root, row, arm, total, collection, count):
    opening = Path(row["model"])
    if digest(opening) != row["model_sha256"]:
        raise ValueError(f"Opening policy changed: {opening}")
    learner = dict(row)
    ghost = arm == "specialist_ghost"
    if ghost:
        student = runner.load_model(opening)
        student.p["W1"] = np.pad(student.p["W1"], ((0, POWER_FEATURES + GHOST_FEATURES), (0, 0)))
        expanded = root / "expanded_parents" / f"seed_{row['seed']}.npz"
        runner.save_model(student, expanded)
        learner.update(model=str(expanded), model_sha256=digest(expanded))

    previous = (replay.REPLAY_PROBABILITY, replay.BRANCH_THRESHOLDS, replay.CachedNavigationTask,
                replay.CHECKPOINT_FIELDS, replay.Snapshot, late.validate)

    def validate(root, result, count, unused_start):
        saved = dict(result, feature_type="ghost" if ghost else "navigation",
                     opening_model=str(opening), opening_sha256=row["model_sha256"],
                     switch_score=SWITCH_SCORE)
        saved["validation"] = {layout: evaluate(root, saved, layout, count, VALIDATION_START, "validation")
                               for layout in (0, 2, 3)}
        write_json(root / arm / f"seed_{row['seed']}" / "result.json", saved)
        return saved

    try:
        replay.REPLAY_PROBABILITY = 1.
        replay.BRANCH_THRESHOLDS = (1500, 1700, 1850)
        if ghost:
            replay.CachedNavigationTask = CACHED_TASKS["ghost"]
            replay.CHECKPOINT_FIELDS += ("available_power", "power_nodes")
            replay.Snapshot = GhostSnapshot
        late.validate = validate
        return replay.fit(root, learner, arm, total, collection, count)
    finally:
        (replay.REPLAY_PROBABILITY, replay.BRANCH_THRESHOLDS, replay.CachedNavigationTask,
         replay.CHECKPOINT_FIELDS, replay.Snapshot, late.validate) = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=320_000)
    parser.add_argument("--collect", type=int, default=60_000)
    parser.add_argument("--episodes", type=int, default=85)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 10_000 <= args.collect < args.budget - 1000:
        parser.error("Use a collection budget inside the total budget.")
    parents = [r for r in json.loads(args.parent.read_text(encoding="utf-8"))["runs"]
               if r["arm"] == late.BASELINE]
    if len(parents) != 3 or len({r["seed"] for r in parents}) != 3 or any(
            set(r["interactions"]) != {"0", "2", "3"} for r in parents):
        parser.error("Expected parents from three training seeds and layouts 0, 2 and 3.")
    root = args.results.resolve()
    (root / "expanded_parents").mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "late_transfer.py", "replay_transfer.py", "refine_transfer.py",
             "part2_run.py", "mspacman.py", "cached_navigation.py", "transfer_power_ppo.py",
             "transfer_ghost_ppo.py", "compare_navigation.py", "reference_policy.py",
             "score_diagnostics.py", "improve_ppo_session.py", "transfer_ppo.py")
    write_json(root / "manifest.json", dict(
        parents=parents, budget=args.budget, collection_budget=args.collect, episodes=args.episodes,
        source=str(args.parent.resolve()), switch=SWITCH_SCORE, validation_seed=VALIDATION_START,
        protocol=runner.PROTOCOL, training_layouts=[0, 2, 3],
        code_sha256={p: digest(Path(__file__).with_name(p)) for p in names}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fit, root, row, arm, args.budget, args.collect, args.episodes)
                for arm in ARMS for row in parents]
        runs = [job.result() for job in as_completed(jobs)]
    means = refine.comparison(runs, (0, 2, 3), ("success_rate", "mean_score", "reached_1500", "conversion"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

