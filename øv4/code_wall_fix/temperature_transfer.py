"""Test whether PPO benefits from sampling more broadly during a score drought."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cached_navigation import CACHED_TASKS
from refine_transfer import comparison, feature_type
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, runner, write_json


START = 982350
BASE = .20
METHODS = ("fixed", "score_drought")


class ScoreDroughtPolicy(TemperaturePolicy):
    def act(self, obs, progress=1., training=False, sampled=True):
        step = int(round((1. - float(obs[8])) * runner.PROTOCOL["max_decisions"]))
        score, lives = float(obs[7]), float(obs[9])
        if step == 0 or lives != self.lives or score > self.score:
            self.last_progress = step
        self.score, self.lives = score, lives
        delay = max(0, step - self.last_progress - 60)
        self.temperature = BASE + .30 * min(delay / 80., 1.)
        return super().act(obs, progress, training, sampled)


def validate(root, row, method, layout, count):
    source = Path(row["model"])
    if digest(source) != row["model_sha256"]:
        raise ValueError(f"Checkpoint contents changed: {source}")
    setting = "B"
    task = CACHED_TASKS[row.get("feature_type", feature_type(row["arm"]))]

    class FixedLayoutTask(task):
        def __init__(self, protocol, setting):
            super().__init__(protocol, setting)
            self.s = dict(self.s, maze=layout)

    policy = ScoreDroughtPolicy if method == "score_drought" else TemperaturePolicy
    agent = policy(runner.load_model(source), BASE)
    path = root / method / f"seed_{row['seed']}" / f"maze_{layout}"
    path.mkdir(parents=True)
    runner.MsPacManTask = FixedLayoutTask
    try:
        result = runner.evaluate(agent, setting, count, seed_start=START, log_dir=path, label="eval")
    finally:
        runner.MsPacManTask = NavigationTask
    print(f"{method}, seed {row['seed']}, maze {layout}: {result['success_rate']:.1%}", flush=True)
    return dict(arm=method, layout=layout, seed=row["seed"], **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-arm", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 150:
        parser.error("Use 1–150 validation episodes.")
    source = json.loads(args.parent.read_text(encoding="utf-8"))
    rows = [r for r in source["runs"] if r["arm"] == args.parent_arm]
    if not rows:
        parser.error("Choose a completed parent arm.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    layouts = (2, 3)
    write_json(root / "manifest.json", dict(
        parent=str(args.parent.resolve()), input_arm=args.parent_arm, input_rows=rows,
        code_sha256={Path(__file__).name: digest(Path(__file__))},
        episodes=args.episodes, seed_start=START, layouts=layouts,
        protocol=runner.PROTOCOL, base_temperature=BASE, drought_patience=60,
        ramp_decisions=80, maximum_temperature=.50))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(validate, root, row, method, layout, args.episodes)
                for row in rows for method in METHODS for layout in layouts]
        scores = [j.result() for j in as_completed(jobs)]
    means = comparison(scores, layouts, ("success_rate", "mean_score"), lambda r, _: r, True)
    write_json(root / "comparison.json", dict(means=means, runs=scores))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

