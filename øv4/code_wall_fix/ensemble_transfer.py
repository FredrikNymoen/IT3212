"""Evaluate a probability ensemble of the three PPO seeds selected by training-layout validation."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cached_navigation import CACHED_TASKS
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, np, runner, write_json
from wrapped_transfer import WrappedNavigationTask


class EnsemblePolicy(TemperaturePolicy):
    def __init__(self, members, temperature):
        super().__init__(members[0], temperature)
        self.members = [TemperaturePolicy(member, temperature) for member in members]

    def policy_value(self, obs):
        estimates = [member.policy_value(obs) for member in self.members]
        probabilities = np.mean([p for p, _ in estimates], axis=0)
        value = float(np.mean([v for _, v in estimates]))
        return probabilities, value


def evaluate(root, rows, layout, count, start, protocol, temperature):
    models = []
    for row in rows:
        file = Path(row["model"])
        if digest(file) != row["model_sha256"]:
            raise ValueError(f"Checkpoint contents changed: {file}")
        models.append(runner.load_model(file))
    kind = rows[0]["feature_type"]
    base = WrappedNavigationTask if kind == "wrapped" else CACHED_TASKS[kind]

    class EvaluationTask(base):
        def __init__(self, protocol, setting):
            super().__init__(protocol, setting)
            self.s = dict(self.s, maze=layout)

    agent = EnsemblePolicy(models, temperature)
    output = root / f"maze_{layout}"
    output.mkdir()
    runner.MsPacManTask = EvaluationTask
    try:
        result = runner.evaluate(agent, "D" if layout == 1 else "B", count, protocol=protocol,
                                 seed_start=start, log_dir=output, label="ensemble", trace_count=1)
    finally:
        runner.MsPacManTask = NavigationTask
    print(f"ensemble maze {layout}: {result['success_rate']:.1%}, "
          f"{result['mean_score']:.0f} points", flush=True)
    return dict(layout=layout, episodes=count, seed_start=start, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="ensemble validation study required for D")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 500:
        parser.error("Use 1–500 episodes.")
    plan = json.loads(args.frozen.read_text(encoding="utf-8"))
    selected = plan["ordered_candidates"][0]
    source = plan["sources"][selected]
    parent_summary = Path(source["results"])
    if digest(parent_summary) != source["sha256"]:
        parser.error("The validation results differ from the frozen plan.")
    parent_manifest = json.loads(parent_summary.with_name("manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in plan["runs"] if r["arm"] == selected]
    if len(rows) != 3 or len({r["feature_type"] for r in rows}) != 1:
        parser.error("The ensemble requires three compatible frozen checkpoints.")
    if args.validation:
        validation = json.loads((args.validation / "comparison.json").read_text(encoding="utf-8"))
        if not validation["eligible_for_test"] or validation["candidate"] != selected:
            parser.error("The ensemble did not meet the training-layout validation criterion.")
        layouts, count, start = (1,), args.episodes, 985000
    else:
        layouts, count, start = (0, 2, 3), parent_manifest["episodes"], parent_manifest["validation_seed"]
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", dict(
        parent_sha256=digest(args.frozen), candidate=selected, runs=rows, layouts=layouts,
        episodes=count, seed_start=start, protocol=plan["protocol"], temperature=plan["temperature"],
        code_sha256=digest(Path(__file__))))
    with ProcessPoolExecutor(len(layouts)) as pool:
        jobs = [pool.submit(evaluate, root, rows, maze, count, start, plan["protocol"],
                            plan["temperature"]) for maze in layouts]
        results = {str(result["layout"]): result for result in (job.result() for job in jobs)}
    baseline = plan["means"][selected]
    eligible = not args.validation and results["0"]["success_rate"] >= .90 and (
        sum(results[str(maze)]["success_rate"] - baseline[str(maze)]["success_rate"]
            for maze in (2, 3)) / 2 >= .03)
    write_json(root / "comparison.json", dict(candidate=selected, results=results,
               reference_validation=baseline, eligible_for_test=bool(eligible)))
    if not args.validation:
        print(f"Ensemble eligible for separate test: {eligible}", flush=True)


if __name__ == "__main__":
    main()

