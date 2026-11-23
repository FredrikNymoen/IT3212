"""Give mixed-layout PPO directional distances to the remaining power pellets."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import transfer_ppo as study
from score_diagnostics import NavigationTask, digest, np, runner, write_json


# Centroids in cropped screen coordinates, measured in layouts 0, 2 and 3.
POWER = np.array([[16., 9.5], [148., 9.5], [15., 149.5], [147., 149.5]])
POWER_FEATURES = 5


class PowerTask(NavigationTask):
    def _read_layout(self):
        super()._read_layout()
        self.available_power = np.ones(len(POWER), dtype=bool)
        self.power_nodes = self.graph.nearest(POWER)

    def _eat(self, position, raw):
        super()._eat(position, raw)
        # The extra 10 or 20 points allow for ordinary dots eaten in
        # the same frameskip. Position disambiguates the four pellets.
        if raw % 100 in (50, 60, 70) and self.available_power.any():
            distance = np.abs(POWER - (np.asarray(position) + 4.5)).sum(axis=1)
            distance[~self.available_power] = np.inf
            index = int(distance.argmin())
            if distance[index] <= 16:
                self.available_power[index] = False

    def features(self, ram):
        current = super().features(ram)
        position = self._actors(ram)[0]
        exits = self.graph.exits(position)
        distance = np.ones(4, dtype=np.float32)
        live = self.power_nodes[self.available_power]
        for action, ((_, first_segment), node) in exits.items():
            path = self.graph.distance[node, live].min(initial=np.inf)
            distance[action] = min(1., (first_segment + path) / 160.)
        return np.concatenate((current, distance,
                               np.array([self.available_power.mean()], dtype=np.float32)))


class TrainPowerTask(study.TransferTrainingTask, PowerTask):
    pass


def pad_checkpoint(parent, path):
    agent = runner.load_model(parent)
    agent.p["W1"] = np.pad(agent.p["W1"], ((0, POWER_FEATURES), (0, 0)))
    runner.save_model(agent, path)


def train(root, seed, parent, budget, count):
    expanded = root / "expanded_parents" / f"seed_{seed}.npz"
    pad_checkpoint(parent, expanded)
    study.TransferTrainingTask = TrainPowerTask
    study.NavigationTask = PowerTask
    study.CONFIG["mixed_power"] = study.CONFIG["mixed"]
    return study.train_and_validate(root, "mixed_power", seed, expanded, budget, count)


def test(root, run, layout, count, start):
    if run["arm"] == "mixed_power":
        study.NavigationTask = PowerTask
    try:
        return study.test(root, run["arm"], run["seed"], run["model"], run["model_sha256"],
                          layout, count, start)
    finally:
        study.NavigationTask = NavigationTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.frozen:
        plan = json.loads(args.frozen.read_text(encoding="utf-8"))
        root = args.results.resolve()
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "frozen.json", plan)
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = [pool.submit(test, root, run, layout, plan["episodes"], plan["seed_start"])
                    for layout in plan["layouts"] for run in plan["runs"]]
            results = [job.result() for job in jobs]
        write_json(root / "comparison.json", dict(runs=results, means={
            arm: {str(layout): {key: mean(r[key] for r in results
                                          if r["arm"] == arm and r["layout"] == layout)
                                for key in ("success_rate", "mean_score", "failure_rate", "timeout_rate")}
                  for layout in plan["layouts"]}
            for arm in sorted({r["arm"] for r in results})}))
        return
    if not args.frontier or args.budget < 1_000 or not 1 <= args.episodes <= 500:
        parser.error("Use a frontier, at least 1000 interactions and 1–500 validation episodes.")
    root = args.results.resolve()
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    parents = {r["seed"]: Path(r["model"]) for r in frontier["runs"] if r["name"] == "cold_30k"}
    if not parents:
        parser.error("The frontier is missing cold_30k checkpoints.")
    (root / "expanded_parents").mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", dict(
        protocol=runner.PROTOCOL, budget=args.budget, validation_seed=study.VALIDATION_SEED,
        validation_layouts=study.TRAIN_LAYOUTS, episodes=args.episodes,
        parents={seed: dict(path=str(path), sha256=digest(path)) for seed, path in parents.items()},
        config=study.CONFIG["mixed"], power_centroids=POWER.tolist(),
        sources={p.name: digest(p) for p in (Path(__file__), Path(study.__file__))}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train, root, seed, parent, args.budget, args.episodes)
                for seed, parent in parents.items()]
        rows = [job.result() for job in jobs]
    write_json(root / "comparison.json", dict(runs=rows, means={
        "mixed_power": {str(layout): {key: mean(r["validation"][layout][key] for r in rows)
                                     for key in ("success_rate", "mean_score")}
                        for layout in study.TRAIN_LAYOUTS}}))


if __name__ == "__main__":
    main()

