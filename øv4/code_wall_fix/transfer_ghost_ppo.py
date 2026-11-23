"""Train PPO with power-pellet paths and decoded ghost scoring opportunities."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import transfer_ppo as study
import transfer_power_ppo as power
from mspacman import FRIGHTENED
from score_diagnostics import NavigationTask, digest, np, runner, write_json


GHOST_FEATURES = 11
NORMAL_MODES = (0, 1, 2, 3, 5, 7)
ARMS = {
    "mixed_ghost": study.CONFIG["mixed"],
    "mixed_ghost_hot": study.CONFIG["mixed_hot"],
    "mixed_ghost_fast": study.CONFIG["mixed_fast"],
}


class GhostTask(power.PowerTask):
    def features(self, ram):
        base = super().features(ram)
        ghost_modes = ram[1:5] >> 4
        frightened = (ghost_modes & 8) != 0
        # Modes 4 and 6 follow an eaten ghost on its return to the house.
        # Treat waiting and newly released ghosts conservatively as threats.
        dangerous = np.isin(ghost_modes, NORMAL_MODES)
        chain, timer = divmod(int(ram[FRIGHTENED]), 64)
        actors = self._actors(ram)
        nodes = self.graph.nearest(actors[1:])
        paths = np.ones((2, 4), dtype=np.float32)
        vulnerable_nodes, dangerous_nodes = nodes[frightened], nodes[dangerous]
        for action, ((_, first_segment), node) in self.graph.exits(actors[0]).items():
            distances = self.graph.distance[node]
            chase = distances[vulnerable_nodes].min(initial=np.inf)
            avoid = distances[dangerous_nodes].min(initial=np.inf)
            paths[0, action] = min(1., (first_segment + chase) / 160.)
            paths[1, action] = min(1., (first_segment + avoid) / 160.)
        phase = np.array([timer / 63., chain / 3., frightened.mean()], dtype=np.float32)
        return np.concatenate((base, phase, paths.ravel()))


class TrainGhostTask(study.TransferTrainingTask, GhostTask):
    pass


def train(root, arm, seed, parent, budget, episodes):
    path = root / "expanded_parents" / f"seed_{seed}.npz"
    study.TransferTrainingTask = TrainGhostTask
    study.NavigationTask = GhostTask
    study.CONFIG[arm] = ARMS[arm]
    return study.train_and_validate(root, arm, seed, path, budget, episodes)


def test(root, run, layout, count, start):
    if run["arm"] not in ARMS:
        return power.test(root, run, layout, count, start)
    study.NavigationTask = GhostTask
    try:
        return study.test(root, run["arm"], run["seed"], run["model"], run["model_sha256"],
                          layout, count, start)
    finally:
        study.NavigationTask = NavigationTask


def evaluate_plan(root, plan, workers):
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "frozen.json", plan)
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(test, root, run, layout, plan["episodes"], plan["seed_start"])
                for layout in plan["layouts"] for run in plan["runs"]]
        rows = [job.result() for job in jobs]
    write_json(root / "comparison.json", dict(runs=rows, means={
        arm: {str(layout): {key: mean(r[key] for r in rows
                                      if r["arm"] == arm and r["layout"] == layout)
                            for key in ("success_rate", "mean_score", "failure_rate", "timeout_rate")}
              for layout in plan["layouts"]}
        for arm in sorted({r["arm"] for r in rows})}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["mixed_ghost"])
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.results.resolve()
    if args.frozen:
        plan = json.loads(args.frozen.read_text(encoding="utf-8"))
        evaluate_plan(root, plan, args.workers)
        return
    if not args.frontier or args.budget < 1000 or not 1 <= args.episodes <= 500:
        parser.error("Use a frontier, at least 1000 interactions and 1–500 validation episodes.")
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    parents = {r["seed"]: Path(r["model"]) for r in frontier["runs"] if r["name"] == "cold_30k"}
    if not parents or len(set(args.arms)) != len(args.arms):
        parser.error("Choose unique arms and a frontier containing cold_30k checkpoints.")
    (root / "expanded_parents").mkdir(parents=True, exist_ok=False)
    modules = (Path(__file__), Path(power.__file__), Path(study.__file__))
    sources = (*modules, *(modules[0].with_name(name) for name in (
        "improve_ppo_session.py", "score_diagnostics.py", "compare_navigation.py",
        "reference_policy.py", "part2_run.py", "mspacman.py")))
    write_json(root / "manifest.json", dict(
        protocol=runner.PROTOCOL, budget=args.budget, validation_seed=study.VALIDATION_SEED,
        validation_layouts=study.TRAIN_LAYOUTS, episodes=args.episodes,
        parents={seed: dict(model=str(path), sha256=digest(path)) for seed, path in parents.items()},
        arms={arm: ARMS[arm] for arm in args.arms},
        code_sha256={path.name: digest(path) for path in sources}))
    for seed, path in parents.items():
        agent = runner.load_model(path)
        added_features = power.POWER_FEATURES + GHOST_FEATURES
        agent.p["W1"] = np.pad(agent.p["W1"], ((0, added_features), (0, 0)))
        runner.save_model(agent, root / "expanded_parents" / f"seed_{seed}.npz")
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train, root, arm, seed, parent, args.budget, args.episodes)
                for arm in args.arms for seed, parent in parents.items()]
        rows = [job.result() for job in jobs]
    write_json(root / "comparison.json", dict(runs=rows, means={
        arm: {str(layout): {key: mean(r["validation"][layout][key] for r in rows if r["arm"] == arm)
                            for key in ("success_rate", "mean_score")}
              for layout in study.TRAIN_LAYOUTS}
        for arm in args.arms}))


if __name__ == "__main__":
    main()

