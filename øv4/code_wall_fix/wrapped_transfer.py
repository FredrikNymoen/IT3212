"""Train mixed-layout PPO with measured side-tunnel connections in its route graph."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import refine_transfer as refine
import transfer_ppo as study
from cached_navigation import CachedNavigationTask, CachedQueries
from score_diagnostics import NavigationTask, digest, np, runner, write_json


ARM = "mixed_wrapped"
original_evaluate = refine.evaluate


class WrappedQueries(CachedQueries):
    def __init__(self, original, walls):
        super().__init__(original)
        self.width = walls.shape[1]
        self.distance = original.distance.copy()
        self.tunnels = []
        nodes = original.nodes
        left, right = float(nodes[:, 1].min()), float(nodes[:, 1].max())
        connected = set(original.connected)
        for row in np.unique(nodes[:, 0]):
            if walls[int(row), :int(left) + 1].any() or walls[int(row), int(right):].any():
                continue
            a = int(np.flatnonzero((nodes[:, 0] == row) & (nodes[:, 1] == left))[0])
            b = int(np.flatnonzero((nodes[:, 0] == row) & (nodes[:, 1] == right))[0])
            if a not in connected or b not in connected:
                continue
            length = self.width - right + left
            forward = self.distance[:, a, None] + length + self.distance[b, None, :]
            self.distance = np.minimum(self.distance, np.minimum(forward, forward.T))
            self.tunnels.append((float(row), (left, a), (right, b)))

    def exits(self, position):
        key = tuple(position)
        if key not in self.exit_cache:
            result = self.graph.exits(position).copy()
            row, col = position
            for tunnel_row, (left, a), (right, b) in self.tunnels:
                offset = abs(row - tunnel_row)
                if offset > 4:
                    continue
                if col <= left:
                    result.setdefault(2, ((offset, self.width + col - right), b))
                    result.setdefault(3, ((offset, left - col), a))
                if col >= right:
                    result.setdefault(2, ((offset, col - right), b))
                    result.setdefault(3, ((offset, self.width - col + left), a))
            self.exit_cache[key] = result
        return self.exit_cache[key]


class WrappedNavigationTask(CachedNavigationTask):
    def _read_layout(self):
        super()._read_layout()
        if not isinstance(self.graph, WrappedQueries):
            self.graph = WrappedQueries(self.graph.graph, self.walls)
            self.layout_graphs[self.s["maze"]] = self._maze_key, self.graph, self._pellet_nodes


class TrainWrappedTask(study.TransferTrainingTask, WrappedNavigationTask):
    pass


def train(root, seed, parent, budget, count):
    study.TransferTrainingTask = TrainWrappedTask
    study.NavigationTask = WrappedNavigationTask
    study.CONFIG[ARM] = study.CONFIG["mixed"]
    row = study.train_and_validate(root, ARM, seed, parent, budget, count)
    return dict(row, feature_type="wrapped")


def evaluate(root, row, layout, count, start):
    refine.CACHED_TASKS["wrapped"] = WrappedNavigationTask
    return original_evaluate(root, row, layout, count, start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.results.resolve()
    if args.frozen:
        refine.evaluate = evaluate
        plan = json.loads(args.frozen.read_text(encoding="utf-8"))
        refine.final_test(root, plan, args.workers)
        return
    if not args.frontier or args.budget < 1000 or not 1 <= args.episodes <= 500:
        parser.error("Use a frontier, at least 1000 interactions and 1–500 validation episodes.")
    baseline = json.loads(args.frontier.read_text(encoding="utf-8"))
    parents = {r["seed"]: Path(r["model"]) for r in baseline["runs"] if r["name"] == "cold_30k"}
    if not parents:
        parser.error("The frontier is missing the cold_30k checkpoints.")
    root.mkdir(parents=True, exist_ok=False)
    sources = ("refine_transfer.py", "cached_navigation.py", "transfer_ppo.py",
               "improve_ppo_session.py", "score_diagnostics.py", "compare_navigation.py",
               "reference_policy.py", "part2_run.py", "mspacman.py", Path(__file__).name)
    write_json(root / "manifest.json", dict(
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in sources},
        parents={seed: dict(path=str(path), sha256=digest(path)) for seed, path in parents.items()},
        config=study.CONFIG["mixed"], protocol=runner.PROTOCOL,
        budget=args.budget, episodes=args.episodes, validation_seed=study.VALIDATION_SEED,
        validation_layouts=study.TRAIN_LAYOUTS))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train, root, seed, parent, args.budget, args.episodes)
                for seed, parent in parents.items()]
        rows = [job.result() for job in jobs]
    means = {ARM: {str(layout): {key: mean(r["validation"][layout][key] for r in rows)
                               for key in ("success_rate", "mean_score")}
                   for layout in study.TRAIN_LAYOUTS}}
    write_json(root / "comparison.json", dict(runs=rows, means=means))


if __name__ == "__main__":
    main()

