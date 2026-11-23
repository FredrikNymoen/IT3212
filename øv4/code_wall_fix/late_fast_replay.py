"""Train a replay specialist while keeping the emulator loaded between branches."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import late_specialist as specialist
import late_teacher as teacher
import late_transfer as late
import refine_transfer as refine
import replay_transfer as replay
from cached_navigation import CachedNavigationTask
from score_diagnostics import digest, np, runner, write_json


ARM = "specialist_fast"


class LoadedSnapshot(replay.Snapshot):
    def resume(self, task, seed):
        # A branch preserves the task's advancing sticky-action RNG. The initial
        # reset seeds that stream, so replay can restore gameplay in place.
        if not hasattr(task, "score"):
            task.env.reset(seed=int(seed))
        task.s = dict(task.s, maze=self.maze)
        obs = teacher.restore(task, self)
        task.layout_graphs[self.maze] = task._maze_key, task.graph, task._pellet_nodes
        ale = task.env.unwrapped.ale
        if int(ale.getRAM()[0]) != self.maze or ale.lives() != self.task_fields["lives"]:
            raise RuntimeError("The restored branch has a different layout or life count.")
        return obs


def fit(root, row, budget, count):
    original = replay.Snapshot
    replay.Snapshot = LoadedSnapshot
    try:
        return specialist.fit(root, row, ARM, budget, 45_000, count)
    finally:
        replay.Snapshot = original


def verify():
    original = CachedNavigationTask(runner.PROTOCOL, "B")
    branch = CachedNavigationTask(runner.PROTOCOL, "B")
    timings = {}
    trajectories = []
    try:
        for mode in replay.refine.WEIGHTS:
            original.s = dict(original.s, maze=mode)
            obs = original.reset(181 + mode)
            for i in range(60):
                obs, _, done, _ = original.step(i % original.n_actions)
                assert not done
            for snapshot in (replay.Snapshot, LoadedSnapshot):
                bank = snapshot.save(original, obs)
                branch.reset(400 + mode)
                start = time.perf_counter()
                samples = []
                for i in range(12):
                    np.testing.assert_array_equal(bank.resume(branch, 1210 + i), obs)
                    samples.append(tuple(branch.step(k % branch.n_actions)[0].tobytes() for k in range(20)))
                timings[snapshot.__name__] = timings.get(snapshot.__name__, 0.) + time.perf_counter() - start
                if snapshot is LoadedSnapshot:
                    trajectories.append(len(set(samples)))
    finally:
        original.close()
        branch.close()
    assert min(trajectories) > 1
    print(f"Restore benchmark across three layouts: {timings}; distinct branches {trajectories}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--budget", type=int, default=240_000)
    parser.add_argument("--episodes", type=int, default=75)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if not args.parent or not args.results or args.budget < 50_000:
        parser.error("Provide parents, a result directory and at least 50,000 interactions.")
    parents = [r for r in json.loads(args.parent.read_text(encoding="utf-8"))["runs"]
               if r["arm"] == late.BASELINE]
    if len(parents) != 3 or any(set(r["interactions"]) != {"0", "2", "3"} for r in parents):
        parser.error("Expected three parents trained on layouts 0, 2 and 3.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "late_specialist.py", "late_teacher.py", "replay_transfer.py",
             "late_transfer.py", "cached_navigation.py", "score_diagnostics.py",
             "mspacman.py", "part2_run.py")
    write_json(root / "manifest.json", dict(
        parents=parents, source=str(args.parent.resolve()), budget=args.budget,
        collection_budget=45_000, episodes=args.episodes, validation_seed=specialist.VALIDATION_START,
        protocol=runner.PROTOCOL, training_layouts=[0, 2, 3],
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in names}))
    with ProcessPoolExecutor(3) as pool:
        jobs = [pool.submit(fit, root, row, args.budget, args.episodes) for row in parents]
        runs = [job.result() for job in jobs]
    means = refine.comparison(runs, (0, 2, 3), ("success_rate", "mean_score", "reached_1500", "conversion"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

