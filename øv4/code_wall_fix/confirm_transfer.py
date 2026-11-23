"""Confirm the frozen D result on a separate episode sample and log scoring events."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

from cached_navigation import CACHED_TASKS
from mspacman import FRIGHTENED
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, runner, write_json, write_rows
from wrapped_transfer import WrappedNavigationTask


TEST_START = 985000


def evaluate(root, row, protocol, temperature, count, start):
    path = root / row["arm"] / f"seed_{row['seed']}"
    path.mkdir(parents=True)
    model = Path(row["model"])
    if digest(model) != row["model_sha256"]:
        raise ValueError(f"Checkpoint contents changed: {model}")
    kind = row["feature_type"]
    base = WrappedNavigationTask if kind == "wrapped" else CACHED_TASKS[kind]
    records = []

    class EventTask(base):
        def reset(self, seed, *args, **kwargs):
            obs = super().reset(seed, *args, **kwargs)
            self.last_phase = int(self.env.unwrapped.ale.getRAM()[FRIGHTENED])
            self.episode = dict(seed=seed, first_power_decision=None, power_pellets=0,
                                ghosts_eaten=0, ghost_points=0)
            return obs

        def step(self, action):
            obs, reward, done, info = super().step(action)
            phase = int(self.env.unwrapped.ale.getRAM()[FRIGHTENED])
            old_chain, old_timer = divmod(self.last_phase, 64)
            chain, timer = divmod(phase, 64)
            raw = int(info["raw_reward"])
            if timer > old_timer and raw in (50, 60, 70):
                self.episode["power_pellets"] += 1
                if self.episode["first_power_decision"] is None:
                    self.episode["first_power_decision"] = self.t
            ghost_value = 200 * 2**old_chain
            if timer and chain == (old_chain + 1) % 4 and raw - ghost_value in (0, 10, 20):
                self.episode["ghosts_eaten"] += 1
                self.episode["ghost_points"] += ghost_value
            self.last_phase = phase
            if done:
                self.episode.update(raw_score=info["raw_score"], outcome=info["outcome"],
                                    length=self.t, lives=info["lives"])
                records.append(self.episode)
            return obs, reward, done, info

    runner.MsPacManTask = EventTask
    try:
        agent = TemperaturePolicy(runner.load_model(model), temperature)
        scores = runner.evaluate(agent, "D", count, protocol=protocol, seed_start=start,
                                 log_dir=path, label="confirm", trace_count=1)
    finally:
        runner.MsPacManTask = NavigationTask
    assert len(records) == count and [r["seed"] for r in records] == list(range(start, start + count))
    write_rows(path / "scoring_events.csv", records)
    first_power = [r["first_power_decision"] for r in records if r["first_power_decision"] is not None]
    reached = sum(r["raw_score"] >= 1500 for r in records)
    result = dict(
        arm=row["arm"], seed=row["seed"], episodes=count, seed_start=start, **scores,
        reached_1500=reached / count, power_pellet_rate=len(first_power) / count,
        median_first_power_decision=median(first_power) if first_power else None,
        mean_power_pellets=mean(r["power_pellets"] for r in records),
        mean_ghost_points=mean(r["ghost_points"] for r in records),
        mean_ghosts_eaten=mean(r["ghosts_eaten"] for r in records))
    print(f"{row['arm']} seed {row['seed']}: {result['success_rate']:.1%}, "
          f"{result['mean_score']:.0f} points", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1000:
        parser.error("Use 1–1000 confirmation episodes per checkpoint.")
    test = args.test.resolve()
    root = args.results.resolve()
    plan = json.loads((test / "frozen.json").read_text(encoding="utf-8"))
    scores_path = test / "comparison.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if TEST_START < plan["seed_start"] + plan["episodes"] and TEST_START + args.episodes > plan["seed_start"]:
        parser.error("Confirmation seeds overlap with the original test seeds.")
    trained_arms = [arm for arm in scores["means"] if arm != "starting"]
    best = max(trained_arms, key=lambda arm: (scores["means"][arm]["1"]["success_rate"],
                                             scores["means"][arm]["1"]["mean_score"]))
    rows = [r for r in plan["runs"] if r["arm"] in ("starting", best)]
    if len(rows) < 2:
        parser.error("Confirmation requires the starting and newly trained cohorts.")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", dict(
        original_test=str(test), test_comparison_sha256=digest(scores_path),
        test_plan_sha256=digest(test / "frozen.json"), seed_start=TEST_START,
        episodes=args.episodes, chosen_arm=best, runs=rows,
        protocol=plan["protocol"], temperature=plan["temperature"],
        code_sha256={**plan["evaluation_code_sha256"], Path(__file__).name: digest(Path(__file__))}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(evaluate, root, r, plan["protocol"], plan["temperature"],
                            args.episodes, TEST_START) for r in rows]
        runs = [job.result() for job in as_completed(jobs)]
    metrics = ("success_rate", "mean_score", "reached_1500", "power_pellet_rate",
               "mean_power_pellets", "mean_ghost_points", "mean_ghosts_eaten")
    means = {arm: {key: mean(r[key] for r in runs if r["arm"] == arm) for key in metrics}
             for arm in ("starting", best)}
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

