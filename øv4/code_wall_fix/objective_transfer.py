"""Compare two PPO objectives with matched mixed-layout continuation."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import refine_transfer as refine
import transfer_ppo as study
from score_diagnostics import digest, runner, write_json


SUCCESS_BONUS = 400.
METHODS = ("balanced", "target_reward")


def train(root, row, method, budget, count):
    original_factory = refine.training_task

    def make_task(base, mirror_chance, exposure):
        class TargetRewardTask(original_factory(base, mirror_chance, exposure)):
            def step(self, action):
                before = self.score
                obs, reward, done, info = super().step(action)
                threshold = self.p["success_score"]
                credited_score = min(self.score, threshold) - min(before, threshold)
                shaped = reward - info["raw_reward"] - info["success_bonus"] + credited_score
                if info["outcome"] == "success":
                    shaped += SUCCESS_BONUS
                return obs, shaped, done, info
        return TargetRewardTask

    if method == "target_reward":
        refine.MIRROR_CHANCE[method] = 0.
        refine.training_task = make_task
    try:
        result = refine.train(root, row, method, budget, count)
    finally:
        refine.training_task = original_factory
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-arm", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=250_000)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.results.resolve()
    previous = json.loads(args.parent.read_text(encoding="utf-8"))
    rows = [dict(r, feature_type=refine.feature_type(r["arm"])) for r in previous["runs"]
            if r["arm"] == args.parent_arm]
    if not rows or len({r["seed"] for r in rows}) != len(rows):
        parser.error("The parent study should contain one checkpoint per training seed.")
    if any(set(r["interactions"]) - {"0", "2", "3"} for r in rows):
        parser.error("The parent trained on a layout outside the training set.")
    if not 1 <= args.episodes <= 350 or args.budget < 1000:
        parser.error("Use at least 1000 interactions and 1–350 episodes.")
    root.mkdir(parents=True, exist_ok=False)
    names = ("refine_transfer.py", "cached_navigation.py", "transfer_ghost_ppo.py",
             "transfer_power_ppo.py", "transfer_ppo.py", "improve_ppo_session.py",
             "score_diagnostics.py", "compare_navigation.py", "reference_policy.py",
             "part2_run.py", "mspacman.py", Path(__file__).name)
    write_json(root / "manifest.json", dict(
        parent_study=str(args.parent.resolve()), parent_arm=args.parent_arm, parents=rows,
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in names},
        budget=args.budget, episodes=args.episodes, validation_seed=refine.VALIDATION_START,
        protocol=runner.PROTOCOL, training_weights=refine.WEIGHTS, methods=METHODS,
        target_reward=dict(clip_score_at_success=True, bonus=SUCCESS_BONUS),
        temperature=refine.TEMPERATURE, learning_rate=refine.LR, kl=refine.KL,
        inference_temperature=study.INFERENCE_TEMPERATURE))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train, root, row, method, args.budget, args.episodes)
                for method in METHODS for row in rows]
        runs = [job.result() for job in as_completed(jobs)]
    means = refine.comparison(runs, study.TRAIN_LAYOUTS, ("success_rate", "mean_score"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

