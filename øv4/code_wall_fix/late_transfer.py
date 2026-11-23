"""Matched PPO continuation with rewards for survival and late scoring."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import improve_ppo_session as search
import refine_transfer as refine
import transfer_ppo as transfer
from cached_navigation import CachedNavigationTask
from score_diagnostics import digest, np, runner, write_json, write_rows


ARMS = ("control", "survival", "finish")
VALIDATION_START = 981400
BASELINE = "mixed_balanced"
LR, TEMPERATURE, KL = 3e-5, .27, .025


def reward_for(arm, score, reward, raw, lost, threshold):
    progress = min(max((score - 1000.) / 750., 0.), 1.)
    if arm != "control":
        reward -= lost * (20. + 90. * progress)
    if arm == "finish":
        credited = min(raw, max(0., threshold - score))
        reward += .8 * progress * credited
    return reward


def install_task(arm, exposure):
    class LateTrainingTask(refine.training_task(CachedNavigationTask, 0., exposure)):
        def step(self, action):
            previous_score, previous_lives = self.score, self.lives
            obs, reward, done, info = super().step(action)
            lost = max(0, previous_lives - info["lives"])
            reward = reward_for(arm, previous_score, reward, info["raw_reward"], lost,
                                self.p["success_score"])
            return obs, reward, done, info

    return LateTrainingTask


def fit(root, row, arm, budget, episodes, offset, start):
    seed = row["seed"]
    checkpoint = Path(row["model"])
    if digest(checkpoint) != row["model_sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    folder = root / arm / f"seed_{seed}"
    folder.mkdir(parents=True)
    exposure = []
    search.NavigationTask = install_task(arm, exposure)
    try:
        model = search.fit(checkpoint, offset + seed, folder, budget,
                           LR, TEMPERATURE, KL, transfer.ROLLOUT)
    finally:
        search.NavigationTask = CachedNavigationTask
    counts = dict.fromkeys(refine.WEIGHTS, 0)
    for index, episode in enumerate(exposure):
        end = exposure[index + 1]["interaction_start"] if index + 1 < len(exposure) else budget
        episode["interactions"] = end - episode["interaction_start"]
        counts[episode["maze"]] += episode["interactions"]
    assert sum(counts.values()) == budget and set(counts) == {0, 2, 3}
    write_rows(folder / "layout_exposure.csv", exposure)
    result = dict(arm=arm, seed=seed, training_seed=offset + seed, model=str(model),
                  model_sha256=digest(model), parent=str(checkpoint), parent_sha256=row["model_sha256"],
                  interactions=counts, validation={}, feature_type="navigation")
    print(f"{arm} seed {seed} trained {counts}", flush=True)
    return validate(root, result, episodes, start)


def validate(root, row, count, start):
    out = dict(row, validation={})
    folder = root / row["arm"] / f"seed_{row['seed']}"
    transfer.NavigationTask = CachedNavigationTask
    for layout in (0, 2, 3):
        scores = transfer.evaluate(Path(row["model"]), layout, count, start, folder / f"validation_maze_{layout}")
        out["validation"][layout] = scores
        print(f"{row['arm']} seed {row['seed']} maze {layout}: {scores['success_rate']:.1%}, "
              f"{scores['mean_score']:.0f} points", flush=True)
    write_json(folder / "result.json", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-arm", default=BASELINE)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget", type=int, default=240_000)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed-offset", type=int, default=31000)
    parser.add_argument("--validation-seed", type=int, default=VALIDATION_START)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    summary = json.loads(args.parent.read_text(encoding="utf-8"))
    parents = [r for r in summary["runs"] if r["arm"] == args.parent_arm]
    if len(parents) != 3 or len({r["seed"] for r in parents}) != 3:
        parser.error("Expected matched parents from three training seeds.")
    if any(set(r["interactions"]) != {"0", "2", "3"} or r["feature_type"] != "navigation" for r in parents):
        parser.error("Expected parents trained on layouts 0, 2 and 3 with navigation features.")
    if len(set(args.arms)) != len(args.arms) or args.episodes < 1 or args.budget < 1000:
        parser.error("Use distinct arms, a positive episode count and at least 1000 interactions.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "refine_transfer.py", "transfer_ppo.py", "cached_navigation.py",
             "improve_ppo_session.py", "part2_run.py", "mspacman.py", "compare_navigation.py",
             "reference_policy.py", "score_diagnostics.py")
    write_json(root / "manifest.json", dict(
        parents=parents, parent_study=str(args.parent.resolve()), budget=args.budget,
        seed_offset=args.seed_offset, validation_seed=args.validation_seed, episodes=args.episodes,
        validation_layouts=[0, 2, 3], training_weights=refine.WEIGHTS, protocol=runner.PROTOCOL,
        learning_rate=LR, temperature=TEMPERATURE, kl=KL, arms=args.arms,
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in names}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fit, root, row, arm, args.budget, args.episodes,
                            args.seed_offset, args.validation_seed) for arm in args.arms for row in parents]
        results = [job.result() for job in as_completed(jobs)]
    means = refine.comparison(results, (0, 2, 3), ("success_rate", "mean_score"),
                             lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=results, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

