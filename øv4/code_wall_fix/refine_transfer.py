"""Refine PPO across training layouts and replay frozen checkpoint comparisons."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import improve_ppo_session as search
import transfer_ppo as study
from cached_navigation import CACHED_TASKS
from mspacman import SCREEN_DX
from score_diagnostics import NavigationTask, digest, np, runner, write_json, write_rows


WEIGHTS = {0: .26, 2: .37, 3: .37}
VALIDATION_START = 982150
TEMPERATURE, LR, KL = .30, 3e-5, .025
MIRROR_CHANCE = {"balanced": 0., "mirror": .35}
LEFT_RIGHT = ([12, 13], [19, 20], [23, 24], [41, 42], [49, 50], [53, 54])


def feature_type(arm):
    if arm.startswith("mixed_ghost"):
        return "ghost"
    if arm == "mixed_power":
        return "power"
    return "navigation"


def training_task(base, mirror_chance, exposure):
    class RefinementTask(base):
        def __init__(self, protocol, setting):
            super().__init__(protocol, setting)
            self.interactions = dict.fromkeys(WEIGHTS, 0)
            opposite = {"LEFT": "RIGHT", "RIGHT": "LEFT",
                        "UPLEFT": "UPRIGHT", "UPRIGHT": "UPLEFT",
                        "DOWNLEFT": "DOWNRIGHT", "DOWNRIGHT": "DOWNLEFT"}
            self.action_map = np.array([self.actions.index(opposite.get(a, a))
                                        for a in self.actions])

        def reset(self, seed, startup_steps=runner.START_DECISIONS, after_startup_step=None):
            if exposure:
                previous = exposure[-1]
                self.interactions[previous["maze"]] += self.decisions - previous["interaction_start"]
            maze = min(WEIGHTS, key=lambda m: self.interactions[m] / WEIGHTS[m])
            self.s = dict(self.s, maze=maze)
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 411]))
            self.mirrored = rng.random() < mirror_chance
            exposure.append(dict(seed=seed, maze=maze, interaction_start=self.decisions,
                                 startup_interactions=startup_steps, mirrored=self.mirrored))
            return super().reset(seed, startup_steps, after_startup_step)

        def step(self, action):
            return super().step(int(self.action_map[action]) if self.mirrored else action)

        def features(self, ram):
            features = super().features(ram)
            if self.mirrored:
                features[0] = (self.walls.shape[1] - 1 - 2*(SCREEN_DX + 4.5))/160 - features[0]
                features[[2, 14, 26, 28]] *= -1
                for left, right in LEFT_RIGHT:
                    if right < len(features):
                        features[left], features[right] = features[right], features[left]
                features[30:39] = features[30:39][self.action_map]
            return features

    return RefinementTask


def evaluate(root, row, layout, count, start):
    if digest(Path(row["model"])) != row["model_sha256"]:
        raise ValueError(f"Checkpoint contents changed: {row['model']}")
    study.NavigationTask = CACHED_TASKS[row["feature_type"]]
    try:
        return study.test(root, row["arm"], row["seed"], row["model"], row["model_sha256"],
                          layout, count, start)
    finally:
        study.NavigationTask = NavigationTask


def train(root, row, method, budget, count):
    seed, parent = row["seed"], Path(row["model"])
    kind = row["feature_type"]
    arm = f"{row['arm']}_{method}"
    if digest(parent) != row["model_sha256"]:
        raise ValueError(f"Parent checkpoint contents changed: {parent}")
    folder = root / arm / f"seed_{seed}"
    folder.mkdir(parents=True)
    exposure = []
    search.NavigationTask = training_task(CACHED_TASKS[kind], MIRROR_CHANCE[method], exposure)
    training_seed = 12000 + seed
    try:
        model = search.fit(parent, training_seed, folder, budget, LR, TEMPERATURE, KL, study.ROLLOUT)
    finally:
        search.NavigationTask = NavigationTask
    totals = dict.fromkeys(WEIGHTS, 0)
    for i, episode in enumerate(exposure):
        end = exposure[i+1]["interaction_start"] if i+1 < len(exposure) else budget
        episode["interactions"] = end - episode["interaction_start"]
        totals[episode["maze"]] += episode["interactions"]
    assert sum(totals.values()) == budget
    write_rows(folder / "layout_exposure.csv", exposure)
    output = dict(arm=arm, seed=seed, training_seed=training_seed, model=str(model),
                  model_sha256=digest(model), parent=str(parent), parent_sha256=digest(parent),
                  interactions=totals, validation={}, feature_type=kind)
    print(f"{arm} seed {seed} trained {totals}", flush=True)
    study.NavigationTask = CACHED_TASKS[kind]
    try:
        for layout in study.TRAIN_LAYOUTS:
            scores = study.evaluate(model, layout, count, VALIDATION_START,
                                    folder / f"validation_maze_{layout}")
            output["validation"][layout] = scores
            print(f"{arm} seed {seed} maze {layout}: {scores['success_rate']:.1%}, "
                  f"{scores['mean_score']:.0f} points", flush=True)
    finally:
        study.NavigationTask = NavigationTask
    write_json(folder / "result.json", output)
    return output


def comparison(rows, layouts, metrics, get_layout, require_layout_field=False):
    return {arm: {str(layout): {metric: mean(get_layout(r, layout)[metric] for r in rows
                if r["arm"] == arm and (not require_layout_field or r["layout"] == layout))
                for metric in metrics} for layout in layouts}
            for arm in sorted({r["arm"] for r in rows})}


def final_test(root, plan, workers):
    rows = plan["runs"]
    identities = {(r["arm"], r["seed"]) for r in rows}
    if len(identities) != len(rows) or not rows or set(plan["layouts"]) - {0, 1}:
        raise ValueError("The frozen plan must contain unique checkpoints and test layouts.")
    count, start = plan["episodes"], plan["seed_start"]
    if not 1 <= count <= 1200 or not 982500 <= start < start + count <= 990000:
        raise ValueError("Use 1–1200 test episodes drawn from the held-out seed range.")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "frozen.json", plan)
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(evaluate, root, row, layout, count, start)
                for layout in plan["layouts"] for row in rows]
        scores = [job.result() for job in as_completed(jobs)]
    scores.sort(key=lambda r: (r["arm"], r["seed"], r["layout"]))
    means = comparison(scores, plan["layouts"], ("success_rate", "mean_score", "failure_rate",
                                                  "timeout_rate"), lambda r, _: r, True)
    write_json(root / "comparison.json", dict(runs=scores, means=means))
    print(json.dumps(means, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-arm")
    parser.add_argument("--method", nargs="+", choices=MIRROR_CHANCE,
                        default=list(MIRROR_CHANCE))
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.results.resolve()
    if args.frozen:
        final_test(root, json.loads(args.frozen.read_text(encoding="utf-8")), args.workers)
        return
    if not args.parent or not args.parent_arm or not 1 <= args.episodes <= 350 or args.budget < 1000:
        parser.error("Use --parent and --parent-arm, at least 1000 interactions and 1–350 episodes.")
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    rows = []
    for r in parent["runs"]:
        if r["arm"] == args.parent_arm:
            if set(r["interactions"]) - {"0", "2", "3"}:
                parser.error("The parent trained on a layout outside the training set.")
            rows.append(dict(r, feature_type=r.get("feature_type", feature_type(r["arm"]))))
    if not rows or len({r["seed"] for r in rows}) != len(rows):
        parser.error("The parent study should contain one checkpoint per training seed.")
    root.mkdir(parents=True, exist_ok=False)
    names = ("cached_navigation.py", "transfer_ghost_ppo.py", "transfer_power_ppo.py",
             "transfer_ppo.py", "improve_ppo_session.py", "score_diagnostics.py",
             "compare_navigation.py", "reference_policy.py", "part2_run.py", "mspacman.py")
    sources = {name: digest(Path(__file__).with_name(name)) for name in (*names, Path(__file__).name)}
    write_json(root / "manifest.json", dict(
        parent_study=str(args.parent.resolve()), parent_arm=args.parent_arm,
        parents=rows, code_sha256=sources, budget=args.budget, episodes=args.episodes,
        validation_seed=VALIDATION_START, protocol=runner.PROTOCOL, training_weights=WEIGHTS,
        methods={m: MIRROR_CHANCE[m] for m in args.method}, temperature=TEMPERATURE,
        learning_rate=LR, kl=KL, inference_temperature=study.INFERENCE_TEMPERATURE))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train, root, r, method, args.budget, args.episodes)
                for method in args.method for r in rows]
        runs = [job.result() for job in as_completed(jobs)]
    means = comparison(runs, study.TRAIN_LAYOUTS, ("success_rate", "mean_score"),
                       lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

