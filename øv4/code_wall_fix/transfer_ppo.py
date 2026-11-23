"""Compare matched PPO continuation on one or several Ms. Pac-Man layouts."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import improve_ppo_session as search
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, np, runner, write_json, write_rows


TRAIN_LAYOUTS = (0, 2, 3)
CONFIG = {
    "b_only": ((0,), .25, 4e-5, .04),
    "mixed": (TRAIN_LAYOUTS, .25, 4e-5, .04),
    "mixed_hot": (TRAIN_LAYOUTS, .40, 4e-5, .025),
    "mixed_fast": (TRAIN_LAYOUTS, .25, 8e-5, .02),
    "mixed_local": (TRAIN_LAYOUTS, .25, 4e-5, .04),
    "mixed_b_weighted": ((0, 2, 0, 3), .25, 4e-5, .04),
}
VALIDATION_SEED = 982000
TEST_SEED = 983000
INFERENCE_TEMPERATURE = .2
ROLLOUT = 512


class TransferTrainingTask(NavigationTask):
    layouts = TRAIN_LAYOUTS
    exposure = None
    hide_coordinates = False

    def reset(self, seed, startup_steps=65, after_startup_step=None):
        # The first reset sizes the network inputs. It uses the same layout for
        # all arms and contributes only opening-animation interactions.
        i = len(self.exposure)
        maze = self.layouts[(i - 1) % len(self.layouts)] if i else 0
        self.s = dict(self.s, maze=maze)
        self.mask_coordinates = self.hide_coordinates and int(seed) % 5 < 3
        self.exposure.append(dict(seed=seed, maze=maze,
                                  interaction_start=self.decisions,
                                  startup_interactions=startup_steps))
        return super().reset(seed, startup_steps, after_startup_step)

    def features(self, ram):
        features = super().features(ram)
        if self.mask_coordinates:
            features[:2] = .5
        return features


def evaluate(model, layout, count, start, folder):
    folder.mkdir(parents=True, exist_ok=True)
    setting = "D" if layout == 1 else "B"

    class FixedLayoutTask(NavigationTask):
        def __init__(self, protocol, condition):
            super().__init__(protocol, condition)
            self.s = dict(self.s, maze=layout)

    runner.MsPacManTask = FixedLayoutTask
    try:
        agent = TemperaturePolicy(runner.load_model(Path(model)), INFERENCE_TEMPERATURE)
        return runner.evaluate(agent, setting, count, seed_start=start,
                               log_dir=folder, label="eval", trace_count=1)
    finally:
        runner.MsPacManTask = NavigationTask


def train_and_validate(root, arm, seed, parent, budget, count):
    folder = root / arm / f"seed_{seed}"
    folder.mkdir(parents=True)
    layouts, temperature, lr, kl = CONFIG[arm]
    task = TransferTrainingTask
    task.layouts, task.exposure = layouts, []
    task.hide_coordinates = arm == "mixed_local"
    search.NavigationTask = task
    try:
        model = search.fit(parent, seed, folder, budget, lr, temperature, kl, ROLLOUT)
    finally:
        search.NavigationTask = NavigationTask
    exposure = task.exposure
    interactions = {}
    for i, row in enumerate(exposure):
        end = exposure[i + 1]["interaction_start"] if i + 1 < len(exposure) else budget
        row["interactions"] = end - row["interaction_start"]
        maze = row["maze"]
        interactions[maze] = interactions.get(maze, 0) + row["interactions"]
    assert 1 not in interactions and sum(interactions.values()) == budget
    write_rows(folder / "layout_exposure.csv", exposure)
    result = dict(arm=arm, seed=seed, model=str(model), model_sha256=digest(model),
                  parent=str(parent), interactions=interactions, validation={})
    print(f"{arm} seed {seed} finished training {interactions}", flush=True)
    for layout in TRAIN_LAYOUTS:
        scores = evaluate(model, layout, count, VALIDATION_SEED,
                          folder / f"validation_maze_{layout}")
        result["validation"][layout] = scores
        print(f"{arm} seed {seed} validation maze {layout}: "
              f"{scores['success_rate']:.1%}, {scores['mean_score']:.0f} points", flush=True)
    write_json(folder / "result.json", result)
    return result


def test(root, arm, seed, model, expected_hash, layout, count, start):
    if digest(Path(model)) != expected_hash:
        raise ValueError(f"Checkpoint contents changed: {model}")
    scores = evaluate(model, layout, count, start, root / arm / f"seed_{seed}" / f"maze_{layout}")
    print(f"{arm} seed {seed} test maze {layout}: "
          f"{scores['success_rate']:.1%}, {scores['mean_score']:.0f} points", flush=True)
    return dict(arm=arm, seed=seed, layout=layout, model=model, model_sha256=expected_hash,
                episodes=count, seed_start=start, **scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--arms", nargs="+", choices=CONFIG, default=["b_only", "mixed"])
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--freeze", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    args = parser.parse_args()
    root = args.results.resolve()
    if args.freeze:
        runs = [r for path in args.freeze for r in json.loads(
            path.read_text(encoding="utf-8"))["runs"]]
        if args.baseline:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            for r in baseline["runs"]:
                if r["name"] == "cold_30k":
                    runs.append(dict(arm="starting", seed=r["seed"], model=r["model"],
                                     model_sha256=digest(Path(r["model"]))))
        runs = [r for r in runs if r["arm"] in args.arms or r["arm"] == "starting"]
        identities = [(r["arm"], r["seed"]) for r in runs]
        if not runs or len(set(identities)) != len(identities):
            parser.error("Choose a single checkpoint per arm and seed.")
        if any("validation" in r and "1" in r["interactions"] for r in runs):
            parser.error("A checkpoint saw the test layout during training.")
        if args.test_seed < VALIDATION_SEED + 500 or args.test_seed + args.episodes > 990000:
            parser.error("Use a held-out seed range from 982500 to 989999.")
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "frozen.json", dict(
            comparisons=[str(p.resolve()) for p in args.freeze], runs=runs,
            seed_start=args.test_seed, episodes=args.episodes, layouts=[0, 1],
            temperature=INFERENCE_TEMPERATURE, protocol=runner.PROTOCOL))
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = [pool.submit(test, root, r["arm"], r["seed"], r["model"],
                                r["model_sha256"], layout, args.episodes, args.test_seed)
                    for layout in (1, 0) for r in runs]
            rows = [job.result() for job in jobs]
        write_json(root / "comparison.json", dict(runs=rows, means={
            arm: {str(layout): {key: mean(r[key] for r in rows
                                          if r["arm"] == arm and r["layout"] == layout)
                                for key in ("success_rate", "mean_score", "failure_rate", "timeout_rate")}
                  for layout in (0, 1)}
            for arm in sorted({r["arm"] for r in rows})}))
        return
    if not args.frontier:
        parser.error("--frontier is required for training.")
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    parents = {r["seed"]: Path(r["model"]) for r in frontier["runs"] if r["name"] == "cold_30k"}
    if not parents or len(set(args.arms)) != len(args.arms) or not 1 <= args.episodes <= 500:
        parser.error("Use unique arms, 1–500 episodes and a frontier containing cold_30k models.")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", dict(
        protocol=runner.PROTOCOL, budget=args.budget, validation_seed=VALIDATION_SEED,
        validation_layouts=TRAIN_LAYOUTS, episodes=args.episodes,
        temperature=INFERENCE_TEMPERATURE, rollout=ROLLOUT,
        arms={arm: CONFIG[arm] for arm in args.arms},
        parents={seed: dict(model=str(path), sha256=digest(path)) for seed, path in parents.items()},
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in (
            Path(__file__).name, "improve_ppo_session.py", "score_diagnostics.py",
            "compare_navigation.py", "reference_policy.py", "part2_run.py", "mspacman.py")}))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(train_and_validate, root, arm, seed, parent, args.budget, args.episodes)
                for arm in args.arms for seed, parent in parents.items()]
        rows = [job.result() for job in jobs]
    write_json(root / "comparison.json", dict(runs=rows, means={
        arm: {str(layout): {key: mean(r["validation"][layout][key] for r in rows if r["arm"] == arm)
                            for key in ("success_rate", "mean_score")}
              for layout in TRAIN_LAYOUTS}
        for arm in args.arms}))


if __name__ == "__main__":
    main()

