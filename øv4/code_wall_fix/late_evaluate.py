"""Screen candidates on common training-layout seeds and evaluate frozen D plans."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import refine_transfer as refine
from late_specialist import evaluate
from score_diagnostics import digest, runner, write_json


TRAIN_LAYOUTS = (0, 2, 3)
VALIDATION_SEED = 980500
TEST_SEED = 987000
CONFIRM_SEED = 988500
METRICS = ("success_rate", "mean_score", "reached_1500", "conversion",
           "late_life_loss", "failure_rate", "timeout_rate")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources(path):
    manifest = load(path.with_name("manifest.json"))
    here = Path(__file__).resolve().parent
    for name, checksum in manifest["code_sha256"].items():
        if digest(here / name) != checksum:
            raise ValueError(f"{name} has changed since {path.parent.name} ran.")


def make_row(row):
    out = {key: row[key] for key in ("arm", "seed", "model", "model_sha256", "feature_type")}
    if "opening_model" in row:
        out.update({key: row[key] for key in ("opening_model", "opening_sha256", "switch_score")})
    return out


def validate_rows(rows):
    identities = {(row["arm"], row["seed"]) for row in rows}
    if not rows or len(rows) != len(identities):
        raise ValueError("Expected unique checkpoints per setup and training seed.")
    for row in rows:
        if digest(Path(row["model"])) != row["model_sha256"]:
            raise ValueError(f"Checkpoint changed: {row['model']}")
        if "opening_model" in row and digest(Path(row["opening_model"])) != row["opening_sha256"]:
            raise ValueError(f"Opening policy changed: {row['opening_model']}")


def run_plan(root, plan, workers, label):
    validate_rows(plan["runs"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "frozen.json", plan)
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(evaluate, root, row, layout, plan["episodes"], plan["seed_start"],
                            label, plan["temperature"])
                for layout in plan["layouts"] for row in plan["runs"]]
        results = [job.result() for job in as_completed(jobs)]
    checkpoints = {(row["arm"], row["seed"]): row for row in plan["runs"]}
    for result in results:
        row = checkpoints[result["arm"], result["seed"]]
        result.update(model=row["model"], model_sha256=row["model_sha256"])
    results.sort(key=lambda r: (r["arm"], r["seed"], r["layout"]))
    means = refine.comparison(results, plan["layouts"], METRICS, lambda r, _: r, True)
    write_json(root / "comparison.json", dict(runs=results, means=means))
    for arm, layouts in means.items():
        print(arm, " ".join(f"maze {m}: {v['success_rate']:.1%}" for m, v in layouts.items()), flush=True)


def screen(args):
    verify_sources(args.source)
    rows = [r for r in load(args.source)["runs"] if not args.arm or r["arm"] == args.arm]
    if not rows or any(set(r["interactions"]) != {str(m) for m in TRAIN_LAYOUTS} for r in rows):
        raise ValueError("Screening expects rows trained only on the three training layouts.")
    if args.alias:
        if not args.arm:
            raise ValueError("--alias requires a single input arm.")
        rows = [dict(r, arm=args.alias) for r in rows]
    plan = dict(protocol=runner.PROTOCOL, temperature=.2, layouts=TRAIN_LAYOUTS,
                seed_start=VALIDATION_SEED, episodes=args.episodes, runs=list(map(make_row, rows)),
                source=dict(path=str(args.source.resolve()), sha256=digest(args.source)))
    run_plan(args.results.resolve(), plan, args.workers, "validation")


def freeze(args):
    runs, means, checks = [], {}, {}
    for folder in args.screens:
        plan_path, comparison_path = folder / "frozen.json", folder / "comparison.json"
        plan, results = load(plan_path), load(comparison_path)
        if plan["seed_start"] != VALIDATION_SEED or plan["layouts"] != list(TRAIN_LAYOUTS):
            raise ValueError(f"{folder} did not use the common training-layout screen.")
        if plan["episodes"] != args.screen_episodes or plan["protocol"] != runner.PROTOCOL:
            raise ValueError(f"{folder} had a different evaluation protocol.")
        if set(results["means"]) & means.keys():
            raise ValueError(f"Duplicate arms in {folder}.")
        if digest(Path(plan["source"]["path"])) != plan["source"]["sha256"]:
            raise ValueError(f"The original training result changed for {folder}.")
        verify_sources(Path(plan["source"]["path"]))
        runs.extend(plan["runs"])
        means.update(results["means"])
        checks[str(folder.resolve())] = dict(plan=digest(plan_path), comparison=digest(comparison_path))
    validate_rows(runs)
    if "anchor" not in means or "control" not in means:
        raise ValueError("The freeze needs the opening checkpoints and the matched continuation control.")
    seeds = {row["seed"] for row in runs if row["arm"] == "anchor"}
    for arm in means:
        if {r["seed"] for r in runs if r["arm"] == arm} != seeds:
            raise ValueError(f"{arm} does not include all matched training seeds.")
    baseline_b = means["anchor"]["0"]["success_rate"]
    floor = baseline_b - .06

    def rank(arm):
        return (mean(means[arm][str(m)]["success_rate"] for m in (2, 3)),
                mean(means[arm][str(m)]["conversion"] for m in (2, 3)),
                mean(means[arm][str(m)]["mean_score"] for m in (2, 3)))

    candidates = sorted(set(means) - {"anchor", "control"}, key=rank, reverse=True)
    eligible = [arm for arm in candidates if means[arm]["0"]["success_rate"] >= floor]
    selected = ["anchor", "control", *eligible[:args.top]]
    # Preserve the highest-ranked phased policy as a targeted test of the same
    # opening reaching the threshold, even when another family wins screening.
    phased = {row["arm"] for row in runs if row.get("opening_model")}
    if eligible_phased := [arm for arm in eligible if arm in phased]:
        if eligible_phased[0] not in selected:
            selected.append(eligible_phased[0])
    source_names = ("late_evaluate.py", "late_specialist.py", "cached_navigation.py",
                    "refine_transfer.py", "score_diagnostics.py", "part2_run.py", "mspacman.py",
                    "compare_navigation.py", "reference_policy.py", "transfer_power_ppo.py",
                    "transfer_ghost_ppo.py")
    here = Path(__file__).resolve().parent
    plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), protocol=runner.PROTOCOL,
                temperature=.2, layouts=[1, 0], seed_start=TEST_SEED, episodes=args.episodes,
                screen_episodes=args.screen_episodes, screen_seed=VALIDATION_SEED,
                screen_sources=checks, screen_means=means, ordered_candidates=candidates,
                b_floor=floor, selected_arms=selected,
                evaluation_code_sha256={name: digest(here / name) for name in source_names},
                runs=[r for r in runs if r["arm"] in selected])
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")
    print("Frozen:", ", ".join(selected), flush=True)


def test(args):
    plan = load(args.frozen)
    here = Path(__file__).resolve().parent
    if plan["seed_start"] != TEST_SEED or set(plan["layouts"]) != {0, 1}:
        raise ValueError("The frozen test needs both B and D and the declared test seed.")
    for name, expected in plan["evaluation_code_sha256"].items():
        if digest(here / name) != expected:
            raise ValueError(f"{name} has changed since the experiment was frozen.")
    run_plan(args.results.resolve(), plan, args.workers, "test")


def confirm(args):
    game = load(args.test / "frozen.json")
    results_path = args.test / "comparison.json"
    means = load(results_path)["means"]
    winner = max((arm for arm in means if arm != "anchor"),
              key=lambda arm: (means[arm]["1"]["success_rate"], means[arm]["1"]["mean_score"]))
    plan = dict(game, seed_start=CONFIRM_SEED, episodes=args.episodes, layouts=[1],
             selected_arms=["anchor", winner],
             primary_test=dict(path=str(args.test.resolve()), sha256=digest(results_path)),
             runs=[r for r in game["runs"] if r["arm"] in {"anchor", winner}])
    run_plan(args.results.resolve(), plan, args.workers, "confirm")


def options(parser, episodes, workers):
    parser.add_argument("--results", type=Path, required=True)
    if episodes:
        parser.add_argument("--episodes", type=int, default=episodes)
    parser.add_argument("--workers", type=int, default=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    screening = modes.add_parser("screen")
    screening.add_argument("--source", type=Path, required=True)
    screening.add_argument("--arm")
    screening.add_argument("--alias")
    options(screening, 70, 6)
    screening.set_defaults(run=screen)

    freezing = modes.add_parser("freeze")
    freezing.add_argument("--screens", type=Path, nargs="+", required=True)
    freezing.add_argument("--output", type=Path, required=True)
    freezing.add_argument("--screen-episodes", type=int, default=70)
    freezing.add_argument("--episodes", type=int, default=150)
    freezing.add_argument("--top", type=int, default=3)
    freezing.set_defaults(run=freeze)

    testing = modes.add_parser("test")
    testing.add_argument("--frozen", type=Path, required=True)
    options(testing, None, 12)
    testing.set_defaults(run=test)

    confirmation = modes.add_parser("confirm")
    confirmation.add_argument("--test", type=Path, required=True)
    options(confirmation, 300, 6)
    confirmation.set_defaults(run=confirm)

    args = parser.parse_args()
    if not 1 <= getattr(args, "episodes", 1) <= 400 or getattr(args, "workers", 1) < 1:
        parser.error("Use 1–400 episodes and at least one worker.")
    args.run(args)


if __name__ == "__main__":
    main()

