"""Freeze the transfer test from validation results before opening Setting D."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import refine_transfer as refine
import transfer_ppo as study
from score_diagnostics import digest, runner


BASELINE_ARM = "starting"
B_MINIMUM = .90


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validation_rank(means):
    return (mean(means[str(m)]["success_rate"] for m in (2, 3)),
            mean(means[str(m)]["mean_score"] for m in (2, 3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", default=["b_only", "mixed"])
    parser.add_argument("--latest", type=int, default=3,
                        help="maximum additional candidates ranked by validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--test-seed", type=int, default=study.TEST_SEED)
    args = parser.parse_args()
    if args.latest < 0 or len(set(args.controls)) != len(args.controls):
        parser.error("Use a nonnegative shortlist size and distinct controls.")
    if not 1 <= args.episodes <= 1200 or not 982500 <= args.test_seed < args.test_seed + args.episodes <= 990000:
        parser.error("Use 1–1200 episodes from the held-out test seed range.")
    baseline = read_json(args.baseline)
    starts = [r for r in baseline["runs"] if r["name"] == "cold_30k"]
    seeds = {r["seed"] for r in starts}
    if not starts or len(seeds) != len(starts):
        parser.error("The baseline must have one cold_30k model per seed.")
    runs = [dict(arm=BASELINE_ARM, feature_type="navigation", seed=r["seed"],
                 model=r["model"], model_sha256=digest(Path(r["model"]))) for r in starts]
    groups, means, sources = {}, {}, {}
    here = Path(__file__).resolve().parent
    for path in args.studies:
        path = path.resolve()
        manifest_path = path.with_name("manifest.json")
        manifest = read_json(manifest_path)
        for name, value in manifest.get("code_sha256", manifest.get("sources", {})).items():
            if digest(here / name) != value:
                parser.error(f"{name} changed since {path.parent.name} ran.")
        if set(manifest.get("validation_layouts", study.TRAIN_LAYOUTS)) != set(study.TRAIN_LAYOUTS):
            parser.error(f"{path} used a layout outside the permitted validation set.")
        if manifest["protocol"] != runner.PROTOCOL:
            parser.error(f"{path} used a different score or termination protocol.")
        summary = read_json(path)
        for name, statistics in summary["means"].items():
            if name in groups:
                parser.error(f"Choose a single completed experiment for {name}.")
            rows = [r for r in summary["runs"] if r["arm"] == name]
            if len(rows) != len(seeds) or {r["seed"] for r in rows} != seeds:
                parser.error(f"Missing matched training seeds in {name}.")
            for r in rows:
                budget = sum(r["interactions"].values())
                if set(r["interactions"]) - {"0", "2", "3"} or budget != manifest["budget"]:
                    parser.error(f"Unexpected training layout or budget in {r['model']}.")
            groups[name] = rows
            means[name] = statistics
            sources[name] = {"results": str(path), "sha256": digest(path),
                             "manifest_sha256": digest(manifest_path)}
    if set(args.controls) - groups.keys():
        parser.error("At least one fixed control is missing.")
    eligible = [arm for arm in groups if arm not in args.controls
                and means[arm]["0"]["success_rate"] >= B_MINIMUM]
    shortlisted = sorted(eligible, key=lambda arm: validation_rank(means[arm]), reverse=True)
    selected = [*args.controls, *shortlisted[:args.latest]]
    for arm in selected:
        for r in groups[arm]:
            kind = r.get("feature_type", refine.feature_type(arm))
            runs.append(dict(arm=arm, feature_type=kind, seed=r["seed"], model=r["model"],
                             model_sha256=r["model_sha256"]))
    for r in runs:
        if digest(Path(r["model"])) != r["model_sha256"]:
            parser.error(f"Checkpoint changed since training: {r['model']}")
    plan = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        protocol=runner.PROTOCOL, temperature=study.INFERENCE_TEMPERATURE, layouts=[1, 0],
        episodes=args.episodes, seed_start=args.test_seed, train_layouts=study.TRAIN_LAYOUTS,
        layout_b_minimum=B_MINIMUM, means=means, ordered_candidates=shortlisted,
        control_arms=args.controls, selected_arms=[BASELINE_ARM, *selected],
        sources=sources, baseline=dict(results=str(args.baseline.resolve()), sha256=digest(args.baseline)),
        evaluation_code_sha256={name: digest(here / name) for name in (
            "freeze_transfer.py", "wrapped_transfer.py", "refine_transfer.py", "cached_navigation.py",
            "transfer_ppo.py", "transfer_power_ppo.py", "transfer_ghost_ppo.py", "score_diagnostics.py",
            "compare_navigation.py", "reference_policy.py", "part2_run.py", "mspacman.py")},
        runs=runs)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print("Frozen arms:", ", ".join(plan["selected_arms"]), flush=True)


if __name__ == "__main__":
    main()

