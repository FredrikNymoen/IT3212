"""Portable, read-only evidence interface for the Part 2 synthesis.

Johan's session API motivates one notebook entry point; Elling's per-agent
dashboard motivates the common records; Mats' protocol checks motivate strict
source/model compatibility. No collaborator's policy or training code is copied.
The registered ordinary-Q/PPO engines and original observations stay unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "evidence.json"
MAIN = "results/assignment4/three_hour_20261122/final_v1"
TRANSFER = "results/assignment4/transfer_boundary_v1"
CURVES = "results/assignment4/three_hour_20261122/main_v1/comparison.csv"
TRAIN_SEEDS = (17, 29, 101, 314, 2718)
ALGORITHMS = ("Q-learning", "PPO")
PREDICTION_SHA = "17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(recorded, root=ROOT):
    """Rebase archived Windows paths, without reading outside this checkout."""
    root = Path(root).resolve()
    parts = PureWindowsPath(str(recorded)).parts
    require(".." not in parts, f"Parent traversal in recorded path: {recorded}")
    if PureWindowsPath(str(recorded)).is_absolute():
        starts = [i for i in range(len(parts) - 1)
                  if parts[i].lower() == "results" and parts[i + 1].lower() == "assignment4"]
        require(len(starts) == 1, f"Unrecognised archived path: {recorded}")
        parts = parts[starts[0]:]
    target = root.joinpath(*parts).resolve()
    require(target.is_relative_to(root), f"Path outside checkout: {recorded}")
    return target


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jobs():
    """All 30 A/B/C cells plus all ten repaired D cells; no ranking/filtering."""
    old = read_json(ROOT / MAIN / "locked.json")
    new = read_json(ROOT / TRANSFER / "protocol.json")
    require(sha(ROOT / MAIN / "locked.json") == new["original_selection_lock_sha256"],
            "D no longer references the original checkpoint lock")
    result = [(job, old) for job in old["jobs"] if job["condition"] in "ABC"]
    result += [(job, new) for job in new["jobs"]]
    require(len(result) == 40, "Expected exactly 40 primary cells")
    require({(j["condition"], j["algorithm"], j["seed"]) for j, _ in result}
            == {(c, a, s) for c in "ABCD" for a in ALGORITHMS for s in TRAIN_SEEDS},
            "Missing/duplicate primary cells")
    for job, lock in result:
        require(len(lock["evaluation_seeds"]) == 200, "Expected 200 held-out episodes")
        if job["condition"] == "D":
            partner = next(j for j in old["jobs"] if j["condition"] == "B"
                           and j["algorithm"] == job["algorithm"] and j["seed"] == job["seed"])
            require(job["model_sha256"] == partner["model_sha256"] and job["model"] == partner["model"],
                    "D must reuse the identical B checkpoint")
    return result


def average(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def spread(values):
    values = [float(v) for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def summaries(agents):
    result = []
    for condition in "ABCD":
        for algorithm in ALGORITHMS:
            group = [r for r in agents if r["condition"] == condition and r["algorithm"] == algorithm]
            require(len(group) == 5, "Five trained agents are the statistical replications")
            row = dict(condition=condition, algorithm=algorithm, trained_agents=5,
                       episodes=sum(r["episodes"] for r in group))
            for field in ["success_rate", "success_upper_bound", "failure_rate", "censored_rate",
                          "mean_return", "mean_score", "success_length", "failure_length",
                          "training_seconds", "evaluation_seconds"]:
                row[field] = average(r[field] for r in group)
                row[field + "_sd"] = spread(r[field] for r in group)
            result.append(row)
    return result


def build_bundle(output=BUNDLE):
    inputs = {}

    def register(path, expected=None):
        path = local_path(path)
        digest = sha(path)
        require(expected is None or digest == expected, f"Source/model hash mismatch: {path}")
        inputs[path.relative_to(ROOT).as_posix()] = digest
        return path

    register("reports/assignment4/Sections/01_predictions.tex", PREDICTION_SHA)
    for name in [MAIN + "/locked.json", TRANSFER + "/protocol.json",
                 TRANSFER + "/verification.json", MAIN + "/native_replay_validation.json", CURVES]:
        register(name)
    agents = []
    for job, lock in jobs():
        for path, digest in lock["source_hashes"].items():
            register(path, digest)
        model = register(job["model"], job["model_sha256"])
        directory = local_path(job["directory"])
        rows = read_csv(register(directory / "evaluation.csv"))
        metrics = read_json(register(directory / "metrics.json"))
        d = job["condition"] == "D"
        trace = register(directory / ("traces.npz" if d else "evaluation_traces.npz"))
        require([int(r["evaluation_seed"]) for r in rows] == lock["evaluation_seeds"],
                "Incomplete or reordered final episode list")
        require(metrics["frozen_parameters_verified"], "Evaluation parameters changed")
        success = [int(r["success"]) for r in rows]
        failure = [int(r["known_failure"]) if d else 1 - int(r["success"]) for r in rows]
        censored = [int(r["censored"]) if d else 0 for r in rows]
        require(all(x in (0, 1) for column in (success, failure, censored) for x in column),
                "Invalid outcome flag")
        require(all(s + f + c == 1 for s, f, c in zip(success, failure, censored)),
                "Outcomes must partition episodes")
        lengths = [int(r["length"]) for r in rows]
        require(all(1 <= n <= 1000 for n in lengths), "Unexpected episode horizon")
        scores = [float(r["observed_score" if d else "score"]) for r in rows]
        returns = [float(r["observed_return" if d else "task_return"]) for r in rows]
        require(all((score >= 2000) == bool(s) for score, s in zip(scores, success)),
                "Success disagrees with observed score")
        require(all(math.isclose(ret, score - (.1 * n if job["condition"] == "C" else 0), abs_tol=1e-6)
                    for ret, score, n in zip(returns, scores, lengths)), "Reward accounting mismatch")
        require(sum(lengths) == metrics["evaluation_interactions"], "Interaction count mismatch")
        require(math.isclose(average(returns), metrics["mean_return"], abs_tol=1e-6),
                "Mean return disagrees with raw episodes")
        require(math.isclose(average(success), metrics["observed_success_rate" if d else "success_rate"]),
                "Success rate disagrees with raw episodes")
        if d:
            require(metrics["further_training_interactions"] == 0 and metrics["training_seconds"] == 0,
                    "D contains training")
        agents.append(dict(
            condition=job["condition"], algorithm=job["algorithm"], training_seed=job["seed"],
            episodes=len(rows), successes=sum(success), failures=sum(failure), censored=sum(censored),
            success_rate=average(success), success_upper_bound=average(s + c for s, c in zip(success, censored)),
            failure_rate=average(failure), censored_rate=average(censored),
            mean_return=average(returns), mean_score=average(scores),
            success_length=average(n for n, s in zip(lengths, success) if s),
            failure_length=average(n for n, f in zip(lengths, failure) if f),
            training_interactions=0 if d else job["training_interactions"],
            source_training_interactions=job["original_training_interactions"] if d else job["training_interactions"],
            checkpoint_interactions=job["checkpoint_interactions"],
            training_seconds=0.0 if d else job["training_seconds"], evaluation_seconds=metrics["evaluation_seconds"],
            evaluation_interactions=sum(lengths), evaluation_seed_first=lock["evaluation_seeds"][0],
            evaluation_seed_last=lock["evaluation_seeds"][-1], model=model.relative_to(ROOT).as_posix(),
            model_sha256=job["model_sha256"], trace=trace.relative_to(ROOT).as_posix(),
            evaluation_csv=(directory / "evaluation.csv").relative_to(ROOT).as_posix()))
    agents.sort(key=lambda r: (r["condition"], r["algorithm"], r["training_seed"]))
    curves = []
    for row in read_csv(ROOT / CURVES):
        curves.append(dict(condition=row["setting"], algorithm=row["algorithm"], training_seed=int(row["seed"]),
                           interactions=int(row["interactions"]), success_rate=float(row["success_rate"])))
    require(len(curves) == 150, "Expected five checkpoints for 30 main models")
    bundle = dict(schema="assignment4-part2-evidence-v1", training_seeds=list(TRAIN_SEEDS),
                  source_studies=dict(ABC="22 November 2026 main experiment", D="23 November 2026 frozen boundary evaluation"),
                  policy="Greedy Q with episode-seeded uniform ties; PPO argmax. No updates.",
                  inference_unit="Five trained agents per method/condition; episodes are repeated evaluation, not training replications.",
                  censoring="D map exits are unresolved; return excludes crossing reward. Bounds are not confidence intervals.",
                  agents=agents, summaries=summaries(agents), curves=curves, inputs=dict(sorted(inputs.items())))
    require(sum(r["episodes"] for r in agents) == 8000, "Expected 8,000 primary episodes")
    require(sum(r["censored"] for r in agents) == 1, "Unexpected censoring count")
    Path(output).write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return bundle


def load_bundle(path=BUNDLE):
    bundle = read_json(path)
    require(bundle["schema"] == "assignment4-part2-evidence-v1", "Unknown evidence schema")
    require(bundle["summaries"] == summaries(bundle["agents"]), "Summary arithmetic changed")
    return bundle


def check_bundle(path=BUNDLE, full=False):
    bundle = load_bundle(path)
    checked = missing = 0
    for name, expected in bundle["inputs"].items():
        source = local_path(name)
        if source.exists():
            require(sha(source) == expected, f"Evidence input changed: {name}")
            checked += 1
        else:
            require(not full, f"Missing raw evidence input: {name}")
            missing += 1
    require(sha(ROOT / "reports/assignment4/Sections/01_predictions.tex") == PREDICTION_SHA,
            "Original predictions changed")
    return dict(checked_inputs=checked, missing_local_inputs=missing, full_local_archive=missing == 0,
                agents=len(bundle["agents"]), episodes=sum(r["episodes"] for r in bundle["agents"]),
                note="A compact snapshot is not a substitute for the original raw traces/models in the submission.")


def dashboard(bundle, output):
    template = (HERE / "dashboard.html").read_text(encoding="utf-8")
    data = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(template.replace("/* EVIDENCE_DATA */null", data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "check", "dashboard"])
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--full", action="store_true", help="Require the complete local raw archive")
    parser.add_argument("--output", type=Path, default=ROOT / "results/assignment4/synthesis/overview.html")
    args = parser.parse_args()
    if args.command == "build":
        b = build_bundle(args.bundle)
        print(json.dumps(dict(agents=len(b["agents"]), episodes=sum(r["episodes"] for r in b["agents"]),
                              input_hashes=len(b["inputs"]))))
    elif args.command == "check":
        print(json.dumps(check_bundle(args.bundle, args.full), indent=2))
    else:
        dashboard(load_bundle(args.bundle), args.output)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
