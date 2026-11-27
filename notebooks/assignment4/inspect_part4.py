"""Select and reproduce the ten Part 4 episodes from the frozen Part 3 test."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ppo import FrozenPolicy
from q_learning import (
    ROOT, TabularQ, evaluate, new_directory, require, sha, verify_sources, write_json,
)


CONFIGURATIONS = (
    ("Q", "Q-learning", "selected"),
    ("PPO6", "PPO", "baseline"),
    ("PPO12", "PPO", "selected"),
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_population(source):
    lock = load_json(source / "locked.json")
    summaries = read_rows(source / "per_seed.csv")
    population = {}
    for label, algorithm, phase in CONFIGURATIONS:
        models = [row for row in summaries
                  if (row["algorithm"], row["label"]) == (algorithm, phase)]
        require(len({row["training_seed"] for row in models}) == len(models) == 5,
                f"Expected five distinct trained agents for {label}")
        episodes = []
        for row in models:
            folder = ROOT / row["run_id"]
            config = load_json(folder / "configuration.json")
            require(config["protocol"]["setting"] == "B"
                    and config["settings"] == lock["comparisons"][phase][algorithm],
                    f"Unexpected configuration in {folder}")
            selected = load_json(folder / "selected.json")
            evaluation = load_json(folder / "held_out" / "locked.json")
            metrics = load_json(folder / "held_out" / "completed.json")
            require(evaluation["selected"] == selected and metrics["frozen_parameters_verified"],
                    f"Evaluation lock does not match model in {folder}")
            raw = read_rows(folder / "held_out" / "evaluation.csv")
            require(len(raw) == 200 and [int(x["evaluation_seed"]) for x in raw] == lock["evaluation_seeds"],
                    f"Unexpected evaluation seeds in {folder}")
            for episode in raw:
                require(int(episode["censored"]) == 0 and
                        int(episode["success"]) == (float(episode["score"]) >= 2000),
                        f"Unexpected outcome in {folder}")
                episodes.append(dict(episode, configuration=label, algorithm=algorithm,
                                     training_seed=int(row["training_seed"]), source_run=row["run_id"]))
        population[label] = episodes
    return population


def seed_order(episode):
    return episode["training_seed"], int(episode["evaluation_seed"])


def typical(episodes):
    values = np.array([[float(e["score"]), int(e["length"])] for e in episodes])
    q1, median, q3 = np.quantile(values, (.25, .5, .75), axis=0)
    scales = np.maximum(q3 - q1, [10, 1])
    distances = (np.abs(values - median) / scales).sum(axis=1)
    reason = Counter(e["reason"] for e in episodes).most_common(1)[0][0]
    index = min((i for i, e in enumerate(episodes) if e["reason"] == reason),
                key=lambda i: (distances[i], seed_order(episodes[i])))
    summary = dict(n=len(episodes), median_score=float(median[0]), median_length=float(median[1]),
                   score_quartiles=[float(q1[0]), float(q3[0])],
                   length_quartiles=[float(q1[1]), float(q3[1])], modal_reason=reason)
    return episodes[index], summary


def select(population):
    choices, distributions = [], {}
    for success, prefix in ((1, "TS"), (0, "TF")):
        for i, (label, _, _) in enumerate(CONFIGURATIONS, 1):
            subset = [e for e in population[label] if int(e["success"]) == success]
            episode, summary = typical(subset)
            choices.append(dict(episode, id=f"{prefix}{i}"))
            distributions[f"{label}_{success}"] = summary

    for i, label in enumerate(("Q", "PPO12"), 1):
        previous = {tuple(seed_order(row)) for row in choices if row["configuration"] == label}
        successes = [e for e in population[label] if int(e["success"]) and seed_order(e) not in previous]
        failures = [e for e in population[label] if not int(e["success"]) and seed_order(e) not in previous]
        fastest = min(successes, key=lambda e: (int(e["length"]), seed_order(e)))
        nearest = min(failures, key=lambda e: (-float(e["score"]), int(e["length"]), seed_order(e)))
        choices.extend((dict(fastest, id=f"SS{i}"), dict(nearest, id=f"SF{i}")))
    return sorted(choices, key=lambda e: (("TS", "TF", "SS", "SF").index(e["id"][:2]), e["id"])), distributions


def replay(case, output):
    run = ROOT / case["source_run"]
    selected = load_json(run / "selected.json")
    verify_sources(selected["source_hashes"])
    model = run / selected["model"]
    require(sha(model) == selected["model_sha256"], f"Checkpoint changed: {model}")
    algorithm, case_id = case["algorithm"], case["id"]
    load_model = TabularQ.load if algorithm == "Q-learning" else FrozenPolicy.load
    metrics = evaluate(load_model(model), algorithm, "B", [int(case["evaluation_seed"])],
                       output, case_id, trace=True)
    [replayed] = read_rows(output / f"{case_id}.csv")
    require(all(case[key] == value for key, value in replayed.items()),
            f"Replay diverges from the frozen evaluation in {case_id}")
    require(sha(model) == selected["model_sha256"], f"Replay changed {model}")
    return dict(case, model=model.relative_to(ROOT).as_posix(), model_sha256=selected["model_sha256"],
                checkpoint_interactions=selected["checkpoint_interactions"],
                trace=f"{case_id}_traces.npz", trace_sha256=sha(output / f"{case_id}_traces.npz"),
                frozen_parameters_verified=metrics["frozen_parameters_verified"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Part 3 frozen-test summary directory")
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory under results/assignment4")
    args = parser.parse_args()
    source = args.source.resolve()
    choices, distributions = select(read_population(source))
    output = new_directory(args.output)
    write_json(output / "selection.json", dict(source=source.relative_to(ROOT).as_posix(), setting="B",
               typical="Minimum IQR-scaled absolute distance to the score and length medians, with modal termination.",
               surprising="Shortest success and highest-scoring failure in Q and PPO12, after selecting typical episodes.",
               distributions=distributions, cases=choices))
    torch.set_num_threads(1)
    receipts = [replay(case, output) for case in choices]
    write_json(output / "completed.json", dict(cases=receipts, decisions=sum(int(e["length"]) for e in receipts)))
    print(f"Verified {len(receipts)} frozen replays in {output.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
