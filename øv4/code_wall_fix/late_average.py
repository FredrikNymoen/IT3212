"""Average matched PPO continuations from the same opening checkpoint."""
import argparse
import json
from pathlib import Path

from late_evaluate import verify_sources
from score_diagnostics import digest, np, runner, write_json


def average(root, siblings):
    seed = siblings[0]["seed"]
    parents = {(r["parent"], r["parent_sha256"]) for r in siblings}
    if len(parents) != 1:
        raise ValueError(f"Seed {seed} does not have a shared parent.")
    parent, parent_hash = parents.pop()
    if digest(Path(parent)) != parent_hash:
        raise ValueError(f"The shared parent changed: {parent}")
    agents = []
    interactions = dict.fromkeys(("0", "2", "3"), 0)
    for row in siblings:
        model = Path(row["model"])
        if row["feature_type"] != "navigation" or digest(model) != row["model_sha256"]:
            raise ValueError(f"Invalid sibling checkpoint: {model}")
        agents.append(runner.load_model(model))
        for maze, count in row["interactions"].items():
            interactions[maze] += count
    averaged = runner.load_model(Path(parent))
    for key in averaged.p:
        values = np.stack([agent.p[key] for agent in agents])
        if values.shape[1:] != averaged.p[key].shape:
            raise ValueError(f"Incompatible parameter shape for {key}")
        averaged.p[key] = values.mean(axis=0)
    folder = root / f"seed_{seed}"
    folder.mkdir()
    model = folder / "model.npz"
    runner.save_model(averaged, model)
    return dict(arm="averaged", seed=seed, model=str(model), model_sha256=digest(model),
                parent=parent, parent_sha256=parent_hash, interactions=interactions,
                feature_type="navigation", inputs=siblings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    paths = {}
    for path in args.inputs:
        verify_sources(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        paths[str(path.resolve())] = digest(path)
        for row in result["runs"]:
            groups.setdefault(row["seed"], []).append(row)
    if len(groups) != 3 or len({len(r) for r in groups.values()}) != 1:
        parser.error("Expected matching arms for three training seeds.")
    matching = [{r["arm"] for r in group} for group in groups.values()]
    if any(s != matching[0] for s in matching[1:]) or len(matching[0]) < 2:
        parser.error("Expected the same two or more siblings for every training seed.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "late_evaluate.py", "part2_run.py", "score_diagnostics.py")
    write_json(root / "manifest.json", dict(
        sources=paths, matched_arms=sorted(matching[0]), additional_interactions=0,
        training_layouts=[0, 2, 3],
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in names}))
    write_json(root / "comparison.json", dict(runs=[average(root, r) for r in groups.values()]))
    print("Saved three averaged PPO checkpoints.", flush=True)


if __name__ == "__main__":
    main()

