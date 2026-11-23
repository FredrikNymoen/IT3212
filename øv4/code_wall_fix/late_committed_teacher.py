"""Improve the branch teacher with longer action requests and held-episode early stopping."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import late_specialist as specialist
import late_teacher as teacher
import late_transfer as late
import refine_transfer as refine
from replay_transfer import Snapshot
from score_diagnostics import digest, np, runner, write_json


ARM = "committed_teacher"
HORIZON = 32
REQUEST_LENGTH = 3


def branches(task, obs, policy, uniforms):
    snap = Snapshot.save(task, obs, include_rng=True)
    score, lives = task.score, task.lives
    delta = np.zeros(task.n_actions)
    losses = np.zeros(task.n_actions, dtype=bool)
    successes = np.zeros(task.n_actions, dtype=bool)
    for action in range(task.n_actions):
        state = teacher.restore(task, snap)
        for i in range(HORIZON):
            if i < REQUEST_LENGTH:
                request = action
            else:
                probabilities, _ = policy.policy_value(state)
                request = min(task.n_actions-1, int(np.searchsorted(np.cumsum(probabilities),
                                                                    uniforms[i], side="right")))
            state, _, done, info = task.step(request)
            lost = info["lives"] < lives
            if done or lost:
                break
        delta[action] = info["raw_score"] - score
        losses[action] = lost
        successes[action] = info["outcome"] == "success"
    if not np.array_equal(teacher.restore(task, snap), obs):
        raise RuntimeError("Branching changed the original observation.")
    return delta, losses, successes


def learn(parent, data, seed):
    agent = runner.load_model(parent)
    temperature = teacher.TEMPERATURE
    utility = np.minimum(data["deltas"] / 180., 3.) - 4. * data["losses"] + 1.5 * data["successes"]
    preferences = runner.softmax(np.log(data["priors"] + .004) + 1.2 * utility)
    targets = .35 * data["priors"] + .65 * preferences
    train = np.flatnonzero(data["episodes"] % 5 != 0)
    validation = np.flatnonzero(data["episodes"] % 5 == 0)
    random = np.random.default_rng(seed)

    def loss():
        logits = agent.forward(data["states"][validation])[2]
        probs = runner.softmax(logits / temperature)
        return float(-np.mean(np.sum(targets[validation] * np.log(probs + 1e-12), axis=1)))

    if not len(train) or not len(validation):
        raise RuntimeError("Need independent training and checking episodes.")
    initial = best_loss = loss()
    best = {name: value.copy() for name, value in agent.p.items()}
    selected_epoch = 0
    for epoch in range(1, 13):
        for batch in np.array_split(random.permutation(train), max(1, len(train) // 64)):
            x = data["states"][batch]
            probs = runner.softmax(agent.forward(x)[2] / temperature)
            gradient = agent.backward(x, (probs-targets[batch]) / (len(batch)*temperature),
                                      np.zeros(len(batch)))
            for name, value in gradient.items():
                agent.opt[name].step(agent.p[name], value, 5e-5)
        value = loss()
        if value < best_loss:
            best_loss, selected_epoch = value, epoch
            best = {name: value.copy() for name, value in agent.p.items()}
    agent.p = best
    changed = np.any(data["losses"], axis=1) & ~np.all(data["losses"], axis=1)
    return agent, dict(train_states=len(train), validation_states=len(validation),
                       validation_cross_entropy_before=initial, validation_cross_entropy_after=best_loss,
                       selected_epoch=selected_epoch, differentiated_life_loss_states=int(changed.sum()))


def fit(root, row, budget, episodes):
    replacements = dict(HORIZON=HORIZON, ARM=ARM, branch_outcomes=branches, learn=learn)
    previous = {name: getattr(teacher, name) for name in replacements}
    for name, value in replacements.items():
        setattr(teacher, name, value)
    try:
        return teacher.fit(root, row, budget, episodes)
    finally:
        for name, value in previous.items():
            setattr(teacher, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=240_000)
    parser.add_argument("--episodes", type=int, default=75)
    args = parser.parse_args()
    parents = [r for r in json.loads(args.parent.read_text(encoding="utf-8"))["runs"]
               if r["arm"] == late.BASELINE]
    if len(parents) != 3 or any(set(r["interactions"]) != {"0", "2", "3"} for r in parents):
        parser.error("Expected three parents trained on layouts 0, 2 and 3.")
    if args.budget < 50_000 or args.episodes < 1:
        parser.error("Use at least 50,000 interactions and a positive episode count.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "late_teacher.py", "late_specialist.py", "replay_transfer.py",
             "late_transfer.py", "cached_navigation.py", "mspacman.py", "part2_run.py",
             "score_diagnostics.py")
    write_json(root / "manifest.json", dict(
        parents=parents, source=str(args.parent.resolve()), budget=args.budget,
        episodes=args.episodes, validation_seed=teacher.VALIDATION_START,
        horizon=HORIZON, first_request_decisions=REQUEST_LENGTH, switch=specialist.SWITCH_SCORE,
        training_layouts=[0,2,3], protocol=runner.PROTOCOL,
        code_sha256={name:digest(Path(__file__).with_name(name)) for name in names}))
    with ProcessPoolExecutor(3) as pool:
        jobs = [pool.submit(fit, root, r, args.budget, args.episodes) for r in parents]
        runs = [job.result() for job in jobs]
    means = refine.comparison(runs, (0, 2, 3), ("success_rate", "mean_score", "reached_1500", "conversion"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

