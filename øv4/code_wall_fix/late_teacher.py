"""Distil short counterfactual action rollouts into a PPO late-game specialist."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import late_transfer as late
import refine_transfer as refine
import replay_transfer as replay
from cached_navigation import CachedNavigationTask
from late_specialist import SWITCH_SCORE, evaluate
from score_diagnostics import TemperaturePolicy, digest, np, runner, write_json


HORIZON = 24
VALIDATION_START = 981750
ARM = "branch_teacher"
TEMPERATURE = .2


def restore(task, snap):
    ale = task.env.unwrapped.ale
    ale.restoreState(snap.emulator)
    for name, value in snap.task_fields.items():
        setattr(task, name, value)
    task.cached_pellets = None
    return task.features(ale.getRAM())


def branch_outcomes(task, obs, policy, uniforms):
    snap = replay.Snapshot.save(task, obs, include_rng=True)
    start_score, start_lives = task.score, task.lives
    delta = np.zeros(task.n_actions)
    losses = np.zeros(task.n_actions, dtype=bool)
    successes = np.zeros(task.n_actions, dtype=bool)
    for action in range(task.n_actions):
        state = restore(task, snap)
        next_action = action
        for step in range(HORIZON):
            state, _, done, info = task.step(next_action)
            lost = info["lives"] < start_lives
            if done or lost:
                break
            # One common sequence of uniforms across branches reduces variation
            # from the opening policy after the counterfactual action.
            probabilities, _ = policy.policy_value(state)
            next_action = min(task.n_actions - 1, int(np.searchsorted(
                np.cumsum(probabilities), uniforms[step], side="right")))
        delta[action] = info["raw_score"] - start_score
        losses[action] = lost
        successes[action] = info["outcome"] == "success"
    restored = restore(task, snap)
    if not np.array_equal(restored, obs):
        raise RuntimeError("Branching changed the original observation.")
    return delta, losses, successes


def collect(model, seed, budget):
    random = np.random.default_rng(seed)
    branch_random = np.random.default_rng(seed + 2)
    policy = TemperaturePolicy(runner.load_model(model), TEMPERATURE)
    policy.rng = np.random.default_rng(seed + 1)
    task = CachedNavigationTask(runner.PROTOCOL, "B")
    counts = dict.fromkeys(refine.WEIGHTS, 0)
    states, priors, deltas, losses, successes, episodes, layouts = ([], [], [], [], [], [], [])
    try:
        while task.decisions < budget:
            start = task.decisions
            maze = replay.next_maze(counts)
            task.s = dict(task.s, maze=maze)
            episode = int(random.integers(1_000_000, 2**31 - 1))
            obs = task.reset(episode, min(runner.START_DECISIONS, budget - start))
            next_branch = 0
            while task.decisions < budget:
                enough = budget - task.decisions > task.n_actions * HORIZON
                if task.score >= SWITCH_SCORE and task.t >= next_branch and enough:
                    before = policy.policy_value(obs)[0]
                    delta, lost, success = branch_outcomes(task, obs, policy, branch_random.random(HORIZON))
                    states.append(obs)
                    priors.append(before)
                    deltas.append(delta)
                    losses.append(lost)
                    successes.append(success)
                    episodes.append(episode)
                    layouts.append(maze)
                    next_branch = task.t + (10 if obs[4] < .2 else 22)
                obs, _, done, _ = task.step(policy.act(obs))
                if done:
                    break
            counts[maze] += task.decisions - start
    finally:
        task.close()
    if sum(counts.values()) != budget or len(set(layouts)) != 3:
        raise RuntimeError(f"Unexpected teacher coverage: {counts}, {set(layouts)}")
    data = dict(states=np.array(states), priors=np.array(priors), deltas=np.array(deltas),
                losses=np.array(losses), successes=np.array(successes),
                episodes=np.array(episodes), layouts=np.array(layouts))
    return data, counts


def learn(parent, data, seed):
    agent = runner.load_model(parent)
    utility = np.minimum(data["deltas"] / 180., 3.) - 3. * data["losses"] + 1.5 * data["successes"]
    preferences = runner.softmax(np.log(data["priors"] + .003) + 1.2 * utility)
    targets = .3 * data["priors"] + .7 * preferences
    ids = np.arange(len(targets))
    random = np.random.default_rng(seed)
    training = ids[data["episodes"] % 5 != 0]
    validation = ids[data["episodes"] % 5 == 0]

    def cross_entropy():
        logits = agent.forward(data["states"][validation])[2]
        probabilities = runner.softmax(logits / TEMPERATURE)
        return float(-np.mean(np.sum(targets[validation] * np.log(probabilities + 1e-12), axis=1)))

    if not len(training) or not len(validation):
        raise RuntimeError("The teacher dataset needs independent episodes for its fit check.")
    before = cross_entropy()
    for _ in range(16):
        shuffled = random.permutation(training)
        for batch in np.array_split(shuffled, max(1, len(shuffled) // 64)):
            x = data["states"][batch]
            probabilities = runner.softmax(agent.forward(x)[2] / TEMPERATURE)
            dlogits = (probabilities - targets[batch]) / (len(batch) * TEMPERATURE)
            gradients = agent.backward(x, dlogits, np.zeros(len(batch)))
            for key, gradient in gradients.items():
                agent.opt[key].step(agent.p[key], gradient, 1e-4)
    info = dict(train_states=len(training), validation_states=len(validation),
                validation_cross_entropy_before=before, validation_cross_entropy_after=cross_entropy(),
                differentiated_life_loss_states=int(np.any(data["losses"], axis=1).sum()
                                                    - np.all(data["losses"], axis=1).sum()),
                mean_probability_shift=float(np.abs(targets - data["priors"]).sum(1).mean() / 2))
    return agent, info


def fit(root, row, budget, episodes):
    seed = row["seed"]
    parent = Path(row["model"])
    if digest(parent) != row["model_sha256"]:
        raise ValueError(f"Checkpoint changed: {parent}")
    folder = root / ARM / f"seed_{seed}"
    folder.mkdir(parents=True)
    data, counts = collect(parent, 57000 + seed, budget)
    np.savez_compressed(folder / "teacher_data.npz", **data)
    print(f"{ARM} seed {seed}: gathered {len(data['states'])} branch states and {counts} interactions", flush=True)
    student, diagnostics = learn(parent, data, 57100 + seed)
    model = folder / "model.npz"
    runner.save_model(student, model)
    result = dict(arm=ARM, seed=seed, model=str(model), model_sha256=digest(model),
                  opening_model=str(parent), opening_sha256=row["model_sha256"],
                  switch_score=SWITCH_SCORE, interactions=counts, feature_type="navigation",
                  teacher=diagnostics)
    result["validation"] = {layout: evaluate(root, result, layout, episodes, VALIDATION_START, "validation")
                            for layout in (0, 2, 3)}
    write_json(folder / "result.json", result)
    return result


def selfcheck():
    t = CachedNavigationTask(runner.PROTOCOL, "B")
    rng = np.random.default_rng(2230)
    try:
        for maze in refine.WEIGHTS:
            t.s = dict(t.s, maze=maze)
            t.reset(843 + maze)
            for action in rng.integers(t.n_actions, size=90):
                obs, _, done, _ = t.step(int(action))
                assert not done
            snap = replay.Snapshot.save(t, obs, True)
            actions = rng.integers(t.n_actions, size=40)
            previous = None
            for _ in range(2):
                np.testing.assert_array_equal(restore(t, snap), obs)
                trial = []
                for action in actions:
                    result = t.step(int(action))
                    trial.append(result)
                    if result[2]:
                        break
                if previous is not None:
                    assert len(previous) == len(trial)
                    for a, b in zip(previous, trial):
                        np.testing.assert_array_equal(a[0], b[0])
                        assert a[1:] == b[1:]
                previous = trial
        print("In-place branch restoration produced identical counterfactual trajectories.")
    finally:
        t.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--budget", type=int, default=260_000)
    parser.add_argument("--episodes", type=int, default=80)
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.parent or not args.results or args.budget < 50_000 or args.episodes < 1:
        parser.error("Provide parent checkpoints, a result directory and a suitable interaction budget.")
    parents = [r for r in json.loads(args.parent.read_text(encoding="utf-8"))["runs"]
               if r["arm"] == late.BASELINE]
    if len(parents) != 3 or any(set(r["interactions"]) != {"0", "2", "3"} for r in parents):
        parser.error("Expected three parents trained on layouts 0, 2 and 3.")
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = (Path(__file__).name, "late_specialist.py", "replay_transfer.py", "late_transfer.py",
             "refine_transfer.py", "cached_navigation.py", "compare_navigation.py",
             "mspacman.py", "reference_policy.py", "part2_run.py", "score_diagnostics.py")
    write_json(root / "manifest.json", dict(
        parents=parents, source=str(args.parent.resolve()), budget=args.budget,
        episodes=args.episodes, validation_seed=VALIDATION_START,
        horizon=HORIZON, switch_score=SWITCH_SCORE, training_layouts=[0, 2, 3],
        protocol=runner.PROTOCOL,
        code_sha256={name: digest(Path(__file__).with_name(name)) for name in names}))
    with ProcessPoolExecutor(3) as pool:
        jobs = [pool.submit(fit, root, row, args.budget, args.episodes) for row in parents]
        runs = [job.result() for job in jobs]
    means = refine.comparison(runs, (0, 2, 3), ("success_rate", "mean_score", "reached_1500", "conversion"),
                              lambda r, layout: r["validation"][layout])
    write_json(root / "comparison.json", dict(runs=runs, means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()

