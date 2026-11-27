"""Part 3 one-factor sweeps using the Part 2 agents and MsPacMan task."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import platform
from tempfile import mkdtemp
from time import perf_counter

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

from ppo import Checkpoints, FrozenPolicy, RewardScale, SharedActionPolicy
from q_learning import (
    ROOT, TabularQ, checkpoint_points, evaluate, lock_selection, make_environment,
    require, sha, source_hashes, verify_sources, write_csv, write_json,
)


PART2_BASELINE = {
    "Q-learning": dict(alpha=.15, gamma=.995, epsilon_start=1., epsilon_end=.05,
                       epsilon_decay_fraction=.8),
    "PPO": dict(gamma=.995, entropy_scale=1., learning_rate_scale=1., n_epochs=6),
}
VARY = {"Q-learning": {"epsilon_end", "gamma", "alpha"},
        "PPO": {"gamma", "entropy_scale", "learning_rate_scale", "n_epochs"}}


def experiment_protocol(setting="B", budget=262144, training_seeds=(17, 29, 101, 314, 2718),
                        development_episodes=10, quick_check=False):
    require(setting in ("A", "B", "C"), "Use a training setting A, B or C.")
    if quick_check:
        budget, training_seeds, development_episodes = 2048, (17,), 1
    seeds = sorted(set(training_seeds))
    require(quick_check or len(seeds) >= 5, "Part 3 requires at least five distinct training seeds.")
    require(seeds and all(type(seed) is int and 0 <= seed < 1000000 for seed in seeds),
            "Use nonnegative integer training seeds below 1,000,000.")
    require(type(budget) is int and budget > 0 and budget % 2048 == 0,
            "Use a positive interaction budget divisible by one PPO rollout (2,048).")
    require(type(development_episodes) is int and 0 < development_episodes <= 1000,
            "Use 1 to 1,000 development episodes per checkpoint.")
    sources = source_hashes()
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha(__file__)
    return dict(
        experiment="part3_one_factor", quick_check=quick_check, setting=setting, budget=budget,
        training_seeds=seeds, checkpoints=checkpoint_points(budget),
        development_seeds=list(range(1350000, 1350000 + development_episodes)),
        evaluation_seeds=list(range(1360000, 1360000 + (1 if quick_check else 200))),
        policy="Greedy Q with seeded uniform ties; PPO argmax",
        source_hashes=sources, python=platform.python_version(), os=platform.platform(),
        cpu=platform.processor() or platform.machine(), device="cpu", numerical_threads=1,
        packages={name: version(name) for name in
                  ("numpy", "pandas", "torch", "gymnasium", "ale-py", "stable-baselines3", "threadpoolctl")},
    )


def record_id(record):
    return sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def study_folder(protocol):
    mode = "quick_check" if protocol["quick_check"] else "study"
    return ROOT / "results/assignment4/part3" / mode / record_id(protocol)


def job_folder(protocol, algorithm, settings, seed):
    return study_folder(protocol) / f"{algorithm}_{record_id(settings)}" / f"seed_{seed}"


def publish(attempt, destination):
    """Make completed work visible atomically and keep interrupted attempts available."""
    try:
        attempt.rename(destination)
    except OSError:
        if not (destination / "completed.json").is_file():
            raise


class EntropyCheckpoints(Checkpoints):
    def __init__(self, budget, points, save, entropy_scale):
        super().__init__(budget, points, save)
        self.entropy_scale = entropy_scale

    def _on_step(self):
        super()._on_step()
        self.model.ent_coef *= self.entropy_scale
        return True


def fit_q(protocol, settings, seed, output, checkpoint):
    budget, points = protocol["budget"], set(protocol["checkpoints"])
    model = TabularQ(seed, alpha=settings["alpha"], gamma=settings["gamma"])
    start, end = settings["epsilon_start"], settings["epsilon_end"]
    decay_steps = settings["epsilon_decay_fraction"] * budget
    env = make_environment(protocol["setting"])
    rows = []
    episode = 0
    try:
        obs, _ = env.reset(seed=seed * 1000)
        checkpoint(model, 0)
        state = model.state(obs)
        for step in range(budget):
            epsilon = max(end, start - (start - end) * step / decay_steps)
            action = model.action(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = model.state(obs)
            done = terminated or truncated
            model.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                rows.append(dict(interactions=step + 1, training_episode_seed=seed * 1000 + episode,
                                 score=info["score"], task_return=info["task_return"],
                                 success=int(info["success"]), length=info["episode_steps"],
                                 reason=info["reason"]))
                episode += 1
                obs, _ = env.reset(seed=seed * 1000 + episode)
                state = model.state(obs)
            if step + 1 in points:
                checkpoint(model, step + 1)
    finally:
        env.close()
    write_csv(output / "training.csv", rows)


def fit_ppo(protocol, settings, seed, output, checkpoint):
    budget = protocol["budget"]
    envs = DummyVecEnv([
        lambda i=i: Monitor(RewardScale(make_environment(protocol["setting"])),
                            str(output / f"training_{i}"),
                            info_keywords=("score", "task_return", "success", "reason", "episode_steps"))
        for i in range(4)
    ])
    lr_scale = settings["learning_rate_scale"]
    ent_scale = settings["entropy_scale"]
    try:
        model = PPO(SharedActionPolicy, envs,
                    learning_rate=lambda progress: lr_scale * (.00005 + (.0003 - .00005) * max(0, progress)),
                    gamma=settings["gamma"], gae_lambda=.97, ent_coef=.02 * ent_scale, n_steps=512,
                    batch_size=256, n_epochs=settings["n_epochs"], clip_range=.15, target_kl=.015,
                    seed=seed, device="cpu", verbose=0)
        points = [point for point in protocol["checkpoints"] if point < budget]
        callback = EntropyCheckpoints(budget, points, lambda point: checkpoint(model, point), ent_scale)
        model.learn(budget, callback=callback)
        checkpoint(model, budget)
    finally:
        envs.close()


def train_one(protocol, algorithm, settings, seed):
    verify_sources(protocol["source_hashes"])
    destination = job_folder(protocol, algorithm, settings, seed)
    config = dict(protocol=protocol, algorithm=algorithm, settings=settings, training_seed=seed,
                  fixed_ppo=dict(envs=4, n_steps=512, batch_size=256, gae_lambda=.97, clip_range=.15,
                                 target_kl=.015, learning_rate=(.0003, .00005), entropy=(.02, .002))
                  if algorithm == "PPO" else None)
    # JSON also gives tuples a consistent representation when checking a cached run.
    config = json.loads(json.dumps(config))
    done = destination / "completed.json"
    if not done.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = Path(mkdtemp(prefix=f".seed_{seed}_", dir=destination.parent))
        write_json(output / "configuration.json", config)
        torch.set_num_threads(1)
        curves = []
        seconds = 0.
        clock = perf_counter()

        def checkpoint(model, point):
            nonlocal seconds, clock
            seconds += perf_counter() - clock
            if algorithm == "Q-learning":
                require(model.update_calls == point, "Q update count differs from interactions.")
                path = output / f"checkpoint_{point}.npz"
                agent, extra = model, {"q_states": len(model.table)}
            else:
                require(model.num_timesteps == point, "PPO interaction count differs from budget.")
                path = output / f"checkpoint_{point}.zip"
                agent, extra = FrozenPolicy(model), {"ppo_epochs_started": int(model._n_updates)}
            model.save(path)
            metrics = evaluate(agent, algorithm, protocol["setting"], protocol["development_seeds"],
                               output, f"development_{point}")
            curves.append(dict(interactions=point, model=path.name, model_sha256=sha(path),
                               training_seconds=seconds, **extra, **metrics))
            clock = perf_counter()

        try:
            with threadpool_limits(limits=1):
                fit = fit_q if algorithm == "Q-learning" else fit_ppo
                fit(protocol, settings, seed, output, checkpoint)
            write_csv(output / "curves.csv", curves)
            selected = lock_selection(output, algorithm, protocol["setting"], seed, protocol["budget"],
                                      curves, protocol["source_hashes"])
            write_json(output / "completed.json", dict(selected=selected, training_seconds=seconds))
            publish(output, destination)
        except Exception as error:
            raise RuntimeError(f"Training failed. Partial records are in {output}") from error
    require(json.loads((destination / "configuration.json").read_text(encoding="utf-8")) == config,
            f"Configuration differs from the cached run at {destination}")
    return destination


def run_jobs(task, jobs, workers):
    require(type(workers) is int and workers >= 1, "Use at least one worker.")
    if workers == 1:
        for job in jobs:
            print(task(*job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(task, *job) for job in jobs]):
                print(future.result(), flush=True)


def validate_values(parameter, values):
    for value in values:
        require(isinstance(value, (int, float)) and isfinite(value),
                f"Use finite numbers for {parameter}.")
        if parameter == "n_epochs":
            require(type(value) is int and value >= 1, "PPO epochs must be positive integers.")
        elif parameter in ("epsilon_end", "gamma", "alpha"):
            require(0 <= value <= 1, f"{parameter} must be between 0 and 1.")
        else:
            minimum = 0 if parameter == "entropy_scale" else 1e-12
            require(value >= minimum, f"{parameter} must be at least {minimum:g}.")


def run_sweep(protocol, algorithm, incumbent, parameter, alternatives, workers=1):
    """Train each candidate on matched seeds and rank development checkpoints only."""
    require(parameter in VARY[algorithm], f"Unknown parameter for {algorithm}: {parameter}")
    require(incumbent.keys() == PART2_BASELINE[algorithm].keys(), "Incomplete agent settings.")
    baseline_value = incumbent[parameter]
    values = list(dict.fromkeys([baseline_value, *alternatives]))
    require(len(values) >= 2, "Add at least one different value to compare.")
    validate_values(parameter, values)
    candidates = {value: {**incumbent, parameter: value} for value in values}
    seeds = protocol["training_seeds"]
    run_jobs(train_one, [(protocol, algorithm, settings, seed)
                         for settings in candidates.values() for seed in seeds], workers)
    records, curves = [], []
    for value, settings in candidates.items():
        for seed in seeds:
            folder = job_folder(protocol, algorithm, settings, seed)
            result = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "curves.csv").assign(value=value, training_seed=seed)
            point = result["selected"]["checkpoint_interactions"]
            record = frame.loc[frame["interactions"].eq(point)].iloc[0].to_dict()
            records.append(dict(record, checkpoint_training_seconds=record["training_seconds"],
                                training_seconds=result["training_seconds"],
                                run_id=str(folder.relative_to(ROOT))))
            curves.append(frame)
    per_seed = pd.DataFrame(records).assign(incumbent=lambda frame: frame["value"].eq(baseline_value))
    baseline = per_seed.loc[per_seed["incumbent"]].set_index("training_seed")["success_rate"]
    per_seed["paired_success_delta"] = per_seed["success_rate"] - per_seed["training_seed"].map(baseline)
    ranking = (per_seed.groupby("value").agg(
        training_seeds=("training_seed", "count"), success_mean=("success_rate", "mean"),
        success_sd=("success_rate", "std"), score_mean=("mean_score", "mean"),
        return_mean=("mean_return", "mean"), failure_mean=("failure_rate", "mean"),
        success_length=("success_length", "mean"), failure_length=("failure_length", "mean"),
        train_seconds=("training_seconds", "mean"), paired_success_delta=("paired_success_delta", "mean"),
        incumbent=("incumbent", "first"),
    ).reset_index().sort_values(["success_mean", "score_mean", "incumbent"],
                               ascending=False, kind="stable"))
    winner = ranking.iloc[0]["value"]
    selected = candidates[winner]
    sweep = dict(algorithm=algorithm, parameter=parameter, incumbent=incumbent, values=values)
    folder = study_folder(protocol) / "sweeps" / f"{algorithm}_{parameter}_{record_id(sweep)}"
    folder.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(folder / "per_seed.csv", index=False)
    ranking.to_csv(folder / "ranking.csv", index=False)
    learning = pd.concat(curves, ignore_index=True)
    learning.to_csv(folder / "curves.csv", index=False)
    write_json(folder / "selection.json", dict(sweep, selected=selected,
               rule="Mean development success, then mean raw score, then prefer incumbent on ties.",
               source_runs=per_seed["run_id"].tolist()))
    return selected, ranking, per_seed, learning, folder


def evaluate_one(protocol, algorithm, settings, seed):
    verify_sources(protocol["source_hashes"])
    folder = job_folder(protocol, algorithm, settings, seed)
    selected = json.loads((folder / "selected.json").read_text(encoding="utf-8"))
    model_path = folder / selected["model"]
    require(sha(model_path) == selected["model_sha256"], f"Checkpoint changed at {model_path}")
    destination = folder / "held_out"
    if not (destination / "completed.json").is_file():
        output = Path(mkdtemp(prefix=".held_out_", dir=folder))
        write_json(output / "locked.json", dict(selected=selected, evaluation_seeds=protocol["evaluation_seeds"],
                   policy=protocol["policy"]))
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            loader = TabularQ.load if algorithm == "Q-learning" else FrozenPolicy.load
            metrics = evaluate(loader(model_path), algorithm, protocol["setting"],
                               protocol["evaluation_seeds"], output, "evaluation")
        write_json(output / "completed.json", metrics)
        publish(output, destination)
    return destination


def evaluate_locked(protocol, final_settings, workers=1):
    """Evaluate the Part 2 and final Part 3 settings after all choices are locked."""
    comparisons = dict(baseline=PART2_BASELINE, selected=final_settings)
    seeds = protocol["training_seeds"]
    unique = {(algorithm, record_id(settings)): settings for methods in comparisons.values()
              for algorithm, settings in methods.items()}
    jobs = [(protocol, algorithm, settings, seed)
            for (algorithm, _), settings in unique.items() for seed in seeds]
    folder = study_folder(protocol) / f"held_out_{record_id(comparisons)}"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "locked.json", dict(comparisons=comparisons,
               source_runs=[str(job_folder(*job).relative_to(ROOT)) for job in jobs],
               evaluation_seeds=protocol["evaluation_seeds"], policy=protocol["policy"]))
    run_jobs(evaluate_one, jobs, workers)
    records = []
    for label, methods in comparisons.items():
        for algorithm, settings in methods.items():
            for seed in seeds:
                run = job_folder(protocol, algorithm, settings, seed)
                metrics = json.loads((run / "held_out/completed.json").read_text(encoding="utf-8"))
                training = json.loads((run / "completed.json").read_text(encoding="utf-8"))
                records.append(dict(label=label, algorithm=algorithm, training_seed=seed, **metrics,
                                    training_interactions=protocol["budget"],
                                    training_seconds=training["training_seconds"],
                                    checkpoint_interactions=training["selected"]["checkpoint_interactions"],
                                    run_id=str(run.relative_to(ROOT))))
    frame = pd.DataFrame(records)
    frame.to_csv(folder / "per_seed.csv", index=False)
    return frame
