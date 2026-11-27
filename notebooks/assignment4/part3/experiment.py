"""Controlled Setting B mechanism experiments and frozen sticky-action diagnostic."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo import FrozenPolicy, RewardScale, SharedActionPolicy
from mspacman_env import TaskConfig
from q_learning import (
    ROOT, TabularQ, checkpoint_points, evaluate, make_environment, new_directory,
    require, sha, source_hashes, verify_sources, write_csv, write_json,
)

SETTING = "B"
BUDGET = 262144
ROLLOUT = 2048
SEEDS = (17, 29, 101, 314, 2718)
DEVELOPMENT = tuple(range(1300000, 1300010))
EVALUATION = tuple(range(1310000, 1310200))
DIAGNOSTIC = tuple(range(1320000, 1320020))
PREDICTIONS = "reports/assignment4/Sections/01_predictions.tex"

BASELINES = {
    "q": dict(alpha=.15, gamma=.995, epsilon_start=1., epsilon_end=.05, epsilon_fraction=.8),
    "ppo": dict(lr_start=.0003, lr_end=.00005, gamma=.995, gae_lambda=.97,
                entropy_start=.02, entropy_end=.002, entropy_scale=1.,
                n_envs=4, n_steps=512, batch_size=256, n_epochs=6, clip_range=.15,
                target_kl=.015),
}
VARIANTS = {
    "q-fast-epsilon": ("q", "exploration", dict(epsilon_fraction=.2)),
    "q-short-gamma": ("q", "planning horizon", dict(gamma=.95)),
    "q-low-alpha": ("q", "update size", dict(alpha=.05)),
    "ppo-short-gamma": ("ppo", "planning horizon", dict(gamma=.95)),
    "ppo-no-entropy": ("ppo", "policy exploration", dict(entropy_scale=0.)),
    "ppo-low-lr": ("ppo", "update size", dict(lr_start=.0001)),
    "ppo-two-epochs": ("ppo", "PPO data reuse", dict(n_epochs=2)),
}
ALGORITHM = {"q": "Q-learning", "ppo": "PPO"}


def configurations():
    configs = {f"{name}-baseline": dict(algorithm=name, parameters=params)
               for name, params in BASELINES.items()}
    for name, (algorithm, factor, change) in VARIANTS.items():
        baseline = BASELINES[algorithm]
        parameters = baseline | change
        require(sum(parameters[key] != baseline[key] for key in baseline) == 1,
                f"{name} must change exactly one parameter")
        configs[name] = dict(algorithm=algorithm, factor=factor, baseline=f"{algorithm}-baseline",
                             change=change, parameters=parameters)
    return configs


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def linear(start, end, fraction_remaining):
    return end + (start - end) * max(0, fraction_remaining)


def cpu_name():
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def train_q(parameters, seed, budget, checkpoints, save, output):
    agent = TabularQ(seed, alpha=parameters["alpha"], gamma=parameters["gamma"])
    env = make_environment(SETTING)
    rows = []
    episode_seed = seed * 1000
    try:
        obs, _ = env.reset(seed=episode_seed)
        save(0, agent)
        epsilon_sum = 0.
        start, end = parameters["epsilon_start"], parameters["epsilon_end"]
        for step in range(budget):
            state = agent.state(obs)
            epsilon = max(end, start - (start - end) * step / (parameters["epsilon_fraction"] * budget))
            action = agent.action(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated
            agent.update(state, action, reward, agent.state(obs), terminal)
            epsilon_sum += epsilon
            if terminal:
                rows.append(dict(interactions=step + 1, episode_seed=episode_seed,
                                 score=info["score"], task_return=info["task_return"],
                                 success=int(info["success"]), length=info["episode_steps"],
                                 reason=info["reason"], mean_epsilon=epsilon_sum / info["episode_steps"]))
                if step + 1 < budget:
                    episode_seed += 1
                    obs, _ = env.reset(seed=episode_seed)
                    epsilon_sum = 0.
            if step + 1 in checkpoints:
                save(step + 1, agent)
        require(agent.update_calls == budget, "Expected one Q update per real transition")
    finally:
        env.close()
    write_csv(output / "training.csv", rows)
    return dict(q_updates=agent.update_calls, visited_q_keys=len(agent.table))


class PPOCheckpoints(BaseCallback):
    def __init__(self, parameters, budget, checkpoints, save):
        super().__init__()
        self.parameters = parameters
        self.budget = budget
        self.checkpoints = checkpoints
        self.save = save

    def _on_rollout_start(self):
        if self.num_timesteps in self.checkpoints:
            self.save(self.num_timesteps, self.model)

    def _on_step(self):
        p = self.parameters
        self.model.ent_coef = p["entropy_scale"] * linear(
            p["entropy_start"], p["entropy_end"], 1 - self.num_timesteps / self.budget)
        return True


def train_ppo(parameters, seed, budget, checkpoints, save, output):
    p = parameters
    start_rate, end_rate = p["lr_start"], p["lr_end"]
    envs = DummyVecEnv([
        lambda i=i: Monitor(RewardScale(make_environment(SETTING)), str(output / f"training_{i}"),
                            info_keywords=("score", "success", "reason", "episode_steps"))
        for i in range(p["n_envs"])
    ])
    try:
        model = PPO(
            SharedActionPolicy, envs,
            learning_rate=lambda remaining: end_rate + (start_rate - end_rate) * max(0, remaining),
            gamma=p["gamma"], gae_lambda=p["gae_lambda"],
            ent_coef=p["entropy_start"] * p["entropy_scale"], n_steps=p["n_steps"],
            batch_size=p["batch_size"], n_epochs=p["n_epochs"], clip_range=p["clip_range"],
            target_kl=p["target_kl"], seed=seed, device="cpu", verbose=0,
        )
        model.learn(budget, callback=PPOCheckpoints(p, budget, checkpoints, save))
        require(model.num_timesteps == budget, "PPO did not use the declared interaction budget")
        save(budget, model)
        return dict(sb3_epoch_counter=int(model._n_updates))
    finally:
        envs.close()


def load_model(path, config, interactions):
    algorithm, p = config["algorithm"], config["parameters"]
    model = TabularQ.load(path) if algorithm == "q" else FrozenPolicy.load(path)
    trained = model.update_calls if algorithm == "q" else model.model.num_timesteps
    require(trained == interactions, f"Unexpected interaction count in {path}")
    if algorithm == "q":
        require((model.alpha, model.gamma) == (p["alpha"], p["gamma"]),
                f"Unexpected Q update parameters in {path}")
    else:
        ppo = model.model
        expected = {key: p[key] for key in ("gamma", "gae_lambda", "n_steps", "batch_size",
                                             "n_epochs", "target_kl", "n_envs")}
        require(all(getattr(ppo, key) == value for key, value in expected.items())
                and np.isclose(ppo.lr_schedule(1), p["lr_start"])
                and np.isclose(ppo.lr_schedule(0), p["lr_end"])
                and np.isclose(ppo.ent_coef, p["entropy_end"] * p["entropy_scale"])
                and np.isclose(ppo.clip_range(1), p["clip_range"]),
                f"Unexpected PPO parameters in {path}")
        ppo.policy.set_training_mode(False)
    return model


def trace_metrics(output, name, algorithm):
    with np.load(output / f"{name}_traces.npz", allow_pickle=False) as trace:
        actions = trace["action"]
        episodes = trace["episode"]
        counts = np.bincount(episodes)
        values = trace["values"]
        within_episode = episodes[1:] == episodes[:-1]
        repeated = actions[1:] == actions[:-1]
        metrics = dict(
            action_repeat_rate=float(np.mean(repeated[within_episode])) if within_episode.any() else None,
            mean_first_100_score_gain=float(np.mean(np.bincount(
                episodes, weights=trace["raw_reward"] * (trace["step"] < 100)))),
            mean_reward_events_ge_200=float(np.mean(np.bincount(
                episodes, weights=trace["raw_reward"] >= 200))),
        )
        if algorithm == "ppo":
            entropy = -(values * np.log(np.maximum(values, 1e-30))).sum(axis=1)
            metrics["mean_policy_entropy"] = float(np.mean(np.bincount(episodes, weights=entropy) / counts))
        return metrics


def job(run, plan, name, seed):
    torch.set_num_threads(1)
    output = run / f"{name}_seed{seed}"
    output.mkdir(exist_ok=True)
    completion = output / "completed.json"
    if completion.exists():
        result = read_json(completion)
        attempt = output / result["attempt"]
        require(result["plan_sha256"] == sha(run / "plan.json")
                and result["configuration"] == name and result["training_seed"] == seed
                and result["training_interactions"] == plan["budget"]
                and all(sha(attempt / filename) == digest for filename, digest in result["files"].items()),
                f"Existing job failed verification: {output}")
        return result

    staging = Path(mkdtemp(prefix="attempt-", dir=output))
    config = plan["configurations"][name]
    algorithm = config["algorithm"]
    budget = plan["budget"]
    points = checkpoint_points(budget)
    curves = []
    elapsed = 0.
    started = clock = perf_counter()

    def save(point, model):
        nonlocal clock, elapsed
        elapsed += perf_counter() - clock
        path = staging / f"checkpoint_{point}{'.npz' if algorithm == 'q' else '.zip'}"
        model.save(path)
        # Development evaluation uses a separate environment and read-only inference.
        # Reload only after training, since loading SB3 reseeds global RNGs.
        frozen = model if algorithm == "q" else FrozenPolicy(model)
        metrics = evaluate(frozen, ALGORITHM[algorithm], SETTING, plan["development_seeds"],
                           staging, f"development_{point}")
        curves.append(dict(interactions=point, model=path.name, model_sha256=sha(path),
                           training_seconds=elapsed, **metrics))
        clock = perf_counter()

    trainer = train_q if algorithm == "q" else train_ppo
    updates = trainer(config["parameters"], seed, budget, set(points), save, staging)
    require([row["interactions"] for row in curves] == points, "Incomplete learning curve")
    write_csv(staging / "curves.csv", curves)
    final = curves[-1]
    path = staging / final["model"]
    require(sha(path) == final["model_sha256"], "Final checkpoint changed")
    frozen = load_model(path, config, budget)
    metrics = evaluate(frozen, ALGORITHM[algorithm], SETTING, plan["evaluation_seeds"],
                       staging, "evaluation", trace=True)
    training_paths = [staging / "training.csv"] if algorithm == "q" else sorted(staging.glob("*.monitor.csv"))
    training_scores = []
    for log in training_paths:
        if not log.exists():
            continue
        with log.open(encoding="utf-8", newline="") as handle:
            if algorithm == "ppo":
                next(handle)  # Monitor metadata line
            training_scores.extend(float(row["score"]) for row in csv.DictReader(handle))
    result = dict(configuration=name, training_seed=seed, algorithm=algorithm, attempt=staging.name,
                  training_interactions=budget, training_seconds=elapsed,
                  training_episodes=len(training_scores),
                  mean_training_return=float(np.mean(training_scores)) if training_scores else None,
                  model=path.name, model_sha256=final["model_sha256"],
                  plan_sha256=sha(run / "plan.json"), **updates, **metrics,
                  **trace_metrics(staging, "evaluation", algorithm))
    result["wall_seconds"] = perf_counter() - started
    result["files"] = {file.name: sha(file) for file in staging.iterdir() if file.is_file()}
    pending = staging / "completed_pending.json"
    write_json(pending, result)
    pending.replace(completion)
    print(f"{name} seed {seed}: {metrics['success_rate']:.1%} success", flush=True)
    return result


def summarize(run, plan, results):
    from matplotlib.figure import Figure

    configs = plan["configurations"]
    by_job = {(row["configuration"], row["training_seed"]): row for row in results}
    seeds = plan["training_seeds"]
    columns = sorted(set().union(*results) - {"files"})
    write_csv(run / "agents.csv", [{k: row.get(k) for k in columns} for row in results])
    fields = ("success_rate", "failure_rate", "mean_return", "success_length", "failure_length",
              "mean_training_return", "training_seconds", "evaluation_seconds", "wall_seconds",
              "mean_first_100_score_gain", "mean_reward_events_ge_200", "action_repeat_rate",
              "mean_policy_entropy")
    summary, paired = [], []
    for name, config in configs.items():
        row = dict(configuration=name, n_training_seeds=len(seeds), interactions=plan["budget"])
        for field in fields:
            values = [by_job[name, seed].get(field) for seed in seeds]
            defined = [value for value in values if value is not None]
            row[f"{field}_mean"] = float(np.mean(defined)) if defined else None
            row[f"{field}_seed_sd"] = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
            row[f"{field}_n_defined"] = len(defined)
        summary.append(row)
        if "baseline" in config:
            baseline = config["baseline"]
            for seed in seeds:
                variant, control = by_job[name, seed], by_job[baseline, seed]
                paired.append(dict(configuration=name, baseline=baseline, training_seed=seed,
                                   **{f"delta_{field}": variant[field] - control[field]
                                      if variant.get(field) is not None and control.get(field) is not None
                                      else None for field in fields}))
    write_csv(run / "summary.csv", summary)
    write_csv(run / "paired_deltas.csv", paired)

    curve_rows = []
    for (name, seed), result in by_job.items():
        for row in read_csv(run / f"{name}_seed{seed}" / result["attempt"] / "curves.csv"):
            curve_rows.append(dict(configuration=name, training_seed=seed, **row))
    write_csv(run / "curves.csv", curve_rows)
    figure = Figure(figsize=(12, 5), layout="constrained")
    for algorithm, axis in zip(ALGORITHM, figure.subplots(1, 2, sharey=True)):
        for name, config in configs.items():
            if config["algorithm"] != algorithm:
                continue
            series = [[float(row["success_rate"]) * 100 for row in curve_rows
                       if row["configuration"] == name and row["training_seed"] == seed] for seed in seeds]
            points = checkpoint_points(plan["budget"])
            line, = axis.plot(points, np.mean(series, axis=0), marker="o", label=name)
            for values in series:
                axis.plot(points, values, color=line.get_color(), alpha=.14, linewidth=.7)
        axis.set(title=ALGORITHM[algorithm], xlabel="Training interactions", ylim=(-2, 102))
        axis.legend(fontsize=8)
    figure.axes[0].set_ylabel("Development success (%)")
    figure.suptitle(f"{run.name} | Setting B | per-seed curves and means, unsmoothed")
    figure.savefig(run / "learning_curves.png", dpi=160)


def run_study(args):
    hashes = source_hashes()
    for source in (Path(__file__).resolve().relative_to(ROOT).as_posix(), PREDICTIONS):
        hashes[source] = sha(ROOT / source)
    plan = dict(
        protocol="assignment4-part3-v1", mode="engineering smoke" if args.smoke else "study",
        setting=SETTING, environment=asdict(TaskConfig(SETTING)) | {
            "environment_id": "ALE/MsPacman-v5", "observation_size": 61,
            "maze": TaskConfig(SETTING).maze, "sticky_probability": TaskConfig(SETTING).sticky,
            "reward": "Native score increment (no per-step penalty in B)",
            "action_meanings": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"],
            "training_reward_divisor": 100,
        },
        budget=ROLLOUT if args.smoke else BUDGET,
        configurations=configurations(), training_seeds=list(SEEDS[:1] if args.smoke else SEEDS),
        development_seeds=list(DEVELOPMENT[:1] if args.smoke else DEVELOPMENT),
        evaluation_seeds=list(EVALUATION[:1] if args.smoke else EVALUATION),
        diagnostic_seeds=list(DIAGNOSTIC[:1] if args.smoke else DIAGNOSTIC),
        training_seeding="Q RNG: training seed; ALE: seed*1000 + episode index. "
                         "PPO: SB3 training seed, first ALE resets seed+i, subsequent unseeded.",
        evaluation_policy="Frozen greedy Q with seeded uniform ties; PPO argmax.",
        checkpoint_rule="Final budget; development checkpoints are descriptive.",
        software={name: version(name) for name in
                  ("numpy", "torch", "gymnasium", "ale-py", "stable-baselines3", "matplotlib")},
        python=platform.python_version(), os=platform.platform(),
        cpu=cpu_name(), logical_cpus=os.cpu_count(),
        compute_device="CPU (no GPU used)", torch_threads_per_worker=1, workers=args.workers,
        source_hashes=hashes,
    )
    run = Path(args.output).resolve()
    if run.exists():
        require(run.is_relative_to(ROOT / "results/assignment4"), "Use results/assignment4/")
        require(read_json(run / "plan.json") == plan, "Existing output has a different plan; use a new output")
        verify_sources(plan["source_hashes"])
    else:
        new_directory(run)
        for source in hashes:
            target = run / "sources" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / source).read_bytes())
        write_json(run / "plan.json", plan)
    jobs = [(name, seed) for name in plan["configurations"] for seed in plan["training_seeds"]]
    if args.workers == 1:
        results = [job(run, plan, name, seed) for name, seed in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(job, run, plan, name, seed) for name, seed in jobs]
            results = [future.result() for future in pending]
    verify_sources(hashes)
    summarize(run, plan, results)


def diagnose(args):
    torch.set_num_threads(1)
    run = Path(args.run).resolve()
    plan = read_json(run / "plan.json")
    verify_sources(plan["source_hashes"])
    variant = plan["configurations"].get(args.variant)
    require(variant and "baseline" in variant, "Choose a mechanism variant")
    seeds, algorithm = plan["training_seeds"], variant["algorithm"]
    baseline = variant["baseline"]
    models = {}
    for name in (baseline, args.variant):
        for seed in seeds:
            job_path = run / f"{name}_seed{seed}"
            record = read_json(job_path / "completed.json")
            path = job_path / record["attempt"] / record["model"]
            require(record["plan_sha256"] == sha(run / "plan.json") and sha(path) == record["model_sha256"],
                    f"Checkpoint provenance mismatch: {path}")
            models[name, seed] = path
    output = new_directory(args.output)
    write_json(output / "diagnostic_plan.json", dict(
        source_run=str(run), source_plan_sha256=sha(run / "plan.json"), mode=plan["mode"],
        variant=args.variant, baseline=baseline, seeds=seeds, evaluation_seeds=plan["diagnostic_seeds"],
        question="Does the selected variant's success gap depend on sticky-action execution?",
        sticky_probabilities=dict(A=0., B=.25), policy=plan["evaluation_policy"],
        models={str(path.relative_to(run)): sha(path) for path in models.values()},
    ))
    results = []
    for (name, seed), path in models.items():
        model = load_model(path, plan["configurations"][name], plan["budget"])
        for condition in ("A", "B"):
            metrics = evaluate(model, ALGORITHM[algorithm], condition, plan["diagnostic_seeds"],
                               output, f"{name}_seed{seed}_{condition}", trace=True)
            results.append(dict(configuration=name, training_seed=seed, condition=condition, **metrics))
    write_csv(output / "agents.csv", results)
    rates = {(row["configuration"], row["training_seed"], row["condition"]): row["success_rate"]
             for row in results}
    deltas = []
    for seed in seeds:
        gap_a = rates[args.variant, seed, "A"] - rates[baseline, seed, "A"]
        gap_b = rates[args.variant, seed, "B"] - rates[baseline, seed, "B"]
        deltas.append(dict(training_seed=seed, variant_minus_baseline_A=gap_a,
                           variant_minus_baseline_B=gap_b, gap_B_minus_A=gap_b - gap_a))
    write_csv(output / "paired_deltas.csv", deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    study = commands.add_parser("run", help="Train, evaluate and summarize every configuration")
    study.add_argument("--output", required=True)
    study.add_argument("--workers", type=int, default=1)
    study.add_argument("--smoke", action="store_true", help="Unassessed engineering check")
    diagnostic = commands.add_parser("diagnose", help="Re-evaluate a frozen variant with/without sticky actions")
    diagnostic.add_argument("--run", required=True)
    diagnostic.add_argument("--variant", choices=VARIANTS, required=True)
    diagnostic.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        require(args.workers > 0, "Use at least one worker")
        run_study(args)
    else:
        diagnose(args)
