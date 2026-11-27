"""Unified Part 2 runner on the local Ryzen/PyTorch 2.15 environment.

Reuses the untouched Part 3 baseline Setting B runs (e8c0c6a8f4071325), trains
and evaluates Settings A and C with the exact same protocol/hardware/software,
runs the clarified Setting D paired observation-obstruction evaluation
(32x32 fixed box: x in [16, 48), y in [24, 56)), and runs the A-vs-B
ghost-approach probe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, HERE / "pilot", HERE / "current", HERE / "improvements", HERE / "three_hour"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import part3_experiments as p3
from time import sleep


def _windows_safe_publish(attempt, destination):
    for i in range(20):
        try:
            attempt.rename(destination)
            return
        except PermissionError:
            if (destination / "completed.json").is_file():
                return
            sleep(0.5)
        except OSError:
            if (destination / "completed.json").is_file():
                return
            raise
    attempt.rename(destination)


p3.publish = _windows_safe_publish


def _train_one_safe(*args):
    p3.publish = _windows_safe_publish
    return p3.train_one(*args)


def _eval_one_safe(*args):
    p3.publish = _windows_safe_publish
    return p3.evaluate_one(*args)
from mspacman_env import FOOD_BYTES, FOOD_BITS, MsPacmanTask, TaskConfig
from navigation_features import ALL_FOOD, CELLS, DISTANCE, GRID, NEIGHBORS, POWER_NODES, XY
from ppo import FrozenPolicy
from q_learning import TabularQ, make_environment, require, sha, verify_sources, write_csv, write_json

OBSTRUCTION_BOX = (16.0, 48.0, 24.0, 56.0)  # x_min, x_max, y_min, y_max
POWER_COORDS = np.array([[150.0, 150.0], [150.0, 18.0], [10.0, 150.0], [10.0, 18.0]])
PART3_STUDY = ROOT / "results/assignment4/part3/study/e8c0c6a8f4071325"
OUT_ROOT = ROOT / "results/assignment4/unified_part2"
D_SEEDS = list(range(1600000, 1600200))


def in_box(xy: np.ndarray, box=OBSTRUCTION_BOX) -> np.ndarray:
    x0, x1, y0, y1 = box
    return (xy[..., 0] >= x0) & (xy[..., 0] < x1) & (xy[..., 1] >= y0) & (xy[..., 1] < y1)


class ObstructedStudyTask(gym.Wrapper):
    """Native Setting B game with optional fixed 32x32 object-feature obstruction.

    When obstructed=False, outputs are bitwise identical to StudyTask('B').
    When obstructed=True, underlying ALE dynamics, RAM, and rewards are untouched;
    object centres inside OBSTRUCTION_BOX are hidden before feature recomputation,
    and a hidden player retains their last visible pose. Static geometry remains.
    """

    def __init__(self, obstructed: bool = True, box=OBSTRUCTION_BOX):
        super().__init__(MsPacmanTask(TaskConfig("B")))
        self.obstructed = bool(obstructed)
        self.box = tuple(map(float, box))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(61,), dtype=np.float32)
        self.previous_player = None
        self.last_visible_player = None
        self.previous_ghosts = None
        self.previous_ghost_visible = None
        self.food_key = None
        self.food_dist = None
        self.power_dist = None
        self.node = 0

    def _compute_obs(self, info) -> np.ndarray:
        ram = self.env.ram
        assert int(ram[0]) == 0
        raw_player = np.array([float(ram[10]) - 8.0, float(ram[16]) + 6.0])
        raw_ghosts = np.column_stack((ram[6:10].astype(float) - 8.0, ram[12:16].astype(float) + 6.0))
        active = (np.arange(4) <= int(ram[19])) & bool(ram[47])
        present = GRID.ravel() & ((ram[FOOD_BYTES] & FOOD_BITS) != 0)
        power = np.array([bool(int(ram[117]) & bit) for bit in (4, 8, 16, 32)])

        if self.obstructed:
            player_hidden = bool(in_box(raw_player, self.box))
            if player_hidden and self.last_visible_player is not None:
                player = self.last_visible_player.copy()
            else:
                player = raw_player.copy()
                if not player_hidden:
                    self.last_visible_player = raw_player.copy()
            ghost_hidden = in_box(raw_ghosts, self.box)
            obs_active = active & (~ghost_hidden)
            obs_present = present & (~in_box(ALL_FOOD, self.box))
            obs_power = power & (~in_box(POWER_COORDS, self.box))
        else:
            player = raw_player.copy()
            self.last_visible_player = raw_player.copy()
            obs_active = active
            obs_present = present
            obs_power = power

        gd = np.abs(raw_ghosts - player).sum(axis=1)
        gd[~obs_active] = np.inf
        nearest = int(np.argmin(gd))
        ghost_delta = raw_ghosts[nearest] - player if obs_active.any() else np.zeros(2)
        ghost_distance = float(gd[nearest]) if obs_active.any() else 320.0

        food = ALL_FOOD[obs_present]
        if len(food):
            delta = food - player
            distance = np.abs(delta).sum(axis=1)
            j = int(np.argmin(distance))
            pellet_delta, pellet_distance = delta[j], float(distance[j])
            direction = np.where(
                np.abs(delta[:, 0]) > np.abs(delta[:, 1]),
                np.where(delta[:, 0] > 0, 1, 3),
                np.where(delta[:, 1] > 0, 2, 0),
            )
            density = np.bincount(direction, minlength=4) / 252.0
        else:
            pellet_delta, pellet_distance, density = np.zeros(2), 320.0, np.zeros(4)

        node = int(np.argmin(np.square((XY - player) / [8, 12]).sum(axis=1)))
        neighbors = NEIGHBORS[node]
        available = neighbors >= 0

        key = obs_present.tobytes() + obs_power.tobytes()
        if key != self.food_key:
            targets = obs_present[CELLS].copy()
            targets[POWER_NODES[obs_power]] = True
            self.food_dist = DISTANCE[:, targets].min(axis=1) if targets.any() else np.full(len(CELLS), 32.0)
            self.power_dist = DISTANCE[:, POWER_NODES[obs_power]].min(axis=1) if obs_power.any() else np.full(len(CELLS), 32.0)
            self.food_key = key

        def regret(dist):
            costs = np.full(4, 32.0)
            costs[available] = 1 + dist[neighbors[available]]
            out = np.clip((costs - costs[available].min()) / 8, 0, 1)
            out[~available] = 1
            return out

        threat = np.zeros(4)
        if obs_active.any():
            ghost_nodes = np.argmin(np.square((XY[:, None, :] - raw_ghosts[None, :, :]) / [8, 12]).sum(axis=2), axis=0)[obs_active]
            path_distance = DISTANCE[:, ghost_nodes].min(axis=1)
            threat[available] = np.exp(-path_distance[neighbors[available]] / 3)

        if self.previous_ghosts is None:
            motion = np.zeros((4, 2))
        else:
            motion = raw_ghosts - self.previous_ghosts
            if self.obstructed:
                valid_motion = obs_active & self.previous_ghost_visible
                motion[~valid_motion] = 0.0
        player_motion = np.zeros(2) if self.previous_player is None else player - self.previous_player
        if np.linalg.norm(player_motion) > 20:
            player_motion[:] = 0
        motion[np.abs(motion) > 20] = 0

        if self.previous_ghosts is None:
            self.previous_ghosts = raw_ghosts.copy()
            self.previous_ghost_visible = obs_active.copy()
        else:
            if self.obstructed:
                self.previous_ghosts[obs_active] = raw_ghosts[obs_active]
                self.previous_ghost_visible = obs_active.copy()
            else:
                self.previous_ghosts = raw_ghosts.copy()
                self.previous_ghost_visible = obs_active.copy()
        self.previous_player = player.copy()

        base = [
            player[0] / 160, player[1] / 210, ghost_delta[0] / 160, ghost_delta[1] / 210, ghost_distance / 320,
            pellet_delta[0] / 160, pellet_delta[1] / 210, pellet_distance / 320, *density,
            min(int(ram[116]), 63) / 63, (int(ram[56]) & 3) / 3, max(0, 1 - self.env.steps / 1000),
        ]
        offset = np.clip((player - XY[node]) / [8, 12], -1, 1)
        obs45 = np.clip(
            np.array(
                [*base, *available.astype(float), *regret(self.food_dist), *threat, *regret(self.power_dist),
                 *offset, *np.clip(motion.ravel() / 8, -1, 1), *np.clip(player_motion / 8, -1, 1),
                 obs_present.sum() / 154, 1 / (1 + np.linalg.norm(player - XY[node]))],
                dtype=np.float32,
            ),
            -1, 1,
        )
        self.node = node

        nearest_motion = obs45[33:41].reshape(4, 2)[nearest] if obs_active.any() else np.zeros(2, dtype=np.float32)
        obs49 = np.r_[obs45, nearest_motion, min(self.env.score / 2000, 1), info["lives"] / 5].astype(np.float32)
        all_delta = np.clip((raw_ghosts - player) / [160, 210], -1, 1)
        all_delta[~obs_active] = 0
        return np.r_[obs49, all_delta.ravel(), obs_active].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self.previous_player = None
        self.last_visible_player = None
        self.previous_ghosts = None
        self.previous_ghost_visible = None
        self.food_key = None
        return self._compute_obs(info), info

    def step(self, action):
        _, reward, term, trunc, info = self.env.step(int(action))
        return self._compute_obs(info), reward, term, trunc, info


def verify_obstruction_wrapper():
    """Verify exact clean parity and native dynamics noninterference over 1,600 steps."""
    ref = make_environment("B")
    clean = ObstructedStudyTask(obstructed=False)
    obstr = ObstructedStudyTask(obstructed=True)
    rng = np.random.default_rng(3212)
    diff_count = 0
    total_steps = 0
    try:
        for ep in range(4):
            seed = 990000 + ep
            o_ref, _ = ref.reset(seed=seed)
            o_cln, _ = clean.reset(seed=seed)
            o_obs, _ = obstr.reset(seed=seed)
            require(np.array_equal(o_ref, o_cln), "Clean ObstructedStudyTask differs from StudyTask at reset")
            for _ in range(400):
                action = int(rng.integers(5))
                o_ref, r_ref, t_ref, tr_ref, i_ref = ref.step(action)
                o_cln, r_cln, t_cln, tr_cln, i_cln = clean.step(action)
                o_obs, r_obs, t_obs, tr_obs, i_obs = obstr.step(action)
                require(np.array_equal(o_ref, o_cln), "Clean ObstructedStudyTask obs differs from StudyTask")
                require(r_ref == r_cln == r_obs and t_ref == t_cln == t_obs and tr_ref == tr_cln == tr_obs,
                        "Obstruction altered native reward or termination")
                require(np.array_equal(ref.unwrapped.ale.getRAM(), obstr.unwrapped.ale.getRAM()),
                        "Obstruction altered native RAM")
                if not np.array_equal(o_cln, o_obs):
                    diff_count += 1
                total_steps += 1
                if t_ref or tr_ref:
                    break
    finally:
        ref.close()
        clean.close()
        obstr.close()
    require(diff_count > 0, "Obstruction box had zero effect on observations")
    return {"checked_transitions": total_steps, "obstructed_obs_changed_steps": diff_count}


def evaluate_on_env(agent, algorithm, env, seeds, output, name):
    before = agent.fingerprint()
    rows = []
    start = perf_counter()
    bcd = lambda v: int(v) // 16 * 10 + int(v) % 16
    native_score = lambda ram: bcd(ram[122]) * 10000 + bcd(ram[121]) * 100 + bcd(ram[120])
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed) + 3000000)
            initial = native_score(env.unwrapped.ale.getRAM())
            for _ in range(1000):
                values = agent.values_for_observation(obs)
                if algorithm == "Q-learning":
                    rng.random()
                    action = int(rng.choice(np.flatnonzero(values == values.max())))
                else:
                    action = int(values.argmax())
                obs, reward, terminated, truncated, info = env.step(action)
                after = env.unwrapped.ale.getRAM()
                require(native_score(after) - initial == info["score"], "Native score disagreement")
                if terminated or truncated:
                    rows.append(dict(
                        evaluation_seed=int(seed), success=int(info["success"]),
                        known_failure=int(not info["success"]), censored=0,
                        score=info["score"], task_return=info["task_return"],
                        length=info["episode_steps"], reason=info["reason"],
                    ))
                    break
            else:
                raise RuntimeError("Episode did not terminate within horizon")
    finally:
        env.close()
    seconds = perf_counter() - start
    require(agent.fingerprint() == before, "Evaluation modified learned parameters")
    write_csv(output / f"{name}.csv", rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    def cond_len(flag):
        vals = [r["length"] for r in rows if r[flag]]
        return float(np.mean(vals)) if vals else None
    return dict(
        episodes=len(rows), success_rate=mean("success"), failure_rate=mean("known_failure"),
        mean_score=mean("score"), mean_return=mean("task_return"),
        success_length=cond_len("success"), failure_length=cond_len("known_failure"),
        evaluation_interactions=sum(r["length"] for r in rows), evaluation_seconds=seconds,
        frozen_parameters_verified=True,
    )


def run_d_one(label, algorithm, run_dir_str, seed, mode):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run_dir = Path(run_dir_str)
        selected = json.loads((run_dir / "selected.json").read_text(encoding="utf-8"))
        verify_sources(selected["source_hashes"])
        model_path = run_dir / selected["model"]
        require(sha(model_path) == selected["model_sha256"], f"Checkpoint changed: {model_path}")
        out_dir = OUT_ROOT / "setting_D" / f"{label}_seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        done_path = out_dir / f"{mode}_completed.json"
        if done_path.is_file():
            return json.loads(done_path.read_text(encoding="utf-8"))
        loader = TabularQ.load if algorithm == "Q-learning" else FrozenPolicy.load
        agent = loader(model_path)
        env = ObstructedStudyTask(obstructed=(mode == "obstructed"))
        metrics = evaluate_on_env(agent, algorithm, env, D_SEEDS, out_dir, mode)
        require(sha(model_path) == selected["model_sha256"], "Model file modified during D evaluation")
        record = dict(
            label=label, algorithm=algorithm, training_seed=seed, mode=mode,
            source_run=str(run_dir.relative_to(ROOT)), model_sha256=selected["model_sha256"],
            checkpoint_interactions=selected["checkpoint_interactions"], **metrics,
        )
        write_json(done_path, record)
        return record


def run_ac_training_and_eval(workers: int):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_records = []
    for setting in ("A", "C"):
        protocol = p3.experiment_protocol(setting=setting)
        study_dir = p3.study_folder(protocol)
        print(f"=== Setting {setting}: study_dir={study_dir.relative_to(ROOT)} ===", flush=True)
        train_jobs = [
            (protocol, alg, p3.PART2_BASELINE[alg], seed)
            for alg in ("Q-learning", "PPO")
            for seed in protocol["training_seeds"]
        ]
        p3.run_jobs(_train_one_safe, train_jobs, workers)
        p3.run_jobs(_eval_one_safe, train_jobs, workers)

        for alg in ("Q-learning", "PPO"):
            settings = p3.PART2_BASELINE[alg]
            for seed in protocol["training_seeds"]:
                run = p3.job_folder(protocol, alg, settings, seed)
                metrics = json.loads((run / "held_out/completed.json").read_text(encoding="utf-8"))
                training = json.loads((run / "completed.json").read_text(encoding="utf-8"))
                summary_records.append(dict(
                    setting=setting, algorithm=alg, training_seed=seed,
                    run_id=str(run.relative_to(ROOT)),
                    training_interactions=protocol["budget"],
                    training_seconds=training["training_seconds"],
                    checkpoint_interactions=training["selected"]["checkpoint_interactions"],
                    **metrics,
                ))
    # Also collect Setting B baseline from existing Part 3 study
    b_proto = p3.experiment_protocol(setting="B")
    for alg in ("Q-learning", "PPO"):
        settings = p3.PART2_BASELINE[alg]
        for seed in b_proto["training_seeds"]:
            run = p3.job_folder(b_proto, alg, settings, seed)
            metrics = json.loads((run / "held_out/completed.json").read_text(encoding="utf-8"))
            training = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            summary_records.append(dict(
                setting="B", algorithm=alg, training_seed=seed,
                run_id=str(run.relative_to(ROOT)),
                training_interactions=b_proto["budget"],
                training_seconds=training["training_seconds"],
                checkpoint_interactions=training["selected"]["checkpoint_interactions"],
                **metrics,
            ))
    df = pd.DataFrame(summary_records)
    df.to_csv(OUT_ROOT / "abc_per_seed.csv", index=False)
    print(f"Saved A/B/C summary to {(OUT_ROOT / 'abc_per_seed.csv').relative_to(ROOT)}", flush=True)
    return df


def run_setting_d(workers: int):
    check = verify_obstruction_wrapper()
    print(f"Obstruction wrapper check passed: {check}", flush=True)
    d_dir = OUT_ROOT / "setting_D"
    d_dir.mkdir(parents=True, exist_ok=True)
    write_json(d_dir / "wrapper_check.json", dict(box=OBSTRUCTION_BOX, **check))

    b_proto = p3.experiment_protocol(setting="B")
    models = [
        ("Q-learning", "Q-learning", p3.PART2_BASELINE["Q-learning"]),
        ("PPO", "PPO", p3.PART2_BASELINE["PPO"]),
        ("PPO12", "PPO", {**p3.PART2_BASELINE["PPO"], "learning_rate_scale": 2.0, "n_epochs": 12}),
    ]
    jobs = []
    for label, alg, settings in models:
        for seed in b_proto["training_seeds"]:
            run_dir = p3.job_folder(b_proto, alg, settings, seed)
            for mode in ("clean", "obstructed"):
                jobs.append((label, alg, str(run_dir), seed, mode))

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_d_one, *j) for j in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(f"D {res['label']} seed {res['training_seed']} ({res['mode']}): {res['success_rate']:.1%}", flush=True)

    df = pd.DataFrame(results).sort_values(["label", "training_seed", "mode"])
    df.to_csv(d_dir / "per_seed.csv", index=False)

    # Paired comparison table
    paired_rows = []
    for label, alg, _ in models:
        for seed in b_proto["training_seeds"]:
            out_dir = d_dir / f"{label}_seed_{seed}"
            c_df = pd.read_csv(out_dir / "clean.csv")
            o_df = pd.read_csv(out_dir / "obstructed.csv")
            require((c_df["evaluation_seed"].values == o_df["evaluation_seed"].values).all(), "Seed mismatch")
            s_to_f = int(((c_df["success"] == 1) & (o_df["success"] == 0)).sum())
            f_to_s = int(((c_df["success"] == 0) & (o_df["success"] == 1)).sum())
            c_sr = float(c_df["success"].mean()) * 100
            o_sr = float(o_df["success"].mean()) * 100
            paired_rows.append(dict(
                label=label, algorithm=alg, training_seed=seed,
                clean_success_pct=c_sr, obstructed_success_pct=o_sr,
                delta_pp=o_sr - c_sr, s_to_f=s_to_f, f_to_s=f_to_s,
                clean_return=float(c_df["task_return"].mean()),
                obstructed_return=float(o_df["task_return"].mean()),
            ))
    pdf = pd.DataFrame(paired_rows)
    pdf.to_csv(d_dir / "paired_summary.csv", index=False)
    print(f"Saved Setting D paired summary to {(d_dir / 'paired_summary.csv').relative_to(ROOT)}", flush=True)
    return df, pdf


def action_approaches_ghost(obs, action: int) -> bool:
    """Return True if requested action moves toward nearest ghost on the chosen axis."""
    if action == 0:
        return False
    dx, dy = float(obs[2] * 160.0), float(obs[3] * 210.0)
    # Actions: 1=UP (dy<0), 2=RIGHT (dx>0), 3=LEFT (dx<0), 4=DOWN (dy>0)
    if action == 1:
        return dy < 0
    if action == 2:
        return dx > 0
    if action == 3:
        return dx < 0
    if action == 4:
        return dy > 0
    return False


def q_expected_approach(obs, values: np.ndarray) -> float:
    best = np.flatnonzero(values == values.max())
    return float(np.mean([action_approaches_ghost(obs, int(a)) for a in best]))


def _probe_recorded_one(setting: str, alg: str, seed: int):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        proto = p3.experiment_protocol(setting=setting)
        run = p3.job_folder(proto, alg, p3.PART2_BASELINE[alg], seed)
        sel = json.loads((run / "selected.json").read_text(encoding="utf-8"))
        loader = TabularQ.load if alg == "Q-learning" else FrozenPolicy.load
        agent = loader(run / sel["model"])
        env = make_environment(setting)
        rng = np.random.default_rng(3212 + seed * 17 + (0 if setting == "A" else 1000) + (0 if alg == "Q-learning" else 5000))
        approaches = []
        eligible_samples = []
        try:
            for ep_seed in range(1360000, 1360200):
                obs, _ = env.reset(seed=ep_seed)
                q_rng = np.random.default_rng(ep_seed + 3000000)
                ep_eligible = []
                for _ in range(1000):
                    vals = agent.values_for_observation(obs)
                    if alg == "Q-learning":
                        q_rng.random()
                        act = int(q_rng.choice(np.flatnonzero(vals == vals.max())))
                    else:
                        act = int(vals.argmax())
                    dist = float(obs[4] * 320.0)
                    if obs[12] == 0 and 1.0 <= dist <= 16.0:
                        approaches.append(float(action_approaches_ghost(obs, act)))
                        ep_eligible.append(obs.copy())
                    obs, _, term, trunc, _ = env.step(act)
                    if term or trunc:
                        break
                if ep_eligible:
                    eligible_samples.append(ep_eligible[int(rng.integers(len(ep_eligible)))])
        finally:
            env.close()
        return dict(
            setting=setting, algorithm=alg, training_seed=seed,
            eligible_steps=len(approaches),
            approach_pct=float(np.mean(approaches)) * 100 if approaches else float("nan"),
        ), eligible_samples


def run_ab_ghost_probe(workers: int = 10):
    """Post-hoc probe of frozen A vs B models near dangerous ghosts (dist 1..16, power==0)."""
    torch.set_num_threads(1)
    probe_dir = OUT_ROOT / "ab_ghost_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    seeds = [17, 29, 101, 314, 2718]

    jobs = [(setting, alg, seed) for setting in ("A", "B") for alg in ("Q-learning", "PPO") for seed in seeds]
    recorded_stats = []
    pool_raw = {(s, a): [] for s in ("A", "B") for a in ("Q-learning", "PPO")}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_probe_recorded_one, *j) for j in jobs]
        for fut in as_completed(futs):
            stat, samples = fut.result()
            recorded_stats.append(stat)
            pool_raw[(stat["setting"], stat["algorithm"])].extend(samples)

    rng = np.random.default_rng(3212)
    common_pool = []
    for key in (("A", "Q-learning"), ("A", "PPO"), ("B", "Q-learning"), ("B", "PPO")):
        items = pool_raw[key]
        n_pick = min(64, len(items))
        if n_pick > 0:
            idx = rng.choice(len(items), size=n_pick, replace=False)
            common_pool.extend(items[int(i)] for i in idx)

    models = {}
    for setting in ("A", "B"):
        proto = p3.experiment_protocol(setting=setting)
        for alg in ("Q-learning", "PPO"):
            models[(setting, alg)] = {}
            for seed in seeds:
                run = p3.job_folder(proto, alg, p3.PART2_BASELINE[alg], seed)
                sel = json.loads((run / "selected.json").read_text(encoding="utf-8"))
                loader = TabularQ.load if alg == "Q-learning" else FrozenPolicy.load
                models[(setting, alg)][seed] = loader(run / sel["model"])

    pool_stats = []
    ppo_disagreements = []
    for alg in ("Q-learning", "PPO"):
        for setting in ("A", "B"):
            for seed in seeds:
                agent = models[(setting, alg)][seed]
                vals_list = [agent.values_for_observation(o) for o in common_pool]
                if alg == "Q-learning":
                    pct = float(np.mean([q_expected_approach(o, v) for o, v in zip(common_pool, vals_list)])) * 100
                else:
                    pct = float(np.mean([action_approaches_ghost(o, int(v.argmax())) for o, v in zip(common_pool, vals_list)])) * 100
                pool_stats.append(dict(setting=setting, algorithm=alg, training_seed=seed, same_obs_approach_pct=pct))

    for seed in seeds:
        a_acts = [int(models[("A", "PPO")][seed].values_for_observation(o).argmax()) for o in common_pool]
        b_acts = [int(models[("B", "PPO")][seed].values_for_observation(o).argmax()) for o in common_pool]
        ppo_disagreements.append(float(np.mean([a != b for a, b in zip(a_acts, b_acts)])) * 100)

    rec_df = pd.DataFrame(recorded_stats).sort_values(["algorithm", "setting", "training_seed"])
    pool_df = pd.DataFrame(pool_stats)
    rec_df.to_csv(probe_dir / "recorded.csv", index=False)
    pool_df.to_csv(probe_dir / "same_obs.csv", index=False)
    summary = dict(
        common_pool_size=len(common_pool),
        ppo_pair_disagreement_pct=float(np.mean(ppo_disagreements)),
        recorded_means=rec_df.groupby(["algorithm", "setting"])["approach_pct"].mean().to_dict(),
        same_obs_means=pool_df.groupby(["algorithm", "setting"])["same_obs_approach_pct"].mean().to_dict(),
    )
    write_json(probe_dir / "summary.json", {
        "common_pool_size": summary["common_pool_size"],
        "ppo_pair_disagreement_pct": summary["ppo_pair_disagreement_pct"],
        "recorded_means": {f"{k[0]}_{k[1]}": v for k, v in summary["recorded_means"].items()},
        "same_obs_means": {f"{k[0]}_{k[1]}": v for k, v in summary["same_obs_means"].items()},
    })
    print("A/B ghost probe summary:", json.dumps({f"{k[0]}_{k[1]}": v for k, v in summary["same_obs_means"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ac", "d", "probe", "all"], default="all")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.stage in ("ac", "all"):
        run_ac_training_and_eval(args.workers)
    if args.stage in ("d", "all"):
        run_setting_d(args.workers)
    if args.stage in ("probe", "all"):
        run_ab_ghost_probe(args.workers)


if __name__ == "__main__":
    main()
