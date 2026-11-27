"""Render native Atari screenshots and run real counterfactual replays for Part 4.

For all 10 selected Part 4 episodes (TS1..SF2) in
results/assignment4/part4/part3_B_054b0e6b63381969:
1. Replays each episode in native Gymnasium with render_mode='rgb_array',
   verifying exact step-by-step score/action/outcome parity with the locked
   NPZ trace and capturing the exact RGB frame immediately before each of the
   3 reviewed decisions.
2. Saves 10 three-panel PDF figures in reports/assignment4/Images/part4_<ID>.pdf.
3. For all 30 reviewed decisions, runs an actual counterfactual replay changing
   that single input to the recorded alternative action, recording:
   - Immediate 1-step movement & reward difference
   - For collision/life-loss decisions (and all 30), a matched continuation
     reusing subsequent recorded inputs for up to 64 decisions to test whether
     the alternative actually prevented life loss.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import ale_py
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gym.register_envs(ale_py)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, HERE / "pilot", HERE / "current", HERE / "improvements", HERE / "three_hour"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mspacman_env import MsPacmanTask, TaskConfig
from q_learning import require, write_csv, write_json

PART4_DIR = ROOT / "results/assignment4/part4/part3_B_054b0e6b63381969"
IMAGES_DIR = ROOT / "reports/assignment4/Images"
ACTION_NAME = {0: "NOOP", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "DOWN"}
LETTER_TO_ID = {"N": 0, "U": 1, "R": 2, "L": 3, "D": 4}
ID_TO_LETTER = {v: k for k, v in LETTER_TO_ID.items()}


def make_rgb_b_env():
    base = gym.make(
        "ALE/MsPacman-v5", obs_type="ram", frameskip=4,
        repeat_action_probability=0.25, mode=0, render_mode="rgb_array",
    )
    return MsPacmanTask(TaskConfig("B"), env=base)


def player_xy(ram):
    return np.array([float(ram[10]) - 8.0, float(ram[16]) + 6.0])


def movement_label(before_xy, after_xy):
    d = after_xy - before_xy
    if np.linalg.norm(d) < 0.5:
        return "stayed"
    if abs(d[0]) >= abs(d[1]):
        return "right" if d[0] > 0 else "left"
    return "down" if d[1] > 0 else "up"


def replay_counterfactual(ep_seed: int, orig_actions: np.ndarray, step_idx: int, alt_action: int, horizon: int = 64):
    """Replay up to step_idx identically, apply alt_action once, then reuse orig_actions."""
    env = make_rgb_b_env()
    try:
        _, info = env.reset(seed=int(ep_seed))
        for t in range(step_idx):
            _, _, term, trunc, info = env.step(int(orig_actions[t]))
            require(not (term or trunc), f"Episode ended before step {step_idx}")
        lives_before = int(info["lives"])
        xy_before = player_xy(env.ram)
        # Step t with alternative
        _, rew1, term, trunc, info = env.step(int(alt_action))
        xy_after1 = player_xy(env.ram)
        alt_move1 = movement_label(xy_before, xy_after1)
        alt_rew1 = float(info["raw_reward"])
        lost_life = int(info["lives"]) < lives_before
        steps_run = 1
        end_reason = info["reason"] if (term or trunc) else "ongoing"
        # Matched continuation up to `horizon` decisions total (including step_idx)
        max_k = min(horizon, len(orig_actions) - step_idx)
        for k in range(1, max_k):
            if term or trunc:
                break
            _, _, term, trunc, info = env.step(int(orig_actions[step_idx + k]))
            steps_run += 1
            if int(info["lives"]) < lives_before:
                lost_life = True
            if term or trunc:
                end_reason = info["reason"]
                break
        return dict(
            alt_move_1step=alt_move1,
            alt_reward_1step=alt_rew1,
            alt_lost_life_within_window=bool(lost_life),
            window_decisions=int(max_k),
            alt_steps_executed=int(steps_run),
            alt_end_reason=end_reason,
            alt_final_score=float(info["score"]),
        )
    finally:
        env.close()


def run_all():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    selection = json.loads((PART4_DIR / "selection.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in selection["cases"]}
    reviewed = pd.read_csv(PART4_DIR / "reviewed_decisions.csv")

    counterfactual_rows = []
    continuation_rows = []

    for case_id in ("TS1", "TS2", "TS3", "TF1", "TF2", "TF3", "SS1", "SS2", "SF1", "SF2"):
        case = cases[case_id]
        ep_seed = int(case["evaluation_seed"])
        trace = np.load(PART4_DIR / f"{case_id}_traces.npz")
        orig_actions = trace["action"]
        orig_scores = trace["score_after"]
        orig_rams = trace["ram"]
        orig_rams_after = trace["ram_after"]
        orig_vals = trace["values"]

        sub = reviewed[reviewed["case"] == case_id].sort_values("step_zero_based")
        target_steps = [int(s) for s in sub["step_zero_based"]]
        frames = {}
        lives_trace = []

        # Exact native replay with RGB capture
        env = make_rgb_b_env()
        try:
            _, info = env.reset(seed=ep_seed)
            for t, act in enumerate(orig_actions):
                lives_trace.append(int(info["lives"]))
                if t in target_steps:
                    # Capture exact RGB frame right before decision t
                    frames[t] = env.unwrapped.render().copy()
                require(np.array_equal(env.ram, orig_rams[t]), f"RAM mismatch at {case_id}/{t}")
                _, _, term, trunc, info = env.step(int(act))
                require(float(info["score"]) == float(orig_scores[t]), f"Score mismatch at {case_id}/{t}")
            lives_trace.append(int(info["lives"]))
            require(int(len(orig_actions)) == int(case["length"]), f"Length mismatch in {case_id}")
        finally:
            env.close()

        # Check each of the 3 reviewed decisions + run counterfactual replay
        panel_titles = []
        for _, row in sub.iterrows():
            t = int(row["step_zero_based"])
            chosen = str(row["chosen"])
            alt = str(row["alternative"])
            alt_id = LETTER_TO_ID[alt]
            chosen_val = float(row["chosen_value"])
            alt_val = float(row["alternative_value"])

            # Check original life loss within next 64 decisions (and immediate/near-term collision)
            lives_at_t = lives_trace[t]
            window = min(64, len(orig_actions) - t)
            future_lives = lives_trace[t + 1 : t + 1 + window]
            orig_lost_life_64 = any(lv < lives_at_t for lv in future_lives)
            first_loss_offset = next((k + 1 for k, lv in enumerate(future_lives) if lv < lives_at_t), None)
            is_collision_case = (first_loss_offset is not None) and (first_loss_offset <= 30)

            cf = replay_counterfactual(ep_seed, orig_actions, t, alt_id, horizon=64)
            orig_move1 = movement_label(player_xy(orig_rams[t]), player_xy(orig_rams_after[t]))

            rec = dict(
                case=case_id,
                agent=case["configuration"],
                step=t,
                chosen=chosen,
                chosen_value=round(chosen_val, 3),
                alternative=alt,
                alternative_value=round(alt_val, 3),
                orig_move_1step=orig_move1,
                orig_reward_1step=float(row["reward"]),
                orig_lost_life_64=bool(orig_lost_life_64),
                orig_first_loss_offset=first_loss_offset if first_loss_offset is not None else -1,
                is_collision_case=bool(is_collision_case),
                **cf,
            )
            counterfactual_rows.append(rec)
            if is_collision_case:
                continuation_rows.append(rec)

            dagger = r"$^{\dagger}$" if is_collision_case else ""
            panel_titles.append(
                f"Step {t}{dagger}: {ACTION_NAME[LETTER_TO_ID[chosen]]} ({chosen_val:.2f}) vs {ACTION_NAME[alt_id]} ({alt_val:.2f})"
            )

        # Render crisp 3-panel PDF figure
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
        for ax, t, title in zip(axes, target_steps, panel_titles):
            # Crop black bottom margin slightly? Keep COMPLETE native frame (210x160) per spec!
            ax.imshow(frames[t])
            ax.set_title(title, fontsize=8.5, fontweight="bold")
            ax.axis("off")
        fig.tight_layout(pad=0.6)
        out_pdf = IMAGES_DIR / f"part4_{case_id}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Rendered {out_pdf.relative_to(ROOT)} + verified 3 counterfactual replays", flush=True)

    write_csv(PART4_DIR / "counterfactual_replays.csv", counterfactual_rows)
    write_csv(PART4_DIR / "continuations.csv", continuation_rows)
    print(f"\nTotal collision cases (life loss <= 30 decisions): {len(continuation_rows)}", flush=True)
    cdf = pd.DataFrame(continuation_rows)
    if len(cdf):
        print(cdf[["case", "step", "chosen", "alternative", "orig_first_loss_offset",
                   "orig_lost_life_64", "alt_lost_life_within_window", "window_decisions"]].to_string(index=False))


if __name__ == "__main__":
    run_all()
