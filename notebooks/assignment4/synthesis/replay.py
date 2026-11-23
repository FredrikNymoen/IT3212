"""Inspect an explicitly chosen recorded episode, with direct Gymnasium replay.

Inspired by Elling's RAM-checked trace dashboard and Mats' per-decision viewer.
No best-test-model selection, model loading, parameter updates or new episodes.
The displayed Q-values/probabilities are recorded evidence, not counterfactuals.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

from part2_evidence import ROOT, HERE, load_bundle, local_path, require, sha


def native_score(ram):
    bcd = lambda v: int(v) // 16 * 10 + int(v) % 16
    return bcd(ram[122]) * 10000 + bcd(ram[121]) * 100 + bcd(ram[120])


def replay(condition, algorithm, training_seed, evaluation_seed, output, stride=4):
    require(stride > 0, "Frame stride must be positive")
    bundle = load_bundle()
    candidates = [r for r in bundle["agents"] if r["condition"] == condition
                  and r["algorithm"] == algorithm and r["training_seed"] == training_seed]
    require(len(candidates) == 1, "Choose one of the registered model/condition pairs")
    case = candidates[0]
    for name, expected in bundle["inputs"].items():
        # Source checks and this case's records, without hashing unrelated traces.
        if not name.startswith("results/") or name in (case["trace"], case["evaluation_csv"], case["model"]):
            require(sha(local_path(name)) == expected, f"Recorded input changed: {name}")
    from part2_evidence import read_csv
    rows = read_csv(local_path(case["evaluation_csv"]))
    matches = [i for i, row in enumerate(rows) if int(row["evaluation_seed"]) == evaluation_seed]
    require(len(matches) == 1, "Evaluation seed was not part of this recorded case")
    episode_index = matches[0]
    row = rows[episode_index]
    with np.load(local_path(case["trace"]), allow_pickle=False) as archive:
        mask = archive["episode"] == episode_index
        arrays = {key: archive[key][mask] for key in archive.files if key != "episode"}
    actions = arrays["action"]
    require(len(actions) == int(row["length"]), "Trace length differs from episode record")
    require(arrays["ram"].shape == (len(actions), 128) and arrays["values"].shape == (len(actions), 5),
            "Incompatible state/action trace")
    require(arrays["observation"].shape == (len(actions), 61), "Incompatible observation protocol")
    require(np.isfinite(arrays["values"]).all(), "Non-finite policy values")
    if algorithm == "PPO":
        require(np.allclose(arrays["values"].sum(axis=1), 1, atol=1e-6)
                and (arrays["values"] >= 0).all(), "Invalid recorded probabilities")
    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", obs_type="ram", frameskip=4,
                   repeat_action_probability=0.0 if condition == "A" else .25, mode=0)
    frames, earned = [], 0.0
    try:
        env.reset(seed=evaluation_seed)
        expected_maze = int(condition == "D")
        env.unwrapped.ale.setRAM(0, expected_maze)
        initial = native_score(env.unwrapped.ale.getRAM())
        labels = env.unwrapped.get_action_meanings()[:5]
        for index, action in enumerate(actions):
            ram = env.unwrapped.ale.getRAM()
            require(int(ram[0]) == expected_maze and np.array_equal(ram, arrays["ram"][index]),
                    f"Native pre-action RAM differs at decision {index + 1}")
            values = arrays["values"][index]
            require(np.isclose(values[int(action)], values.max()), "Action is not greedy in recorded values")
            if algorithm == "PPO":
                require(int(action) == int(values.argmax()), "PPO action differs from argmax")
            capture = index % stride == 0 or index == len(actions) - 1
            if capture:
                buffer = io.BytesIO()
                Image.fromarray(env.unwrapped.ale.getScreenRGB()).save(buffer, format="PNG")
                frame = dict(decision=index + 1, rgb=base64.b64encode(buffer.getvalue()).decode(),
                             action=int(action), values=values.tolist(), score_before=earned,
                             observation=arrays["observation"][index].tolist())
            after, reward, terminated, truncated, _ = env.step(int(action))
            earned += reward
            require(reward == arrays["raw_reward"][index], "Native reward differs")
            require(native_score(after) - initial == earned == arrays["score_after"][index],
                    "Native score registers differ")
            require(np.isclose(arrays["reward"][index], reward - (.1 if condition == "C" else 0)),
                    "Task reward differs")
            if "ram_after" in arrays:
                require(np.array_equal(after, arrays["ram_after"][index]), "Native post-action RAM differs")
            if int(after[0]) != expected_maze:
                require(condition == "D" and int(row["censored"]) == 1 and index == len(actions) - 1,
                        "Unrecorded map exit or action after a map exit")
            if index + 1 < len(actions):
                require(not terminated and not truncated and earned < 2000, "Action after task ending")
            if capture:
                frame.update(raw_reward=float(reward), task_reward=float(arrays["reward"][index]),
                             score_after=earned, maze_after=int(after[0]))
                frames.append(frame)
    finally:
        env.close()
    require(earned == float(row["native_score" if condition == "D" else "score"]), "Final native score differs")
    payload = dict(case=case, evaluation_seed=evaluation_seed, outcome=row, labels=labels, frames=frames,
                   stride=stride, verified_native_decisions=len(actions),
                   disclaimer="Recorded greedy-policy episode; no model selection, learning or counterfactual outcome is performed.")
    template = (HERE / "replay.html").read_text(encoding="utf-8")
    output = Path(output)
    require(not output.exists(), "Choose a fresh output path; existing inspection is preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("/* REPLAY_DATA */null", json.dumps(payload, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    receipt = dict(condition=condition, algorithm=algorithm, training_seed=training_seed,
                   evaluation_seed=evaluation_seed, verified_native_decisions=len(actions),
                   source_trace_sha256=sha(local_path(case["trace"])),
                   model_sha256=case["model_sha256"], output_sha256=sha(output),
                   outcome=row, new_training=False, new_evaluation=False,
                   native_replay_bypasses_custom_wrappers=True)
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=list("ABCD"), required=True)
    parser.add_argument("--algorithm", choices=["Q-learning", "PPO"], required=True)
    parser.add_argument("--training-seed", type=int, choices=[17, 29, 101, 314, 2718], required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.condition, args.algorithm, args.training_seed,
                            args.evaluation_seed, args.output, args.stride), indent=2))
