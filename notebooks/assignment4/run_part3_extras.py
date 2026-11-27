"""Part 3 diagnostics, same-state prediction probes, and publication figures.

Runs:
1. Frozen execution diagnostic (argmax vs. sampled) on the 5 baseline PPO and
   5 zero-entropy PPO models at the fixed 262,144-step cap across 50 fresh
   paired seeds (1700000..1700049), measuring success rate and visited-state
   action entropy (nats).
2. Training-return vs. evaluation-success rank correlation across A, B, C and
   Part 3 B cohorts.
3. Same-state probes for Part 1's three predicted state types using real
   recorded Setting B evaluation observations.
4. Generates:
   - reports/assignment4/Images/part2_all_settings.png
   - reports/assignment4/Images/part3_mechanism_curves.png
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from ppo import FrozenPolicy
from q_learning import TabularQ, make_environment, require, sha, verify_sources, write_csv, write_json

PART3_STUDY = ROOT / "results/assignment4/part3/study/e8c0c6a8f4071325"
UNIFIED_DIR = ROOT / "results/assignment4/unified_part2"
EXTRA_DIR = ROOT / "results/assignment4/part3_extras"
IMAGES_DIR = ROOT / "reports/assignment4/Images"
DIAG_SEEDS = list(range(1700000, 1700050))


def eval_exec_mode_one(label: str, run_dir_str: str, seed: int, mode: str):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run_dir = Path(run_dir_str)
        model_path = run_dir / "checkpoint_262144.zip"
        agent = FrozenPolicy.load(model_path)
        before = agent.fingerprint()
        env = make_environment("B")
        out_dir = EXTRA_DIR / "exec_diagnostic"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        entropies = []
        try:
            for ep_seed in DIAG_SEEDS:
                obs, _ = env.reset(seed=ep_seed)
                rng = np.random.default_rng(ep_seed + 5000000)
                for _ in range(1000):
                    probs = agent.values_for_observation(obs)
                    probs = np.clip(probs, 1e-12, 1.0)
                    probs = probs / probs.sum()
                    entropies.append(float(-np.sum(probs * np.log(probs))))
                    if mode == "argmax":
                        action = int(probs.argmax())
                    else:
                        action = int(rng.choice(5, p=probs))
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc:
                        rows.append(dict(
                            evaluation_seed=ep_seed, success=int(info["success"]),
                            score=info["score"], length=info["episode_steps"], reason=info["reason"],
                        ))
                        break
        finally:
            env.close()
        require(agent.fingerprint() == before, "Model changed during execution diagnostic")
        write_csv(out_dir / f"{label}_{mode}_seed_{seed}.csv", rows)
        return dict(
            label=label, mode=mode, training_seed=seed,
            episodes=len(rows), success_rate=float(np.mean([r["success"] for r in rows])),
            mean_score=float(np.mean([r["score"] for r in rows])),
            mean_entropy_nats=float(np.mean(entropies)),
            model_sha256=sha(model_path),
            source_run=str(run_dir.relative_to(ROOT)),
        )


def run_exec_diagnostic(workers: int = 10):
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = EXTRA_DIR / "exec_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("PPO_baseline", PART3_STUDY / "PPO_50ed71056d1034d6"),
        ("PPO_zero_entropy", PART3_STUDY / "PPO_8d1ce00dd6d355b3"),
    ]
    seeds = [17, 29, 101, 314, 2718]
    jobs = [(label, str(base / f"seed_{seed}"), seed, mode)
            for label, base in configs for seed in seeds for mode in ("argmax", "sampled")]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_exec_mode_one, *j) for j in jobs]
        for f in as_completed(futs):
            res = f.result()
            rows.append(res)
            print(f"Exec diag {res['label']} {res['mode']} seed {res['training_seed']}: "
                  f"{res['success_rate']:.1%}, H={res['mean_entropy_nats']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["label", "mode", "training_seed"])
    df.to_csv(out_dir / "per_seed.csv", index=False)
    summary = df.groupby(["label", "mode"]).agg(
        success_mean=("success_rate", "mean"),
        success_sd=("success_rate", "std"),
        entropy_mean=("mean_entropy_nats", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def run_train_eval_rank_check():
    """Check Spearman rank correlation between training return and evaluation success."""
    abc = pd.read_csv(UNIFIED_DIR / "abc_per_seed.csv")
    rows = []
    for (setting, alg), grp in abc.groupby(["setting", "algorithm"]):
        tr_means = []
        for _, r in grp.iterrows():
            run = ROOT / r["run_id"]
            if alg == "Q-learning":
                tdf = pd.read_csv(run / "training.csv")
                tr_ret = float(tdf["task_return"].mean())
            else:
                parts = [pd.read_csv(run / f"training_{i}.monitor.csv", skiprows=1) for i in range(4)]
                tdf = pd.concat(parts, ignore_index=True)
                tr_ret = float(tdf["task_return"].mean())
            tr_means.append(dict(
                setting=setting, algorithm=alg, training_seed=int(r["training_seed"]),
                train_mean_return=tr_ret, eval_success=float(r["success_rate"]),
                eval_return=float(r["mean_return"]),
            ))
        sub = pd.DataFrame(tr_means)
        rho = float(sub["train_mean_return"].corr(sub["eval_success"], method="spearman"))
        top_train = int(sub.sort_values("train_mean_return", ascending=False).iloc[0]["training_seed"])
        max_eval = float(sub["eval_success"].max())
        top_eval_seeds = set(sub.loc[sub["eval_success"] == max_eval, "training_seed"].astype(int))
        rows.append(dict(
            setting=setting, algorithm=alg, spearman_rho=rho,
            train_leader_seed=top_train, train_leader_is_eval_leader=bool(top_train in top_eval_seeds),
        ))
    out_dir = EXTRA_DIR / "rank_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out_dir / "summary.csv", index=False)
    print(rdf.to_string(index=False), flush=True)


def run_same_state_probes():
    """Probe the 5 frozen B Q models and 5 frozen B PPO models on identical B states
    matching Part 1's State 1 (ghost near right shortcut, up safe) and State 3
    (close pellet on right, larger pellet cluster on left, no immediate threat).
    State 2 (frightened ghost with precise timer) cannot be verified with the capped RAM proxy.
    """
    torch.set_num_threads(1)
    seeds = [17, 29, 101, 314, 2718]
    q_models, ppo_models, ppo12_models = [], [], []
    for seed in seeds:
        for base, target, loader in (
            (PART3_STUDY / "Q-learning_1e35db1b778e5790", q_models, TabularQ.load),
            (PART3_STUDY / "PPO_50ed71056d1034d6", ppo_models, FrozenPolicy.load),
            (PART3_STUDY / "PPO_3b1630567f4b85f5", ppo12_models, FrozenPolicy.load),
        ):
            run = base / f"seed_{seed}"
            sel = json.loads((run / "selected.json").read_text(encoding="utf-8"))
            target.append(loader(run / sel["model"]))

    # Collect real states from Part 4 recorded B traces (10 episodes, ~6,000+ states)
    trace_dir = ROOT / "results/assignment4/part4/part3_B_054b0e6b63381969"
    all_obs = []
    for npz_path in sorted(trace_dir.glob("*_traces.npz")):
        with np.load(npz_path) as d:
            all_obs.append(d["observation"])
    obs_arr = np.concatenate(all_obs, axis=0)

    # State 1: junction with UP (obs[15]>0.5) and RIGHT (obs[16]>0.5) available,
    # unpowered (obs[12]==0), nearest ghost to the right (dx>0, |dx|>|dy|, dist<=40),
    # right threat high (obs[24]>0.36), up threat low (obs[23]<0.25), pellets right (obs[9]>0)
    s1_mask = (
        (obs_arr[:, 15] > 0.5) & (obs_arr[:, 16] > 0.5) &
        (obs_arr[:, 12] == 0) &
        (obs_arr[:, 2] > 0) & (np.abs(obs_arr[:, 2] * 160) >= np.abs(obs_arr[:, 3] * 210)) &
        (obs_arr[:, 4] * 320 <= 48) &
        (obs_arr[:, 24] > 0.36) & (obs_arr[:, 23] < 0.25) &
        (obs_arr[:, 9] > 0)
    )
    # State 3: RIGHT (obs[16]>0.5) and LEFT (obs[17]>0.5) available, no immediate ghost threat
    # (obs[4]*320 >= 60), nearest pellet to the right (obs[5]>0, obs[7]*320 <= 24),
    # larger pellet group to the left (obs[11] >= obs[9] + 0.04)
    s3_mask = (
        (obs_arr[:, 16] > 0.5) & (obs_arr[:, 17] > 0.5) &
        (obs_arr[:, 4] * 320 >= 60) &
        (obs_arr[:, 5] > 0) & (obs_arr[:, 7] * 320 <= 24) &
        (obs_arr[:, 11] >= obs_arr[:, 9] + 0.04)
    )

    def summarize(mask, actions_of_interest):
        sub = obs_arr[mask]
        if len(sub) == 0:
            return {"n_states": 0}
        res = {"n_states": int(len(sub))}
        for name, group, is_q in (("Q", q_models, True), ("PPO", ppo_models, False), ("PPO12", ppo12_models, False)):
            counts = {a: [] for a in actions_of_interest}
            for m in group:
                probs_a = {a: 0.0 for a in actions_of_interest}
                for o in sub:
                    v = m.values_for_observation(o)
                    if is_q:
                        best = np.flatnonzero(v == v.max())
                        for a in actions_of_interest:
                            probs_a[a] += float(a in best) / len(best)
                    else:
                        act = int(v.argmax())
                        for a in actions_of_interest:
                            probs_a[a] += float(act == a)
                for a in actions_of_interest:
                    counts[a].append(probs_a[a] / len(sub) * 100.0)
            res[name] = {str(a): float(np.mean(counts[a])) for a in actions_of_interest}
        return res

    # Actions: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN
    out = {
        "total_recorded_states": int(len(obs_arr)),
        "state1_ghost_right_shortcut_pred_Q_RIGHT2_PPO_UP1": summarize(s1_mask, (1, 2)),
        "state2_frightened_ghost_timer": "Unverifiable: RAM 116 proxy does not measure remaining edible duration.",
        "state3_few_pellets_right_cluster_left_pred_Q_RIGHT2_PPO_LEFT3": summarize(s3_mask, (2, 3)),
    }
    out_dir = EXTRA_DIR / "same_state_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", out)
    print("Same-state probes:", json.dumps(out, indent=2), flush=True)


def make_part2_figure():
    """Create 4-panel Part 2 figure (A, B, C learning curves + D clean vs obstructed)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [17, 29, 101, 314, 2718]
    colors = {"Q-learning": "#1f77b4", "PPO": "#d62728"}

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), sharey=True)
    for ax, setting, title in zip(axes[:3], ("A", "B", "C"), ("A: Deterministic", "B: Stochastic", "C: Step penalty")):
        proto = p3.experiment_protocol(setting=setting)
        for alg in ("Q-learning", "PPO"):
            dfs = []
            for seed in seeds:
                run = p3.job_folder(proto, alg, p3.PART2_BASELINE[alg], seed)
                c = pd.read_csv(run / "curves.csv")
                dfs.append(c[["interactions", "success_rate"]].assign(seed=seed))
                ax.plot(c["interactions"] / 1000.0, c["success_rate"] * 100.0,
                        color=colors[alg], alpha=0.22, linewidth=0.9)
            all_c = pd.concat(dfs, ignore_index=True)
            agg = all_c.groupby("interactions")["success_rate"].agg(["mean", "std"]).reset_index()
            x = agg["interactions"] / 1000.0
            m = agg["mean"] * 100.0
            s = agg["std"].fillna(0) * 100.0
            ax.plot(x, m, color=colors[alg], linewidth=2.2, marker="o", markersize=3.5, label=alg)
            ax.fill_between(x, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=colors[alg], alpha=0.14)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Interactions (k)", fontsize=9)
        ax.set_xlim(0, 265)
        ax.set_ylim(-2, 104)
        ax.grid(True, alpha=0.25)

    # Panel D: paired clean vs obstructed
    ax_d = axes[3]
    pdf = pd.read_csv(UNIFIED_DIR / "setting_D/paired_summary.csv")
    for alg in ("Q-learning", "PPO"):
        sub = pdf[pdf["label"] == alg]
        for _, r in sub.iterrows():
            ax_d.plot([0, 1], [r["clean_success_pct"], r["obstructed_success_pct"]],
                      color=colors[alg], alpha=0.28, linewidth=1.0, marker="o", markersize=3)
        ax_d.plot([0, 1], [sub["clean_success_pct"].mean(), sub["obstructed_success_pct"].mean()],
                  color=colors[alg], linewidth=2.4, marker="o", markersize=5, label=alg)
    ax_d.set_title("D: Observation robustness", fontsize=10, fontweight="bold")
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Clean", "Obstructed"], fontsize=9)
    ax_d.set_xlim(-0.2, 1.2)
    ax_d.grid(True, alpha=0.25)

    axes[0].set_ylabel("Success rate (%)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = IMAGES_DIR / "part2_all_settings.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}", flush=True)


def make_part3_figure():
    """Create multi-panel Part 3 mechanism learning curves from sweep curves.csv files."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sweeps_dir = PART3_STUDY / "sweeps"
    panels = [
        ("Q-learning_epsilon_end_44e262f0407f6f5a", r"Q: $\epsilon_{\mathrm{end}}$", 0.05),
        ("Q-learning_gamma_a7dfc09b21386364", r"Q: $\gamma$", 0.995),
        ("Q-learning_alpha_7c1bd8c148ebbb73", r"Q: $\alpha$", 0.15),
        ("PPO_gamma_58c70af0b0ce6aa4", r"PPO: $\gamma$", 0.995),
        ("PPO_entropy_scale_540ec07d218ba7dc", "PPO: entropy scale", 1.0),
        ("PPO_learning_rate_scale_c153bc0b479a9167", "PPO: LR scale", 1.0),
        ("PPO_n_epochs_b45bbccce63e7f52", r"PPO: epochs (at $2\times$ LR)", 6),
    ]
    palette = ["#2ca02c", "#1f77b4", "#d62728"]

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.8), sharey=True)
    flat = axes.ravel()
    for ax, (folder_name, title, inc) in zip(flat, panels):
        df = pd.read_csv(sweeps_dir / folder_name / "curves.csv")
        vals = sorted(df["value"].unique())
        for idx, val in enumerate(vals):
            sub = df[df["value"] == val]
            agg = sub.groupby("interactions")["success_rate"].agg(["mean", "std"]).reset_index()
            x = agg["interactions"] / 1000.0
            m = agg["mean"] * 100.0
            s = agg["std"].fillna(0) * 100.0
            is_inc = abs(float(val) - float(inc)) < 1e-9
            lbl = f"{val:g}" + (" (base)" if is_inc else "")
            ls = "-" if is_inc else "--"
            lw = 2.2 if is_inc else 1.7
            col = "#222222" if is_inc else palette[idx % len(palette)]
            ax.plot(x, m, color=col, linestyle=ls, linewidth=lw, marker="o", markersize=3, label=lbl)
            ax.fill_between(x, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=col, alpha=0.11)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlim(0, 265)
        ax.set_ylim(-2, 104)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left", frameon=True, framealpha=0.85)

    # Panel 8: Early-learning diagnostic bar comparison (65,536 interactions, 50 fresh episodes)
    ax_diag = flat[7]
    diag_df = pd.read_csv(PART3_STUDY / "diagnostic_early_3142fac3111db314/per_seed.csv")
    order = ["Q-learning", "PPO-6", "PPO-12"]
    means = [diag_df[diag_df["configuration"] == c]["success_pct"].mean() for c in order]
    sds = [diag_df[diag_df["configuration"] == c]["success_pct"].std() for c in order]
    bar_cols = ["#1f77b4", "#ff7f0e", "#d62728"]
    ax_diag.bar(range(3), means, yerr=sds, capsize=4, color=bar_cols, alpha=0.85)
    ax_diag.set_xticks(range(3))
    ax_diag.set_xticklabels(["Q", "PPO (6 ep)", "PPO (12 ep)"], fontsize=8.5)
    ax_diag.set_title("Early diagnostic (65k, 50 eps)", fontsize=9.5, fontweight="bold")
    ax_diag.grid(True, axis="y", alpha=0.25)

    for ax in flat[4:7]:
        ax.set_xlabel("Interactions (k)", fontsize=8.5)
    flat[0].set_ylabel("Dev. success (%)", fontsize=9)
    flat[4].set_ylabel("Dev. success (%)", fontsize=9)
    fig.tight_layout()
    out_path = IMAGES_DIR / "part3_mechanism_curves.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["exec", "ranks", "probes", "figs", "all"], default="all")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.stage in ("exec", "all"):
        run_exec_diagnostic(args.workers)
    if args.stage in ("ranks", "all"):
        run_train_eval_rank_check()
    if args.stage in ("probes", "all"):
        run_same_state_probes()
    if args.stage in ("figs", "all"):
        make_part2_figure()
        make_part3_figure()


if __name__ == "__main__":
    main()
