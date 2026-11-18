"""Render report figures from fixed Assignment 3 exports; never refit/select models.

Run after the combined notebook and, for the current report, finalize_lda_report.py:
    python notebooks/assignment3/report_tasks_09_10.py
Vector PDFs are saved once in report/Images/; local PNGs support notebook previews.
"""

from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_RESULTS = ROOT / "results/assignment3/combined"
FINAL_RESULTS = ROOT / "results/assignment3/lda_selected"
BASE_FIGURES = ROOT / "figures/assignment3/combined"
FIGURES = BASE_FIGURES
IMAGES = ROOT / "report/Images"
NAVY, TEAL, GREY, ORANGE = "#183b56", "#007f82", "#82929f", "#c47936"


def save(fig, stem):
    for directory in (FIGURES, IMAGES):
        directory.mkdir(parents=True, exist_ok=True)
    for destination in (IMAGES / f"{stem}.pdf", FIGURES / f"{stem}.png"):
        fig.savefig(destination, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return str(FIGURES / f"{stem}.png")


def render(results_dir=None):
    global FIGURES
    if results_dir is None:
        results_dir = FINAL_RESULTS if (FINAL_RESULTS / 'summary.json').is_file() else BASE_RESULTS
    RESULTS = Path(results_dir)
    FIGURES = ROOT / 'figures/assignment3/lda_selected' if RESULTS == FINAL_RESULTS else BASE_FIGURES
    IMAGES.mkdir(parents=True, exist_ok=True)
    # Keep the feature figure and optional appendix assets synchronized too.
    for stem in ("01_features_and_representations", "02_basic_development_pr",
                 "05_cost_and_input_dependence", "06_probability_reliability"):
        shutil.copy2(BASE_FIGURES / f"{stem}.png", IMAGES / f"{stem}.png")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "grid.alpha": .22,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    locked = json.loads((RESULTS / "locked_choices.json").read_text())
    selected = locked["selected_model"]
    meta = json.loads((RESULTS / "dnn_metadata.json").read_text())
    trials = pd.read_csv(RESULTS / "search_Deep_neural_network.csv")
    history = pd.read_csv(RESULTS / "dnn_stopping_history.csv")
    test = pd.read_csv(RESULTS / "test_metrics.csv", index_col="model")
    development = pd.read_csv(RESULTS / "development_models.csv", index_col="model")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    capacity = pd.read_csv(RESULTS / "capacity_sensitivity.csv")
    truth = predictions.dropout_label.to_numpy(dtype=int)
    n, positive = len(truth), int(truth.sum())
    assert n == meta["test_rows"] and positive > 0
    assert len(trials) == 8 and np.isfinite(trials.mean_test_AP).all()
    assert np.isclose(test.loc[selected, "AP"],
                      json.loads((RESULTS / "summary.json").read_text())["test_AP"])
    # Check every exported count against the frozen scores; this is validation,
    # not threshold selection or retraining.
    scores = predictions[f"{selected}_score"].to_numpy()
    for row in capacity.itertuples():
        k = int(np.ceil(row.capacity * n))
        reached = truth[np.argsort(-scores, kind="stable")[:k]].sum()
        assert row.contacts == k and row.dropouts_reached == reached
    paths = []

    # One axes per figure: no compound panels or a/b/c labels.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    labels = []
    for i, row in enumerate(trials.itertuples()):
        params = json.loads(row.params)
        widths = "-".join(map(str, params["hidden"]))
        chosen = params == locked["models"]["Deep neural network"]["parameters"]
        color = TEAL if chosen else GREY
        ax.errorbar(row.mean_test_AP, i, xerr=row.std_test_AP,
                    fmt="D" if chosen else "o", color=color,
                    capsize=3, markersize=6, linewidth=1.5)
        labels.append(f"{widths} | {params['lr']:g} | {params['dropout']:.1f}")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="Mean validation AP (higher is better)",
           title="DNN settings: validation results")
    ax.grid(axis="x")
    paths.append(save(fig, "09_dnn_hyperparameter_comparison"))

    best = meta["best_epoch"]
    for column, title, ylabel, stem in [
        ("train_weighted_loss", "DNN training loss", "Training loss",
         "09_dnn_training_loss"),
        ("stopping_AP", "DNN internal validation", "Internal validation AP",
         "09_dnn_stopping_performance"),
    ]:
        fig, ax = plt.subplots(figsize=(4.4, 3.0), layout="constrained")
        ax.plot(history.epoch, history[column], color=TEAL, linewidth=2)
        ax.axvline(best, color=ORANGE, linestyle="--", linewidth=1.4,
                   label=f"Selected epoch: {best}")
        ax.set(title=title, ylabel=ylabel, xlabel="Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y")
        ax.legend(frameon=False, fontsize=10)
        if column == "stopping_AP":
            value = history.loc[history.epoch == best, column].iloc[0]
            ax.scatter([best], [value], s=38, color=ORANGE, zorder=3)
            ax.annotate(f"Best AP: {value:.4f}", (best, value), xytext=(8, -30),
                        textcoords="offset points", fontsize=10, color=NAVY)
        paths.append(save(fig, stem))

    # Development AP determines display order, not the interpretability preference.
    order = development.drop(index="Prior baseline").index.tolist()
    plot = test.loc[order]
    short = {"Deep neural network": "DNN", "Tree bagging control": "Single-tree control"}
    colors = [NAVY if name == selected else TEAL if name == "Deep neural network" else GREY
              for name in order]
    ypos = np.arange(len(order))
    for column, xlabel, title, stem in [
        ("AP", "Test AP (higher is better)", "Ranking on test students",
         "10_test_average_precision"),
        ("search_seconds", "Search time (seconds)", "Time spent testing model settings",
         "10_model_search_time"),
        ("score_ms_per_student", "Prediction time (ms per student; log scale)",
         "Time to score students", "10_model_inference_time"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
        ax.set_yticks(ypos, [short.get(name, name) for name in order])
        ax.invert_yaxis()
        ax.grid(axis="x")
        if column == "search_seconds":
            ax.barh(ypos, plot[column], color=colors, height=.48, zorder=2)
            ax.set_xlim(0, plot[column].max()*1.18)
        else:
            ax.scatter(plot[column], ypos, c=colors, s=45, zorder=3)
        if column == "AP":
            ax.set_xlim(.89, .968)
        if column == "score_ms_per_student":
            ax.set(xscale="log", xlim=(min(.005, plot[column].min()*.65), max(1.2, plot[column].max()*2)))
        for i, value in enumerate(plot[column]):
            label = f"{value:.1f}" if column == "search_seconds" else f"{value:.3f}"
            ax.annotate(label, (value, i), xytext=(6, -3), textcoords="offset points", fontsize=10)
        ax.set(xlabel=xlabel, title=title)
        paths.append(save(fig, stem))

    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    budgets = np.linspace(0, .5, 201)
    ax.plot(budgets, np.minimum(np.ceil(budgets*n)/positive, 1), color=GREY,
            linestyle=":", label="Best possible result for this budget")
    ax.plot(budgets, np.ceil(budgets*n)/n, color=GREY, linestyle="--", label="Random contacts (expected)")
    ax.plot(capacity.capacity, capacity.recall_at_capacity, marker="o", color=NAVY,
            linewidth=2, label=selected)
    policy = capacity.loc[np.isclose(capacity.capacity, locked["capacity"])].iloc[0]
    ax.scatter([policy.capacity], [policy.recall_at_capacity], s=85, color=ORANGE, zorder=5)
    ax.annotate(f"20% budget: {int(policy.contacts)} contacts\n"
                f"{int(policy.dropouts_reached)}/{positive} cases ({policy.recall_at_capacity:.1%})",
                (policy.capacity, policy.recall_at_capacity), xytext=(.24, .30),
                arrowprops={"arrowstyle": "-", "color": ORANGE}, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 3})
    ax.set(xlim=(0, .51), ylim=(0, 1.04), xlabel="Share of students contacted",
           ylabel="Share of dropout cases reached", title="Student-support contact budget")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    paths.append(save(fig, "10_advisor_contact_capacity"))

    alert = predictions[f"{selected}_alert"].to_numpy(dtype=bool)
    matrix = np.array([[np.sum((truth == a) & (alert == b)) for b in [0, 1]] for a in [0, 1]])
    fig, ax = plt.subplots(figsize=(5.8, 4.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    cell_labels = [["Correct Graduate", "False alert"],
                   ["Missed Dropout", "Correct Dropout"]]
    for (i, j), value in np.ndenumerate(matrix):
        fraction = value / matrix.max()
        ax.text(j, i, f"{value}\n{cell_labels[i][j]}", ha="center", va="center",
                fontsize=12, color="white" if fraction > .55 else NAVY)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Predicted Graduate\nNo alert", "Predicted Dropout\nAlert"],
           yticklabels=["Graduate", "Dropout"], xlabel="Model prediction",
           ylabel="Actual recorded outcome", title=f"{selected}: actual vs predicted outcomes")
    paths.append(save(fig, "10_alert_confusion_matrix"))

    print("Validated report figures:")
    print("\n".join(paths))
    return paths


if __name__ == "__main__":
    REPORT_FIGURES = render()
