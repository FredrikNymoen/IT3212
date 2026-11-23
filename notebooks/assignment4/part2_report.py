"""Render measured Part-2 exports into inline LaTeX and one vector figure.

Only call after the whole locked experiment has finished. No model fitting,
parameter selection or PDF report compilation occurs here.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def mean_sd(values, digits=1, percent=False):
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return '--'
    if percent:
        values = values*100
    mean = f'{values.mean():.{digits}f}'
    return '$' + mean + (f' \\pm {values.std(ddof=1):.{digits}f}' if len(values) > 1 else '') + '$'


def render(results):
    results = Path(results)
    original = json.loads((results / 'final_summary.json').read_text())
    protocol = json.loads((results / 'protocol.json').read_text())
    if not protocol.get('controlled_settings'):
        print('Preserved initial per-setting pilot; report requires the corrected controlled comparison.', flush=True)
        return results
    hardware = json.loads((results / 'environment.json').read_text())
    assert len(original) == 18 and all(r['evaluation_episodes'] == 200 for r in original)
    rows = []
    curves = []
    for row in original:
        rows.append(row)
        if row['setting'] == 'B':
            assert row['parameter_fingerprint'] == row['transfer']['parameter_fingerprint']
            assert row['transfer']['evaluation_episodes'] == 200
            rows.append(dict(row, **{**row['transfer'], 'setting': 'D', 'train_seconds': 0,
                                      'training_interactions': 0}))
        curve = pd.read_csv(Path(row['directory']) / 'learning_curve.csv')
        curve['setting'], curve['algorithm'], curve['seed'] = row['setting'], row['algorithm'], row['seed']
        curves.append(curve)
    curve = pd.concat(curves, ignore_index=True)
    quality, costs, aggregates = [], [], []
    for setting in 'ABCD':
        for algorithm in ('Q-learning', 'PPO'):
            group = [r for r in rows if r['setting'] == setting and r['algorithm'] == algorithm]
            assert len(group) == 3
            quality.append(' & '.join([setting, algorithm,
                mean_sd([r['success_rate'] for r in group], percent=True),
                mean_sd([r['failure_rate'] for r in group], percent=True),
                mean_sd([r['mean_return'] for r in group])]) + r' \\')
            costs.append(' & '.join([setting, algorithm,
                mean_sd([r['success_length'] for r in group], 0),
                mean_sd([r['failure_length'] for r in group], 0),
                mean_sd([r['train_seconds'] for r in group]),
                mean_sd([r['evaluation_seconds'] for r in group])]) + r' \\')
            aggregates.append(dict(setting=setting, algorithm=algorithm,
                success_rate=np.mean([r['success_rate'] for r in group]),
                mean_score=np.mean([r['mean_score'] for r in group]),
                mean_return=np.mean([r['mean_return'] for r in group]),
                training_interactions=group[0]['training_interactions']))
    pd.DataFrame(aggregates).to_csv(results / 'condition_means.csv', index=False)
    images = ROOT / 'reports/assignment4/Images'
    images.mkdir(parents=True, exist_ok=True)
    preview = results / 'previews'
    preview.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), layout='constrained', sharey=True)
    for ax, algorithm in zip(axes, ('Q-learning', 'PPO')):
        for setting, color in zip('ABC', ('#007f82', '#c47936', '#545b87')):
            grouped = curve.loc[curve.algorithm.eq(algorithm) & curve.setting.eq(setting)].groupby('interactions').mean_score
            means, sd = grouped.mean(), grouped.std()
            x = means.index.to_numpy()
            ax.plot(x, means, marker='o', markersize=3, color=color, label=setting)
            ax.fill_between(x, means-sd, means+sd, color=color, alpha=.13)
        ax.set(title=algorithm, xlabel='Training interactions (decisions)')
        ax.grid(alpha=.2)
        ax.legend(title='Setting', frameon=False)
    axes[0].set_ylabel('Development evaluation score (game points)')
    fig.savefig(images / 'part2_learning_curves.pdf', bbox_inches='tight')
    fig.savefig(preview / 'part2_learning_curves.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    data = pd.DataFrame(aggregates).set_index(['setting', 'algorithm'])
    interpretations = []
    for algorithm in ('Q-learning', 'PPO'):
        a, b, c, d = [data.loc[(s, algorithm)] for s in 'ABCD']
        interpretations.append(
            f"For {algorithm}, mean evaluation score changed from {a.mean_score:.1f} points in A "
            f"to {b.mean_score:.1f} in B. With the step penalty in C it was {c.mean_score:.1f} points, "
            f"with success changing from {100*b.success_rate:.1f}\\% in B to {100*c.success_rate:.1f}\\%. "
            f"On the held-out layout D, the unchanged B model scored {d.mean_score:.1f} new points "
            f"and succeeded in {100*d.success_rate:.1f}\\% of episodes on average across trained seeds.")
    all_zero = all(r['success_rate'] == 0 for r in rows)
    interpretation = '\n\n'.join(interpretations)
    if all_zero:
        interpretation += ('\n\nNeither workflow reached the 2,000-point target during held-out evaluation. '
            'The score differences therefore show partial progress, not successful task completion. '
            'These results do not establish that either agent is reliable under the chosen budget.')
    selected = json.loads((results / 'selected_parameters.json').read_text())
    rates = '; '.join(f"{s}: Q-learning {selected[s+'_Q-learning']['learning_rate']:g}, "
                     f"PPO {selected[s+'_PPO']['learning_rate']:g}" for s in 'ABC')
    cpu = hardware['cpu'].replace('_', r'\_')
    text = r'''% Generated from locked Part 2 exports; keep measured tables synchronized.
% Source run: RUN_ID; raw per-seed/per-episode records are internal reproducibility evidence.
\setcounter{section}{1}
\section{Compare learning settings}
\label{sec:learning-settings}

\subsection{Environment and comparison protocol}
We use Gymnasium's \texttt{ALE/MsPacman-v5} with five actions: no-op and the
four cardinal directions. An interaction is one agent decision, normally four
emulator frames. Success means earning at least 2,000 points before losing all
lives or reaching HORIZON decisions. The horizon is part of the task, with time
remaining included in the observation. An unsuccessful timeout is a terminal
failure. The score target is an operational definition, not proof of clearing a level.

Following the team's instructor clarification, A/B use game-score increments
as rewards, retaining Part 1's assumption instead of the brief's terminal-only
success reward. A has no sticky actions. B repeats the previous action with
probability 0.25 at the emulator-frame level. C retains B and subtracts 0.1 game
points per decision. Both algorithms use the same reward scale, dividing by 100
internally. Raw reported rewards remain in points.

Observations describe position, nearest ghost and pellet offsets/distances,
directional pellet counts, the power-pellet countdown, facing direction and time
left. Q-learning discretises these into position bins, bearings, distance bands,
dominant pellet direction and coarse timer bins. PPO receives the 15 continuous
features and uses two 64-unit tanh layers. This compares the two workflows from
our predictions; it does not isolate algorithm effects from representation.

Each final agent receives BUDGET training interactions, with seeds 42, 7 and
2026 shared across algorithms. This bounded budget tests early learning rather
than convergence. Each algorithm/setting receives two learning-rate candidates,
16,384 training interactions each, one separate tuning seed and 50 development
evaluation episodes per candidate. Mean development success across A/B/C, then
mean game score, selects ONE common learning rate per algorithm, keeping other
settings fixed when the environment changes. Selected learning rates are RATES.
Q-learning uses discount 0.99 and
epsilon decaying from 1 to 0.05 over 80\% of its budget. PPO uses discount 0.99,
GAE 0.95, 512-step rollouts, four epochs, 128-step minibatches, clipping 0.2 and
entropy coefficient 0.01.

The controlled comparison corrects an initial per-setting tuning design after
partial evaluation had been inspected. Only development data select the common
rates; unchanged runs are reused with recorded provenance. These evaluation
seeds were already used, so this is a revised comparison, not a new independent
test. Each trained agent has 200 evaluation episodes per condition.
Q-learning acts greedily with seeded uniform tie-breaking;
PPO selects its highest-probability action. No parameters update, Q-table entries
are not added, and evaluation does not alter the training RNG. Final seeds
900000--900199 are disjoint from training and development seeds. Shared seeds
do not imply identical trajectories once actions differ.

\subsection{Settings A--D and measured results}
D evaluates the unchanged B agents on a different ROM maze layout, selected
before the first action. The training and development layout is held fixed;
native level transitions are checked to prevent accidental exposure. Action and
feature formats stay identical. Relative object features offer some basis for
transfer, while absolute position bins can alias different locations. The map
intervention may also change layout-dependent game behaviour, so D measures
transfer to that modified environment rather than an isolated geometry effect.

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrr@{}}
\toprule
Setting & Agent & Success (\%) & Failure (\%) & Return (points) \\
\midrule
QUALITY_ROWS
\bottomrule
\end{tabular}
\caption{Final evaluation: mean $\pm$ sample SD across three independently
trained agents, each evaluated for 200 episodes per condition. D reuses B agents.
C returns include the step penalty and cannot be directly compared with A/B/D returns.}
\label{tab:part2-outcomes}
\end{table}

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrrr@{}}
\toprule
Setting & Agent & Success length & Failure length & Train (s) & Evaluate (s) \\
\midrule
COST_ROWS
\bottomrule
\end{tabular}
\caption{Mean $\pm$ sample SD of per-agent means; lengths are decisions,
conditional on outcome. A dash means no episode of that outcome occurred.
SD is omitted if only one seed has that outcome. Each A/B/C agent trains for
BUDGET interactions; D adds zero training interactions. Training times exclude
curve evaluations; evaluation times cover 200 episodes and trace collection,
excluding archive compression.}
\label{tab:part2-costs}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{Images/part2_learning_curves.pdf}
\caption{Learning curves for source run \texttt{RUN_ID/final/}: A/B/C,
Q-learning/PPO, seeds 42/7/2026. Each checkpoint evaluates 20 separate development
episodes per agent, every QUARTER training interactions including zero and the
final budget. Lines show mean game score across trained seeds; shading is one
sample SD. No smoothing is applied. Final held-out episodes are excluded.}
\label{fig:part2-learning}
\end{figure}

\subsection{Interpretation and limits}
INTERPRETATION

We compare B/C using success, failure and newly earned game points, since the
reward definitions differ. Episode lengths are conditional on outcome and
cannot alone establish safer behaviour. Results are summarised across trained
agents, not by treating all evaluation episodes as independent training runs.
Three training seeds and a two-candidate search give limited evidence about
robustness. The coarse Q state can merge different situations; the feature vector
also omits parts of the emulator state. Runtime was measured on CPU_NAME with
one numerical thread per worker and up to WORKERS concurrent workers, so timings
describe shared-compute runs rather than isolated inference benchmarks. Reused
agents retain their original timing provenance.
'''
    replacements = {'RUN_ID': results.name.replace('_', r'\_'), 'HORIZON': f"{protocol['horizon']:,}",
        'BUDGET': f"{protocol['main_budget']:,}", 'QUARTER': f"{protocol['main_budget']//4:,}",
        'RATES': rates, 'QUALITY_ROWS': '\n'.join(quality), 'COST_ROWS': '\n'.join(costs),
        'INTERPRETATION': interpretation, 'CPU_NAME': cpu, 'WORKERS': str(hardware['max_concurrent_workers'])}
    for key, value in replacements.items():
        text = text.replace(key, value)
    (ROOT / 'reports/assignment4/Sections/02_learning_settings.tex').write_text(text, encoding='utf-8')
    print('Rendered measured Part 2 report and learning curve:', results.name, flush=True)
    return results
