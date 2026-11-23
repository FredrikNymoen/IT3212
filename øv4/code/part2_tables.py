"""Report-ready Part 2 tables from part2_run.py's seed-level results.

    python part2_tables.py --results ../results/assignment4_v8

An explicit experiment directory is required so older results cannot be selected
silently. Writes part2_tables.md and learning curves beside the source logs.
Nothing is estimated or imputed: a condition with no completed runs is reported as
missing, and a measure that never occurred stays blank rather than becoming zero.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def find_results(given):
    """The results folder: given explicitly, or searched for upward from this script."""
    if given:
        if not given.is_dir():
            raise SystemExit(f'No such folder: {given}')
        return given
    raise SystemExit('Pass --results explicitly to select one experiment.')

# Spec "Required evidence", in the order the brief lists it.
EVIDENCE = [('success_rate', 'Success rate', '{:.3f}'),
            ('mean_score', 'Mean score', '{:.0f}'),
            ('mean_return', 'Mean return', '{:.1f}'),
            ('unseen_state_fraction', 'Unseen states', '{:.3f}'),
            ('cleared_rate', 'Cleared rate', '{:.3f}'),
            ('failure_rate', 'Failure rate', '{:.3f}'),
            ('success_length', 'Len (success)', '{:.1f}'),
            ('failure_length', 'Len (failure)', '{:.1f}'),
            ('timeout_rate', 'Timeout rate', '{:.3f}'),
            ('agent_decisions', 'Agent decisions', '{:.0f}'),
            ('environment_interactions', 'Training env. steps', '{:.0f}'),
            ('training_seconds', 'Training (s)', '{:.1f}'),
            ('curve_seconds', 'Curve eval. (s)', '{:.1f}'),
            ('final_eval_seconds', 'Final eval. (s)', '{:.1f}'),
            ('greedy_eval_seconds', 'Extra greedy eval. (s)', '{:.1f}')]
# Settings B and C define reward differently, so only outcome measures compare.
COMMON = ['success_rate', 'cleared_rate', 'failure_rate', 'timeout_rate', 'mean_score',
          'success_length', 'failure_length']


def cell(values, fmt, spread=True):
    """mean ± sample SD over seeds, or '-' when the measure never occurred."""
    v = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if v.empty:
        return '-'
    if not spread or len(v) < 2:
        return fmt.format(v.mean())
    # A standard deviation is never negative, so it never carries a sign.
    return f'{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1)).lstrip("+")}'


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |'
    return '\n'.join([line(header), '|' + '|'.join('-'*(w+2) for w in widths) + '|',
                      *(line(r) for r in rows)])


def required_evidence(main, evidence=EVIDENCE):
    rows = []
    for condition in ['A', 'B', 'C', 'D']:
        for algorithm in ['Q', 'PPO']:
            g = main[(main.condition == condition) & (main.algorithm == algorithm)]
            if g.empty:
                rows.append([condition, algorithm, 'no completed runs'] + ['-']*len(evidence))
                continue
            rows.append([condition, algorithm, f'{len(g)} seeds']
                        + [cell(g[key], fmt) for key, _, fmt in evidence])
    return table(rows, ['Setting', 'Agent', 'n'] + [label for _, label, _ in evidence])


def greedy_evidence(main):
    evidence = [('greedy_success_rate', 'Success rate', '{:.3f}'),
                ('greedy_mean_score', 'Mean score', '{:.0f}'),
                ('greedy_mean_return', 'Mean return', '{:.1f}')]
    rows = []
    for condition in 'ABC':
        g = main[(main.algorithm == 'PPO') & (main.condition == condition)]
        if g.empty:
            continue
        rows.append([condition, len(g)] + [cell(g.get(key, []), fmt)
                                          for key, _, fmt in evidence])
    return table(rows, ['Setting', 'Trained seeds', *[label for _, label, _ in evidence]])


def paired(df, left, right, label, caveat):
    """Per-seed differences on shared seeds only; unmatched seeds are never averaged in."""
    out = []
    for algorithm in ['Q', 'PPO']:
        a = df[(df.condition == left) & (df.algorithm == algorithm)].set_index('seed')
        b = df[(df.condition == right) & (df.algorithm == algorithm)].set_index('seed')
        shared = sorted(set(a.index) & set(b.index))
        if not shared:
            out.append([algorithm, 0, *['-']*len(COMMON)])
            continue
        deltas = [cell(b.loc[shared, m].values - a.loc[shared, m].values,
                       '{:+.0f}' if m == 'mean_score' else
                       '{:+.1f}' if m.endswith('_length') else '{:+.3f}')
                  for m in COMMON]
        out.append([algorithm, len(shared), *deltas])
    head = [f'{m.replace("_", " ")} Δ' for m in COMMON]
    return f'{label} (paired on shared training seeds; {caveat})\n\n' + table(
        out, ['Agent', 'seeds', *head])


def plot_curves(curves, directory):
    """One figure per setting: each algorithm's mean curve with an SD band."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    written = []
    # The x axis counts agent decisions. Actual environment interactions are higher,
    # because each episode spends START_DECISIONS steps on the opening animation;
    # both numbers are in seed_results (training_interactions, environment_interactions).
    use_env_steps = 'environment_interactions' in curves and curves.environment_interactions.notna().all()
    if use_env_steps:
        x_column = 'environment_interactions'
    else:
        x_column = 'decisions'
        if 'interactions' in curves:
            curves = curves.copy()
            curves[x_column] = curves.get(x_column, curves.interactions).fillna(curves.interactions)
    # Success sits near zero at the declared 2000-point threshold, so a success-only
    # figure hides the change. Score is plotted beside it.
    for condition in sorted(curves.condition.unique()):
        series = curves.loc[curves.condition == condition].copy()
        if 'mean_score' not in series:
            series['mean_score'] = np.nan
        if condition in ('A', 'B') and not use_env_steps:
            # Archived A and B awarded raw score only.
            series['mean_score'] = series.mean_score.fillna(series.mean_return)
        has_score = series.mean_score.notna().all()
        score = 'mean_score' if has_score else 'mean_return'
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        drawn = False
        for algorithm in ('Q', 'PPO'):
            group = series[series.algorithm == algorithm]
            if group.empty:
                continue
            for axes, column, clip in ((top, score, False),
                                       (bottom, 'success_rate', True)):
                stats = group.groupby(x_column)[column]
                mean, sd = stats.mean(), stats.std()
                axes.plot(mean.index, mean.values, marker='o',
                          label=f'{algorithm} (n={group.seed.nunique()})')
                low, high = mean-sd, mean+sd
                if clip:   # a rate cannot leave [0, 1]
                    low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
                axes.fill_between(mean.index, low, high, alpha=.18)
            drawn = True
        if not drawn:
            plt.close(figure)
            continue
        label = 'raw game score' if has_score else 'shaped return'
        top.set(ylabel=f'Checkpoint mean {label}',
                title=f'Setting {condition}: mean and one SD across trained seeds')
        top.legend()
        axis_label = 'Training environment interactions' if use_env_steps else 'Agent decisions (startup excluded)'
        # Include all SD bands, with at least a 5% range for rare successes.
        upper = series.groupby(['algorithm', x_column]).success_rate.agg(['mean', 'std'])
        ceiling = (upper['mean'] + upper['std'].fillna(0)).max()
        bottom.set(xlabel=axis_label, ylabel='Checkpoint success rate',
                   ylim=(0, min(1., max(.05, float(ceiling)*1.15))))
        bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        figure.tight_layout()
        name = f'learning_curve_{condition}.png'
        figure.savefig(directory/name, dpi=160)
        plt.close(figure)
        written.append(f'{condition}: mean {label} and success rate. Sources: '
                       + ', '.join(sorted(series.run_id.unique())) + f'.\n\n![{condition}]({name})')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='one experiment directory written by part2_run.py')
    OUTPUT = find_results(parser.parse_args().results)
    # Shards run concurrently and each writes its own file; read whatever is there.
    shards = sorted(OUTPUT.glob('seed_results*.csv'))
    if not shards:
        raise SystemExit(f'No seed_results*.csv in {OUTPUT}. '
                         'Run part2_run.py first; no numbers are invented here.')
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    if df.run_id.duplicated().any():
        raise SystemExit('Duplicate run IDs across shards. Select a single experiment.')
    protocols = df.get('protocol_id', pd.Series('legacy', index=df.index)).fillna('legacy').unique()
    if len(protocols) != 1:
        raise SystemExit('This directory mixes experimental protocols.')
    legacy = protocols[0] == 'legacy'
    if legacy:
        df['agent_decisions'] = df['training_interactions']
        # Old training_seconds included curve evaluation. Only newer timing
        # schemas can supply comparable training and primary evaluation times.
        has_curve_time = df.get('curve_seconds', pd.Series(np.nan, index=df.index)).notna()
        df.loc[~has_curve_time, 'training_seconds'] = np.nan
    if 'evaluation_seconds' in df:
        df['final_eval_seconds'] = df.get('final_eval_seconds', df.evaluation_seconds).fillna(df.evaluation_seconds)
    for key, _, _ in EVIDENCE:
        if key not in df:
            df[key] = np.nan
    parts = ['# Part 2 tables',
             f'Source: {len(shards)} shard file(s), {len(df)} run summaries. '
             'Cells are mean ± sample SD across trained seeds. '
             'A dash indicates unavailable data or no episodes with that outcome.',
             f'Protocol: `{protocols[0]}`. Training rows: '
             f'{len(df[df.stage == "main"])}/30; transfer rows: '
             f'{len(df[df.stage == "transfer"])}/10 for the standard five-seed study.',
             '## Outcomes, Settings A/B/C/D',
             required_evidence(df[df.stage.isin(['main', 'transfer'])], EVIDENCE[:3] + EVIDENCE[4:6] + EVIDENCE[8:9]),
             '## Episode lengths and state coverage',
             required_evidence(df[df.stage.isin(['main', 'transfer'])], EVIDENCE[6:8] + EVIDENCE[3:4]),
             'Lengths are agent decisions, excluding startup. Conditional lengths omit '
             'seeds with no matching outcomes; B-C length differences compare different '
             'sets of successful/failed episodes, not identical trajectories. Unseen '
             'state coverage applies to Q only; PPO has no lookup table.',
             '## Computation',
             required_evidence(df[df.stage.isin(['main', 'transfer'])], EVIDENCE[9:]),
             '## Additional greedy PPO evaluation',
             greedy_evidence(df[df.stage == 'main']),
             'Primary PPO results use sampled actions. Greedy results are a separate '
             'evaluation mode and are not pooled with primary results.',
             '## Setting B versus Setting C',
             paired(df[df.stage == 'main'], 'B', 'C', 'Step penalty effect, C minus B',
                    'returns are omitted because B and C define the reward differently')]

    transfer = df[df.stage == 'transfer']
    parts.append('## Setting A versus Setting B')
    parts.append(paired(df[df.stage == 'main'], 'A', 'B',
                        'Uncertainty effect, B minus A',
                        'same reward, so score and return both compare; only the '
                        'sticky-action probability changes. Route choice is not '
                        'established by these numbers, it needs episode traces'))
    parts.append('## Setting D, transfer from Setting B')
    if transfer.empty:
        parts.append('No transfer evaluations on record.')
    else:
        combined = pd.concat([df[(df.stage == 'main') & (df.condition == 'B')], transfer])
        description = ('original experiment: changed game mode on the training maze'
                       if legacy else 'frozen B models on held-out maze 1, same game mode, '
                       'reward and stickiness. Maze 1 uses a modified Atari start state')
        parts.append(paired(combined, 'B', 'D', 'Transfer effect, D minus B', description))

    curve_files = sorted(OUTPUT.glob('curves*.csv'))
    parts.append('## Learning curves')
    if curve_files:
        names = plot_curves(pd.concat([pd.read_csv(p) for p in curve_files],
                                      ignore_index=True), OUTPUT)
        parts.append('Points are unsmoothed means across training seeds. Bands show '
                     'one sample standard deviation when at least two seeds are available.\n\n'
                     + '\n\n'.join(names))
    else:
        parts.append('No curves*.csv; run part2_run.py to produce them.')

    text = '\n\n'.join(parts) + '\n'
    (OUTPUT/'part2_tables.md').write_text(text, encoding='utf-8')
    print(text)
    print(f'Written to {OUTPUT/"part2_tables.md"}', file=sys.stderr)


if __name__ == '__main__':
    main()
