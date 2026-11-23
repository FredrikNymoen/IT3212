"""Render current Part 2 LaTeX and a PNG learning curve from measured exports.

No model training/selection, report-PDF compilation or original-prediction edits.
"""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]


def fmt(values,digits=1,multiplier=1):
    values=np.asarray([v for v in values if v is not None and not pd.isna(v)],dtype=float)*multiplier
    if not len(values):return '--'
    if len(values)==1:return f'${values[0]:.{digits}f}$'
    return f'${values.mean():.{digits}f} \\pm {values.std(ddof=1):.{digits}f}$'


def collect(output):
    protocol=json.loads((output/'protocol.json').read_text())
    selected=json.loads((output/'selected.json').read_text())
    rows=[];curves=[];checks=[]
    for setting in 'ABC':
        for algorithm in ('Q-learning','PPO'):
            for seed in protocol['training_seeds']:
                directory=output/'final'/f'{setting}_{algorithm}_{seed}'
                result=json.loads((directory/'metrics.json').read_text())
                assert result['rate']==selected[algorithm] and result['budget']==protocol['main_budget']
                model=directory/f'checkpoint_262144{".pkl" if algorithm=="Q-learning" else ".zip"}'
                assert hashlib.sha256(model.read_bytes()).hexdigest()==result['model_sha256']
                for condition,name,metrics in [(setting,'evaluation',result)]+([('D','transfer',result['transfer'])] if setting=='B' else []):
                    frame=pd.read_csv(directory/f'{name}.csv')
                    assert len(frame)==200 and frame.evaluation_seed.tolist()==protocol['final_evaluation_seeds']
                    assert metrics['learning_disabled_verified']
                    assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
                    assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
                    assert np.isclose(frame.task_return.mean(),metrics['mean_return'])
                    assert np.isclose(frame.score.mean(),metrics['mean_score'])
                    assert np.isclose(frame.success.mean(),metrics['success_rate'])
                    assert np.isclose(1-frame.success.mean(),metrics['failure_rate'])
                    for success,key in [(True,'success_length'),(False,'failure_length')]:
                        lengths=frame.loc[frame.success.astype(bool)==success,'length']
                        if len(lengths):assert np.isclose(lengths.mean(),metrics[key])
                        else:assert metrics[key] is None
                    expected_return=frame.score-(.1*frame.length if condition=='C' else 0)
                    assert np.allclose(frame.task_return,expected_return,atol=1e-6)
                    if condition=='D':assert metrics['training_interactions']==0 and metrics['model_sha256']==result['model_sha256']
                    rows.append(dict(setting=condition,algorithm=algorithm,training_seed=seed,
                        rate=selected[algorithm],training_seconds=0. if condition=='D' else result['training_seconds'],
                        training_interactions=0 if condition=='D' else result['training_interactions'],
                        evaluation_seconds=metrics['evaluation_seconds'],success_rate=metrics['success_rate'],
                        failure_rate=metrics['failure_rate'],mean_score=metrics['mean_score'],mean_return=metrics['mean_return'],
                        success_length=metrics['success_length'],failure_length=metrics['failure_length'],
                        evaluation_interactions=metrics['evaluation_interactions'],model_sha256=result['model_sha256'],
                        reused_training=result['reused_training'],provenance=result['provenance']))
                    archive=np.load(directory/f'{name}_traces.npz')
                    arrays={key:archive[key] for key in ('ram','observation','next_observation','scores','action','episode','raw_reward','reward')}
                    assert arrays['ram'].shape==(int(frame.length.sum()),128)
                    assert arrays['observation'].shape==arrays['next_observation'].shape==(int(frame.length.sum()),45)
                    assert arrays['scores'].shape==(int(frame.length.sum()),5)
                    assert np.all((arrays['action']>=0)&(arrays['action']<5))
                    assert np.array_equal(np.bincount(arrays['episode'],minlength=200),frame.length.to_numpy())
                    assert np.allclose(np.bincount(arrays['episode'],weights=arrays['raw_reward'],minlength=200),frame.score)
                    assert np.allclose(np.bincount(arrays['episode'],weights=arrays['reward'],minlength=200),frame.task_return,atol=1e-3)
                    chosen=arrays['scores'][np.arange(len(arrays['action'])),arrays['action']]
                    assert np.allclose(chosen,arrays['scores'].max(axis=1))
                    checks.append(dict(condition=condition,algorithm=algorithm,seed=seed,episodes=200,
                        interactions=int(frame.length.sum()),trace_rows=len(arrays['action'])))
                    archive.close()
                    del arrays
                curve=pd.read_csv(directory/'curves.csv')
                assert curve.interactions.tolist()==protocol['curve_checkpoints']
                curve['setting']=setting;curve['algorithm']=algorithm;curve['training_seed']=seed
                curves.append(curve)
    for name,expected in protocol['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    table=pd.DataFrame(rows).sort_values(['setting','algorithm','training_seed'])
    table.to_csv(output/'comparison.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(output/'learning_curves.csv',index=False)
    (output/'validation.json').write_text(json.dumps(dict(
        trained_models=12,conditions=16,evaluation_episodes=3200,trace_checks=checks,
        frozen_model_and_source_hashes_verified=True,common_evaluation_seeds_verified=True,
        ordinary_q_only=True,transfer_uses_frozen_B_without_training=True),indent=2))
    return protocol,selected,table,pd.concat(curves,ignore_index=True)


def render(output):
    output=Path(output)
    protocol,selected,table,curves=collect(output)
    environment=json.loads((output/'environment.json').read_text())
    images=ROOT/'reports/assignment4/Images';images.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'figures/assignment4/current';figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.3),sharex=True,sharey=True)
    for ax,setting in zip(axes,'ABC'):
        for algorithm,color in [('Q-learning','tab:blue'),('PPO','tab:orange')]:
            subset=curves[(curves.setting==setting)&(curves.algorithm==algorithm)]
            grouped=subset.groupby('interactions').mean_score.agg(['mean','std']).reset_index()
            x=grouped.interactions.to_numpy();mean=grouped['mean'].to_numpy();sd=grouped['std'].to_numpy()
            ax.plot(x,mean,'o-',color=color,label=algorithm,markersize=3)
            ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.16)
        ax.set_title(f'Setting {setting}');ax.set_xlabel('Training interactions');ax.grid(alpha=.25)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0].set_ylabel('Development game score (points)');axes[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(images/'part2_learning_curves_v3.png',dpi=220)
    fig.savefig(figures/'part2_learning_curves_v3.png',dpi=220);plt.close(fig)
    outcome=[];cost=[]
    for setting in 'ABCD':
        for algorithm in ('Q-learning','PPO'):
            group=table[(table.setting==setting)&(table.algorithm==algorithm)]
            outcome.append(f'{setting} & {algorithm} & '+ ' & '.join([
                fmt(group.success_rate,multiplier=100),
                fmt(group.mean_score),fmt(group.mean_return)])+r' \\')
            cost.append(f'{setting} & {algorithm} & '+' & '.join([
                fmt(group.success_length,digits=0),fmt(group.failure_length,digits=0),
                fmt(group.training_seconds),fmt(group.evaluation_seconds)])+r' \\')
    # NaN conditional means are missing outcomes, not numerical measurements.
    outcome='\n'.join(outcome)
    cost='\n'.join(cost).replace('$nan \\pm nan$','--').replace('$nan$','--')
    cpu=environment['cpu'].replace('_',r'\_')
    source=output.name.replace('_',r'\_')
    reused=table[(table.setting=='B')&table.reused_training].shape[0]>0
    reuse_text=('Exactly matching B checkpoints were reused with source/model checks; their original '
        'training times are retained and they receive zero additional training here. '
        'All reported evaluations were rerun. ' if reused else '')
    template=r'''% Generated from verified current ordinary-Q/PPO exports.
% Source run: SOURCE. Do not replace with archived Double Q/pilot results.
\setcounter{section}{1}
\section{Compare learning settings}
\label{sec:learning-settings}

\subsection{Environment, agents and interaction budget}
We use Gymnasium's native Atari \texttt{ALE/MsPacman-v5}, with no-op and four
cardinal actions. One interaction is one agent decision, normally four emulator
frames. Success is at least 2,000 newly earned game points before all lives are
lost or 1,000 decisions pass. A timeout is a terminal task failure, with time
remaining observable. Reaching the score target does not establish level completion.
The instructor-approved adaptation retains Part~1's game-score rewards in A/B
instead of the brief's terminal-only success reward. Both methods divide rewards
by 100 internally; all reported scores and returns use raw game points.

Both receive the same corrected RAM-derived information: object locations,
food-path and corridor features, ghost proximity, motion and a power-related
signal (45 values). Ordinary Q-learning maps this to compact food-relative bins
and makes one update in one Q-table per new transition. It uses discount 0.99
and epsilon decreasing from 1 to 0.05 over 80\% of training. No Double Q,
replay or eligibility traces enter this comparison. PPO uses a learned shared
action scorer (18 direction-relative inputs per action, two 32-unit tanh layers)
and a separate value network (45 inputs, two 64-unit layers). All five actions
remain available to both agents. PPO settings are discount 0.995, GAE 0.95, four
512-step environments, minibatches of 256, up to four epochs, clipping 0.2,
entropy 0.01 and target KL 0.02. This compares the specified workflows,
including their different representations, rather than isolating algorithm alone.

Every A/B/C agent receives 262,144 training interactions, with training seeds
42 and 7 shared across methods. This bounded ceiling follows development
pilots that needed more than the initial 65,536 decisions; it is compatible
with PPO's 2,048-transition rollouts and is not a convergence claim.
The new matched search gives each method two 65,536-interaction candidates,
one separate training seed (31415), and 50 B-development episodes per candidate.
Q rates are 0.10/0.15 and PPO rates 0.0001/0.0003. Development success, then
mean game score, selects one common rate per algorithm: Q \(\alpha=QRATE\),
PPO learning rate \(PRATE\). These remain fixed across A/B/C.
Earlier exploratory search was unequal and informed this revision; its outcomes
and the rejected Double Q variants are excluded from the current tables.

\subsection{Settings A--D and evaluation}
A uses deterministic transitions (sticky probability 0). B changes only that
probability to 0.25 at emulator-frame level. C retains B and subtracts 0.1 game
points per decision. D evaluates the unchanged B models on native ROM maze
layout 1, with zero further training. A/B/C use layout 0; natural level changes
are checked to protect the holdout. The 45-value observation and five-action
formats and their scales remain compatible. Local, food-relative Q states and
PPO's shared action scoring offer transferable structure, while both still
omit parts of the emulator state. Feature extraction uses the new layout's
fixed geometry while the learned table/network stays unchanged. The ROM intervention can also alter
layout-dependent behaviour, so D is not a geometry-only causal test.

Each trained agent has 200 final episodes per evaluated condition, using
790000--790199, separate from training, tuning and curve seeds. Q acts greedily
with seeded uniform ties; PPO chooses its highest-probability action. Q entries
and neural parameters stay frozen. Model fingerprints, source hashes and
per-decision traces are checked. No final episodes select settings or checkpoints.

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrr@{}}
\toprule
Setting & Agent & Success (\%) & Game score & Return \\
\midrule
OUTCOMES
\bottomrule
\end{tabular}
\caption{Source: \texttt{SOURCE/final/}. Mean \(\pm\) sample SD across two
independently trained agents, each evaluated for 200 episodes per condition.
Failure rate is 100\% minus success rate. Game scores and returns are in points.
D reuses B weights. C returns include the step penalty; use success rates and
common game scores when comparing B/C.
The SD describes variation between trained agents, not a confidence interval.}
\label{tab:part2-outcomes}
\end{table}

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrrr@{}}
\toprule
Setting & Agent & Success length & Failure length & Train (s) & Evaluate (s) \\
\midrule
COSTS
\bottomrule
\end{tabular}
\caption{Mean \(\pm\) sample SD across two trained agents.
Lengths are per-agent mean decisions conditional on outcome; a dash means no such outcome.
SD is omitted when only one agent has the outcome. Each A/B/C model trains for
262,144 interactions; D adds zero. Evaluation time covers 200 episodes and
trace collection, excluding compression. Training time excludes curve evaluation.}
\label{tab:part2-costs}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{Images/part2_learning_curves_v3.png}
\caption{Source: \texttt{SOURCE/final/*/curves.csv}; A/B/C, Q-learning/PPO,
training seeds 42/7. Each checkpoint uses the same 20 development episodes
(780000--780019) at 0 and every 65,536 interactions. Lines are mean raw game
scores across trained agents; shading is one sample SD, clipped at zero.
No smoothing or final evaluation episodes are used.}
\label{fig:part2-learning}
\end{figure}

\subsection{Interpretation and limits}
INTERPRETATION

Two training seeds give limited evidence about training variability; the
evaluation episodes are not independent training replications. Route safety
cannot be inferred from mean score or conditional episode length alone.
Deterministic A can repeat identical trajectories without adding training replications.
The inherited RAM-116 feature is capped and is not a verified precise remaining
power-time estimate, limiting the corresponding Part~1 explanation.
Runtime was recorded on CPU, with one numerical thread per worker and up to
WORKERS concurrent workers, using CPU_NAME. REUSE
These are shared-compute measurements, not isolated inference benchmarks.
The five-seed mechanism study and individual-episode analysis remain separate work.
'''
    statements=[]
    for algorithm in ('Q-learning','PPO'):
        means=table[table.algorithm==algorithm].groupby('setting').mean(numeric_only=True)
        statements.append(f"For {algorithm}, mean success was {means.loc['A','success_rate']*100:.1f}\\% in A and "
            f"{means.loc['B','success_rate']*100:.1f}\\% in B. Adding the penalty gave {means.loc['C','success_rate']*100:.1f}\\% "
            f"in C, with mean raw game score {means.loc['C','mean_score']:.1f} points. Frozen B agents achieved "
            f"{means.loc['D','success_rate']*100:.1f}\\% success and {means.loc['D','mean_score']:.1f} points on D.")
    replacements=dict(SOURCE=source,QRATE=f"{selected['Q-learning']:.2f}",PRATE=f"{selected['PPO']:.4f}",
        OUTCOMES=outcome,COSTS=cost,INTERPRETATION='\n\n'.join(statements),
        WORKERS=str(environment['max_concurrent_workers']),CPU_NAME=cpu,REUSE=reuse_text)
    for key,value in replacements.items():template=template.replace(key,value)
    template='\n'.join(line.rstrip() for line in template.splitlines())+'\n'
    assert not re.search(r'\b(?:OUTCOMES|COSTS|INTERPRETATION|CPU_NAME|QRATE|PRATE)\b',template)
    assert template.count(r'\begin{table}')==template.count(r'\end{table}')==2
    assert template.count(r'\begin{figure}')==template.count(r'\end{figure}')==1
    target=ROOT/'reports/assignment4/Sections/02_learning_settings.tex'
    previous=output/'previous_report/02_learning_settings.tex'
    if target.exists() and target.read_bytes()!=previous.read_bytes():
        assert target.read_text(encoding='utf-8').startswith('% Generated from verified current ordinary-Q/PPO exports.'), 'Report was edited during the run; preserve and merge it before rendering.'
    target.write_text(template,encoding='utf-8')
    (output/'rendered_02_learning_settings.tex').write_text(template,encoding='utf-8')
    (output/'report_renderer.py').write_bytes(Path(__file__).read_bytes())
    manifest_paths=[output/'comparison.csv',output/'learning_curves.csv',target,
        images/'part2_learning_curves_v3.png',Path(__file__)]
    (output/'report_manifest.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_paths},indent=2))
    print('Updated verified ordinary-Q/PPO A--D report:',output.name,flush=True)
    print(table[['setting','algorithm','training_seed','success_rate','mean_score','mean_return']].to_string(index=False),flush=True)
    return table
