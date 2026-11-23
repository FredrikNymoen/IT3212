"""Verify complete frozen exports and refresh LaTeX/PNG only; no training/PDF."""
import argparse
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
REPORT=ROOT/'reports/assignment4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(values,digits=1,multiplier=1):
    values=np.asarray([float(v)*multiplier for v in values if v is not None and not pd.isna(v)])
    if not len(values):return '--'
    if len(values)==1:return f'${values[0]:.{digits}f}$'
    return f'${values.mean():.{digits}f} \\pm {values.std(ddof=1):.{digits}f}$'
def escape(value):return str(value).replace('_',r'\_')

def verify(output,training):
    lock=json.loads((output/'locked.json').read_text())
    interrupted=(output/'interrupted.json').exists()
    assert ((output/'completed.json').exists() or interrupted) and (training/'completed.json').exists()
    excluded=json.loads((output/'interrupted.json').read_text())['excluded_cases'] if interrupted else []
    assert not interrupted or excluded==[dict(condition='D',algorithm='Q-learning',seed=314)]
    for path,expected in lock['source_hashes'].items():assert sha(ROOT/path)==expected,path
    assert sha(training/'comparison.csv')==lock['development_export_sha256']
    rows=[];episodes=0;decisions=0
    for job in lock['jobs']:
        directory=Path(job['directory'])
        case={key:job[key] for key in ('condition','algorithm','seed')}
        if case in excluded:
            assert not (directory/'metrics.json').exists()
            diagnostic=json.loads((output/'map_guard_native_flags.json').read_text())
            assert diagnostic['native_score']==1760 and diagnostic['native_maze']==0
            assert not diagnostic['terminated'] and not diagnostic['game_over']
            continue
        metrics=json.loads((directory/'metrics.json').read_text())
        assert metrics['frozen_parameters_verified'] and metrics['native_score_verified_every_step']
        assert sha(ROOT/job['model'])==job['model_sha256']
        frame=pd.read_csv(directory/'evaluation.csv');assert len(frame)==200
        assert frame.evaluation_seed.tolist()==lock['evaluation_seeds']
        assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
        assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
        for column,key in [('success','success_rate'),('score','mean_score'),('task_return','mean_return')]:
            assert np.isclose(frame[column].mean(),metrics[key])
        assert np.allclose(frame.task_return,frame.score-(.1*frame.length if job['condition']=='C' else 0))
        with np.load(directory/'evaluation_traces.npz') as z:
            n=int(frame.length.sum());assert z['ram'].shape==(n,128)
            assert z['observation'].shape==z['next_observation'].shape==(n,61) and z['values'].shape==(n,5)
            assert np.array_equal(np.bincount(z['episode'],minlength=200),frame.length)
            assert np.allclose(np.bincount(z['episode'],weights=z['raw_reward'],minlength=200),frame.score)
            assert np.allclose(np.bincount(z['episode'],weights=z['reward'],minlength=200),frame.task_return,atol=1e-3)
            assert np.allclose(z['reward'],z['raw_reward']-(.1 if job['condition']=='C' else 0))
            assert ((z['action']>=0)&(z['action']<5)).all()
            bcd=lambda v:(v.astype(int)//16)*10+v.astype(int)%16
            raw=bcd(z['ram'][:,122])*10000+bcd(z['ram'][:,121])*100+bcd(z['ram'][:,120])
            assert np.array_equal(raw+z['raw_reward'],z['score_after'])
            chosen=z['values'][np.arange(n),z['action']]
            assert np.allclose(chosen,z['values'].max(axis=1))
            if job['algorithm']=='PPO':assert np.array_equal(z['action'],z['values'].argmax(axis=1))
        if job['condition']=='D':
            assert metrics['training_seconds']==0 and metrics['further_training_interactions']==0
            partner=next(x for x in lock['jobs'] if x['condition']=='B' and x['algorithm']==job['algorithm'] and x['seed']==job['seed'])
            assert job['model']==partner['model'] and job['model_sha256']==partner['model_sha256']
        rows.append({key:value for key,value in metrics.items() if key!='source_hashes'})
        episodes+=200;decisions+=n
    table=pd.DataFrame(rows)
    assert not table.duplicated(['condition','algorithm','seed']).any()
    nseeds=table.seed.nunique();assert len(table)==8*nseeds-len(excluded)
    table.to_csv(output/'verified_comparison.csv',index=False)
    verification=dict(trained_seeds=nseeds,conditions=len(table),episodes=episodes,native_decisions=decisions,
        source_and_model_hashes_verified=True,all_trace_shapes_rewards_scores_actions_verified=True,
        frozen_evaluation_verified=True,transfer_reuses_frozen_B=True,pdf_compiled=False,
        complete_A_B_C_and_PPO=True,complete_Q_D=not interrupted,excluded_cases=excluded)
    (output/'report_validation.json').write_text(json.dumps(verification,indent=2))
    return lock,table,verification

def render(args):
    output=ROOT/'results/assignment4'/args.final_run;training=ROOT/'results/assignment4'/args.training_run
    lock,table,verification=verify(output,training)
    protocol=json.loads((training/'protocol.json').read_text());jobs=protocol['jobs']
    budgets={job['budget'] for job in jobs};assert len(budgets)==1;budget=budgets.pop()
    seeds=sorted(table.seed.unique().tolist());nseeds=len(seeds)
    variants={algorithm:table.loc[table.algorithm==algorithm,'variant'].unique().tolist() for algorithm in ['Q-learning','PPO']}
    assert all(len(v)==1 for v in variants.values());qvariant=variants['Q-learning'][0];pvariant=variants['PPO'][0]
    outcomes=[];costs=[]
    for setting in 'ABCD':
        for algorithm in ['Q-learning','PPO']:
            part=table[(table.condition==setting)&(table.algorithm==algorithm)]
            if setting=='D' and algorithm=='Q-learning' and len(part)!=nseeds:
                assert len(part)==nseeds-1 and not verification['complete_Q_D']
                outcomes.append('D & Q-learning & -- & -- & --'+r' \\')
                costs.append('D & Q-learning & -- & -- & -- & --'+r' \\')
                continue
            assert len(part)==nseeds
            outcomes.append(f'{setting} & {algorithm} & '+ ' & '.join([fmt(part.success_rate,multiplier=100),fmt(part.mean_score),fmt(part.mean_return)])+r' \\')
            costs.append(f'{setting} & {algorithm} & '+ ' & '.join([fmt(part.success_length,0),fmt(part.failure_length,0),fmt(part.training_seconds),fmt(part.evaluation_seconds)])+r' \\')
    curves=pd.read_csv(training/'comparison.csv');fig,axes=plt.subplots(1,3,figsize=(11.8,3.4),sharex=True,sharey=True)
    for ax,setting in zip(axes,'ABC'):
        for algorithm,color in [('Q-learning','tab:blue'),('PPO','tab:orange')]:
            part=curves[(curves.setting==setting)&(curves.algorithm==algorithm)]
            group=part.groupby('interactions').success_rate.agg(['mean','std']).reset_index()
            x=group.interactions.to_numpy();mean=100*group['mean'].to_numpy();sd=100*group['std'].fillna(0).to_numpy()
            ax.plot(x,mean,'o-',color=color,label=algorithm,markersize=3)
            ax.fill_between(x,np.clip(mean-sd,0,100),np.clip(mean+sd,0,100),color=color,alpha=.16)
        ax.set_title(f'Setting {setting}');ax.set_xlabel('Training interactions');ax.grid(alpha=.2)
        ax.set_ylim(0,102);ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0].set_ylabel('Development success (%)')
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.tight_layout(rect=(0,0,1,.91))
    name='part2_learning_curves_20261122.png';image=REPORT/'Images'/name
    image.parent.mkdir(parents=True,exist_ok=True);fig.savefig(image,dpi=180)
    figdir=ROOT/'figures/assignment4/three_hour';figdir.mkdir(parents=True,exist_ok=True)
    (figdir/name).write_bytes(image.read_bytes());plt.close(fig)
    qdescription={
        'baseline':'the original compact food-relative key',
        'motion':'the original key plus a nearby relative-closing ghost bit',
        'gamma995':'the original compact food-relative key, with discount changed to 0.995',
        'gamma999':'the original compact food-relative key, with discount changed to 0.999',
        'bearing':'a coarse nearest-ghost bearing and distance code replacing the directional danger mask',
        'alignment':'signed food-relative corridor-offset bins replacing the alignment bit',
        'compact':'a smaller key retaining food/corridor/danger/power bins and a stopped/moving bit'}[qvariant]
    pgamma=.999 if pvariant=='long' else .995
    epochs=4 if pvariant=='careful' else 6;clip=.1 if pvariant=='careful' else .15;kl=.01 if pvariant=='careful' else .015
    rate='0.0002' if pvariant=='careful' else '0.0003'
    entropy='0.04 to 0.01' if pvariant=='explore' else '0.02 to 0.002'
    actor_inputs=43 if pvariant=='allghost' else 23;critic_inputs=61 if pvariant=='allghost' else 49
    actor_width=16 if pvariant=='small' else 32;critic_width=32 if pvariant=='small' else 64
    power_note=('PPO uses an experimental low-six-bit power proxy. It is not a verified precise timer.' if pvariant=='powerclock'
        else 'The inherited capped RAM-116 signal is a power proxy, not a verified precise remaining timer.')
    information_note=('The selected all-ghost PPO consumes these extra channels.' if pvariant=='allghost' else
        'The all-ghost candidate used the extra twelve channels; selected PPO retains the first 49 and Q coarsens that prefix, so those unused channels do not explain a gain.')
    checkpoint_range=lambda algorithm:f'{int(table.loc[table.algorithm==algorithm,"checkpoint_interactions"].min()):,}--{int(table.loc[table.algorithm==algorithm,"checkpoint_interactions"].max()):,}'
    points=sorted(curves.interactions.unique());curveepisodes=int(curves.episodes.unique()[0])
    guard_note=r'''One Q/D test (training seed 314, evaluation seed 1150094) hit the map guard
at decision 975: the native maze changed from 1 to 0 at 1,760 points, with
two lives remaining and no native termination. It is neither a verified
failure nor a completed 200-episode test. Its partial run is preserved.
We suppress the five-seed Q/D aggregate rather than silently dropping that
agent or changing the task after observing this event. The other 39 cells
contain 7,800 completed episodes. A complete Q transfer comparison requires
an explicitly revised boundary protocol and new evaluation; no model is
selected from this event. Independent native replay confirmed the guard.'''
    if verification['complete_Q_D']:guard_note=''
    text=r'''% Generated from verified three-hour-study exports; source records remain frozen.
\setcounter{section}{1}
\section{Compare learning settings}
\label{sec:learning-settings}

\subsection{Task, agents and controlled budget}
We use Gymnasium's native \texttt{ALE/MsPacman-v5}, RAM observations and five
actions: no-op, up, right, left and down. One interaction is one agent decision,
normally four emulator frames. Success means at least 2,000 newly earned game
points before all lives are lost or 1,000 decisions pass. The finite horizon
is a terminal task failure with time remaining observable. This target does
not establish level completion. The instructor-approved adaptation retains
Part~1's score-increment rewards in A/B; both agents scale rewards by 100
internally, while all reported scores and returns are raw points.

Both workflows use observed object locations, food/corridor paths, motion,
score and lives. The observation format has 61 values, adding all four ghost
offsets and activity flags to the earlier 49. Q coarsens these inputs; PPO
uses direction-relative continuous features. @@INFORMATION@@ No action mask, teacher policy,
look-ahead search or extra reward is used. Ordinary Q has one table and one
Bellman update per real transition, without Double Q, replay or eligibility
traces. Its selected variant uses @@QDESCRIPTION@@, alpha 0.15 and epsilon
decreasing from 1 to 0.05 over the first 80\% of the training cap.

PPO uses the \texttt{@@PVARIANT@@} package: a learned shared action scorer with
@@ACTOR@@ inputs per action and two @@ACTORWIDTH@@-unit tanh layers, plus a separate
@@CRITIC@@-input critic with two @@CRITICWIDTH@@-unit layers. It uses discount @@PGAMMA@@,
GAE 0.97, four 512-step environments, minibatches of 256, up to @@EPOCHS@@
epochs, clipping @@CLIP@@ and target KL @@KL@@. Learning rate decreases
linearly from @@RATE@@ to 0.00005, and entropy coefficient from @@ENTROPY@@.
@@POWERNOTE@@ Different representations and update packages make this a
workflow comparison, not an isolated algorithm effect.

Every A/B/C model consumes @@BUDGET@@ training interactions, with common
training seeds @@SEEDS@@. The cap is a bounded development choice compatible
with PPO's 2,048-transition rollout; it does not claim convergence. The new
matched search gives each algorithm seven A/B candidates at 262,144 decisions
per setting, seed 42, then repeats its top two candidates on seed 7. Each
checkpoint has 20 separate development episodes. Mean A/B success, then raw
score and earlier checkpoint, locks one variant per algorithm before main
training. The new search allowance is 4,718,592 training interactions per
algorithm. Earlier unequal exploration and inspected outcomes informed these
hypotheses; this is a further development revision, not an independent restart.

All main runs consume the full common cap. The identical stopping rule selects
one saved checkpoint per A/B/C trained model using only its source-condition
development success, then game score, then earlier point. Every training seed
is retained, including poor runs. Selected weights contain @@QSTOPS@@
transitions for Q and @@PSTOPS@@ for PPO, so checkpoint age must be distinguished
from the full training cost. D inherits the selected B checkpoint; no D or
final-evaluation outcome selects a variant, parameter or checkpoint.

\subsection{Settings and frozen evaluation}
A has deterministic transitions (sticky probability zero). B changes only
frame-level stickiness to 0.25. C retains B and subtracts 0.1 raw points per
decision. A/B/C use native layout 0. D freezes B agents and sets the ROM maze
selector to layout 1 before the first frame, with compatible feature/action
formats and zero additional training. Fixed geometry extracts features for the
current layout; learned weights/tables remain unchanged. Natural level changes
are checked to protect the holdout. The intervention can also alter
layout-dependent behaviour, so it is not a geometry-only causal experiment.

Each trained model has 200 final episodes per condition, with seeds
1150000--1150199 separate from training and development. Q acts greedily with
episode-seeded uniform ties; PPO uses its highest-probability action.
In A, deterministic dynamics, a fixed start and PPO's argmax policy can produce
identical evaluation trajectories despite different reset seeds; the 200
episodes are not 200 independent learned agents or necessarily distinct paths.
Native BCD score registers are checked at every decision. Model/source hashes,
parameter fingerprints and complete decision traces are verified. Final
outcomes never feed back into selection. New episode seeds do not make the
previously inspected D map a new independent domain.

@@GUARDNOTE@@

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrr@{}}
\toprule
Setting & Agent & Success (\%) & Game score & Return \\
\midrule
@@OUTCOMES@@
\bottomrule
\end{tabular}
\caption{Updated comparison with the unresolved Q/D row suppressed, source:
\texttt{@@FINALRUN@@}. Mean \(\pm\) sample SD across @@NSEEDS@@ independently
trained agents, 200 frozen episodes each per condition. @@INCOMPLETETABLE@@ The SD describes
training-seed variation, not a confidence interval. Failure rate is 100\%
minus success rate. Game score and return are points; C returns include the
step penalty, so compare B/C using common scores and success. D reuses B weights.}
\label{tab:part2-outcomes}
\end{table}

\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Setting & Agent & Success length & Failure length & Train (s) & Evaluate (s) \\
\midrule
@@COSTS@@
\bottomrule
\end{tabular}
\caption{Lengths are per-agent mean decisions conditional on outcome;
a dash means no such outcome or the unresolved Q/D row; SD is omitted if only one trained agent has
that outcome. Training costs cover the full @@BUDGET@@-interaction run,
excluding development evaluation and search. D adds zero training. Evaluation
covers 200 episodes, trace collection and native checks, excluding compression.
Same sources as Table~\ref{tab:part2-outcomes}. Shared-compute timings do not
establish an isolated speed advantage.}
\label{tab:part2-costs}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{Images/@@IMAGE@@}
\caption{Source: \texttt{@@TRAINRUN@@/comparison.csv}. Raw, unsmoothed learning
curves at @@POINTS@@ interactions. Each checkpoint uses @@CURVEEPISODES@@
development episodes per trained agent, with the same fixed development seeds.
Lines are mean success across @@NSEEDS@@ trained agents; shading is one sample
SD clipped to 0--100\%, not a confidence interval. Final episodes are excluded.
Final policies can use earlier development-selected checkpoints rather than
the end of these curves.}
\label{fig:part2-learning}
\end{figure}

\subsection{Reliability and interpretation limits}
Episode replication estimates performance of a fixed policy, not variability
between independently trained agents. Deterministic A can repeat trajectories;
its episode count does not add training replications. Conditional successful
lengths alone do not show safer or more efficient behaviour. RAM features omit
parts of the emulator state, and the power-time premise in Part~1 remains
limited by the proxy above. Source-only selection and frozen D testing support
conditional transfer evidence, not general reliability on unfamiliar maps.
Training and evaluation use the Intel Core Ultra 7 258V CPU, one numerical
thread per worker, on shared compute. Exact software versions, concurrency,
seeds and model hashes are retained with the experiment records. The required
five-seed mechanism comparisons and human review remain separate tasks.
'''
    substitutions=dict(QDESCRIPTION=qdescription,PVARIANT=escape(pvariant),ACTOR=actor_inputs,CRITIC=critic_inputs,
        ACTORWIDTH=actor_width,CRITICWIDTH=critic_width,
        PGAMMA=pgamma,EPOCHS=epochs,CLIP=clip,KL=kl,RATE=rate,ENTROPY=entropy,POWERNOTE=power_note,
        INFORMATION=information_note,GUARDNOTE=guard_note,
        INCOMPLETETABLE=('The unresolved Q/D row is suppressed.' if guard_note else ''),
        BUDGET=f'{budget:,}',SEEDS='/'.join(map(str,seeds)),QSTOPS=checkpoint_range('Q-learning'),PSTOPS=checkpoint_range('PPO'),
        OUTCOMES='\n'.join(outcomes),COSTS='\n'.join(costs),NSEEDS=nseeds,FINALRUN=escape(args.final_run),
        TRAINRUN=escape(args.training_run),IMAGE=name,POINTS=', '.join(f'{int(p):,}' for p in points),CURVEEPISODES=curveepisodes)
    for key,value in substitutions.items():text=text.replace(f'@@{key}@@',str(value))
    assert '@@' not in text
    target=REPORT/'Sections/02_learning_settings.tex';before=sha(target)
    archive=output/'previous_report';archive.mkdir(exist_ok=True)
    for name in ['Sections/02_learning_settings.tex','Sections/02_discussion.tex','README.md']:
        destination=archive/Path(name).name
        if not destination.exists():destination.write_bytes((REPORT/name).read_bytes())
    assert sha(target)==before,'Report changed during rendering; preserve and merge.'
    target.write_text(text,encoding='utf-8')
    (output/'rendered_learning_settings.tex').write_text(text,encoding='utf-8')
    (output/'report_renderer.py').write_bytes(Path(__file__).read_bytes())
    manifest={str(p.relative_to(ROOT)):sha(p) for p in [target,image,training/'comparison.csv',output/'verified_comparison.csv',Path(__file__)]}
    (output/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(verification,indent=2));print('Updated LaTeX and PNG only:',target,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',required=True);parser.add_argument('--final-run',required=True)
    render(parser.parse_args())
