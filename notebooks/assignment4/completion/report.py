"""Build the evidence brief from verified records; never train/select a policy."""
import json
from pathlib import Path
import shutil
import re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
REPORT=ROOT/'reports/assignment4'; BASE=ROOT/'results/assignment4'
def read(p):return json.loads(Path(p).read_text())
def write(p,s):
    # Consistent spacing in editorial prose/captions; never called on original Part1.
    protected=[]
    def protect(m):protected.append(m[0]);return f'@@{len(protected)-1}@@'
    s=re.sub(r'\\(?:includegraphics(?:\[[^]]*\])?|input|label|ref)\{[^}]*\}',protect,s)
    s=re.sub(r'([A-Za-z]{2,})(?=\d|\.\d)',lambda m:m[0] if m[0] in {'TS','TF','SS','SF'} else m[0]+' ',s)
    s=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',s)
    s=re.sub(r"(?<=Q)(?=\d|\.\d)|(?<='s)(?=\d|\.\d)",' ',s)
    s=s.replace('to(','to (').replace('position(','position (')
    s=s.replace('reward-','reward -')
    s=re.sub(r'([A-Za-z%]),(?=\d)',r'\1, ',s)
    s=s.replace('IT 3212','IT3212').replace('258 V','258V').replace('Stable-Baselines 3','Stable-Baselines3').replace('a4 paper','a4paper')
    s=re.sub(r'@@(\d+)@@',lambda m:protected[int(m[1])],s)
    Path(p).write_text(s.strip()+'\n',encoding='utf-8')
def pm(mean,sd,digits=1,scale=1):
    return ('$'+f'{mean*scale:.{digits}f}'+(r' \pm '+f'{sd*scale:.{digits}f}' if sd is not None else '')+'$')
def table(header,rows,caption,label,spec):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{@{}'+spec+r'@{}}',r'\toprule',header+r' \\',r'\midrule',
        *[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}'])

P2=r"""
\setcounter{section}{1}
\section{Compare learning settings}
\subsection{Common task and controls}
We compare ordinary tabular Q-learning with PPO in Gymnasium's native
\texttt{ALE/MsPacman-v5}. Success means earning 2,000 new points before game
over or 1,000 decisions, not clearing a level. The observable horizon is
terminal. One interaction is one decision, normally four emulator frames,
summed across PPO environments. Our instructor-approved A/B adaptation uses
score increments rather than terminal-only rewards. C subtracts 0.1 points
per decision; both learners divide rewards by 100. Reported returns use points.

A disables sticky actions; B changes only frame-level stickiness to 0.25;
C adds the penalty to B. Following the clarification, D freezes B agents
and obstructs observations while retaining B's underlying game.
Both receive 61 RAM-derived object features; selected agents use the first
49. Q discretises a subset and omits time, score and lives; PPO retains
finer information. This compares learning workflows, including representations.
The power signal is a capped proxy, not the precise timer predicted originally;
the approximate geometry omits tunnel links. Table~\ref{tab:config} explains
the numerical choices and limitations.

Each A/B/C run consumes 262,144 interactions: 128 complete PPO rollouts,
chosen for bounded resources rather than convergence. Both methods use roots
17/29/101/314/2718, fixed arbitrary identifiers, not favourable seeds.
Shared seeds do not ensure shared experience. Matched new development search
allows seven packages per method on A/B at seed 42, repeating the best two
at seed 7: 4,718,592 interactions per method. Earlier unequal exploration
informed candidates; this is not equal cumulative tuning.
Source-development success, then score, then earlier checkpoint selects
policies; selected weights contain 131,072--262,144 interactions, but costs
charge the full run. Final/D outcomes never select configurations or checkpoints.

Unsmoothed development curves use ten episodes at zero and every 65,536
interactions. Search development uses separate twenty-episode sets.
Final A/B/C seeds 1150000--1150199 supply 200 frozen episodes per trained
agent. D uses paired clean/obstructed seeds 1600000--1600199. Greedy Q uses
seeded uniform ties; PPO takes its highest-probability action. Fingerprints
verify no updates; native score registers verify earned points. Results
average five trained agents, not pooled episodes as independent replications.
Deterministic A can repeat trajectories; episode count does not replace retraining.

\subsection{A: Learning under predictable dynamics}
Does predictable movement permit efficient learning? At 65,536 interactions,
PPO reaches20\% development success versus Q's2\%, contradicting a general
early Q advantage. Final success is Q32.9\%, PPO80\%, but PPO comprises four
perfect agents and one zero-success agent. Its larger across-root SD also
contradicts our predicted lower PPO variability here. Determinism makes
trajectory repetition possible, not retraining reliable. The learning curve
therefore needs the individual-root outcomes, not only its average.

\subsection{B: Learning under transition uncertainty}
Does stickiness change performance? PPO leads Q by35.2 percentage points,
in four of five roots, but still fails32.6\% of episodes. Early development
instead gives Q6\% versus PPO0\%. Sticky transitions reduce PPO success by12.6 points;
Q's mean change of $-0.7$ hides paired changes from $-71$ to $+48$.
Q varies more across roots in B, supporting that prediction only conditionally.
These outcomes neither establish safer routes nor support the predicted
larger Q loss; action timing and learned trajectories both matter.

\subsection{C: Learning under changed incentives}
Does a speed incentive preserve goal attainment?
C reduces mean success by 5.3 points for Q and 14.7 for PPO, although two
roots improve per method. Successful episodes shorten, but condition on
different successes. With failure avoidance prioritised, this penalty is
unsupported; one magnitude cannot test the predicted dose dependence.
Raw C scores are Q $1534.8\pm201.0$, PPO $1873.4\pm283.5$;
penalised returns cannot be compared directly with B.

\subsection{D: Robustness without further learning}
What survives missing observations? D fixes a32-by-32 playfield region,
$x\in[16,48),y\in[24,56)$, before outcomes:3.05\% of the frame, avoiding
a score-only change while bounding the intervention.
Because policies consume objects rather than pixels, centres inside it are
hidden before recomputing object features; an obscured player retains its
last visible pose. Static geometry and nonspatial status remain available.
Engineering verified1,600 unchanged native transitions and hidden-object
noninterference. This is an object-feature analogue, not a pixel-CNN test.

The same ten B models receive200 paired clean/obstructed episodes each;
weights, native dynamics, rewards and action formats remain fixed.
Clean success is Q29.5\%, PPO65.9\%; obstruction gives23.2\% and48.5\%.
Table~\ref{tab:robustness-pairs} isolates these within-model changes; comparing
against the separate B evaluation would mix observation and seed changes.
PPO retains higher absolute success, while Q loses fewer percentage points.
Neither ranking alone establishes reliability. Q root2718 falls49.5\% to19\%,
whereas PPO root101 improves64.5\% to73\%: the mean conceals opposite responses.
Information removal can redirect trajectories beneficially in particular
policies; these improvements do not establish that less information helps
generally. Changed visited states compete with simple loss-of-information
explanations. Relative feature formats remain usable, but cannot recover
hidden objects. This tests frozen observation robustness, not unseen-map
generalisation; additional locations/durations need independent confirmation.
"""

CONFIG=r"""
\begin{table}[htbp]
\centering\small
\begin{tabularx}{\linewidth}{@{}p{.22\linewidth}XX@{}}
\toprule Choice & Configuration & Reason and scope \\
\midrule
Task/actions & 2,000 points; 1,000 decisions; NOOP/up/right/left/down & Score objective from predictions; cap bounds cost. All five controls remain available; corridor indicators are not action masks. \\
Native dynamics/reward & Four-frame repeat; sticky probability A=0, B/C/D=.25; C cost .1; rewards /100 & Native repeat/sticky defaults; isolate A/B uncertainty. Maximum cost 100 points is 5\% of target; scaling moderates targets equally. \\
Q updates & One table, $\alpha=.15$, $\gamma=.995$; epsilon $1\to.05$ over 80\% cap & One Bellman update per actual transition, no terminal bootstrap, replay or Double Q. Relative bins reuse experience but alias states; moderate alpha smooths targets, epsilon retains coverage. \\
PPO architecture & Shared action scorer: 23 inputs/action, two 32-unit tanh layers; critic: 49 inputs, two 64-unit layers & Directional parameter sharing limits model size; independent critic estimates returns. Richer representation confounds an algorithm-only comparison. \\
PPO optimisation & Four $512$-step streams; batch256; at most6 epochs; clip .15; target KL .015; GAE .97; $\gamma=.995$ & 2,048 transitions/rollout, eight minibatches/epoch. GAE trades variance/bootstrapping; clip, KL stop and bounded reuse restrain update drift. \\
PPO schedules & Linear LR $.0003\to.00005$, entropy $.02\to.002$ & Reduce update size and training diversity late; not asserted optimal. Part3 tests alternatives. \\
Discount & $.995$ in both workflows & Rough discounted horizon $1/(1-\gamma)=200$ decisions retains delayed credit; task horizon remains1,000. \\
Monitoring/replication & Ten main-development episodes every32 rollouts; 200 final episodes; five training roots & Bound monitoring cost; meet final-test requirement and expose retraining variability. Numeric seed ranges label disjoint phases, not statistical quality. \\
\bottomrule
\end{tabularx}
\caption{Learning-settings study configurations, held fixed across A/B/C; D applies the same B policies to clean and obstructed observations. Constants are justified design choices, not independent optima. Main development seeds1130000--1130009; search seeds1110000--1110019 and1120000--1120019.}
\label{tab:config}
\end{table}
"""

P3=r"""
\setcounter{section}{2}
\section{Investigate the mechanisms}
We trained both baselines and every one-factor alternative afresh on the
same host/software as Part2: nine configurations, five roots each,
11,796,480 interactions. Native B, representation, actions, rewards,
budget, remaining hyperparameters and final-checkpoint rule stay fixed.
No arm is discarded. Development uses seeds1480000--1480009; primary
evaluation uses 200 frozen episodes/model, seeds1500000--1500199.
Table~\ref{tab:mechanisms} gives all changes and outcomes.

At matching full-budget checkpoints all ten baseline models are
bitwise identical to their original counterparts. Part2 selected earlier
weights by development; Part3 fixes the final checkpoint and uses new
episodes. PPO's 37.6\% versus Part2's 67.4\% therefore does not identify a
hardware/library effect or a directly controlled algorithm difference.

Faster epsilon decay reduces Q success by 18.6 points on average,
consistent with inadequate coverage, but two roots improve. Lower alpha
loses 15.9 points; updates may adapt too slowly within this cap.
Gamma .99 changes the rough credit horizon from 200 to100 decisions;
effects are mixed (Q $+2.4$, PPO $-4.4$ points), so no uniform shorter-horizon
advantage emerges. Lower PPO learning rate and two rather than six epochs
lose 28.7 and31.8 points, respectively, in all five roots. This supports
adequate update progress/data reuse here, not unrestricted reuse or an optimal
epoch count; KL stopping also limits actual optimisation.

Zero entropy improves argmax success by27.6 points in four roots. A competing
explanation to ``training diversity hinders useful learning'' is that argmax
execution happens to favour the sharper learned policy. Root2718 contradicts
a simple benefit: success falls67.5\% to38.5\%.
We locked all ten policies and tested argmax versus sampled execution on
50 fresh paired episodes/model/mode, seeds1700000--1700049, without learning.
The zero-entropy advantage is25.6 points under argmax but only0.8 under
sampling (Table~\ref{tab:diagnostic}). This supports execution dependence over
a general improvement in policy distributions. Sampling changes visited
states too; the diagnostic does not isolate a unique internal cause.
It remains a separately labelled secondary mode, not a replacement primary test.

Original predictions are retained with separate annotations. Training-return
ranks correlate positively with evaluation ranks within all six A/B/C-method
groups, but the training leader is an evaluation leader in only two;
exploratory training return is no deployment guarantee. Same-state probes
also fail to establish the three predicted action differences; the frightened-
ghost premise cannot be verified with our power proxy.
"""

P4=r"""
\setcounter{section}{3}
\section{Inspect what aggregate scores hide}
We retain the original requirement of ten episodes: three typical successes,
three typical failures, two surprising successes and two surprising failures.
Within predeclared B/C-method strata, typical cases minimise distance to
conditional median length, then score. Surprises use the model with the
rarest nonzero relevant outcome, then its conditional median; duplicates
are excluded. Table~\ref{tab:episodes} identifies all cases and rarity.
This is outcome-conditioned analysis, not an unbiased prevalence estimate
or a claim about A/D.

Each episode has three trace-driven decisions: largest score increment,
controllable approach before first life loss, and small value gap near a
ghost. The decision figures provide state, all five actual Q-values/probabilities,
observed movement/reward and a tested alternative. Each alternative replays
the entire seeded action prefix in native Gymnasium before one different
input; all prefix RAM/rewards agree. One coupled step cannot identify
expected survival or the eventual alternative outcome.

TF3/301 misses an immediately available10 points; TS3/401 gains800 and
reaches the goal. Conversely, SF1/326 gains200 in an eventual failure.
TS1/318 has an absent Q key and five zero estimates: its eventual success
does not validate an informed choice. SF1/625 approaches a ghost with
probability.836, but a later collision alone cannot distinguish a poor
expected-risk choice from an unlucky continuation. Sticky inputs and corridor
timing can make different controls realise identical movement.
Success percentages hide these value gaps, coverage holes and productive
actions inside failures; neither a high action probability nor a favourable
single fork is a calibrated success probability.
"""

P6=r"""
\setcounter{section}{5}
\section{Make three recommendations}
\paragraph{Rapid classroom demonstration.}
Use a frozen, source-development-selected B PPO policy with the baseline
configuration; display probabilities and native transitions. Two supports
are its67.4\% five-root B success and A's20\% development success at65,536
interactions versus Q's2\%. The weakness is retraining variation: A includes
a zero-success PPO model, and full-budget B baselines average37.6\%.
If live training from scratch is mandatory, promise no successful short
demonstration. Reverse preference if matched short-budget Q replications
deliver higher success with lower wall cost. Before use beyond simulation,
test repeated fixed-time starts and independent perception errors with a
human abort mechanism.

\paragraph{Repeated operation in uncertainty.}
Recommend neither tested workflow for consequential autonomous use; for
further simulated development choose baseline PPO without the C penalty.
PPO's B failure rate32.6\% is lower than Q's67.8\%, while C lowers PPO
success by14.7 points rather than improving the prioritised outcome.
The weakness is substantial failure and seed dependence despite the mean
advantage. Change this decision only after a prespecified acceptable
failure bound holds across fresh retrainings and operating conditions.
Before leaving simulation, run stress tests with controlled hazard rates,
matched perception, human intervention and reported failure severity;
episode lengths alone do not establish safety.

\paragraph{Operation after an environment change.}
Recommend neither for reliable unfamiliar operation. Under fixed observation
obstruction PPO retains48.5\% success versus Q's23.2\%, but loses17.4 points
against paired clean observations, versus Q's6.3-point loss. These are two
supports for caution: higher retained success and greater absolute degradation
answer different questions. The observed weakness is incomplete robustness
even to one small feature obstruction; static geometry remains available.
Reverse the decision only if locked policies satisfy the required reliability
threshold on multiple independently chosen shifts. Before deployment, test
unseen obstruction locations, missing-object durations and genuine pixel
perception on a protected test platform. Directional sharing can preserve
feature meaning; these results do not establish unseen-map transfer or
universal algorithm superiority.
"""

def main():
    summary=read(BASE/'completion_analysis_v1/summary.json'); agents=pd.read_csv(BASE/'completion_analysis_v1/agents.csv')
    groups=summary['groups']
    group=lambda study,condition,arm:next(x for x in groups if x['study']==study and x['condition']==condition and x['arm']==arm)
    archive=BASE/'completion_report_before';archive.mkdir(exist_ok=True)
    for name in ['main.tex','ai_assistance.tex','README.md','Sections/02_learning_settings.tex','Sections/03_mechanisms.tex','Sections/04_episode_inspection.tex','Sections/05_human_review.tex','Sections/06_recommendations.tex']:
        target=archive/name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPORT/name,target)
    outcomes=[];costs=[]
    for setting in 'ABCD':
        for alg in ['Q-learning','PPO']:
            row=group('D' if setting=='D' else 'P2',setting,alg)
            outcomes.append([setting,alg,pm(row['success_rate'],row['success_rate_sd'],scale=100),pm(row['mean_return'],row['mean_return_sd'])])
            costs.append([setting,alg,pm(row['success_length'],row['success_length_sd'],0),pm(row['failure_length'],row['failure_length_sd'],0),pm(row['training_seconds'],row['training_seconds_sd']),pm(row['evaluation_seconds'],row['evaluation_seconds_sd'])])
    p2=P2+table(r'Setting & Agent & Success (\%) & Return',outcomes,
        r'Learning-settings study: A/B/C learning evaluation and D frozen observation-robustness evaluation. Mean $\pm$ sample SD across five trained agents,200 episodes each. Failure is exactly100\% minus success; there is no censoring. D uses identical source-selected B models. Returns are points; C includes the penalty. SD is training variation, not a confidence interval.',
        'tab:outcomes','llrr')
    p2+=table('Setting & Agent & Success length & Failure length & Train (s) & Eval. (s)',costs,
        r'Same studies/models as Table~\ref{tab:outcomes}. Lengths are conditional mean decisions per agent, then mean $\pm$ sample SD across agents with that outcome. A/PPO has only one failing agent. Training charges the full262,144 interactions, excluding tuning/development. D adds zero training. Evaluation covers200 episodes, native checks and trace collection; loading/compression excluded. Known user-suspension time is removed from derived D wall cost; raw timings remain recorded. Shared-host timings do not establish isolated computational speed.',
        'tab:costs','llrrrr')
    p2+=CONFIG
    p2+=r'''
\par\noindent\begin{minipage}{\linewidth}
Runs used an Intel Core Ultra7 258V CPU, Windows11, one numerical thread per
worker (up to16 during historical phase overlap). Part3 used up to6 workers;
D one. Equal interactions do not imply equal compute.
Software was Python3.13.7, Gymnasium1.4.0, ALE0.12.1, PyTorch2.14.1+cpu
and Stable-Baselines3 2.9.2.
\end{minipage}\par
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{Images/part2_all_settings.png}
\caption{Learning-settings study, configurations in Table~\ref{tab:config}, roots17/29/101/314/2718. A/B/C: unsmoothed ten-episode development checks at zero and every65,536 interactions; shading is one sample SD across agents, clipped to0--100\%. Final tests excluded; Table~\ref{tab:outcomes} uses development-selected weights. D: the same frozen B weights under paired clean/obstructed observations,200 final episodes/model/mode. Thin lines show individual agents; thick lines show across-root means. D has zero additional training, so its horizontal axis shows observation condition rather than interactions.}
\end{figure}
'''
    rows=[]
    for seed in [17,29,101,314,2718]:
        values=[str(seed)]
        for setting in 'ABCD':
            for alg in ['Q-learning','PPO']:
                r=agents[(agents.study.eq('D' if setting=='D' else 'P2'))&agents.condition.eq(setting)&agents.arm.eq(alg)&agents.seed.eq(seed)].iloc[0]
                values.append(f'{100*r.success_rate:g}')
        rows.append(values)
    p2+=table(r'Root & A/Q & A/PPO & B/Q & B/PPO & C/Q & C/PPO & D/Q & D/PPO',rows,
        'Every training root in the main A/B/C and clarified D studies; success percentages,200 episodes/cell. D uses exactly the B checkpoint in that row. No root is omitted.',
        'tab:roots','rrrrrrrrr')
    rows=[]
    d=agents[agents.study.eq('D')]
    for alg in ['Q-learning','PPO']:
        for seed in [17,29,101,314,2718]:
            selected=d[d.algorithm.eq(alg)&d.seed.eq(seed)].set_index('condition')
            clean=pd.read_csv(ROOT/selected.loc['D_clean','evaluation_csv'])
            masked=pd.read_csv(ROOT/selected.loc['D','evaluation_csv'])
            assert clean.evaluation_seed.tolist()==masked.evaluation_seed.tolist()==list(range(1600000,1600200))
            before=float(selected.loc['D_clean','success_rate']*100);after=float(selected.loc['D','success_rate']*100)
            lost=int((clean.success.eq(1)&masked.success.eq(0)).sum());gained=int((clean.success.eq(0)&masked.success.eq(1)).sum())
            assert np.isclose(after-before,(gained-lost)/2)
            rows.append([alg,str(seed),f'{before:g}',f'{after:g}',f'{after-before:+.1f}',str(lost),str(gained)])
    p2+=table(r'Agent & Root & Clean S\% & Obstructed S\% & $\Delta$ (pp) & $S\to F$ & $F\to S$',rows,
        r'Frozen observation-robustness study D, paired seeds1600000--1600199,200 episodes per model/mode. $\Delta$ is obstructed minus clean success. Final columns count episodes that change from success to failure and vice versa on the same seed; policies can visit different trajectories after inputs change. This is a within-model observation contrast, not a comparison of newly trained agents or an independent unseen-map test.',
        'tab:robustness-pairs','lrrrrrr')
    shutil.copy2(BASE/'completion_figures/part2_all_settings.png',REPORT/'Images/part2_all_settings.png')
    write(REPORT/'Sections/02_learning_settings.tex',p2)

    labels={'Q_base':'Q baseline','Q_fast_epsilon':r'Q epsilon decay80\%$\to$20\%', 'Q_gamma99':r'Q $\gamma=.99$', 'Q_alpha05':r'Q $\alpha=.05$',
            'PPO_base':'PPO baseline','PPO_gamma99':r'PPO $\gamma=.99$', 'PPO_entropy0':'PPO zero entropy',
            'PPO_lr1e4':r'PPO LR $.0001\to.00005$', 'PPO_epochs2':'PPO two epochs'}
    rows=[]
    for arm,label in labels.items():
        row=group('M3','B',arm); per=agents[agents.study.eq('M3')&agents.arm.eq(arm)].set_index('seed')
        rows.append([label,pm(row['success_rate'],row['success_rate_sd'],scale=100),', '.join(f'{per.loc[s,"success_rate"]*100:g}' for s in [17,29,101,314,2718]),pm(row['training_seconds'],row['training_seconds_sd'])])
    p3=P3+table(r'Configuration & Success (\%) & Five roots (\%) & Train (s)',rows,
        r'Five-seed mechanism study, B, fixed262,144-step weights;200 frozen final episodes/model. Root order17/29/101/314/2718. All other settings match Table~\ref{tab:config}. Epsilon endpoints stay1/.05; only decay duration changes. Zero entropy changes the single entropy-schedule factor. Training wall times exclude development and the logged user pause; they are concurrent-host costs, not isolated CPU benchmarks.',
        'tab:mechanisms','lrlr')
    d=pd.DataFrame(summary['diagnostic']);rows=[]
    for arm in ['PPO_base','PPO_entropy0']:
        for mode in ['argmax','sampled']:
            x=d[d.arm.eq(arm)&d['mode'].eq(mode)]
            rows.append([labels[arm],mode,pm(x.success_rate.mean(),x.success_rate.std(ddof=1),scale=100),f'{x.mean_action_entropy.mean():.3f}'])
    p3+=table(r'Training configuration & Execution & Success (\%) & Action entropy',rows,
        r'Frozen execution diagnostic: all five fixed-full-budget baseline and zero-entropy PPO agents,50 fresh paired episodes/model/mode,1,000 total. Success is mean $\pm$ sample SD across five models; entropy is mean visited-state action-distribution entropy in nats. This secondary mode has50, not200 episodes/model and supplies no policy selection.',
        'tab:diagnostic','llrr')
    shutil.copy2(BASE/'completion_figures/part3_mechanism_curves.png',REPORT/'Images/part3_mechanism_curves.png')
    p3+=r'''
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{Images/part3_mechanism_curves.png}
\caption{Mechanism study; B, nine configurations in Table~\ref{tab:mechanisms}, roots17/29/101/314/2718. Each panel compares one factor to its own algorithm baseline. Unsmoothed ten-episode development checks at zero and every65,536 interactions; mean $\pm$ sample SD across five training roots, clipped to0--100\%. These development curves do not select Part3 checkpoints; primary results use the fixed final cap.}
\end{figure}
'''
    write(REPORT/'Sections/03_mechanisms.tex',p3)

    inspection=read(BASE/'completion_inspection_v4/inspection.json'); comments={x['id']:x for x in read(BASE/'completion_inspection_v4/annotations.json')['annotations']}
    rows=[]
    for e in inspection['episodes']:
        c=e['case'];j=c['job']
        rows.append([c['id'],c['category'].replace('surprising','rare'),('Q' if j['algorithm']=='Q-learning' else 'PPO')+'/'+j['condition'],str(j['seed']),str(c['evaluation_seed']),str(c['length']),f'{c["score"]:g}',f'{100*c["model_success_rate"]:g}'])
    p4=P4+table(r'ID & Category & Agent/setting & Root & Eval. seed & Length & Score & Model S\%',rows,
        r'Learning-settings study recorded B/C episodes, development-selected configurations in Table~\ref{tab:config}. TS/TF denote typical success/failure; SS/SF denote rare success/failure. Model S\% uses all200 recorded episodes of that trained agent, making surprises relative to that model. Conditional medians determine typicality, not visual appeal.',
        'tab:episodes','lllrrrrr')
    panels=read(BASE/'completion_figures/episode_figures.json')
    assert panels['decisions']==30
    p4+=r'\FloatBarrier'+'\n'
    for index,figure in enumerate(panels['figures']):
        path=ROOT/figure['pdf'];shutil.copy2(path,REPORT/'Images'/path.name)
        caption='Recorded decisions for '+ ' and '.join(figure['episodes'])+r', learning-settings study sources in Table~\ref{tab:episodes}. '
        if index==0:
            caption+=('All five controls N/U/R/L/D (NOOP/up/right/left/down) are available. Blue marks the chosen action; orange the tested alternative. '
                'Each image is a64-by-64 native pre-action crop. Coordinates use the full frame, positive $y$ downward; ghost/food offsets are relative. '
                '$p$ is the imperfect power proxy. Q-values use scaled training reward; PPO values are probabilities. '
                'Bars scale within each decision; numerical labels are rounded to four decimals, with full precision retained in experiment records. '
                'Gain is native score increment; task includes the C penalty. ')
        else:caption+=r'Conventions as Figure~\ref{fig:episode-pairs-1}. '
        caption+=' '.join(r'\textbf{'+point['id']+':} '+point['interpretation'] for point in figure['decisions'])
        p4+='\n'+r'\begin{figure}[p]\centering'+'\n'+r'\includegraphics[width=\linewidth]{Images/'+path.name+'}\n'+r'\caption{'+caption+'}\n'+r'\label{fig:episode-pairs-'+str(index+1)+'}\n'+r'\end{figure}'+'\n'
    write(REPORT/'Sections/04_episode_inspection.tex',p4)
    write(REPORT/'Sections/06_recommendations.tex',P6)

    preamble=r'''
%TC:envir table 0 ignore
%TC:envir longtable 0 ignore
%TC:envir figure 0 ignore
\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[english]{babel}\usepackage[a4paper,margin=23mm]{geometry}
\usepackage{microtype,graphicx,booktabs,tabularx,longtable,amsmath,amssymb}
\usepackage[section]{placeins}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\renewcommand{\thesection}{Part \arabic{section}}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}}
'''
    write(REPORT/'main.tex',preamble+r'''
\title{IT3212 Assignment 4: Evidence Brief\\MsPacMan}
\author{MsPacMan team}\date{}
\begin{document}\maketitle
\input{Sections/02_learning_settings.tex}
\input{Sections/03_mechanisms.tex}
\input{Sections/04_episode_inspection.tex}
\input{Sections/06_recommendations.tex}
\end{document}
''')
    write(REPORT/'predictions.tex',preamble+r'''
\title{IT3212 Assignment 4: Predictions and Later Annotations}
\author{MsPacMan team}\date{}
\begin{document}\maketitle
\noindent The original sheet below is preserved unchanged. Its original submission
timestamp must be retained with the team's submitted record; this document
does not reconstruct or certify that timestamp.
\input{Sections/01_predictions.tex}
\clearpage\input{Sections/01_prediction_annotations.tex}
\end{document}
''')
    annotations=r'''
\section*{Post-experiment annotations}
These annotations follow the results and do not replace prior predictions.
The 2,000-point target is a task threshold, not verified level completion.
The actual power feature is an imperfect capped proxy, not a precise timer.

\begin{enumerate}
\item \textbf{Early improvement: mixed, general Q advantage unsupported.}
At65,536 interactions A development success is Q2\%, PPO20\%; B is Q6\%, PPO0\%.
These are ten-episode checks per model, five training roots, not evidence of
convergence or a universally faster algorithm.
\item \textbf{Seed variation: condition-dependent.} Main evaluation SD in B is
Q21.3 versus PPO18.7 percentage points; C13.9 versus32.1, and A39.2 versus44.7.
Thus the predicted larger Q variation holds in B only. PPO A includes one0\%
agent and four100\% agents; an80\% mean is not stable retraining.
\item \textbf{Stochastic routes: not demonstrated.} A-to-B mean success changes
by$-.7$ points for Q and$-12.6$ for PPO, contradicting the predicted larger
Q decline at the mean level. Q's paired changes span$-71$ to$+48$.
Outcome changes do not establish that agents chose safer routes. Recorded
local approaches and delayed losses do not measure route-level expected risk.
\item \textbf{Step penalty: proposed benefit not established.} C lowers mean
success5.3 points for Q and14.7 for PPO. Conditional successful lengths shorten
655 to625 and578 to533 decisions, respectively, but different episodes
qualify as successes. Two roots improve per method. Only one penalty magnitude
was tested; the predicted dose-response remains untested.
\item \textbf{Training/evaluation ranking: association, not a guarantee.}
Using the last20 completed training episodes before each selected checkpoint,
within-condition five-root Spearman correlations are A Q.70/PPO.71,
B Q.90/PPO.21, C Q.10/PPO.70. The highest-training-return model is also an
evaluation leader only for PPO A/C (2/6 groups). In B, Q's training leader
succeeds33.5\% against the evaluation leader66.5\%; PPO64.5\% against92.5\%.
Training is exploratory/nonstationary and five ranks are descriptive, not a
reliable predictive model. Never compare penalised C returns directly with B.
\item \textbf{Initial PPO preference: supported for B simulation, qualified elsewhere.}
PPO B success67.4\% exceeds Q32.2\%, but failure32.6\% and substantial
training variation rule out reliable consequential deployment. Clarified
D is observation robustness, not unseen-maze transfer: PPO48.5\%, Q23.2\%
under obstruction. Lower Q degradation does not imply higher retained success.
\end{enumerate}

\subsection*{The three state-action predictions}
We selected the first observable matching state in fixed recorded traversal
order, then queried all five original frozen B agents on the identical
61-value observation. Selection rules locked before these policy queries.
These are observable analogues, not proof that every original premise holds.
Numbers below count an exclusive maximum matching the predicted action;
ties are not counted as a confirmed preference.
\begin{center}\begin{tabular}{lrr}\toprule
State & Q prediction matched & PPO prediction matched\\\midrule
S1: right/up &2/5&0/5\\
S2: up/right &0/5&0/5\\
S3: right/left &1/5&0/5\\\bottomrule
\end{tabular}\end{center}
The proposed systematic action split is not supported in these probes.
S2 requires enough remaining frightened time to pass safely, which our
proxy cannot establish; it is therefore not a clean falsification of that
conditional hypothesis. A single S1/S3 analogue cannot determine prevalence
over all matching states. Full action vectors, state observations, native
frames and model fingerprints accompany the experiment records.

\subsection*{Mechanism findings that changed our interpretation}
Five-root one-factor tests show shorter Q exploration loses18.6 points on
average but improves two roots. Zero PPO entropy gains27.6 points under
primary argmax execution but loses29 points at root2718. A frozen paired
execution diagnostic reduces its mean advantage from25.6 points under
argmax to.8 under sampled actions. The apparent entropy benefit depends
strongly on execution mode; it does not show that diversity always harms
learning. No final-test winner is promoted into the original comparison.
'''
    write(REPORT/'Sections/01_prediction_annotations.tex',annotations)
    from review import render_review
    review_state,review_text=render_review(REPORT)
    write(REPORT/'Sections/05_human_review.tex',review_text)
    write(REPORT/'review.tex',preamble+r'''
\title{IT3212 Assignment 4: Human Review}\author{MsPacMan team}\date{}
\begin{document}\maketitle\input{Sections/05_human_review.tex}\end{document}
''')
    write(REPORT/'ai_assistance.tex',r'''
\section*{Consequential AI assistance record}
Codex assisted with experimental design,
Gymnasium wrappers, ordinary tabular Q-learning, PPO, experiment runners,
debugging, evidence analysis and report drafting. The team authorised score
increment rewards after instructor clarification, requested ordinary Q
consistent with the original predictions, and rejected DQN. Double Q/replay
exploration was not accepted as the ordinary-Q baseline. Historical pilot
outcomes and failed candidates remain distinct from the assessed study.

Codex proposed the bounded interaction cap, relative features, matched
development search and source-only checkpoint selection. Verification checked
one Q update per transition, terminal targets, native action mapping, frozen
parameter hashes, native BCD scores and directly replayed recorded actions.
The imperfect power proxy and representation differences remain disclosed.

Codex read both briefs, implemented primary D as a fixed
object-feature occlusion analogue, and implemented45 fresh same-host,
five-seed one-factor models plus a frozen execution-mode diagnostic. Native
parity/noninterference checks verified D leaves the game unchanged. Original
and fresh full-budget baseline weights were compared bitwise rather than
attributing checkpoint differences to hardware or library versions.

Codex selected episodes by documented outcome distributions, reproduced30
decision alternatives through complete seeded action prefixes, and generated
tables/figures from measured records. Snapshot restoration alone failed a
sticky-action continuation check and was rejected. One-step alternatives are
not claimed as expected-return estimates. Report checks cover numerical
provenance, original-prediction checksum, word limit, LaTeX compilation and
rendered layout.

The team requested date-free report prose and equal analytical attention to
A/B/C/D. Codex added parallel subsections, a four-setting figure and a paired
D outcome table, verified against the original records without new training.
Raw provenance timestamps and the original prediction sheet remain intact.

Codex initially supplied an evidence-based Part5 proposal. The team then
provided Fredrik, Johan, Elling and Mats's individual initial rankings, its
final priority order and rationale, and broad support for the draft's
evidence assessments. Codex incorporated those reported inputs into the
review. Broad support is not presented as unqualified agreement with every
interpretation. No member-specific arguments, voting procedure or meeting
date were invented. The authors remain responsible for the final claims.
''')
    write(REPORT/'ai_record.tex',preamble+r'''
\title{IT3212 Assignment 4: AI Assistance}\author{MsPacMan team}\date{}
\begin{document}\maketitle\input{ai_assistance.tex}\end{document}
''')
    print('Evidence brief, annotated predictions, completed group review and AI record written.')

if __name__=='__main__':main()
