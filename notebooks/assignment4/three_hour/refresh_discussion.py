"""Reviewable numerical prediction discussion from complete fixed exports."""
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/three_hour_20261122'
REPORT=ROOT/'reports/assignment4'

def main(args):
    final=ROOT/'results/assignment4'/args.final_run
    assert (final/'completed.json').exists() or (final/'interrupted.json').exists()
    frame=pd.read_csv(final/'verified_comparison.csv')
    curves=pd.read_csv(ROOT/'results/assignment4'/args.training_run/'comparison.csv')
    values=lambda condition,algorithm:frame[(frame.condition==condition)&(frame.algorithm==algorithm)]
    mean=lambda condition,algorithm:100*values(condition,algorithm).success_rate.mean()
    early=lambda algorithm:100*curves[(curves.setting=='B')&(curves.algorithm==algorithm)&(curves.interactions==65536)].success_rate.mean()
    q,p=early('Q-learning'),early('PPO')
    early_claim='supports' if q>p else ('contradicts' if q<p else 'does not distinguish')
    qrange=100*values('A','Q-learning').success_rate;prange=100*values('A','PPO').success_rate
    best='PPO' if mean('B','PPO')>mean('B','Q-learning') else 'Q-learning'
    text=r'''% Prediction discussion from verified frozen five-seed-study exports.
\paragraph{Implications for our predictions.}
'''
    text+=f'At 65,536 interactions, B-development success averaged {q:.1f}\\% for Q and {p:.1f}\\% for PPO. This {early_claim} our predicted early Q advantage in B. The final selected B policies averaged {mean("B","Q-learning"):.1f}\\% and {mean("B","PPO"):.1f}\\%, respectively, favouring {best} for this particular workflow. The original algorithm recommendation remains conditional on the setting and training procedure.\n\n'
    text+=f'In A, Q training-seed success ranged from {qrange.min():.1f}\\% to {qrange.max():.1f}\\%, and PPO from {prange.min():.1f}\\% to {prange.max():.1f}\\%. Development curves show that successful checkpoints can regress with more training. Keeping all five fresh training seeds exposes this instability; choosing an earlier checkpoint does not make the full training run cheaper.\n\n'
    qsd=100*values('B','Q-learning').success_rate.std(ddof=1)
    psd=100*values('B','PPO').success_rate.std(ddof=1)
    direction='supports' if qsd>psd else ('contradicts' if qsd<psd else 'does not distinguish')
    text+=f'B success-rate SD across training seeds was {qsd:.1f} percentage points for Q and {psd:.1f} for PPO. This {direction} our predicted larger Q variation in B, but five seeds provide limited precision. Aggregate outcomes alone cannot establish safer routes or test the three predicted state-specific actions.\n\n'
    text+=r'''The earlier 94.5\% PPO claim concerned two fixed B-only models, not all
settings or random retraining. A fresh native check gave those models
98.5\%/98.0\% on 200 episodes each. These fixed-policy results and the new
five-seed average answer different questions; the best two seeds must not
replace the full benchmark.

'''
    for algorithm in ['Q-learning','PPO']:
        before=mean('B',algorithm);after=mean('C',algorithm)
        text+=f'For {algorithm}, the step-penalty comparison changed mean success from {before:.1f}\\% in B to {after:.1f}\\% in C. '
    text+='Successful-episode lengths condition on which episodes succeeded, so shorter successful episodes alone do not establish safer routes or a reliable efficiency gain. Common raw scores and outcomes are needed because the reward definitions differ.\n\n'
    incomplete=len(values('D','Q-learning'))<5
    if incomplete:
        text+=f'On D, frozen PPO achieved {mean("D","PPO"):.1f}\\% success across all five agents. The corresponding Q aggregate is unavailable because one protocol-guard interruption prevented a complete test. The four completed Q agents must not silently replace the planned population. '
    else:
        text+=f'On D, frozen Q and PPO achieved {mean("D","Q-learning"):.1f}\\% and {mean("D","PPO"):.1f}\\% success. '
    unknown=100*values('D','Q-learning').unseen_state_fraction.mean()
    text+=f'Among the {len(values("D","Q-learning"))} completed Q/D agents, {unknown:.1f}\\% of decisions encountered unseen table keys. '
    if unknown<10:
        text+='A familiar compact key can still merge physically different situations, so table-key coverage does not establish that the learned values transfer correctly. '
    else:text+='This indicates a coverage gap, while failures can also reflect inaccurate values in familiar keys. '
    text+='Shared local features provide transferable structure, but these results concern one previously inspected ROM-layout intervention. They do not establish reliability on unfamiliar environments. The precise-power-time premise in Part~1 also remains unverified by the retained proxy.\n'
    selection_path=BASE/'main_operating_selection.json'
    if selection_path.exists() and args.training_run=='three_hour_20261122/main_v1':
        selection=json.loads(selection_path.read_text());rows=[]
        for condition in 'ABCD':
            cells=[condition]
            for algorithm in ['Q-learning','PPO']:
                chosen=next(row for row in selection['jobs'] if row['condition']==condition and row['algorithm']==algorithm)
                actual=frame[(frame.condition==condition)&(frame.algorithm==algorithm)&(frame.seed==chosen['seed'])].iloc[0]
                assert actual.model_sha256==chosen['model_sha256']
                cells.extend([str(chosen['seed']),f'{100*actual.success_rate:.1f}'])
            rows.append(' & '.join(cells)+r' \\')
        text+=r'''

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Setting & Q seed & Q success (\%) & PPO seed & PPO success (\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Fixed operating policies selected from the five fresh main seeds
using only source-development success, score and checkpoint age. The rule was
recorded before inspecting final summaries. These are the corresponding
200-episode test results, \(n=1\) trained policy per cell. The primary table
uses the five-seed design and suppresses any unresolved aggregate. D inherits the B choice. Source:
\texttt{main\_operating\_selection.json} and \texttt{final\_v1}. This demonstrates
the chosen policies, not reliable performance after arbitrary retraining.}
\label{tab:part2-operating}
\end{table}
'''
    target=REPORT/'Sections/02_discussion.tex'
    archive=final/'previous_report/02_discussion.tex'
    if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(target.read_bytes())
    target.write_text(text,encoding='utf-8')
    (final/'rendered_discussion.tex').write_text(text,encoding='utf-8')
    print('Updated measured prediction discussion and source-selected operating-policy table.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-run',default='three_hour_20261122/main_v1')
    parser.add_argument('--final-run',default='three_hour_20261122/final_v1')
    main(parser.parse_args())
