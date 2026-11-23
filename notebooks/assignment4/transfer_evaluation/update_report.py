"""Insert verified D results into the reviewed report; no model selection."""
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'three_hour'))
import run_study_v2 as study

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/transfer_boundary_v1'
REPORT=ROOT/'reports/assignment4'

def fmt(values,multiplier=1,digits=1,mark=False):
    values=np.asarray([float(v)*multiplier for v in values if not pd.isna(v)])
    if not len(values):return '--'
    suffix=r'\;{}^{\dagger}' if mark else ''
    sd=f' \\pm {values.std(ddof=1):.{digits}f}' if len(values)>1 else ''
    return f'${values.mean():.{digits}f}{sd}{suffix}$'

def main():
    verification=json.loads((BASE/'verification.json').read_text())
    assert verification['episodes']==2000 and verification['further_training_interactions']==0
    frame=pd.read_csv(BASE/'verified_comparison.csv')
    q=frame[frame.algorithm.eq('Q-learning')];p=frame[frame.algorithm.eq('PPO')]
    assert len(q)==len(p)==5
    path=REPORT/'Sections/02_learning_settings.tex';text=path.read_text(encoding='utf-8')
    tables=list(re.finditer(r'\\begin\{tabular\}.*?\\end\{tabular\}',text,re.S));assert len(tables)==2
    for index,match in reversed(list(enumerate(tables))):
        table=match.group();lines=table.splitlines()
        for line_index,line in enumerate(lines):
            for algorithm,group in [('Q-learning',q),('PPO',p)]:
                if not line.startswith(f'D & {algorithm} &'):continue
                if index==0:
                    cells=[fmt(group.observed_success_rate,100,mark=True),fmt(group.mean_return)]
                else:cells=[fmt(group.success_length,digits=0),fmt(group.failure_length,digits=0),'$0$',fmt(group.evaluation_seconds)]
                lines[line_index]='D & '+algorithm+' & '+' & '.join(cells)+r' \\'
        text=text[:match.start()]+'\n'.join(lines)+text[match.end():]
    text=text.replace("episode fails. The horizon is terminal, with remaining time observable.",
        "completed in-map episode fails. Map exits in D are recorded separately as\nexternal censoring. The horizon is terminal, with remaining time observable.")
    text=text.replace("same B models on fresh episodes under the explicit map-boundary protocol.",
        "same B models on fresh episodes. \\(\\dagger\\) D success counts verified\ngoal attainment before any map exit; bounds for censored outcomes are reported\nbelow. Censored returns cover only the verified in-map prefix.")
    text=text.replace('one agent. Dashes mark the unresolved Q/D aggregate. Training covers the full',
        'one agent. D failure lengths exclude censored episodes. A dash means no\nepisode of that outcome. Training covers the full')
    text=text.replace('(decisions) and wall-clock costs in the same main experiment.',
        '(decisions) and wall-clock costs in the same A/B/C and D evaluations.')
    insertion=r'''\paragraph{Transfer measurement.}
'''
    insertion+=f'D known-failure rates are {100*q.known_failure_rate.mean():.1f}\\% for Q and {100*p.known_failure_rate.mean():.1f}\\% for PPO. '
    insertion+=f'Map exits censor {int(q.map_exits.sum())} of 1,000 Q episodes and {int(p.map_exits.sum())} of 1,000 PPO episodes. '
    insertion+=f'The across-agent mean Q success bound is [{100*q.success_lower_bound.mean():.1f}\\%, {100*q.success_upper_bound.mean():.1f}\\%]. '
    if p.censored_rate.sum()>0:
        insertion+=f'The PPO bound is [{100*p.success_lower_bound.mean():.1f}\\%, {100*p.success_upper_bound.mean():.1f}\\%]. '
    insertion+='These bounds reflect unknown outcomes after censoring, not confidence intervals. Success, known failure and censoring sum to 100\\%; censored episodes are not counted as agent failures.\n\n'
    marker='\\begin{table}[htbp]\n\\centering\\small\\setlength'
    assert marker in text
    if '\\paragraph{Transfer measurement.}' in text:
        text=re.sub(r'\\paragraph\{Transfer measurement\.\}.*?(?=\\begin\{table\})',lambda _:insertion,text,flags=re.S)
    else:text=text.replace(marker,insertion+marker,1)
    path.write_text(text,encoding='utf-8')
    discussion=REPORT/'Sections/02_discussion.tex';body=discussion.read_text(encoding='utf-8')
    begin=body.index('D preserves local feature meanings:')
    replacement=r'''D preserves local feature meanings: food-relative directions, corridor
structure and danger bins let Q reuse learned values across positions,
while PPO shares learned action-scoring parameters. Compact keys may still
merge different situations, and networks can encounter unfamiliar feature
combinations. '''
    replacement+=f'On the fresh transfer episodes, observed success averages {100*q.observed_success_rate.mean():.1f}\\% for Q and {100*p.observed_success_rate.mean():.1f}\\% for PPO. '
    replacement+='These are the same B agents; no transfer episode updates their weights or selects a checkpoint. The source B success averages remain 32.2\\% and 67.4\\%, respectively.\n\n'
    replacement+=r'''\paragraph{Map-boundary protocol.}
The original Q/D interruption occurred at 1,760 points with two lives left:
the native game changed from the transfer maze to the training maze before
the goal was reached. Continuing would mix two environments in one transfer
episode. The revised Gymnasium evaluation therefore stops at any map exit,
marks it as external truncation, and records the unresolved outcome separately.
The crossing action's reward is excluded from the verified in-map return
prefix; native rewards and RAM remain recorded. No action is taken on the
other maze. A censored episode is neither verified success nor known failure.
If \(S\) episodes succeed and \(C\) are censored out of \(N\), success for
the original task is bounded by \([S/N,(S+C)/N]\). The measured rate \(S/N\)
describes verified goal attainment on the transfer maze; it is not a claim
about what a censored agent would subsequently do.

This boundary rule was fixed before reevaluating all ten original B agents
on 200 fresh episodes each. It repairs the original evaluation interruption,
not the learned policies, and is explicitly a post-inspection protocol revision.
The transfer maze had already been inspected, so fresh episode seeds do not
turn it into an independent new domain. Neither the ROM intervention nor the
power proxy justifies general reliability claims beyond this simulation.
'''
    discussion.write_text(body[:begin]+replacement,encoding='utf-8')
    study.write(BASE/'report_update.json',dict(episodes=2000,source_hashes_verified=True,training_interactions=0,
        report_hashes={str(file.relative_to(ROOT)):study.sha(file) for file in [path,discussion]},
        updater_sha256=study.sha(Path(__file__))))
    print('Updated both D rows, prefix-return/censoring explanation and transfer discussion.',flush=True)

if __name__=='__main__':main()
