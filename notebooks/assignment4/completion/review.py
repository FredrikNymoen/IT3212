"""Part5 review using the rankings, rationale and broad support supplied by the team."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

def render_review(report):
    state=json.loads((report/'review_input.json').read_text())
    assert state['status']=='reviewed'
    codes={'Failure avoidance':'F','Robustness to change':'R','Explainability':'E','Time efficiency':'T'}
    assert all(len(v)==4 and set(v)==set(codes) for v in state['initial_rankings'].values())
    assert len(state['final_priorities'])==4 and set(state['final_priorities'])==set(codes)
    assert state['actual_negotiation_record'] and state['group_rationale_norwegian'] and state['assessment_endorsement_norwegian']
    text=r'''
\section*{Part 5: Human review}
The review participants were Fredrik, Johan, Elling and Mats. The group
broadly supported the evidence assessments below and considered the final
priorities well justified.

\subsection*{Strongest supporting evidence}
The strongest support for preferring PPO in the uncertain simulation is the
matched B comparison: mean success67.4\% versus Q32.2\%, across five training
roots with the same interaction cap and200 frozen episodes per trained
agent. PPO leads in four of five roots, so the conclusion does not rest on
one impressive run. Nevertheless, the different state representations mean
this supports the tested workflows rather than PPO in isolation.

The paired D experiment provides stronger control of observation sensitivity:
the exact same policies are tested on the same200 seeds with clean and
obstructed inputs, without training or game changes. Mean PPO success falls
65.9\% to48.5\%; Q falls29.5\% to23.2\%. Higher retained performance and
smaller degradation are different criteria. Neither establishes adequate
reliability, and the opposite effects in individual roots prevent a universal
claim about information removal.

\subsection*{A claim that would exceed the evidence}
A tempting extension of the initial PPO recommendation is ``PPO generalises
reliably to unfamiliar environments.'' This exceeds our evidence: PPO still
fails51.5\% of obstructed episodes, only one obstruction region was tested,
and the underlying maze and static geometry remain available. A defensible
replacement is: the tested PPO workflow retains a higher mean success rate
than Q under this fixed object-feature obstruction. It does not establish
pixel-perception robustness, unseen-map transfer or deployment safety.

\subsection*{A competing explanation}
Zero entropy's primary success advantage could reflect how argmax execution
concentrates trajectories, rather than a generally better learned action
distribution. The frozen diagnostic supports this alternative: the advantage
is25.6 percentage points under argmax but only0.8 under sampled execution.
Root2718 also contradicts a universal training benefit. Sampling changes the
states visited as well as individual actions, so the diagnostic distinguishes
execution dependence without isolating a unique internal cause. Promoting
the highest final-test configuration would require fresh confirmation.

\subsection*{What changes from the original predictions?}
The predicted early Q advantage holds in B at the early checkpoint but is
contradicted in A. Greater Q seed variation holds in B, not A/C. Aggregate
scores do not establish the predicted safer routes, and the predicted larger
Q loss from stochasticity is unsupported at the mean level. The tested step
penalty reduces average success; shorter successful episodes alone do not
prove improved efficiency. Our power proxy cannot verify the precise-timer
premise behind the frightened-ghost prediction. The initial preference for
PPO therefore becomes conditional: it is preferable for further B simulation,
while neither workflow is supported for consequential autonomous operation.

\subsection*{Initial rankings and final priorities}
The members reported the following initial rankings, from most to least
important: F denotes failure avoidance, R robustness to change,
E explainability and T time efficiency.
__RANKING_TABLE__
The final group order was __FINAL_ORDER__.
Three members initially placed failure avoidance first; Johan placed
robustness first. Explainability and time efficiency also differed in rank.
The group resolved these trade-offs by prioritising avoidance of failures
and stable operation, with explainability and time efficiency less decisive.
The final order therefore places failure avoidance first and robustness
second, while retaining explainability ahead of time efficiency.

This reliability-first ordering supports the conditional recommendation:
PPO is the stronger B workflow for further simulation, but neither tested
workflow is adequately supported for consequential autonomous operation.
Explainability remains useful for diagnosing why decisions fail; shorter
runtime does not compensate for the observed failure rates. A classroom
demonstration can weight these secondary priorities differently from deployment.
'''
    table=[r'\begin{center}\begin{tabular}{ll}\toprule',r'Member & Initial priority order\\\midrule']
    for name in state['members_named_by_user']:
        table.append(name+r' & $'+'>'.join(codes[x] for x in state['initial_rankings'][name])+r'$\\')
    table.extend([r'\bottomrule\end{tabular}\end{center}'])
    text=text.replace('__RANKING_TABLE__','\n'.join(table))
    text=text.replace('__FINAL_ORDER__','$'+'>'.join(codes[x] for x in state['final_priorities'])+'$')
    return state,text
