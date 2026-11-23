"""Substantive Part5 draft; never invent members' actual rankings or feedback."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

def render_review(report):
    state=json.loads((report/'review_input.json').read_text())
    assert state['status']=='draft','Preserve the actual reviewed text; do not regenerate an approved review as a proposal.'
    assert state['actual_negotiation_record'] is None
    assert all(v is None for v in state['initial_rankings'].values())
    text=r'''
\section*{Part 5: Human review draft}
\noindent\textbf{Status.} The team named Fredrik, Johan, Elling and Mats as
participants. The assessment below is a proposed review for their validation,
not verified minutes. Individual initial rankings and the actual negotiation
have not been supplied; no personal opinions or agreement are invented.

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

\subsection*{Priorities proposed for discussion}
A reasonable proposed order for consequential operation is failure avoidance,
robustness to change, explainability, then time efficiency. The first two
follow from the observed failure rates; explainability helps diagnose failures,
while speed should not compensate for missing the goal. A classroom
demonstration can give explainability and short setup time greater weight.
This is a proposal, not a claim that the four members negotiated this order.

\begin{center}\begin{tabular}{ll}\toprule
Member & Actual initial ranking\\\midrule
Fredrik & Not supplied\\
Johan & Not supplied\\
Elling & Not supplied\\
Mats & Not supplied\\\bottomrule
\end{tabular}\end{center}
Before finalising, each member must record their actual initial ranking;
the group must state its adopted final order and how disagreements were
resolved. These cannot be reconstructed as historical views from this draft.
'''
    return state,text
