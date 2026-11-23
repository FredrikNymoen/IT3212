"""Refresh current pointers after the verified November 22 report is complete."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/three_hour_20261122'

def main():
    assert (BASE/'final_v1/delivery_validation.json').exists()
    report=ROOT/'reports/assignment4/README.md'
    text=report.read_text(encoding='utf-8')
    start=text.index('Part 2 now combines')
    end=text.index('English LaTeX writing project')
    text=text[:start]+'''The current Part 2 is the complete five-training-seed A–D study in
[the November 22 implementation guide](../../notebooks/assignment4/three_hour/README.md).
It compares ordinary single-table Q-learning (`gamma995`) with PPO (`stable`),
using fresh training seeds 17/29/101/314/2718, a common 262,144-interaction
cap, and 200 frozen test episodes per trained agent/condition. Both methods
use the same source-development checkpoint rule. D inherits B weights without
training. All five agents remain in the primary table, including failed runs.

The older complete v3 benchmark and B-only/A/D follow-ups remain archived.
The historical 94.5% PPO result concerned two fixed revised B policies. It
was never the complete A–D benchmark or five-seed training reliability. The
new study independently checks those policies, then tests fresh training
variability separately. Double Q variants remain historical side experiments.

'''+text[end:]
    start=text.index('`02_learning_settings.tex` is now')
    end=text.index('Headings follow the brief')
    text=text[:start]+'''`02_learning_settings.tex` contains the latest verified numerical comparison;
`02_discussion.tex` interprets it. Include both Part 2 input lines when
extracting the evidence brief. Use the verification/rendering commands in
`three_hour/README.md`; historical v2/v3 renderers must not replace these
sections. The old report is archived under the latest final run.

'''+text[end:]
    start=text.index('The original runner produced')
    end=text.index('## Submission components and limits')
    text=text[:start]+'''The latest curve is `Images/part2_learning_curves_20261122.png`; it shows
unsmoothed development success and variation across five newly trained agents.
No report PDF or ZIP snapshot is rebuilt. Part 3--6 placeholders remain;
five seeds in Part 2 alone do not complete Part 3's controlled mechanism study.
Part 5 must record an actual team review. Original predictions are unchanged.
Fresh evaluation seeds do not make the previously inspected D layout an
independent new domain. Prior cumulative search was unequal and is disclosed;
this study gives both methods the same new candidate/interaction allowance.

'''+text[end:]
    text=text.replace('in `figures/assignment4/current/`.','in `figures/assignment4/three_hour/`.')
    report.write_text(text,encoding='utf-8')
    main_readme=ROOT/'notebooks/assignment4/README.md'
    text=main_readme.read_text(encoding='utf-8')
    heading='# Assignment 4: MsPacMan, Part 2\n'
    introduction='''
The current complete A–D comparison is the [November 22 five-seed study](three_hour/README.md).
Use its commands to reproduce the selected ordinary-Q/PPO workflows and the
200-episode-per-agent frozen evaluation. The report now uses these results.
Historical descriptions and reproduction commands below retain their original
scope; their renderers must not overwrite the new report.

## Historical studies
'''
    main_readme.write_text(text.replace(heading,heading+introduction,1),encoding='utf-8')
    old=ROOT/'notebooks/assignment4/current/README.md'
    text=old.read_text(encoding='utf-8')
    text=text.replace('# Current Part 2: ordinary Q-learning versus PPO, settings A--D',
        '# Historical v3 Part 2: ordinary Q-learning versus PPO, settings A--D',1)
    index=text.index('\n\n')
    text=text[:index]+'''\n\nThe latest complete report uses the [November 22 five-seed workflow](../three_hour/README.md).
This directory preserves the older two-seed v3 benchmark and its source hashes.
Its report renderer does not reproduce the latest comparison.
'''+text[index:]
    start=text.index('This is the registered complete v3 benchmark workflow.')
    end=text.index('Run from the repository root with')
    text=text[:start]+'''This registered v3 benchmark and the later B-only/A/D studies are historical
records. Their models, numerical exports and source hashes remain intact.
The report guide points to the latest complete study. Do not use this old
renderer to replace the latest report or bypass its overwrite guard.

'''+text[end:]
    old.write_text(text,encoding='utf-8')
    hardware=BASE/'environment.json'
    if hardware.exists():
        record=json.loads(hardware.read_text())
        record['concurrency_note']='Up to eight single-thread training workers and eight frozen evaluation workers overlapped (peak sixteen processes). Timings are shared-compute costs, not isolated algorithm speed measurements.'
        record['thread_policy']='One numerical thread per worker. Initial screen peak six, replication eight, main eight, final eight; main/final overlap peak sixteen processes.'
        record['phase_concurrency']['overlap_peak']=16
        record['wall_window_note']='An apparent system pause consumed approximately 26 wall-clock minutes without checkpoint progress. The main cap was set to 262144 before main training, rather than the earlier target 393216; no result-based extension or final/D tuning.'
        hardware.write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':main()
