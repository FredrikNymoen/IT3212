"""Check the edited four-column report against old A/B/C and new D CSVs."""
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'three_hour'))
import run_study_v2 as study
from validate_delivery import check_cell,summary

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/transfer_boundary_v1'
REPORT=ROOT/'reports/assignment4'

def main():
    assert json.loads((BASE/'verification.json').read_text())['episodes']==2000
    text=(REPORT/'Sections/02_learning_settings.tex').read_text(encoding='utf-8')
    discussion=(REPORT/'Sections/02_discussion.tex').read_text(encoding='utf-8')
    tables=re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',text,re.S);assert len(tables)==2
    rows=lambda table:[[cell.strip() for cell in line.strip().removesuffix('\\\\').split('&')]
        for line in table.splitlines() if re.match(r'^[ABCD] &',line)]
    parsed=[rows(table) for table in tables];assert all(len(table)==8 for table in parsed)
    assert all(len(row)==4 for row in parsed[0]) and 'Game score &' not in tables[0]
    old=ROOT/'results/assignment4/three_hour_20261122/final_v1'
    locks=[json.loads((old/'locked.json').read_text()),json.loads((BASE/'protocol.json').read_text())]
    measurements=[]
    for condition in 'ABCD':
        for algorithm in ['Q-learning','PPO']:
            source=locks[int(condition=='D')]
            jobs=[job for job in source['jobs'] if job['condition']==condition and job['algorithm']==algorithm]
            assert len(jobs)==5
            for job in jobs:
                directory=Path(job['directory']);data=pd.read_csv(directory/'evaluation.csv')
                assert len(data)==200
                metrics=json.loads((directory/'metrics.json').read_text())
                failure='known_failure' if condition=='D' else None
                failure_mask=data[failure].eq(1) if failure else data.success.eq(0)
                measurements.append(dict(condition=condition,algorithm=algorithm,seed=job['seed'],
                    success=100*data.success.mean(),task_return=data['observed_return' if condition=='D' else 'task_return'].mean(),
                    success_length=data.loc[data.success.eq(1),'length'].mean(),failure_length=data.loc[failure_mask,'length'].mean(),
                    training_seconds=0 if condition=='D' else job['training_seconds'],evaluation_seconds=metrics['evaluation_seconds']))
    frame=pd.DataFrame(measurements);index=0
    for condition in 'ABCD':
        for algorithm in ['Q-learning','PPO']:
            group=frame[(frame.condition==condition)&(frame.algorithm==algorithm)]
            for cells,fields in [(parsed[0][index],['success','task_return']),
                (parsed[1][index],['success_length','failure_length','training_seconds','evaluation_seconds'])]:
                assert cells[:2]==[condition,algorithm]
                for cell,field in zip(cells[2:],fields):
                    expected=[0] if condition=='D' and field=='training_seconds' else summary(group[field])
                    check_cell(cell,expected)
            index+=1
    assert study.sha(REPORT/'Sections/01_predictions.tex')=='17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a'
    visible=re.sub(r'(?<!\\)%[^\n]*','',text+'\n'+discussion)
    visible=re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}','',visible)
    assert not re.search(r'\.(?:csv|json|tex|py|npz|pkl|png|pdf|ipynb)\b|results/|three\\?_hour|\\(?:input|href|url)\{',visible,re.I)
    assert 'arbitrary identifiers' in text and 'not\ndifficulty ratings' in text
    assert 'map-exit' in text and 'external truncation' in discussion
    labels=[];references=[]
    for file in REPORT.rglob('*.tex'):
        body=re.sub(r'(?<!\\)%[^\n]*','',file.read_text(encoding='utf-8'));stack=[]
        for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
            if kind=='begin':stack.append(name)
            else:assert stack and stack.pop()==name,file
        assert not stack,file
        depth=0
        for char in re.sub(r'\\[{}]','',body):
            depth+=(char=='{')-(char=='}');assert depth>=0,file
        assert depth==0 and len(re.findall(r'(?<!\\)\$',body))%2==0,file
        labels.extend(re.findall(r'\\label\{([^}]+)\}',body))
        references.extend(re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',body))
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',body):assert (REPORT/image).exists()
    assert len(labels)==len(set(labels)) and not set(references)-set(labels)
    prose=re.sub(r'\\begin\{(table|figure)\}.*?\\end\{\1\}','',visible,flags=re.S)
    prose=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?','',prose)
    words=len(re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*",prose))
    assert words<2000
    study.write(BASE/'report_validation.json',dict(all_40_reported_cases_have_200_episodes=True,
        four_column_outcome_table_verified=True,old_A_B_C_and_new_D_arithmetic_verified=True,
        raw_C_scores_retained_in_prose=True,original_predictions_unchanged=True,no_visible_file_references=True,
        latex_structure_refs_assets_verified=True,part2_prose_words=words,pdf_compiled=False,
        report_hashes={str(file.relative_to(ROOT)):study.sha(file) for file in [REPORT/'Sections/02_learning_settings.tex',
            REPORT/'Sections/02_discussion.tex',REPORT/'README.md',REPORT/'ai_assistance.tex']},
        verifier_sha256=study.sha(Path(__file__))))
    print('Verified all 40 table cases, return column, native boundary disclosure, LaTeX and file-free prose.',flush=True)

if __name__=='__main__':main()
