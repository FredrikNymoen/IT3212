"""Audit the concise Part 2 revision and every new analytical statement."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'results/assignment4/report_review_current'
HERE.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'reports/assignment4'
OLD=ROOT/'results/assignment4/three_hour_20261122/final_v1'
NEW=ROOT/'results/assignment4/transfer_boundary_v1'
sys.path.insert(0,str(ROOT/'notebooks/assignment4/three_hour'))
from validate_delivery import check_cell,summary

sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
main_path=REPORT/'Sections/02_learning_settings.tex'
discussion_path=REPORT/'Sections/02_discussion.tex'
text=main_path.read_text(encoding='utf-8');discussion=discussion_path.read_text(encoding='utf-8')
tables=lambda body:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',body,re.S)
rows=lambda table:[[cell.strip() for cell in line.strip().removesuffix('\\\\').split('&')]
    for line in table.splitlines() if re.match(r'^[ABCD] &',line)]
current=tables(text)
before=tables((HERE/'before/02_learning_settings.tex').read_text(encoding='utf-8'))
assert len(current)==2 and [rows(t) for t in current]==[rows(t) for t in before]
assert len(re.findall(r'\\begin\{table\}',text))==3
assert '\\begin{tabularx}' in text and '\\label{tab:part2-configuration}' in text
assert all(len(row)==4 for row in rows(current[0])) and 'Game score &' not in current[0]

old_lock=json.loads((OLD/'locked.json').read_text())
new_lock=json.loads((NEW/'protocol.json').read_text())
for lock in [old_lock,new_lock]:
    for name,expected in lock['source_hashes'].items():assert sha(ROOT/name)==expected,name
assert sha(OLD/'locked.json')==new_lock['original_selection_lock_sha256']
assert json.loads((NEW/'verification.json').read_text())['episodes']==2000
records=[]
for condition in 'ABCD':
    for algorithm in ['Q-learning','PPO']:
        lock=new_lock if condition=='D' else old_lock
        jobs=[job for job in lock['jobs'] if job['condition']==condition and job['algorithm']==algorithm]
        assert len(jobs)==5
        for job in jobs:
            folder=Path(job['directory']);data=pd.read_csv(folder/'evaluation.csv')
            assert len(data)==200 and data.evaluation_seed.tolist()==lock['evaluation_seeds']
            assert sha(ROOT/job['model'])==job['model_sha256']
            metrics=json.loads((folder/'metrics.json').read_text())
            failure=data.known_failure.eq(1) if condition=='D' else data.success.eq(0)
            if condition=='D':
                assert (data[['success','known_failure','censored']].sum(axis=1)==1).all()
                partner=next(row for row in old_lock['jobs'] if row['condition']=='B' and row['algorithm']==algorithm and row['seed']==job['seed'])
                assert partner['model_sha256']==job['model_sha256']
            records.append(dict(condition=condition,algorithm=algorithm,seed=job['seed'],success=100*data.success.mean(),
                failure=100*failure.mean(),censored=100*data.censored.mean() if condition=='D' else 0,
                task_return=data['observed_return' if condition=='D' else 'task_return'].mean(),
                success_length=data.loc[data.success.eq(1),'length'].mean(),failure_length=data.loc[failure,'length'].mean(),
                training_seconds=0 if condition=='D' else job['training_seconds'],evaluation_seconds=metrics['evaluation_seconds']))
frame=pd.DataFrame(records);assert len(frame)==40
index=0
for condition in 'ABCD':
    for algorithm in ['Q-learning','PPO']:
        group=frame[(frame.condition==condition)&(frame.algorithm==algorithm)]
        for table,fields in [(current[0],['success','task_return']),
            (current[1],['success_length','failure_length','training_seconds','evaluation_seconds'])]:
            cells=rows(table)[index];assert cells[:2]==[condition,algorithm]
            for cell,field in zip(cells[2:],fields):
                check_cell(cell,[0] if condition=='D' and field=='training_seconds' else summary(group[field]))
        index+=1

analysis=json.loads((HERE/'paired_seed_analysis.json').read_text())
for name,expected in analysis['source_hashes'].items():assert sha(ROOT/name)==expected,name
assert analysis['B_PPO_beats_Q_seeds']==4
assert np.isclose(analysis['B_PPO_minus_Q_mean_percentage_points'],35.2)
assert np.allclose(analysis['B_gap_omission_range'],[29.25,44.5])
q_ab=next(row for row in analysis['setting_changes'] if row['algorithm']=='Q-learning' and row['before']=='A' and row['after']=='B')
assert min(q_ab['per_seed'].values())==-71 and np.isclose(max(q_ab['per_seed'].values()),48)
assert np.isclose(q_ab['mean_percentage_points'],-.7)
raw_pairs=frame.pivot(index=['seed','algorithm'],columns='condition',values='success')
raw_gap=raw_pairs.xs('PPO',level='algorithm').B-raw_pairs.xs('Q-learning',level='algorithm').B
assert np.isclose(raw_gap.mean(),analysis['B_PPO_minus_Q_mean_percentage_points'])
for seed in raw_gap.index:
    assert np.isclose(raw_gap.drop(seed).mean(),analysis['B_gap_omitting_each_paired_seed'][str(seed)])
for row in analysis['setting_changes']:
    raw_values=raw_pairs.xs(row['algorithm'],level='algorithm')
    delta=raw_values[row['after']]-raw_values[row['before']]
    assert np.isclose(delta.mean(),row['mean_percentage_points'])
    for seed,value in delta.items():assert np.isclose(value,row['per_seed'][str(seed)])
for algorithm,drop in [('Q-learning',-5.3),('PPO',-14.7)]:
    change=next(row for row in analysis['setting_changes'] if row['algorithm']==algorithm and row['before']=='B' and row['after']=='C')
    assert np.isclose(change['mean_percentage_points'],drop) and change['seeds_improved']==2
for row in analysis['setting_changes']:
    if row['after']=='D':assert row['seeds_worsened']==5

assert sha(REPORT/'Sections/01_predictions.tex')=='17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a'
labels=[];references=[]
for file in REPORT.rglob('*.tex'):
    content=re.sub(r'(?<!\\)%[^\n]*','',file.read_text(encoding='utf-8'));stack=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',content):
        if kind=='begin':stack.append(name)
        else:assert stack and stack.pop()==name,file
    assert not stack,file
    depth=0
    for char in re.sub(r'\\[{}]','',content):
        depth+=(char=='{')-(char=='}');assert depth>=0,file
    assert depth==0 and len(re.findall(r'(?<!\\)\$',content))%2==0,file
    labels.extend(re.findall(r'\\label\{([^}]+)\}',content));references.extend(re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',content))
    for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',content):assert (REPORT/image).exists()
assert len(labels)==len(set(labels)) and not set(references)-set(labels)
old_report_manifest=json.loads((OLD/'report_manifest.json').read_text())
for name,expected in old_report_manifest.items():
    if name.endswith('part2_learning_curves_20261122.png'):assert sha(ROOT/name)==expected
visible=re.sub(r'(?<!\\)%[^\n]*','',text+'\n'+discussion)
visible=re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}','',visible)
assert not re.search(r'\.(?:csv|json|tex|py|npz|pkl|png|pdf|ipynb)\b|results/|three\\?_hour|\\(?:input|href|url)\{',visible,re.I)
prose=re.sub(r'\\begin\{(table|figure)\}.*?\\end\{\1\}','',visible,flags=re.S)
prose=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?','',prose)
words=len(re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*",prose))
assert words<=1150

requirements={
    'A':'Early sample efficiency compared; final mean distinguished from retraining reliability.',
    'B':'Transition change isolated in the workflow; paired-seed shifts, comparison robustness and residual failure discussed.',
    'C':'Success-first objective stated; penalty trade-off assessed using common outcomes and conditional-length caveat.',
    'D':'Frozen B models, compatible formats, meaningful representation, source/transfer counterexample and explicit censor bounds.',
    'rules1to8':'Budget justification; common seeds; frozen evaluation; separate seeds and200 episodes; evaluation policies; equal new search; no heldout tuning; runtime/hardware.',
    'required_metrics':'Success,return,failure,conditional lengths,interactions,runtime,across-training-seed SD retained.',
    'curves':'Original figure retained; source experiment identified, no smoothing,10 episodes every65536 interactions.',
    'limitations':'Five training replications,small development checks,unequal earlier search,power proxy,post-inspection boundary repair and non-independent domain disclosed.',
}
receipt=dict(requirements=requirements,raw_csv_table_arithmetic_verified=True,numerical_rows_unchanged=True,
    paired_seed_analysis_verified=True,all_40_conditions_have_200_evaluation_episodes=True,
    D_censored_Q_episodes=1,no_visible_file_references=True,latex_structure_refs_assets_verified=True,
    original_predictions_runtime_sources_and_models_unchanged=True,part2_prose_words=words,
    new_training=False,pdf_compiled=False,verifier_sha256=sha(Path(__file__)),
    report_hashes={str(file.relative_to(ROOT)):sha(file) for file in [main_path,discussion_path,REPORT/'README.md',REPORT/'ai_assistance.tex']})
(HERE/'requirements_and_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
after=HERE/'after';after.mkdir(exist_ok=True)
for file in [main_path,discussion_path]:(after/file.name).write_bytes(file.read_bytes())
print(json.dumps(receipt,indent=2))
