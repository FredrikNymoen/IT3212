"""Independent raw-CSV aggregation and static LaTeX checks; no PDF build."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as stats
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'reports/assignment4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_cell(cell,values):
    numbers=re.findall(r'-?\d+(?:\.\d+)?',cell)
    assert len(numbers)==len(values),(cell,values)
    for number,value in zip(numbers,values):
        decimals=len(number.split('.')[1]) if '.' in number else 0
        assert abs(float(number)-value)<=.5*10**(-decimals)+1e-6,(cell,value)
def summary(values):
    values=[float(value) for value in values if value is not None and not pd.isna(value)]
    if not values:return []
    return [stats.mean(values)]+([stats.stdev(values)] if len(values)>1 else [])

def main(args):
    output=ROOT/'results/assignment4'/args.final_run
    lock=json.loads((output/'locked.json').read_text())
    training=ROOT/'results/assignment4'/lock['training_run']
    protocol=json.loads((training/'protocol.json').read_text())
    assert sha(ROOT/protocol['selection_lock'])==protocol['selection_lock_sha256']
    selected=json.loads((ROOT/protocol['selection_lock']).read_text())
    for path,expected in selected['development_sources'].items():assert sha(ROOT/path)==expected,path
    for path,expected in lock['source_hashes'].items():assert sha(ROOT/path)==expected,path
    assert sha(REPORT/'Sections/01_predictions.tex')=='17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a'
    rows=[]
    for job in lock['jobs']:
        if not (Path(job['directory'])/'metrics.json').exists():
            assert (output/'interrupted.json').exists() and job['condition']=='D' and job['algorithm']=='Q-learning' and job['seed']==314
            continue
        frame=pd.read_csv(Path(job['directory'])/'evaluation.csv')
        rows.append(dict(condition=job['condition'],algorithm=job['algorithm'],seed=job['seed'],
            success=100*frame.success.mean(),score=frame.score.mean(),task_return=frame.task_return.mean(),
            success_length=frame.loc[frame.success==1,'length'].mean(),failure_length=frame.loc[frame.success==0,'length'].mean(),
            training_seconds=job['training_seconds'],evaluation_seconds=json.loads((Path(job['directory'])/'metrics.json').read_text())['evaluation_seconds']))
    raw=pd.DataFrame(rows);text=(REPORT/'Sections/02_learning_settings.tex').read_text(encoding='utf-8')
    tables=re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',text,re.S)
    assert len(tables)>=2
    parsed=lambda table:[[cell.strip() for cell in line.strip().removesuffix('\\\\').split('&')]
        for line in table.splitlines() if re.match(r'^[ABCD] &',line)]
    table_rows=[parsed(tables[0]),parsed(tables[1])];assert all(len(rows)==8 for rows in table_rows)
    index=0
    for condition in 'ABCD':
        for algorithm in ['Q-learning','PPO']:
            group=raw[(raw.condition==condition)&(raw.algorithm==algorithm)]
            if condition=='D' and algorithm=='Q-learning' and len(group)<5:
                assert len(group)==4
                assert table_rows[0][index][2:]==['--']*3 and table_rows[1][index][2:]==['--']*4
                index+=1;continue
            for cells,fields in [(table_rows[0][index],['success','score','task_return']),
                                (table_rows[1][index],['success_length','failure_length','training_seconds','evaluation_seconds'])]:
                assert cells[:2]==[condition,algorithm]
                for cell,field in zip(cells[2:],fields):check_cell(cell,summary(group[field]))
            index+=1
    labels=[];references=[]
    for path in REPORT.rglob('*.tex'):
        content=re.sub(r'(?<!\\)%[^\n]*','',path.read_text(encoding='utf-8'));stack=[]
        for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',content):
            if kind=='begin':stack.append(name)
            else:assert stack.pop()==name,path
        assert not stack,path
        braces=re.sub(r'\\[{}]','',content);depth=0
        for char in braces:
            depth+=(char=='{')-(char=='}');assert depth>=0,path
        assert depth==0 and len(re.findall(r'(?<!\\)\$',content))%2==0,path
        labels.extend(re.findall(r'\\label\{([^}]+)\}',content));references.extend(re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',content))
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',content):assert (REPORT/image).exists(),image
    assert len(labels)==len(set(labels)) and not set(references)-set(labels)
    prose=text+(REPORT/'Sections/02_discussion.tex').read_text(encoding='utf-8')
    prose=re.sub(r'(?<!\\)%[^\n]*','',prose)
    prose=re.sub(r'\\begin\{(table|figure)\}.*?\\end\{\1\}','',prose,flags=re.S)
    prose=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?','',prose)
    words=len(re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*",prose))
    manifest=dict(raw_episode_tables_independently_verified=True,latex_structure_assets_refs_verified=True,
        original_predictions_unchanged=True,main_seeds=protocol['training_seeds'],main_budget=protocol['budget'],
        final_episodes=len(rows)*200,approximate_part2_prose_words=words,pdf_compiled=False,
        report_hashes={str(path.relative_to(ROOT)):sha(path) for path in [REPORT/'Sections/02_learning_settings.tex',
            REPORT/'Sections/02_discussion.tex',REPORT/'ai_assistance.tex',REPORT/'README.md']})
    (output/'delivery_validation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final-run',required=True);main(parser.parse_args())
