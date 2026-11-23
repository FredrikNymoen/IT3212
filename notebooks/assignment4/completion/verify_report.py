"""Audit all primary report tables, full brief word count and prediction integrity."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]; REPORT=ROOT/'reports/assignment4'
BASE=ROOT/'results/assignment4'; OUT=BASE/'completion_analysis_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numbers(cell):return [float(s) for s in re.findall(r'-?\d+(?:\.\d+)?',cell)]
def check(cell,values):
    raw=re.findall(r'-?\d+(?:\.\d+)?',cell);assert len(raw)==len(values),(cell,values)
    for s,value in zip(raw,values):
        precision=len(s.split('.')[1]) if '.' in s else 0
        assert abs(float(s)-value)<=.5*10**(-precision)+1e-7,(cell,values)
def main():
    a=pd.read_csv(OUT/'agents.csv'); summary=json.loads((OUT/'summary.json').read_text())
    names=['02_learning_settings','03_mechanisms','04_episode_inspection','06_recommendations']
    contents=[(REPORT/'Sections'/f'{x}.tex').read_text(encoding='utf-8') for x in names]
    p2,p3,p4,p6=contents
    tables=lambda t:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',t,re.S)
    lines=lambda t:[[x.strip() for x in s.removesuffix('\\\\').strip().split('&')] for s in t.splitlines() if ' & ' in s]
    out=lines(tables(p2)[0])[1:];cost=lines(tables(p2)[1])[1:]
    assert len(out)==len(cost)==8
    for i,(condition,algorithm) in enumerate((c,g) for c in 'ABCD' for g in ['Q-learning','PPO']):
        row=next(r for r in summary['groups'] if r['study']==('D' if condition=='D' else 'P2') and r['condition']==condition and r['algorithm']==algorithm)
        assert out[i][:2]==cost[i][:2]==[condition,algorithm]
        for cell,field,m in [(out[i][2],'success_rate',100),(out[i][3],'mean_return',1),
                *[(cost[i][j+2],field,1) for j,field in enumerate(['success_length','failure_length','training_seconds','evaluation_seconds'])]]:
            vals=[row[field]*m]+([row[field+'_sd']*m] if row[field+'_sd'] is not None else [])
            check(cell,vals)
    roots=lines(tables(p2)[2])[1:];assert len(roots)==5
    for cells,seed in zip(roots,[17,29,101,314,2718]):
        assert int(cells[0])==seed
        for i,(condition,algorithm) in enumerate((c,g) for c in 'ABCD' for g in ['Q-learning','PPO']):
            row=a[a.study.eq('D' if condition=='D' else 'P2')&a.condition.eq(condition)&a.algorithm.eq(algorithm)&a.seed.eq(seed)].iloc[0]
            check(cells[i+1],[row.success_rate*100])
    paired=lines(tables(p2)[3])[1:];assert len(paired)==10
    for cells,(alg,seed) in zip(paired,[(g,s) for g in ['Q-learning','PPO'] for s in [17,29,101,314,2718]]):
        g=a[a.study.eq('D')&a.algorithm.eq(alg)&a.seed.eq(seed)].set_index('condition')
        clean=pd.read_csv(ROOT/g.loc['D_clean','evaluation_csv']);masked=pd.read_csv(ROOT/g.loc['D','evaluation_csv'])
        assert clean.evaluation_seed.tolist()==masked.evaluation_seed.tolist()==list(range(1600000,1600200))
        assert cells[:2]==[alg,str(seed)]
        check(cells[2],[100*clean.success.mean()]);check(cells[3],[100*masked.success.mean()])
        check(cells[4],[100*(masked.success.mean()-clean.success.mean())])
        check(cells[5],[int((clean.success.eq(1)&masked.success.eq(0)).sum())])
        check(cells[6],[int((clean.success.eq(0)&masked.success.eq(1)).sum())])
    assert all(r'\subsection{'+setting+':' in p2 for setting in 'ABCD')
    assert 'Images/part2_all_settings.png' in p2
    mechanism=lines(tables(p3)[0])[1:];assert len(mechanism)==9
    for cells,arm in zip(mechanism,['Q_base','Q_fast_epsilon','Q_gamma99','Q_alpha05','PPO_base','PPO_gamma99','PPO_entropy0','PPO_lr1e4','PPO_epochs2']):
        row=next(r for r in summary['groups'] if r['study']=='M3' and r['arm']==arm)
        check(cells[1],[row['success_rate']*100,row['success_rate_sd']*100])
        group=a[a.study.eq('M3')&a.arm.eq(arm)].set_index('seed')
        check(cells[2],[group.loc[s,'success_rate']*100 for s in [17,29,101,314,2718]])
        check(cells[3],[row['training_seconds'],row['training_seconds_sd']])
    diagnostic=lines(tables(p3)[1])[1:];assert len(diagnostic)==4
    d=pd.DataFrame(summary['diagnostic'])
    for cells,(arm,mode) in zip(diagnostic,[(x,m) for x in ['PPO_base','PPO_entropy0'] for m in ['argmax','sampled']]):
        g=d[d.arm.eq(arm)&d['mode'].eq(mode)]
        check(cells[2],[100*g.success_rate.mean(),100*g.success_rate.std(ddof=1)])
        check(cells[3],[g.mean_action_entropy.mean()])
    inspection=json.loads((BASE/'completion_inspection_v4/inspection.json').read_text())
    annotations=json.loads((BASE/'completion_inspection_v4/annotations.json').read_text())['annotations']
    panels=json.loads((BASE/'completion_figures/episode_figures.json').read_text())
    assert panels['decisions']==30 and len(panels['figures'])==1
    assert panels['source_sha256']==sha(BASE/'completion_inspection_v4/inspection.json')
    assert panels['annotation_sha256']==sha(BASE/'completion_inspection_v4/annotations.json')
    plotted={p['id']:p for p in panels['ledger']}
    assert len(plotted)==30 and p4.count(r'\begin{table}')==1 and p4.count(r'\begin{longtable}')==1
    assert p4.count(r'\begin{figure}')==1
    captions=re.findall(r'\\caption\{([^\n]+)',p4)
    assert all(len(c.split())<=40 for c in captions)
    for f in panels['figures']:
        pdf=REPORT/'Images'/Path(f['pdf']).name
        assert sha(pdf)==f['pdf_sha256']
        visible=' '.join(page.extract_text() for page in PdfReader(pdf).pages)
        for point in f['decisions']:
            assert point['id'] in visible
            assert all(label in visible for label in point['value_labels'])
            assert sha(ROOT/point['native_frame'])==point['native_frame_sha256']
    assert len(inspection['episodes'])==10 and len(annotations)==30
    for e in inspection['episodes']:
        assert len(e['decisions'])==3
        for point in e['decisions']:
            assert point['alternative']['full_seeded_action_prefix_verified']
            identity=e['case']['id']+'/'+str(point['decision'])
            assert p4.count(r'\textbf{'+identity+':}')==1
            row=p4.split(r'\textbf{'+identity+':}')[1].splitlines()[0]
            assert plotted[identity]['values']==point['values']
            assert plotted[identity]['value_labels']==[f'{v:.4f}' for v in point['values']]
            assert plotted[identity]['chosen_action']==['N','U','R','L','D'][point['action']]
            assert plotted[identity]['alternative_action']==['N','U','R','L','D'][point['alternative']['action']]
            assert next(x['interpretation'] for x in annotations if x['id']==identity)==plotted[identity]['interpretation']
            assert plotted[identity]['short_interpretation'] in row
            assert ', '.join(f'{v:.4f}' for v in point['values']) in row
            coord=lambda v:'('+','.join(f'{x:.0f}' for x in v)+')'
            for label,field in [('P','player_before'),('G','nearest_ghost_offset'),('F','food_offset')]:assert label+coord(point[field]) in row
            assert 'Chosen '+plotted[identity]['chosen_action']+r'$\to$'+coord(point['player_after']) in row
            assert 'Alt '+plotted[identity]['alternative_action']+r'$\to$'+coord(point['alternative']['player']) in row
            assert f'$p={point["encoded_power_proxy"]:.2f}$' in row and f'$s={point["score_before"]:g}$' in row
            assert f'gain {point["raw_reward"]:g}; L{point["lives_after"]}' in row
            assert f'gain {point["alternative"]["reward"]:g}; L{point["alternative"]["lives"]}' in row
    prose_counts={}
    for name,text in zip(names,contents):
        text=re.sub(r'(?<!\\)%[^\n]*','',text)
        text=re.sub(r'\\begin\{(table|figure|longtable)\}.*?\\end\{\1\}','',text,flags=re.S)
        text=re.sub(r'\\(?:label|setcounter)\{[^}]*\}(?:\{[^}]*\})?','',text)
        text=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?','',text)
        prose_counts[name]=len(re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*",text))
    # Conservative count includes headings, reference labels, inline math and numeric tokens.
    assert sum(prose_counts.values())+20<=2000,prose_counts
    main=(REPORT/'main.tex').read_text();brief=(REPORT/'evidence_brief.tex').read_text()
    combined_sections=['02_learning_settings','03_mechanisms','04_episode_inspection','05_human_review','06_recommendations']
    assert re.findall(r'\\input\{Sections/([^}]+)\.tex\}',main)==combined_sections
    assert re.findall(r'\\input\{Sections/([^}]+)\.tex\}',brief)==names
    active=[REPORT/'main.tex',REPORT/'evidence_brief.tex',REPORT/'Sections/05_human_review.tex',*[REPORT/'Sections'/f'{x}.tex' for x in names]]
    labels=[];refs=[]
    for path in active:
        text=path.read_text();assert not re.search(r'\[(?:Describe|Insert|Recommendation|Author)',text)
        assert not re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d{4}-\d{2}-\d{2}\b',text)
        assert not re.search(r'(?:results/|notebooks/|\.py\b|\.csv\b|\.npz\b|\.pkl\b)',text)
        stack=[]
        for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
            if kind=='begin':stack.append(env)
            else:assert stack.pop()==env,path
        assert not stack
        labels+=re.findall(r'\\label\{([^}]+)\}',text);refs+=re.findall(r'\\ref\{([^}]+)\}',text)
        for img in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):assert (REPORT/img).exists()
    assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
    for path in REPORT.rglob('*.tex'):
        if path.name=='01_predictions.tex':continue
        text=path.read_text()
        assert not re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\\date\{[^}]+\}',text),path
    assert sha(REPORT/'Sections/01_predictions.tex')=='17fcfc5a3d12c7abca0f7bc0ec49a3e4972e413c2cb356039db435acbc27961a'
    proof=json.loads((OUT/'records_verification.json').read_text());assert proof['primary_episodes']==19000 and proof['secondary_episodes']==1000
    review=(REPORT/'Sections/05_human_review.tex').read_text()
    review_state=json.loads((REPORT/'review_input.json').read_text())
    review_words=len(re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*",re.sub(r'\\[A-Za-z]+','',review)))
    assert review_words<=1000 and review_state['status']=='reviewed'
    codes={'Failure avoidance':'F','Robustness to change':'R','Explainability':'E','Time efficiency':'T'}
    for name,ranking in review_state['initial_rankings'].items():
        assert len(ranking)==4 and set(ranking)==set(codes)
        assert name+' & $'+'>'.join(codes[x] for x in ranking)+'$' in review
    assert '$'+'>'.join(codes[x] for x in review_state['final_priorities'])+'$' in review
    assert review_state['actual_negotiation_record'] and review_state['group_rationale_norwegian']
    assert review_state['assessment_endorsement_norwegian']
    assert all(name in review for name in review_state['members_named_by_user'])
    assert 'broadly supported' in review and 'Not supplied' not in review and 'review draft' not in review
    result=dict(primary_tables_independently_recomputed=True,all_training_roots_retained=True,decisions=30,original_predictions_unchanged=True,
        calendar_dates_absent_from_report=True,parallel_A_B_C_D_sections=True,paired_D_table_verified=True,
        human_review_in_combined_report=True,separate_evidence_brief_retained=True,
        part4_findings_figures=1,part4_recorded_decisions=30,part4_caption_word_limit=40,
        prose_words_by_part=prose_counts,conservative_brief_words=sum(prose_counts.values())+20,
        counting_rule='Excludes table and longtable contents and figure captions; includes headings, inline mathematics/numbers and conservative title allowance.',
        human_review='Completed using member rankings, final group priorities, rationale and broad support supplied by the user.',review_words=review_words,
        review_input_sha256=sha(REPORT/'review_input.json'),review_text_sha256=sha(REPORT/'Sections/05_human_review.tex'),
        report_hashes={str(p.relative_to(ROOT)):sha(p) for p in active},verifier_sha256=sha(Path(__file__)))
    (OUT/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
