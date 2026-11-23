"""Package required records, not all optional traces; verify every ZIP entry."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/assignment4'
sys.path.insert(0,str(ROOT/'notebooks/assignment4/synthesis'))
from part2_evidence import local_path
TARGET=BASE/'submission/assignment4_submission.zip'
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def stream_digest(f):
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def required_traces():
    inspection=read(BASE/'completion_inspection_v4/inspection.json')
    result=[]
    for e in inspection['episodes']:
        result.append((local_path(e['case']['job']['directory'])/'evaluation_traces.npz').resolve())
    assert len(result)==10 and len(set(result))==8
    return set(result)
def verify():
    with zipfile.ZipFile(TARGET) as z:
        manifest=json.loads(z.read('submission_manifest.json'));names=set(z.namelist())
        assert len(names)==len(z.namelist())
        assert names=={x['path'] for x in manifest['files']}|{'submission_manifest.json','START_HERE.md'}
        for row in manifest['files']:
            assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
            assert z.getinfo(row['path']).file_size==row['bytes']
            with z.open(row['path']) as file:assert stream_digest(file)==row['sha256'],row['path']
        inspection=json.loads(z.read('results/assignment4/completion_inspection_v4/inspection.json'))
        for episode in inspection['episodes']:
            path=local_path(episode['case']['job']['directory'])/'evaluation_traces.npz'
            relative=path.relative_to(ROOT).as_posix()
            with np.load(io.BytesIO(z.read(relative))) as arrays:
                mask=arrays['episode']==episode['case']['episode_index']
                assert int(mask.sum())==episode['case']['length']
                assert arrays['ram'][mask].shape[1]==128 and arrays['values'][mask].shape[1]==5
                assert float(arrays['raw_reward'][mask].sum())==episode['case']['score']
        agents=pd.read_csv(io.BytesIO(z.read('results/assignment4/completion_analysis_v1/agents.csv')))
        assert len(agents)==95
        for r in agents.to_dict('records'):
            key=Path(r['evaluation_csv']).as_posix();model=Path(r['model']).as_posix()
            assert hashlib.sha256(z.read(model)).hexdigest()==r['model_sha256']
            data=pd.read_csv(io.BytesIO(z.read(key)))
            assert len(data)==200 and np.isclose(data.success.mean(),r['success_rate'])
            assert hashlib.sha256(z.read(key)).hexdigest()==r['evaluation_csv_sha256']
    receipt=dict(zip=str(TARGET.relative_to(ROOT)),bytes=TARGET.stat().st_size,sha256=digest(TARGET),
                 entries=len(names),primary_episode_tables=95,primary_episodes=19000,
                 required_full_trace_files=8,required_analyzed_episodes=10,all_entries_SHA256_verified=True,human_review='deferred')
    TARGET.with_suffix('.verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));return receipt

def build():
    report_check=read(BASE/'completion_analysis_v1/report_verification.json')
    assert report_check['conservative_brief_words']<=2000 and report_check['decisions']==30
    records=read(BASE/'completion_analysis_v1/records_verification.json')
    assert records['primary_episodes']==19000 and records['secondary_episodes']==1000
    # Portable compact evidence remains tracked even though raw results are ignored.
    summary=read(BASE/'completion_analysis_v1/summary.json')
    agents=pd.read_csv(BASE/'completion_analysis_v1/agents.csv').replace({np.nan:None}).to_dict('records')
    snapshot=dict(summary=summary,agents=agents,report_verification=report_check,
                  trace_audit=dict(primary_decisions=records['primary_decisions'],secondary_decisions=records['secondary_decisions'],
                                   native_replays=records['native_replays'],complete_archive_trace_files=115),
                  scope='Derived compact evidence; raw episode logs, models and ten analyzed full traces accompany the ZIP. Human review deferred.')
    (ROOT/'notebooks/assignment4/completion/evidence.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    files=set();suffixes={'.py','.ipynb','.json','.md','.txt','.tex','.png','.pdf'}
    for directory in [ROOT/'notebooks/assignment4',ROOT/'reports/assignment4',ROOT/'output/pdf']:
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix.lower() in suffixes and not any(x in p.parts for x in ['__pycache__','.ipynb_checkpoints']):
                if p.parent==ROOT/'reports/assignment4' and p.suffix=='.pdf':continue
                files.add(p.resolve())
    files.update((ROOT/n).resolve() for n in ['requirements.txt','IT3212 - Assignment 4.pdf','clarification-assingment4 (1).pdf'])
    required=required_traces()
    directories=['three_hour_20261122','clarified_D_v1','completion_part3_v1','completion_diagnostic_v1',
                 'completion_inspection_v4','completion_prediction_probes_v1','completion_training_ranks_v1',
                 'completion_analysis_v1','completion_engineering_v1','completion_report_before']
    for name in directories:
        for p in (BASE/name).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if (p.name.endswith('_traces.npz') or p.name=='traces.npz') and p.resolve() not in required:continue
            files.add(p.resolve())
    for name in ['completion_pause_state.json','completion_active_work.json']:
        if (BASE/name).exists():files.add((BASE/name).resolve())
    files.add((BASE/'completion_latex_build/compiled.json').resolve())
    assert required<=files
    for r in agents:
        assert (ROOT/r['model']).resolve() in files and (ROOT/r['evaluation_csv']).resolve() in files
    file_rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=file_rows,
        primary_episodes=19000,secondary_episodes=1000,required_analyzed_episodes=10,annotated_decisions=30,
        omitted='Optional step-level arrays for non-analyzed models; complete local full-audit SHA receipt retained.',
        submission_status='Part5 human review is explicitly deferred; do not present this ZIP as a completed human review.')
    start='''# Assignment 4 evidence package\n\nOpen output/pdf/assignment4_evidence_brief.pdf first. The original predictions\nplus later annotations and the AI record are separate PDFs. The human review\nis explicitly DEFERRED at the team\'s request: complete the actual discussion,\nindividual priority rankings and negotiated priorities before submission.\nRetain the original prediction submission timestamp separately.\n\nFor code, reproduction, exact run mapping and all limitations, read\nnotebooks/assignment4/completion/README.md and reports/assignment4/README.md.\nThe actual algorithms are q_learning.py and ppo.py. No DQN is used.\n\nPrimary D is the instructor-clarified fixed OBJECT-FEATURE observation\nobstruction, not historical forced-map D. The game stays unchanged.\nThe package retains full step traces for all TEN analyzed episodes and raw\nepisode-level metrics for all19,000 primary and1,000 secondary episodes.\nOptional full traces for other episodes remain in the local archive; their\nchecksums and the complete independent verification receipt are included.\n\nExact dependency versions are recorded; no virtual environment, credentials,\n.git directory or ALE ROM is included. Reproduce from the extracted root on\nthe documented Windows CPU environment. Supplied seeds reproduce existing\nexperiments, not independent new tests. The canonical report is never tuned\nor overwritten by the training reproduction command.\n'''
    TARGET.parent.mkdir(parents=True,exist_ok=True);temporary=TARGET.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',allowZip64=True) as z:
        z.writestr('START_HERE.md',start,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('submission_manifest.json',json.dumps(manifest,indent=2),compress_type=zipfile.ZIP_DEFLATED)
        for p,row in zip(sorted(files),file_rows):
            z.write(p,row['path'],compress_type=zipfile.ZIP_STORED if p.suffix in {'.npz','.zip','.png','.pdf'} else zipfile.ZIP_DEFLATED,compresslevel=6)
    temporary.replace(TARGET);verify()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else build()
