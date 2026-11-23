"""Record the known protocol interruption and report verified complete cases."""
import argparse,json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/assignment4/three_hour_20261122'

def run(name,*args):
    print('REPORT STAGE '+name,flush=True)
    subprocess.run([sys.executable,str(HERE/name),*args],cwd=ROOT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-run',default='three_hour_20261122/final_v1')
    parser.add_argument('--training-run',default='three_hour_20261122/main_v1')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    output=ROOT/'results/assignment4'/args.final_run;lock=json.loads((output/'locked.json').read_text())
    assert lock['training_run']==args.training_run
    records=[];missing=[]
    for job in lock['jobs']:
        path=Path(job['directory'])/'metrics.json'
        if path.exists():records.append(json.loads(path.read_text()))
        else:missing.append({key:job[key] for key in ('condition','algorithm','seed')})
    assert len(records)==39 and missing==[dict(condition='D',algorithm='Q-learning',seed=314)]
    if not (output/'map_guard_native_flags.json').exists():
        run('diagnose_map_guard_v2.py','--final-run',args.final_run)
    flags=json.loads((output/'map_guard_native_flags.json').read_text())
    assert flags['native_score']==1760 and not flags['terminated'] and flags['info']['lives']==2
    interruption=dict(excluded_cases=missing,complete_experiment=False,completed_cases=39,
        completed_episodes=7800,reason='Native maze1->0 at1760points, decision975, seed1150094, lives2, nonterminal. Original guard correctly rejects playing on another map. No posthoc failure label or replacement model.',
        full_Q_D_aggregate_suppressed=True,new_training=False,original_protocol_preserved=True)
    (output/'interrupted.json').write_text(json.dumps(interruption,indent=2))
    pd.DataFrame([{key:value for key,value in row.items() if key!='source_hashes'} for row in records]).to_csv(output/'comparison.csv',index=False)
    final=['--final-run',args.final_run]
    run('verify_native_replay.py',*final)
    if args.verify_only:
        from render_results import verify
        verify(output,ROOT/'results/assignment4'/args.training_run)
        print('Verified interrupted experiment; report unchanged.');sys.exit(0)
    run('render_results.py','--training-run',args.training_run,*final)
    run('refresh_discussion.py','--training-run',args.training_run,*final)
    run('validate_delivery.py',*final)
    (output.parent/'report_pipeline_completed.json').write_text(json.dumps(dict(report_completed=True,experiment_complete=False,excluded_cases=missing,pdf_compiled=False),indent=2))
