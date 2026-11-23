"""Reproduce the complete corrected search/main/frozen-evaluation sequence.

Use a new run prefix. This reproduces the registered seed lists; it does not
create a new independent test. Report updates require --update-report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def run(script,*args):
    command=[sys.executable,str(HERE/script),*map(str,args)]
    print('RUN '+subprocess.list2cmdline(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)

def main(args):
    if args.update_report:
        report=ROOT/'reports/assignment4/Sections/02_learning_settings.tex'
        if report.exists() and report.read_text(encoding='utf-8').startswith('% Evidence-focused'):
            raise ValueError('Preserve the reviewed report. Omit --update-report and use the report_review checks.')
    prefix=args.run_id.rstrip('/');directory=ROOT/'results/assignment4'/prefix
    if directory.exists():raise FileExistsError('Choose a new run prefix; historical records are preserved.')
    directory.mkdir(parents=True)
    plan=dict(arguments=vars(args),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Further development/reproduction with original prediction sheet preserved; final seeds reproduce the existing protocol, not independent validation.')
    (directory/'launch.json').write_text(json.dumps(plan,indent=2))
    screen=f'{prefix}/screen';replica=f'{prefix}/replica';training=f'{prefix}/main';final=f'{prefix}/final'
    run('cohort.py','--run-id',screen,'--workers',args.workers,'--settings','AB','--seeds','42',
        '--q-variants','baseline,motion,gamma995,bearing,alignment,gamma999,compact',
        '--ppo-variants','stable,long,allghost,careful,powerclock,explore,small',
        '--budget',262144,'--episodes',20,'--dev-start',1110000)
    run('select.py','shortlist','--runs',screen,'--output',f'{prefix}/shortlist.json')
    shortlist=json.loads((directory/'shortlist.json').read_text())['algorithms']
    q=','.join(item['variant'] for item in shortlist['Q-learning']);p=','.join(item['variant'] for item in shortlist['PPO'])
    run('cohort.py','--run-id',replica,'--workers',args.workers,'--settings','AB','--seeds','7',
        '--q-variants',q,'--ppo-variants',p,'--budget',262144,'--episodes',20,'--dev-start',1120000)
    run('select.py','lock','--runs',screen,replica,'--output',f'{prefix}/selected.json')
    selected=json.loads((directory/'selected.json').read_text())['algorithms']
    run('main_cohort.py','--selection-lock',f'{prefix}/selected.json','--run-id',training,'--workers',args.workers,
        '--seeds',args.main_seeds,'--budget',args.main_budget)
    try:
        run('evaluate_final.py','--training-run',training,'--run-id',final,'--workers',args.workers,'--stopping-rule','per_model')
    except subprocess.CalledProcessError:
        # The exact known D protocol interruption is diagnosed and recorded;
        # the strict finisher refuses any other missing cell or native flags.
        options=[] if args.update_report else ['--verify-only']
        run('finish_partial_report.py','--training-run',training,'--final-run',final,*options)
        return
    run('verify_native_replay.py','--final-run',final)
    if args.update_report:
        run('render_results.py','--training-run',training,'--final-run',final)
        run('refresh_discussion.py','--training-run',training,'--final-run',final)
        run('validate_delivery.py','--final-run',final)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--workers',type=int,default=min(8,os.process_cpu_count() or 1))
    parser.add_argument('--main-budget',type=int,choices=[262144,393216,524288],default=262144)
    parser.add_argument('--main-seeds',default='17,29,101,314,2718');parser.add_argument('--update-report',action='store_true')
    main(parser.parse_args())
