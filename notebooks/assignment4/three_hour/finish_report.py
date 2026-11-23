"""Verify and render the completed fixed study; never train/select from test data."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/assignment4/three_hour_20261122'

def run(name,*args):
    print('REPORT STAGE '+name,flush=True)
    subprocess.run([sys.executable,str(HERE/name),*args],cwd=ROOT,check=True)

def main():
    while not (BASE/'final_v1/completed.json').exists():
        if (BASE/'final_v1/interrupted.json').exists():
            raise RuntimeError('Known interrupted protocol: use the verified partial-report workflow; do not wait for a fabricated completion marker.')
        time.sleep(2)
    final=(' --final-run three_hour_20261122/final_v1').split()
    run('verify_native_replay.py',*final)
    run('render_results.py','--training-run','three_hour_20261122/main_v1',*final)
    run('refresh_discussion.py')
    run('validate_delivery.py',*final)
    run('update_guides.py')
    run('validate_delivery.py',*final)
    (BASE/'report_pipeline_completed.json').write_text(json.dumps(dict(completed=True,pdf_compiled=False),indent=2))

if __name__=='__main__':main()
