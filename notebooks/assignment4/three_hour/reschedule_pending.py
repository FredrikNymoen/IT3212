"""Run not-yet-started pilot jobs concurrently through the metadata-fixed engine.

Preserves the original job parameters and all frozen v1 sources. The only v2
engine change is dictionary construction after training; actual extra sources
and reassignment are recorded before execution. Existing job folders are never
overwritten, so an already running job cannot be duplicated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import cohort

study=cohort.study

def main(args):
    root=study.ROOT/'results/assignment4'/args.run_id
    original=json.loads((root/'protocol.json').read_text())
    first=(Path(__file__).with_name('run_study.py')).read_text()
    fixed=(Path(__file__).with_name('run_study_v2.py')).read_text()
    assert first.replace('result=dict(**job,training_interactions=budget','result=dict(job,training_interactions=budget')==fixed
    actual=study.sources();actual[str(Path(__file__).relative_to(study.ROOT))]=study.sha(__file__)
    hashes=dict(original['source_hashes']);hashes.update(actual)
    jobs=[]
    for old in original['jobs']:
        if old['setting']!=args.setting or (args.algorithm and old['algorithm']!=args.algorithm):continue
        if Path(old['directory']).exists():continue
        job=dict(old,source_hashes=hashes,runtime_reassignment='Metadata-fixed v2; exactly identical training logic, original seed/configuration and target directory.')
        jobs.append(job)
    for path,expected in hashes.items():assert study.sha(study.ROOT/path)==expected,path
    manifest=dict(original_protocol_sha256=study.sha(root/'protocol.json'),jobs=jobs,workers=args.workers,
        actual_source_hashes=hashes,reason='Use free CPU capacity for queued jobs; avoid serial tail. No existing job directories are overwritten.')
    path=root/f'reassignment_{args.setting}_{args.algorithm or "both"}.json'
    if path.exists():raise FileExistsError('Preserve the previous reassignment manifest.')
    study.write(path,manifest)
    for name in hashes:
        target=root/'reassignment_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes((study.ROOT/name).read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(study.worker,job) for job in jobs]):
            result=future.result()
            print('REASSIGNED DONE '+json.dumps({key:result[key] for key in ('algorithm','variant','setting','seed','training_seconds')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--setting',default='B')
    parser.add_argument('--algorithm',choices=['Q-learning','PPO']);parser.add_argument('--workers',type=int,default=2)
    main(parser.parse_args())
