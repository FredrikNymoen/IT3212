"""Recover completed v1 training metadata after a duplicate-keyword error.

Models, sources, native transitions, development CSVs and curves are never
edited. Every checkpoint and episode CSV is checked before metadata recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import candidates,additions,compact_candidates

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def recover(name):
    directory=ROOT/'results/assignment4'/name
    protocol=json.loads((directory/'protocol.json').read_text());recovered=[];pending=[];checks={}
    for path,expected in protocol['source_hashes'].items():assert sha(ROOT/path)==expected,path
    for job in protocol['jobs']:
        folder=Path(job['directory']);curve_path=folder/'curves.csv'
        if not curve_path.exists():pending.append(str(folder));continue
        curve=pd.read_csv(curve_path)
        if curve.interactions.tolist()!=job['checkpoints'] or (job['algorithm']=='Q-learning' and not (folder/'training.csv').exists()):
            pending.append(str(folder));continue
        checkpoints=curve.astype(object).where(pd.notna(curve),None).to_dict('records')
        for point in checkpoints:
            model=ROOT/point['model'];assert sha(model)==point['model_sha256']
            frame=pd.read_csv(folder/f'development_{point["interactions"]}.csv')
            assert len(frame)==20 and frame.evaluation_seed.tolist()==job['development_seeds']
            assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
            assert np.isclose(frame.success.mean(),point['success_rate']) and np.isclose(frame.score.mean(),point['mean_score'])
            assert np.isclose(frame.length.mean(),point['mean_length'])
            checks[str(model.relative_to(ROOT))]=point['model_sha256']
            if job['algorithm']=='Q-learning':
                with model.open('rb') as handle:agent=pickle.load(handle)
                assert type(agent).update is candidates.TabularQ.update and agent.update_calls==point['interactions']
                assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
        result=dict(job,training_interactions=job['budget'],training_seconds=float(curve.training_seconds.iloc[-1]),
            checkpoints=checkpoints,ordinary_q=True if job['algorithm']=='Q-learning' else None,
            recovered_metadata=True,recovery_reason='Duplicate checkpoints keyword after full training. Verified unchanged saved checkpoints and all development episodes; no retraining or outcome edits.')
        target=folder/'metrics.json'
        if target.exists():
            current=json.loads(target.read_text())
            if current.get('recovered_metadata'):
                assert current==result
            else:
                actual_job=json.loads((folder/'job.json').read_text())
                for key in ('algorithm','variant','setting','seed','budget','directory','development_seeds'):
                    assert current[key]==actual_job[key]==job[key],key
                assert current['training_interactions']==job['budget']
                assert np.isclose(current['training_seconds'],result['training_seconds'])
                assert current['checkpoints']==result['checkpoints']
                for path,expected in current['source_hashes'].items():assert sha(ROOT/path)==expected,path
        else:write(target,result)
        recovered.append(job)
    manifest=dict(recovery_source_sha256=sha(__file__),recovered_jobs=len(recovered),pending_jobs=pending,
        verified_model_hashes=checks,original_sources_and_outcomes_unchanged=True)
    write(directory/'metadata_recovery_v2.json',manifest)
    if not pending:
        frame=pd.concat([pd.read_csv(Path(job['directory'])/'curves.csv').assign(algorithm=job['algorithm'],
            variant=job['variant'],setting=job['setting'],seed=job['seed']) for job in protocol['jobs']],ignore_index=True)
        frame.to_csv(directory/'comparison.csv',index=False)
        write(directory/'completed.json',dict(jobs=len(recovered),training_interactions=sum(job['budget'] for job in recovered),
            metadata_recovery=True,all_checkpoints_and_development_episodes_verified=True))
    print(json.dumps(dict(run=name,recovered=len(recovered),pending=len(pending))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',required=True)
    for name in parser.parse_args().runs:recover(name)
