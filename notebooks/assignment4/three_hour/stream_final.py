"""Evaluate completed models while independent training jobs finish.

Each checkpoint is chosen only from that model's source-development CSV and
locked before its first test episode. Global plan/code/seed lists precede all
test episodes. Test outcomes never enter another model's selection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import time
import pandas as pd
import evaluate_final as evaluator
from fast_evaluation_v2 import FastStudyTask,FastLowBitsTask

study=evaluator.study
evaluator.StudyTask=FastStudyTask;evaluator.LowBitsTask=FastLowBitsTask

def worker(job):return evaluator.work(job)

def main(args):
    training=study.ROOT/'results/assignment4'/args.training_run
    protocol=json.loads((training/'protocol.json').read_text())
    check_path=study.ROOT/'results/assignment4/three_hour_20261122/engineering/fast_evaluation_v2.json'
    check=json.loads(check_path.read_text())
    for path,expected in check['sources'].items():assert study.sha(study.ROOT/path)==expected,path
    hashes=dict(protocol['source_hashes'])
    for path in [Path(__file__),Path(evaluator.__file__),Path(__file__).with_name('fast_evaluation.py'),Path(__file__).with_name('fast_evaluation_v2.py')]:
        hashes[str(path.relative_to(study.ROOT))]=study.sha(path)
    for path,expected in hashes.items():assert study.sha(study.ROOT/path)==expected,path
    output=study.ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    plan=dict(training_run=args.training_run,training_protocol_sha256=study.sha(training/'protocol.json'),
        source_hashes=hashes,evaluation_seeds=list(evaluator.SEEDS),stopping_rule='per_model',
        selection='Each completed model uses only source-condition development success, game score, earlier checkpoint. D inherits B. Per-case lock precedes test episodes; all five seeds retained.',
        scope='Streaming changes scheduling only, not the predeclared per-model stopping rule, task, policy or test seeds.',workers=args.workers)
    plan_path=output/'evaluation_plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:study.write(plan_path,plan)
    registered=set();jobs=[];results=[];pending={};checkpoints={'Q-learning':{},'PPO':{}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(results)<8*len(protocol['training_seeds']):
            for source in protocol['jobs']:
                key=(source['algorithm'],source['setting'],source['seed'])
                metrics_path=Path(source['directory'])/'metrics.json'
                if key in registered or not metrics_path.exists():continue
                metrics=json.loads(metrics_path.read_text())
                points=[point for point in metrics['checkpoints'] if point['interactions']>0]
                row=sorted(points,key=lambda x:(-x['success_rate'],-x['mean_score'],x['interactions']))[0]
                checkpoints[source['algorithm']][f'{source["seed"]}_{source["setting"]}']=row['interactions']
                for condition in [source['setting']]+(['D'] if source['setting']=='B' else []):
                    job=dict(algorithm=source['algorithm'],variant=source['variant'],seed=source['seed'],condition=condition,
                        training_setting=source['setting'],training_interactions=source['budget'],
                        checkpoint_interactions=row['interactions'],training_seconds=metrics['training_seconds'] if condition!='D' else 0.,
                        model=row['model'],model_sha256=row['model_sha256'],source_hashes=hashes,
                        directory=str(output/f'{condition}_{source["algorithm"]}_{source["seed"]}'))
                    assert study.sha(study.ROOT/job['model'])==job['model_sha256']
                    study.write(output/'case_locks'/f'{condition}_{source["algorithm"]}_{source["seed"]}.json',job)
                    jobs.append(job);pending[pool.submit(worker,job)]=job
                registered.add(key)
            if pending:
                done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    job=pending.pop(future);result=future.result();results.append(result)
                    print('FINAL '+json.dumps({key:result[key] for key in ('condition','algorithm','seed','success_rate','mean_score','evaluation_seconds')}),flush=True)
            else:time.sleep(2)
            if len(registered)==len(protocol['jobs']) and (training/'comparison.csv').exists() and not (output/'locked.json').exists():
                study.write(output/'locked.json',dict(training_run=args.training_run,source_hashes=hashes,jobs=jobs,
                    evaluation_seeds=list(evaluator.SEEDS),checkpoints=checkpoints,stopping_rule='per_model',
                    selection=plan['selection'],development_export_sha256=study.sha(training/'comparison.csv'),
                    scope=plan['scope'],per_case_pre_evaluation_locks=True))
        assert len(registered)==len(protocol['jobs']) and len(results)==40
    assert (output/'locked.json').exists()
    pd.DataFrame([{key:value for key,value in result.items() if key!='source_hashes'} for result in results]).to_csv(output/'comparison.csv',index=False)
    study.write(output/'completed.json',dict(conditions=len(results),episodes=sum(x['episodes'] for x in results),
        native_scores_verified=True,frozen_parameters_verified=True,per_case_locks_before_evaluation=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--workers',type=int,default=8);main(parser.parse_args())
