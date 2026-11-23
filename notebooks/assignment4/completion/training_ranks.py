"""Test the prior training-return ranking prediction using original selected agents.

Use the last20 completed training episodes BEFORE each selected checkpoint.
PPO's four monitor streams are aligned by4*cumulative stream decisions, not
wall timestamps. Compare ranks within condition/algorithm (five seeds), never
pool C's different reward with A/B. This is descriptive, not model selection.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'synthesis'))
from part2_evidence import sha,local_path


def main():
    base=ROOT/'results/assignment4/three_hour_20261122';lock_path=base/'final_v1/locked.json'
    lock=json.loads(lock_path.read_text());training=json.loads((base/'main_v1/protocol.json').read_text())
    rows=[];hashes={str(lock_path.relative_to(ROOT)):sha(lock_path)}
    for job in lock['jobs']:
        if job['condition'] not in 'ABC':continue
        source=next(r for r in training['jobs'] if r['algorithm']==job['algorithm'] and r['setting']==job['condition'] and r['seed']==job['seed'])
        directory=local_path(source['directory']);point=job['checkpoint_interactions']
        if job['algorithm']=='Q-learning':
            file=directory/'training.csv';frame=pd.read_csv(file);hashes[str(file.relative_to(ROOT))]=sha(file)
            frame=frame.rename(columns={'length':'l'});frame['finish_interactions']=frame.interactions;frame['stream']=0
        else:
            streams=[]
            for index in range(4):
                file=directory/f'training_{index}.monitor.csv';part=pd.read_csv(file,comment='#')
                hashes[str(file.relative_to(ROOT))]=sha(file);part['finish_interactions']=4*part.l.cumsum();part['stream']=index
                streams.append(part)
            frame=pd.concat(streams,ignore_index=True)
        frame=frame[frame.finish_interactions.le(point)].sort_values(['finish_interactions','stream']).tail(20)
        assert len(frame)==20
        returns=frame.score-(.1*frame.l if job['condition']=='C' else 0)
        test=pd.read_csv(local_path(job['directory'])/'evaluation.csv');assert len(test)==200
        rows.append(dict(condition=job['condition'],algorithm=job['algorithm'],seed=job['seed'],checkpoint=point,
            training_window_episodes=20,training_return=float(returns.mean()),training_game_score=float(frame.score.mean()),
            evaluation_success=float(test.success.mean()),evaluation_return=float(test.task_return.mean()),
            selected_model_sha256=job['model_sha256']))
    frame=pd.DataFrame(rows);summaries=[]
    for (condition,algorithm),group in frame.groupby(['condition','algorithm']):
        a=group.training_return.rank(method='average');b=group.evaluation_success.rank(method='average')
        rho=float(a.corr(b)) if a.nunique()>1 and b.nunique()>1 else None
        winner=group.sort_values(['training_return','seed'],ascending=[False,True]).iloc[0]
        summaries.append(dict(condition=condition,algorithm=algorithm,n_training_seeds=5,descriptive_spearman=rho,
            top_training_seed=int(winner.seed),its_evaluation_success=float(winner.evaluation_success),
            best_evaluation_success=float(group.evaluation_success.max()),
            top_training_is_also_evaluation_leader=bool(winner.evaluation_success==group.evaluation_success.max())))
    output=ROOT/'results/assignment4/completion_training_ranks_v1';output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'agents.csv',index=False)
    (output/'analysis.json').write_text(json.dumps(dict(groups=summaries,source_hashes=hashes,source_sha256=sha(__file__),
        caveat='Changing policies/exploration during training and small five-seed groups. This is not a frozen-policy training test, significance claim, or held-out model selection.'),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
