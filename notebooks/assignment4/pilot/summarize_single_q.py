"""Validate and summarize the ordinary-Q replacement, separately from Double Q."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabular_q import TabularQ

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'results/assignment4/single_q_v1'


def main():
    rows=[];curves=[]
    reference=json.loads((ROOT/'results/assignment4/further_exploration/locked_validation_protocol.json').read_text())
    for seed in (42,7):
        directory=OUTPUT/f'B_single_q_{seed}_262144'
        if not (directory/'summary.json').exists():
            print('Ordinary-Q run still incomplete:',seed);return
        protocol=json.loads((directory/'protocol.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        final=json.loads((directory/'final_protocol.json').read_text())
        episodes=pd.read_csv(directory/'final_episodes.csv')
        assert protocol['updates_per_interaction']==1
        assert not any(protocol[k] for k in ('double_q','replay','eligibility_traces'))
        assert summary['training_updates']==summary['training_interactions']==262144
        assert episodes.seed.tolist()==list(range(740000,740200))
        assert len(episodes)==200 and np.array_equal(episodes.success.astype(bool),episodes.score>=2000)
        assert episodes.length.between(1,1000).all() and np.allclose(episodes.score%10,0)
        assert np.isclose(episodes.success.mean(),summary['success_rate'])
        assert np.isclose(episodes.score.mean(),summary['mean_score'])
        assert summary['learning_disabled_verified']
        path=directory/'checkpoint_262144.pkl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==final['model_sha256']==summary['model_sha256']
        with path.open('rb') as handle:agent=pickle.load(handle)
        assert type(agent) is TabularQ
        assert agent.update_calls==262144 and len(agent.table)==summary['states']
        assert not any(hasattr(agent,name) for name in ('other','buffer','traces'))
        assert all(v.shape==(5,) and np.isfinite(v).all() for v in agent.table.values())
        rows.append(dict(algorithm='Ordinary tabular Q-learning',training_seed=seed,
            interactions=262144,successes=int(episodes.success.sum()),episodes=200,
            success_rate=float(episodes.success.mean()),mean_score=float(episodes.score.mean()),
            mean_success_length=float(episodes[episodes.success==1].length.mean()),
            mean_failure_length=float(episodes[episodes.success==0].length.mean()),
            model_sha256=summary['model_sha256']))
        curve=pd.read_csv(directory/'progress.csv');curve['training_seed']=seed;curves.append(curve)
        ppo_dir=ROOT/'results/assignment4/further_exploration/confirmation'/f'aligned_ppo_{seed}_740000'
        if not (ppo_dir/'summary.json').exists():
            print('Paired PPO evaluation still incomplete:',seed);return
        ppo_protocol=json.loads((ppo_dir/'protocol.json').read_text())
        ppo_summary=json.loads((ppo_dir/'summary.json').read_text())
        ppo_episodes=pd.read_csv(ppo_dir/'episodes.csv')
        record=next(r for r in reference['models'] if r['kind']=='sb3' and r['training_seed']==seed)
        assert ppo_protocol['model_sha256']==record['sha256']
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
        assert ppo_protocol['seeds']==episodes.seed.tolist()==ppo_episodes.seed.tolist()
        assert len(ppo_episodes)==200 and ppo_summary['learning_disabled_verified']
        assert np.array_equal(ppo_episodes.success.astype(bool),ppo_episodes.score>=2000)
        assert ppo_episodes.length.between(1,1000).all() and np.allclose(ppo_episodes.score%10,0)
        assert np.isclose(ppo_episodes.success.mean(),ppo_summary['success_rate'])
        assert np.isclose(ppo_episodes.score.mean(),ppo_summary['mean_score'])
        rows.append(dict(algorithm='PPO (unchanged shared actor)',training_seed=seed,
            interactions=262144,successes=int(ppo_episodes.success.sum()),episodes=200,
            success_rate=float(ppo_episodes.success.mean()),mean_score=float(ppo_episodes.score.mean()),
            mean_success_length=float(ppo_episodes[ppo_episodes.success==1].length.mean()),
            mean_failure_length=float(ppo_episodes[ppo_episodes.success==0].length.mean()),
            model_sha256=record['sha256']))
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT/'comparison.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUTPUT/'learning_curves.csv',index=False)
    original=json.loads((ROOT/'results/assignment4/part2_controlled_v2/protocol.json').read_text())
    for name,expected in original['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    checks=dict(ordinary_q_models=2,ppo_models=2,episodes=800,common_evaluation_seeds=[740000,740199],
        one_q_table=True,q_updates_equal_environment_interactions=True,replay=False,double_q=False,
        eligibility_traces=False,all_final_models_frozen=True,ppo_weights_unchanged=True,
        original_registered_sources_and_predictions_unchanged=True,
        scope='B-only pilot with two training seeds; A/C/D and five-seed mechanism study not rerun.')
    (OUTPUT/'validation.json').write_text(json.dumps(checks,indent=2))
    figures=ROOT/'figures/assignment4/single_q_v1';figures.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(len(table));bars=ax.bar(x,table.success_rate*100,
        color=['tab:blue' if a.startswith('Ordinary') else 'tab:orange' for a in table.algorithm])
    ax.set_xticks(x,[f'{"Ordinary Q" if a.startswith("Ordinary") else "PPO"}\ntraining seed {s}'
                     for a,s in zip(table.algorithm,table.training_seed)])
    ax.set_ylim(0,105);ax.set_ylabel('Success rate (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.bar_label(bars,labels=[f'{n}/200' for n in table.successes],padding=4)
    ax.set_title('Part 1-aligned B pilot: fixed 262,144 interactions; 200 new episodes/model',fontsize=10)
    fig.text(.5,.01,'Sources: single_q_v1/comparison.csv and saved protocols; episode seeds 740000–740199',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(figures/'validation_success.png',dpi=160);plt.close(fig)
    print(table.to_string(index=False))
    print('Validated ordinary single-table Q and unchanged PPO; original predictions preserved.')


if __name__=='__main__':main()
