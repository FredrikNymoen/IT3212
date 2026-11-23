"""Summarize measured exploratory runs without selecting from final-test data."""
import hashlib
import json
import sys
import platform
from importlib import metadata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
OUTPUT=ROOT/'results/assignment4/further_exploration'


def main():
    rows=[];selected=[]
    for directory in sorted(OUTPUT.iterdir()):
        path=directory/'progress.csv'
        if not path.is_file():continue
        frame=pd.read_csv(path);frame.insert(0,'run',directory.name)
        frame['complete']=(directory/'summary.csv').exists()
        rows.extend(frame.to_dict('records'))
        greedy=frame[frame['mode']=='greedy'] if 'mode' in frame else frame
        winner=greedy.sort_values(['success_rate','mean_score','interactions'],ascending=[False,False,True]).iloc[0]
        selected.append(winner.to_dict())
    all_rows=pd.DataFrame(rows)
    all_rows.to_csv(OUTPUT/'all_progress.csv',index=False)
    pd.DataFrame(selected).to_csv(OUTPUT/'selected_by_development.csv',index=False)
    confirmations=[]
    for path in sorted((OUTPUT/'confirmation').glob('*/summary.json')):
        confirmations.append(json.loads(path.read_text()))
    pd.DataFrame(confirmations).to_csv(OUTPUT/'confirmation_summary.csv',index=False)
    checks=[]
    for directory in sorted(OUTPUT.iterdir()):
        if not (directory/'progress.csv').is_file():continue
        progress=pd.read_csv(directory/'progress.csv')
        for row in progress.to_dict('records'):
            mode=f'_{row["mode"]}' if 'mode' in row else ''
            episode_path=directory/f'development_{int(row["interactions"])}{mode}.csv'
            episodes=pd.read_csv(episode_path)
            assert len(episodes)==20
            assert np.array_equal(episodes.success.astype(bool),episodes.score>=2000)
            assert episodes.length.between(1,1000).all()
            assert np.allclose(episodes.score%10,0)
            assert np.isclose(episodes.score.mean(),row['mean_score'])
            assert np.isclose(episodes.success.mean(),row['success_rate'])
            checks.append(str(episode_path.relative_to(ROOT)))
        sources=[*directory.glob('*.py'),*directory.glob('maze_geometry.json')]
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (directory/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUTPUT/'validation_summary.json').write_text(json.dumps(dict(
        validated_development_files=len(checks),validated_episodes=20*len(checks),
        score_success_horizon_and_means_consistent=True,files=checks),indent=2))
    environment=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        packages={name:metadata.version(name) for name in ('numpy','pandas','torch','gymnasium','ale-py','stable-baselines3')},
        ppo_torch_threads=1,ppo_numerical_thread_limit=1,q_thread_limit='Not explicitly limited',
        timing_note='Concurrent development jobs; not isolated computation-time comparisons.')
    (OUTPUT/'environment.json').write_text(json.dumps(environment,indent=2))
    protocol=json.loads((ROOT/'results/assignment4/part2_controlled_v2/protocol.json').read_text())
    verified={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
              for name,expected in protocol['source_hashes'].items()}
    assert all(verified.values()),verified
    (OUTPUT/'registered_sources_unchanged.json').write_text(json.dumps(verified,indent=2))
    sys.path.insert(0,str(HERE))
    from mspacman_env import FOOD_MASKS
    geometry=json.loads((Path(__file__).with_name('maze_geometry.json')).read_text())
    correct=np.asarray(geometry['grids'],dtype=bool).reshape(2,-1)
    mismatch=[np.flatnonzero(FOOD_MASKS[i]!=correct[i]).tolist() for i in range(2)]
    (OUTPUT/'geometry_audit/mask_differences.json').write_text(json.dumps(dict(
        mismatched_cells=mismatch,counts=[len(x) for x in mismatch],
        original_registered_sources_preserved=True),indent=2))
    figure_dir=ROOT/'figures/assignment4/further_exploration';figure_dir.mkdir(parents=True,exist_ok=True)
    styles=[('q_relative_lambda0.0_42_524288','Q one-step, relative, seed 42','tab:gray'),
        ('q_phase_lambda0.8_42_524288','Q traces + phase, seed 42','tab:blue'),
        ('q_phase_lambda0.8_7_524288','Q traces + phase, seed 7','tab:cyan'),
        ('q_replay_lambda0.0_42_262144','Double Q + replay, seed 42','tab:purple'),
        ('q_replay_lambda0.0_7_262144','Double Q + replay, seed 7','tab:pink'),
        ('q_goal_replay_lambda0.0_42_262144','Food-relative Double Q, seed 42','tab:olive'),
        ('q_goal_replay_lambda0.0_7_262144','Food-relative Double Q, seed 7','tab:brown'),
        ('reference_base_42_262144','PPO base features, seed 42','tab:gray'),
        ('reference_rich_42_524288','PPO corrected/rich, seed 42','tab:orange'),
        ('reference_rich_7_524288','PPO corrected/rich, seed 7','tab:red'),
        ('shared_rich_42_262144','PPO shared actor, seed 42','tab:green'),
        ('shared_rich_7_262144','PPO shared actor, seed 7','tab:olive')]
    for metric,ylabel in [('mean_score','Mean game score (points)'),('success_rate','Success rate (≥2,000 points)')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for name,label,color in styles:
            frame=all_rows[all_rows.run==name]
            if 'mode' in frame:frame=frame[(frame['mode']=='greedy')|frame['mode'].isna()]
            if len(frame):
                ax=axes[0 if name.startswith('q_') else 1]
                ax.plot(frame.interactions,frame[metric],marker='o',markersize=3,label=label,color=color)
        for ax,title in zip(axes,['Tabular Q-learning','PPO']):
            ax.set_title(title);ax.set_xlabel('Training interactions');ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(alpha=.25)
            ax.legend(fontsize=7,loc='best')
            if metric=='success_rate':ax.set_ylim(-.03,1.03)
        fig.suptitle('Exploratory B development: 20 episodes/checkpoint, greedy policies, no smoothing',fontsize=10)
        fig.tight_layout();fig.savefig(figure_dir/f'{metric}.png',dpi=160);plt.close(fig)
    locked_path=OUTPUT/'locked_validation_protocol.json'
    if locked_path.exists():
        locked=json.loads(locked_path.read_text());locked_rows=[]
        for record in locked['models']:
            seed=record['training_seed'];is_q=record['kind']=='q'
            label=f'locked_goal_q_{seed}_262144' if is_q else f'locked_shared_ppo_{seed}_262144'
            directory=OUTPUT/'confirmation'/label
            if not (directory/'summary.json').exists():continue
            evaluation=json.loads((directory/'protocol.json').read_text())
            summary=json.loads((directory/'summary.json').read_text())
            episodes=pd.read_csv(directory/'episodes.csv')
            assert evaluation['model_sha256']==record['sha256']
            assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
            assert evaluation['seeds']==locked['seeds'] and episodes.seed.tolist()==locked['seeds']
            assert len(episodes)==200 and evaluation['mode']=='greedy' and evaluation['epsilon']==0
            assert summary['learning_disabled_verified']
            assert np.array_equal(episodes.success.astype(bool),episodes.score>=2000)
            assert episodes.length.between(1,1000).all() and np.allclose(episodes.score%10,0)
            assert np.isclose(episodes.score.mean(),summary['mean_score'])
            assert np.isclose(episodes.success.mean(),summary['success_rate'])
            locked_rows.append(dict(algorithm='Tabular Double Q' if is_q else 'Shared-action PPO',
                training_seed=seed,training_interactions=262144,successes=int(episodes.success.sum()),
                failures=int((episodes.success==0).sum()),episodes=200,success_rate=float(episodes.success.mean()),
                mean_score=float(episodes.score.mean()),
                mean_success_length=float(episodes[episodes.success==1].length.mean()),
                mean_failure_length=float(episodes[episodes.success==0].length.mean()),
                model_sha256=record['sha256'],label=label))
        if len(locked_rows)==4:
            final=pd.DataFrame(locked_rows).sort_values(['algorithm','training_seed'])
            final.to_csv(OUTPUT/'locked_validation_summary.csv',index=False)
            (OUTPUT/'locked_validation_checks.json').write_text(json.dumps(dict(
                models=4,episodes=800,all_model_hashes_match=True,all_seed_lists_match=True,
                all_rewards_outcomes_and_horizons_consistent=True,learning_disabled=True),indent=2))
            fig,ax=plt.subplots(figsize=(8,4))
            x=np.arange(4)
            colors=['tab:orange' if a=='Shared-action PPO' else 'tab:blue' for a in final.algorithm]
            bars=ax.bar(x,final.success_rate*100,color=colors,width=.65)
            ax.set_xticks(x,[f'{a}\ntraining seed {s}' for a,s in zip(final.algorithm,final.training_seed)],fontsize=9)
            ax.set_ylabel('Success rate (%)');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.bar_label(bars,labels=[f'{n}/200' for n in final.successes],padding=4)
            ax.set_title('Locked B pilot: 200 new episodes/model, 262,144 training interactions',fontsize=10)
            fig.text(.5,.01,'Source: locked_validation_protocol.json; seeds 710000–710199; frozen final checkpoints',ha='center',fontsize=7)
            fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(figure_dir/'locked_validation_success.png',dpi=160);plt.close(fig)
            print('\nLocked validation (not used for further tuning):\n'+final.to_string(index=False))
    print(pd.DataFrame(confirmations).to_string(index=False))
    print('Registered source hashes verified:',all(verified.values()))


if __name__=='__main__':main()
