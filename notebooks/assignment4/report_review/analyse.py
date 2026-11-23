"""Descriptive paired-seed sensitivity checks; no training or model selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'results/assignment4/report_review_current'
HERE.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'results/assignment4'
old=BASE/'three_hour_20261122/final_v1'
main=pd.read_csv(old/'verified_comparison.csv');abc=main[main.condition.ne('D')]
transfer=pd.read_csv(BASE/'transfer_boundary_v1/verified_comparison.csv')
transfer=transfer.rename(columns={'observed_success_rate':'success_rate'})
combined=pd.concat([abc[['condition','algorithm','seed','success_rate']],transfer[['condition','algorithm','seed','success_rate']]])
assert len(combined)==40 and not combined.duplicated(['condition','algorithm','seed']).any()
pairs=combined.pivot(index=['seed','algorithm'],columns='condition',values='success_rate')
setting_changes=[]
for algorithm in ['Q-learning','PPO']:
    values=pairs.xs(algorithm,level='algorithm')
    for before,after in [('A','B'),('B','C'),('B','D')]:
        delta=100*(values[after]-values[before])
        setting_changes.append(dict(algorithm=algorithm,before=before,after=after,
            mean_percentage_points=float(delta.mean()),per_seed={str(seed):float(value) for seed,value in delta.items()},
            seeds_improved=int((delta>0).sum()),seeds_worsened=int((delta<0).sum())))
gap=100*(pairs.xs('PPO',level='algorithm').B-pairs.xs('Q-learning',level='algorithm').B)
loo={str(seed):float(gap.drop(seed).mean()) for seed in gap.index}
curves_path=BASE/'three_hour_20261122/main_v1/comparison.csv'
curves=pd.read_csv(curves_path)
early=curves[curves.interactions.eq(65536)].groupby(['setting','algorithm']).success_rate.mean()
assert np.isclose(early.loc['A','PPO'],.20) and np.isclose(early.loc['A','Q-learning'],.02)
assert np.isclose(early.loc['B','PPO'],0) and np.isclose(early.loc['B','Q-learning'],.06)
source=abc[(abc.condition=='B')&(abc.algorithm=='PPO')&(abc.seed==314)].iloc[0]
destination=transfer[(transfer.algorithm=='PPO')&(transfer.seed==314)].iloc[0]
assert source.model_sha256==destination.model_sha256
assert np.isclose(source.success_rate,.925) and np.isclose(destination.success_rate,.02)
selection=json.loads((BASE/'three_hour_20261122/main_operating_selection.json').read_text())
selected=next(row for row in selection['jobs'] if row['condition']=='B' and row['algorithm']=='PPO')
assert selected['seed']==314 and selected['model_sha256']==source.model_sha256
assert sorted(abc[(abc.condition=='A')&(abc.algorithm=='PPO')].success_rate.tolist())==[0,1,1,1,1]
assert int(transfer.map_exits.sum())==1
result=dict(setting_changes=setting_changes,B_PPO_minus_Q_mean_percentage_points=float(gap.mean()),
    B_PPO_beats_Q_seeds=int((gap>0).sum()),B_gap_omitting_each_paired_seed=loo,
    B_gap_omission_range=[min(loo.values()),max(loo.values())],
    source_selected_PPO_seed314=dict(B_success=.925,D_success=.02,identical_model_sha256=source.model_sha256),
    D_censoring_maximum_effect_on_mean_Q_percentage_points=.1,
    interpretation='Descriptive sensitivity of these five fixed training seeds, not a confidence interval, statistical significance test or tuning criterion.',
    source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [old/'verified_comparison.csv',BASE/'transfer_boundary_v1/verified_comparison.csv',curves_path,
         BASE/'three_hour_20261122/main_operating_selection.json',Path(__file__)]})
(HERE/'paired_seed_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(100*pairs).to_csv(HERE/'per_seed_success_percent.csv')
print(json.dumps(result,indent=2))
