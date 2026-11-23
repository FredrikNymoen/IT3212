"""Part 4: outcome-distribution selection, 30 native decision inspections/forks.

No training or model selection. Ten original A/B/C recorded episodes: three
typical successes/failures and two rare successes/failures. Surprising means
rare within that trained model's 200-episode distribution, not an attractive
video selected by hand. Alternative actions are one RNG-coupled native step;
they cannot establish expected risk or the result of a whole alternative route.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import q_learning as q
ROOT=q.ROOT
LABELS=['N','U','R','L','D']
STRATA=[('TS1','typical success','Q-learning','B',1,False),
        ('TS2','typical success','PPO','B',1,False),
        ('TS3','typical success','PPO','C',1,False),
        ('TF1','typical failure','Q-learning','B',0,False),
        ('TF2','typical failure','PPO','B',0,False),
        ('TF3','typical failure','Q-learning','C',0,False),
        ('SS1','surprising success','Q-learning','C',1,True),
        ('SS2','surprising success','Q-learning','B',1,True),
        ('SF1','surprising failure','PPO','B',0,True),
        ('SF2','surprising failure','PPO','C',0,True)]


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def select(output):
    lock_path=ROOT/'results/assignment4/three_hour_20261122/final_v1/locked.json'
    registry=json.loads(lock_path.read_text());populations={};source_hashes={}
    for job in registry['jobs']:
        if job['condition'] not in 'ABC':continue
        file=Path(job['directory'])/'evaluation.csv';source_hashes[str(file.relative_to(ROOT))]=q.sha(file)
        rows=read_csv(file);assert len(rows)==200
        rate=sum(int(r['success']) for r in rows)/200
        for index,row in enumerate(rows):
            values=dict(job=job,episode_index=index,evaluation_seed=int(row['evaluation_seed']),
                        success=int(row['success']),length=int(row['length']),score=float(row['score']),
                        task_return=float(row['task_return']),model_success_rate=rate)
            populations.setdefault((job['algorithm'],job['condition']),[]).append(values)
    selected=[];used=set()
    for identity,category,algorithm,condition,success,rare in STRATA:
        population=populations[(algorithm,condition)]
        group=[row for row in population if row['success']==success]
        if rare:
            eligible=sorted({(r['model_success_rate'] if success else 1-r['model_success_rate'],r['job']['seed']) for r in group})
            probability,seed=eligible[0]
            group=[r for r in group if r['job']['seed']==seed]
        median_length=float(np.median([r['length'] for r in group]));median_score=float(np.median([r['score'] for r in group]))
        group.sort(key=lambda r:(abs(r['length']-median_length),abs(r['score']-median_score),r['job']['seed'],r['evaluation_seed']))
        row=next(r for r in group if (algorithm,condition,r['job']['seed'],r['evaluation_seed']) not in used)
        used.add((algorithm,condition,row['job']['seed'],row['evaluation_seed']))
        selected.append(dict(id=identity,category=category,conditional_population=len(group),
                             conditional_median_length=median_length,conditional_median_score=median_score,**row))
    result=dict(episodes=selected,source_episode_csv_hashes=source_hashes,original_lock_sha256=q.sha(lock_path),
        selection_rule='Fixed ten strata; conditional median length then score, seeded identity ties. Rare strata select lowest positive outcome probability trained model, then its conditional median. Skip exact duplicates.',
        decision_rule='Three states separated by eight decisions: largest score increment; two decisions before last observed movement preceding first life loss (else late goal approach); smallest action-value/probability gap near a ghost during observed movement. Opening 70 decisions excluded. Applied uniformly after native replay engineering.',
        alternative_rule='Highest-valued different advisory-cardinal option, then action ID; fall back to another available control. Advisory graph flags are not native legality masks.',
        source_sha256=q.sha(__file__))
    path=output/'selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:q.write_json(path,result)
    print(json.dumps([dict(id=r['id'],category=r['category'],algorithm=r['job']['algorithm'],condition=r['job']['condition'],training_seed=r['job']['seed'],episode_seed=r['evaluation_seed'],length=r['length'],model_success_rate=r['model_success_rate']) for r in selected],indent=2))
    return result


def decisions(arrays):
    n=len(arrays['action']);eligible=np.arange(70,n)
    assert len(eligible)>=3
    reward=arrays['raw_reward'];obs=arrays['observation'];nxt=arrays['next_observation']
    first=int(eligible[np.argmax(reward[eligible])])
    drops=np.flatnonzero(nxt[:,48]<obs[:,48]-.05)
    if len(drops):
        window=np.arange(max(70,int(drops[0])-80),max(71,int(drops[0])))
        moving=window[np.linalg.norm(obs[window,41:43],axis=1)>.01]
        second=max(70,int(moving[-1])-2) if len(moving) else int(window[np.argmin(obs[window,4])])
    else:second=int(eligible[-1])
    selected=[first]
    if abs(second-first)<8:
        second=next(int(i) for i in eligible[::-1] if abs(int(i)-first)>=8)
    selected.append(second)
    remaining=np.array([i for i in eligible if all(abs(i-s)>=8 for s in selected)])
    if not len(remaining):remaining=np.array([i for i in eligible if i not in selected])
    near=remaining[(obs[remaining,4]*320<=48)&(np.linalg.norm(obs[remaining,41:43],axis=1)>.01)]
    if not len(near):near=remaining
    ordered=np.sort(arrays['values'][near],axis=1)
    third=int(near[np.argmin(ordered[:,-1]-ordered[:,-2])]);selected.append(third)
    return sorted(selected)


def inspect(output,selection):
    gym.register_envs(ale_py);records=[]
    for case in selection['episodes']:
        job=case['job'];directory=Path(job['directory']);file=directory/'evaluation_traces.npz'
        assert q.sha(ROOT/job['model'])==job['model_sha256']
        with np.load(file,allow_pickle=False) as z:
            mask=z['episode']==case['episode_index']
            arrays={key:z[key][mask] for key in ['ram','observation','next_observation','values','action','raw_reward','reward','score_after']}
        chosen=decisions(arrays);assert len(chosen)==3
        env=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,
                     repeat_action_probability=0.0 if job['condition']=='A' else .25,mode=0)
        points=[]
        try:
            env.reset(seed=case['evaluation_seed']);earned=0.
            for step,action in enumerate(arrays['action']):
                ram=env.unwrapped.ale.getRAM().copy()
                assert np.array_equal(ram,arrays['ram'][step]),(case['id'],step,'RAM before')
                if step in chosen:
                    values=arrays['values'][step];obs=arrays['observation'][step]
                    controls=np.flatnonzero(obs[15:19]>.5)+1
                    alternatives=[int(a) for a in controls if a!=action] or [a for a in range(5) if a!=action]
                    alternative=min(alternatives,key=lambda a:(-float(values[a]),a))
                    image=env.unwrapped.ale.getScreenRGB().copy()
                    # Reconstruct the full action prefix in another seeded game.
                    # RAM-only clone restoration did not reproduce the next
                    # recorded sticky-action transition in engineering v2.
                    # Prefix replay also reconstructs controller/RNG history.
                    fork=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,
                                  repeat_action_probability=0.0 if job['condition']=='A' else .25,mode=0)
                    try:
                        fork.reset(seed=case['evaluation_seed'])
                        for prior in range(step):
                            assert np.array_equal(fork.unwrapped.ale.getRAM(),arrays['ram'][prior])
                            fork_ram,fork_reward,_,_,_=fork.step(int(arrays['action'][prior]))
                            assert fork_reward==arrays['raw_reward'][prior]
                        assert np.array_equal(fork.unwrapped.ale.getRAM(),ram)
                        alt_ram,alt_reward,alt_term,alt_trunc,alt_info=fork.step(alternative)
                        alternate=dict(action=alternative,reward=float(alt_reward),lives=int(alt_info['lives']),
                                       player=[int(alt_ram[10])-8,int(alt_ram[16])+6],terminated=bool(alt_term),truncated=bool(alt_trunc),
                                       full_seeded_action_prefix_verified=True)
                    finally:fork.close()
                    path=output/f'{case["id"]}_decision_{step+1}.png';Image.fromarray(image).save(path)
                after,reward,term,trunc,info=env.step(int(action));earned+=reward
                assert reward==arrays['raw_reward'][step] and earned==arrays['score_after'][step]
                if step+1<len(arrays['action']):assert np.array_equal(after,arrays['ram'][step+1]) and not term and not trunc
                if step in chosen:
                    obs=arrays['observation'][step];nxt=arrays['next_observation'][step]
                    future_drops=np.flatnonzero(arrays['next_observation'][step:,48]<arrays['observation'][step:,48]-.05)
                    points.append(dict(decision=step+1,available_actions=LABELS,action=int(action),values=values.tolist(),
                        player_before=[int(ram[10])-8,int(ram[16])+6],player_after=[int(after[10])-8,int(after[16])+6],
                        nearest_ghost_offset=(obs[2:4]*[160,210]).tolist(),ghost_distance=float(obs[4]*320),
                        food_offset=(obs[5:7]*[160,210]).tolist(),food_distance=float(obs[7]*320),
                        encoded_power_proxy=float(obs[12]),score_before=float(earned-reward),score_after=float(earned),
                        lives_before=int(round(float(obs[48])*5)),lives_after=int(info['lives']),
                        raw_reward=float(reward),task_reward=float(arrays['reward'][step]),
                        steps_to_next_life_loss=int(future_drops[0])+1 if len(future_drops) else None,
                        alternative=alternate,rgb=path.relative_to(ROOT).as_posix(),one_step_fork_only=True))
            assert earned==case['score'] and len(arrays['action'])==case['length']
        finally:env.close()
        result=dict(case=case,decisions=points,trace_sha256=q.sha(file),native_episode_replayed=True,
                    baseline_trace_unaffected_by_forks=True,scope='One seeded-prefix native step per alternative; no expected-outcome or whole-route counterfactual claim.')
        q.write_json(output/f'{case["id"]}.json',result);records.append(result)
    assert len(records)==10 and sum(len(r['decisions']) for r in records)==30
    q.write_json(output/'inspection.json',dict(episodes=records,source_sha256=q.sha(__file__),
                 native_decisions=sum(r['case']['length'] for r in records),alternative_native_steps=30))
    print('Verified ten episodes, thirty selected decisions and thirty RNG-coupled one-step alternatives.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--run-id',default='completion_inspection_v4')
    args=cli.parse_args();output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    selection=select(output);inspect(output,selection)
