"""Same-observation probes of the three original state-action predictions.

Select by observable scene criteria BEFORE querying model action values.
The power-state probe is only a positive-proxy analogue: the submitted premise
of a verified precise remaining timer cannot be established by this extractor.
"""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
import gymnasium as gym
import ale_py
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'three_hour'),str(HERE.parent/'pilot')]
import q_learning as q
import ppo as p
ROOT=q.ROOT


def main():
    output=ROOT/'results/assignment4/completion_prediction_probes_v1';output.mkdir(parents=True,exist_ok=True)
    original=ROOT/'results/assignment4/three_hour_20261122/final_v1/locked.json'
    lock=json.loads(original.read_text())
    jobs=sorted([j for j in lock['jobs'] if j['condition']=='B'],key=lambda j:(j['algorithm'],j['seed']))
    plan=dict(criteria={
        'S1':'Moving after decision70; right and up corridor flags; ghost8..40 pixels right, |dy|<=12 and observed closing x motion; food right; power proxy0.',
        'S2':'Moving after decision70; right and up corridor flags; ghost4..40 pixels right, |dy|<=12; nearest food beyond ghost to right; positive power proxy. Precise safety-time premise is NOT established.',
        'S3':'Moving after decision70; left/right corridor flags; nearest food right within24pixels; left food density greater than right; nearest active ghost>=64pixels.'},
        selection='First matching recorded state in fixed algorithm/seed/episode/step order, before any same-state model outputs are queried.',
        expected=dict(S1={'Q-learning':2,'PPO':1},S2={'Q-learning':1,'PPO':2},S3={'Q-learning':2,'PPO':3}),
        original_lock_sha256=q.sha(original),source_sha256=q.sha(__file__),scope='Frozen descriptive policy probes, not performance estimates, parameter tuning or reconstructed prior predictions.')
    file=output/'plan.json'
    if file.exists():assert json.loads(file.read_text())==plan
    else:q.write_json(file,plan)
    chosen={}
    for job in jobs:
        path=Path(job['directory'])/'evaluation_traces.npz'
        with np.load(path,allow_pickle=False) as archive:
            obs=archive['observation'];steps=archive['step']
            common=(steps>=70)&(np.linalg.norm(obs[:,41:43],axis=1)>.01)
            right_up=(obs[:,15]>.5)&(obs[:,16]>.5)
            ghost_right=(obs[:,2]*160>=8)&(obs[:,2]*160<=40)&(abs(obs[:,3]*210)<=12)
            masks={
                'S1':common&right_up&ghost_right&(obs[:,45]<-.05)&(obs[:,5]>0)&(obs[:,12]==0),
                'S2':common&right_up&(obs[:,2]*160>=4)&(obs[:,2]*160<=40)&(abs(obs[:,3]*210)<=12)&(obs[:,5]>obs[:,2])&(obs[:,12]>0),
                'S3':common&(obs[:,16]>.5)&(obs[:,17]>.5)&(obs[:,5]>0)&(obs[:,7]*320<=24)&(obs[:,11]>obs[:,9])&(obs[:,4]*320>=64)}
            for identity,mask in masks.items():
                if identity in chosen:continue
                indexes=np.flatnonzero(mask)
                if not len(indexes):continue
                i=int(indexes[0]);episode=int(archive['episode'][i]);episode_seed=lock['evaluation_seeds'][episode]
                rows=archive['episode']==episode
                chosen[identity]=dict(source_job=job,episode_seed=episode_seed,decision=int(steps[i])+1,observation=obs[i].tolist(),
                    raw_RAM=archive['ram'][i].tolist(),source_trace_sha256=q.sha(path),prefix_actions=archive['action'][rows][:int(steps[i])].tolist())
        if len(chosen)==3:break
    assert set(chosen)=={'S1','S2','S3'},'Do not substitute a state after querying policies; report any unsupported archetype.'
    q.write_json(output/'selected_states.json',chosen)
    torch.set_num_threads(1);answers=[]
    for job in jobs:
        path=ROOT/job['model'];assert q.sha(path)==job['model_sha256']
        if job['algorithm']=='Q-learning':
            with path.open('rb') as handle:agent=pickle.load(handle)
        else:agent=p.FrozenPolicy.load(path)
        before=q.sha(path)
        for identity,case in chosen.items():
            obs=np.asarray(case['observation'],dtype=np.float32)
            values=agent.values(agent.state(obs)) if job['algorithm']=='Q-learning' else agent.values_for_observation(obs)
            maxima=np.flatnonzero(values==values.max()).tolist()
            answers.append(dict(state=identity,algorithm=job['algorithm'],training_seed=job['seed'],values=values.tolist(),
                maximisers=maxima,exclusive_predicted_action=maxima==[plan['expected'][identity][job['algorithm']]],model_sha256=before))
        assert q.sha(path)==before
    gym.register_envs(ale_py)
    for identity,case in chosen.items():
        env=gym.make('ALE/MsPacman-v5',obs_type='ram',frameskip=4,repeat_action_probability=.25,mode=0)
        try:
            env.reset(seed=case['episode_seed'])
            for action in case['prefix_actions']:env.step(action)
            assert env.unwrapped.ale.getRAM().tolist()==case['raw_RAM']
            Image.fromarray(env.unwrapped.ale.getScreenRGB()).save(output/f'{identity}.png')
        finally:env.close()
    q.write_json(output/'answers.json',dict(answers=answers,selected_states_sha256=q.sha(output/'selected_states.json'),plan_sha256=q.sha(file),source_sha256=q.sha(__file__)))
    print(json.dumps({s:{a:sum(r['exclusive_predicted_action'] for r in answers if r['state']==s and r['algorithm']==a) for a in ['Q-learning','PPO']} for s in ['S1','S2','S3']},indent=2))


if __name__=='__main__':main()
