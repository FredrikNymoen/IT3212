"""Current ordinary-Q/PPO A--D experiment. Run from the repository root.

python notebooks/assignment4/current/experiment.py run
python notebooks/assignment4/current/experiment.py report
No report PDF is compiled. Original v2 and pilot sources/results are preserved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from importlib import metadata
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ASSIGNMENT=HERE.parent
ROOT=ASSIGNMENT.parents[1]
PILOT=ASSIGNMENT/'pilot'
sys.path[:0]=[str(ASSIGNMENT),str(PILOT)]
from environment import ExperimentTask
from tabular_q import TabularQ
from shared_policy import SharedActionPolicy
from sb3_numpy import FrozenActor

RUN_ID='part2_current_v3'
TRAIN_SEEDS=(42,7)
TUNING_SEEDS=tuple(range(760000,760050))
CURVE_SEEDS=tuple(range(780000,780020))
FINAL_SEEDS=tuple(range(790000,790200))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def hashes():
    paths=[Path(__file__),HERE/'environment.py',ASSIGNMENT/'mspacman_env.py',
        *[PILOT/name for name in ('tabular_q.py','shared_policy.py','sb3_numpy.py','navigation_features.py','maze_geometry.json')],
        ROOT/'IT3212 - Assignment 4.pdf',ROOT/'reports/assignment4/Sections/01_predictions.tex']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fingerprint(agent,algorithm):
    if algorithm=='Q-learning':return agent.fingerprint()
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in agent.policy.state_dict().values())).hexdigest()


class ScaleReward(gym.RewardWrapper):
    def reward(self,reward):return reward/100


def make_ppo(setting,seed,rate,directory):
    envs=DummyVecEnv([lambda i=i:Monitor(ScaleReward(ExperimentTask(setting)),
        str(directory/f'training_{i}'),info_keywords=('score','success','reason','task_return','episode_steps')) for i in range(4)])
    model=PPO(SharedActionPolicy,envs,learning_rate=rate,n_steps=512,batch_size=256,n_epochs=4,
        gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.01,target_kl=.02,seed=seed,
        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])),device='cpu',verbose=0)
    return model,envs


def evaluate(agent,algorithm,setting,seeds,directory,name,trace=False):
    before=fingerprint(agent,algorithm)
    rng=agent.rng if algorithm=='Q-learning' else None
    actor=FrozenActor(agent) if algorithm=='PPO' else None
    env=ExperimentTask(setting);rows=[]
    traces={k:[] for k in ('ram','observation','next_observation','scores','action','episode','step','reward','raw_reward','score_after')}
    start=perf_counter();logp_error=0.
    try:
        for episode,seed in enumerate(seeds):
            obs,_=env.reset(seed=int(seed))
            if algorithm=='Q-learning':agent.rng=np.random.default_rng(seed+2000000)
            for step in range(1000):
                state=agent.state(obs) if actor is None else None
                values=agent.values(state) if actor is None else actor.probabilities(obs)
                if actor is not None and episode==0 and step<20:
                    with torch.no_grad():actual=agent.policy.get_distribution(torch.as_tensor(obs[None,:])).distribution.probs.cpu().numpy()[0]
                    logp_error=max(logp_error,float(np.max(np.abs(actual-values))))
                    assert np.allclose(actual,values,atol=1e-5) and actual.argmax()==values.argmax()
                action=agent.action(state) if actor is None else int(values.argmax())
                if trace:
                    traces['ram'].append(env.env.ram.copy());traces['observation'].append(obs.copy())
                    traces['scores'].append(values.copy());traces['action'].append(action)
                    traces['episode'].append(episode);traces['step'].append(step)
                next_obs,reward,term,trunc,info=env.step(action)
                if trace:
                    traces['next_observation'].append(next_obs.copy());traces['reward'].append(reward)
                    traces['raw_reward'].append(info['raw_reward']);traces['score_after'].append(info['score'])
                obs=next_obs
                if term or trunc:
                    assert np.isclose(info['task_return'],info['score']-(.1*info['episode_steps'] if setting=='C' else 0))
                    rows.append(dict(episode=episode,evaluation_seed=seed,success=int(info['success']),
                        failure=int(not info['success']),score=info['score'],task_return=info['task_return'],
                        length=info['episode_steps'],reason=info['reason']))
                    break
        seconds=perf_counter()-start
    finally:
        env.close()
        if algorithm=='Q-learning':agent.rng=rng
    assert fingerprint(agent,algorithm)==before,'Evaluation updated the learned model'
    frame=pd.DataFrame(rows)
    assert len(frame)==len(seeds) and np.array_equal(frame.success.astype(bool),frame.score>=2000)
    assert frame.length.between(1,1000).all() and np.allclose(frame.score%10,0)
    frame.to_csv(directory/f'{name}.csv',index=False)
    if trace:
        dtypes=dict(ram=np.uint8,observation=np.float32,next_observation=np.float32,scores=np.float32,
            action=np.uint8,episode=np.uint16,step=np.uint16,reward=np.float32,raw_reward=np.float32,score_after=np.float32)
        np.savez_compressed(directory/f'{name}_traces.npz',**{k:np.asarray(v,dtype=dtypes[k]) for k,v in traces.items()})
    conditional=lambda flag:float(frame.loc[frame.success.eq(flag),'length'].mean()) if frame.success.eq(flag).any() else None
    return dict(success_rate=float(frame.success.mean()),failure_rate=float(frame.failure.mean()),
        mean_score=float(frame.score.mean()),mean_return=float(frame.task_return.mean()),
        success_length=conditional(1),failure_length=conditional(0),evaluation_seconds=seconds,
        evaluation_episodes=len(frame),evaluation_interactions=int(frame.length.sum()),
        time_limit_failures=int(frame.reason.eq('time_limit').sum()),learning_disabled_verified=True,
        numpy_torch_max_error=logp_error)


def reused_b(job,directory):
    if job['setting']!='B' or job['budget']!=262144:return None
    seed=job['seed'];algorithm=job['algorithm']
    if algorithm=='Q-learning' and job['rate']==.15:
        source=ROOT/'results/assignment4/single_q_v1'/f'B_single_q_{seed}_262144'
        if not (source/'summary.json').exists():return None
        protocol=json.loads((source/'protocol.json').read_text())
        for name,expected in protocol['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
        summary=json.loads((source/'summary.json').read_text())
        for stop in (65536,131072,196608,262144):shutil.copy2(source/f'checkpoint_{stop}.pkl',directory/f'checkpoint_{stop}.pkl')
        shutil.copy2(source/'training.csv',directory/'training.csv')
        return dict(training_seconds=summary['training_seconds'],training_interactions=262144,
            provenance=str(source.relative_to(ROOT)),reused_training=True)
    if algorithm=='PPO' and job['rate']==.0003:
        source=ROOT/'results/assignment4/further_exploration'/f'shared_rich_{seed}_262144'
        if not (source/'summary.csv').exists():return None
        assert (source/'shared_policy.py').read_bytes()==(PILOT/'shared_policy.py').read_bytes()
        assert (source/'navigation_features.py').read_bytes()==(PILOT/'navigation_features.py').read_bytes()
        progress=pd.read_csv(source/'summary.csv')
        for stop in (65536,131072,196608,262144):shutil.copy2(source/f'checkpoint_{stop}.zip',directory/f'checkpoint_{stop}.zip')
        for file in source.glob('training_*.monitor.csv'):shutil.copy2(file,directory/file.name)
        return dict(training_seconds=float(progress.training_seconds.max()),training_interactions=262144,
            provenance=str(source.relative_to(ROOT)),reused_training=True)
    return None


def load_model(directory,algorithm,step):
    path=directory/f'checkpoint_{step}{".pkl" if algorithm=="Q-learning" else ".zip"}'
    if algorithm=='PPO':return PPO.load(path,device='cpu')
    with path.open('rb') as handle:return pickle.load(handle)


def train(job,directory):
    algorithm=job['algorithm'];budget=job['budget'];seed=job['seed'];rate=job['rate'];setting=job['setting']
    stops=(0,budget) if job['stage']=='tuning' else (0,65536,131072,196608,262144)
    seconds=0.;episodes=[];metrics=[]
    if algorithm=='Q-learning':
        agent=TabularQ(seed,alpha=rate,gamma=.99)
        with (directory/'checkpoint_0.pkl').open('wb') as f:pickle.dump(agent,f)
        env=ExperimentTask(setting);episode=0;steps=0;obs,_=env.reset(seed=seed*1000)
        try:
            for stop in stops[1:]:
                start=perf_counter()
                while steps<stop:
                    state=agent.state(obs);epsilon=max(.05,1-.95*steps/(budget*.8))
                    action=agent.action(state,epsilon)
                    nxt,reward,term,trunc,info=env.step(action)
                    agent.update(state,action,reward,agent.state(nxt),term or trunc);steps+=1;obs=nxt
                    if term or trunc:
                        episodes.append(dict(interactions=steps,score=info['score'],task_return=info['task_return'],
                            success=int(info['success']),length=info['episode_steps']))
                        episode+=1;obs,_=env.reset(seed=seed*1000+episode)
                seconds+=perf_counter()-start
                assert agent.update_calls==steps and not hasattr(agent,'other') and not hasattr(agent,'buffer')
                with (directory/f'checkpoint_{stop}.pkl').open('wb') as f:pickle.dump(agent,f)
            pd.DataFrame(episodes).to_csv(directory/'training.csv',index=False)
        finally:env.close()
    else:
        agent,envs=make_ppo(setting,seed,rate,directory);agent.save(directory/'checkpoint_0')
        try:
            previous=0
            for stop in stops[1:]:
                start=perf_counter();agent.learn(total_timesteps=stop-previous,reset_num_timesteps=False)
                seconds+=perf_counter()-start;previous=stop
                assert agent.num_timesteps==stop;agent.save(directory/f'checkpoint_{stop}')
                metrics.append(dict(interactions=stop,**{k:float(v) for k,v in agent.logger.name_to_value.items()
                    if k.startswith('train/') and np.isscalar(v)}))
            write_json(directory/'optimizer_updates.json',metrics)
        finally:envs.close()
    return dict(training_seconds=seconds,training_interactions=budget,reused_training=False,provenance='current run')


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return worker_body(job)


def worker_body(job):
    directory=Path(job['directory']);done=directory/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    if directory.exists():raise FileExistsError(f'Incomplete job preserved: {directory}')
    directory.mkdir(parents=True);write_json(directory/'job.json',job)
    reused=reused_b(job,directory) if job['stage']=='final' else None
    if reused:
        if job['algorithm']=='Q-learning':zero=TabularQ(job['seed'],alpha=job['rate'])
        else:
            initialization=directory/'initialization';initialization.mkdir()
            zero,envs=make_ppo(job['setting'],job['seed'],job['rate'],initialization);envs.close()
        if job['algorithm']=='Q-learning':
            with (directory/'checkpoint_0.pkl').open('wb') as handle:pickle.dump(zero,handle)
        else:zero.save(directory/'checkpoint_0')
        training=reused
    else:training=train(job,directory)
    agent=load_model(directory,job['algorithm'],job['budget'])
    if job['algorithm']=='Q-learning':
        assert type(agent) is TabularQ and agent.update_calls==job['budget']
        assert not hasattr(agent,'other') and not hasattr(agent,'buffer')
    if job['stage']=='tuning':
        result=dict(**job,**training,**evaluate(agent,job['algorithm'],'B',TUNING_SEEDS,directory,'development'))
    else:
        curve=[]
        for stop in (0,65536,131072,196608,262144):
            checkpoint=load_model(directory,job['algorithm'],stop)
            curve.append(dict(interactions=stop,**evaluate(checkpoint,job['algorithm'],job['setting'],CURVE_SEEDS,directory,f'curve_{stop}')))
        pd.DataFrame(curve).to_csv(directory/'curves.csv',index=False)
        final_path=directory/f'checkpoint_{job["budget"]}{".pkl" if job["algorithm"]=="Q-learning" else ".zip"}'
        checksum=hashlib.sha256(final_path.read_bytes()).hexdigest()
        result=dict(**job,**training,model_sha256=checksum,
            **evaluate(agent,job['algorithm'],job['setting'],FINAL_SEEDS,directory,'evaluation',trace=True))
        if job['setting']=='B':
            result['transfer']=dict(setting='D',training_interactions=0,model_sha256=checksum,
                **evaluate(agent,job['algorithm'],'D',FINAL_SEEDS,directory,'transfer',trace=True))
        assert hashlib.sha256(final_path.read_bytes()).hexdigest()==checksum
    write_json(done,result)
    return result


def run_jobs(jobs,workers):
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({k:result[k] for k in ('stage','algorithm','setting','seed','rate','success_rate','mean_score','training_seconds')}),flush=True)
    return results


def run(args):
    output=ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    protocol=dict(run_id=args.run_id,main_budget=262144,tuning_budget=65536,workers=args.workers,
        training_seeds=list(TRAIN_SEEDS),tuning_training_seed=31415,
        tuning_evaluation_seeds=list(TUNING_SEEDS),curve_evaluation_seeds=list(CURVE_SEEDS),
        final_evaluation_seeds=list(FINAL_SEEDS),curve_checkpoints=[0,65536,131072,196608,262144],
        rates={'Q-learning':[.1,.15],'PPO':[.0001,.0003]},
        selection='B-only development success, then mean game score, then candidate index; same chosen rate across A/B/C',
        fixed=dict(q_gamma=.99,q_epsilon='1 to .05 over first 80%',q_updates_per_transition=1,
            ppo_gamma=.995,ppo_gae=.95,ppo_epochs=4,ppo_rollout_per_env=512,ppo_envs=4,
            ppo_batch=256,ppo_clip=.2,ppo_entropy=.01,ppo_target_kl=.02,
            ppo_actor=[32,32],ppo_value=[64,64],reward_scale=.01),
        source_hashes=hashes(),scope='Ordinary Q/PPO controlled A-D revision after exploratory pilot inspection',
        history='Previous pilot search effort was unequal; this new prospective grid gives each method two 65536-step candidates and 50 development episodes each. Prior inspected evaluations are not new tests.',
        reward_approval='User confirmed instructor approval of game-score rewards in A/B.',
        limitations=['Two training seeds, not the five-seed Part 3 study.',
            'RAM116 remains a capped power signal, not verified precise remaining power time.'])
    file=output/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Protocol/source mismatch: use a new run ID.'
    else:
        write_json(file,protocol)
        snapshot=output/'sources';snapshot.mkdir()
        for name in protocol['source_hashes']:
            source=ROOT/name;destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        archive=output/'previous_report';archive.mkdir()
        for name in ('Sections/02_learning_settings.tex','Sections/02_discussion.tex'):
            source=ROOT/'reports/assignment4'/name
            shutil.copy2(source,archive/source.name)
    cpu=platform.processor()
    if sys.platform=='win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    write_json(output/'environment.json',dict(cpu=cpu,platform=platform.platform(),python=sys.version,
        packages={n:metadata.version(n) for n in ('numpy','pandas','torch','gymnasium','ale-py','stable-baselines3','matplotlib')},
        max_concurrent_workers=args.workers,numerical_threads=1,ppo_torch_threads=1))
    tuning=[dict(stage='tuning',algorithm=algorithm,setting='B',seed=31415,rate=rate,budget=65536,
        directory=str(output/'tuning'/f'{algorithm}_{index}'),candidate_index=index)
        for algorithm,rates in protocol['rates'].items() for index,rate in enumerate(rates)]
    tuned=run_jobs(tuning,args.workers)
    chosen={}
    for algorithm in protocol['rates']:
        candidates=[r for r in tuned if r['algorithm']==algorithm]
        chosen[algorithm]=sorted(candidates,key=lambda r:(-r['success_rate'],-r['mean_score'],r['candidate_index']))[0]['rate']
    write_json(output/'selected.json',chosen)
    pd.DataFrame(tuned).to_csv(output/'tuning_summary.csv',index=False)
    print('COMMON RATES '+json.dumps(chosen),flush=True)
    jobs=[dict(stage='final',algorithm=algorithm,setting=setting,seed=seed,rate=chosen[algorithm],budget=262144,
        directory=str(output/'final'/f'{setting}_{algorithm}_{seed}'))
        for setting in 'ABC' for algorithm in protocol['rates'] for seed in TRAIN_SEEDS]
    results=run_jobs(jobs,args.workers)
    write_json(output/'all_metrics.json',results)
    from report import render
    render(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','report'))
    parser.add_argument('--run-id',default=RUN_ID);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.command=='run':run(args)
    else:
        from report import render
        render(ROOT/'results/assignment4'/args.run_id)
