"""Fresh native verification of the disputed historical B PPO success rates."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent/'improvements'), str(HERE.parent/'pilot')]
from agents import ImprovedTask, FrozenMotion
from sb3_numpy import FrozenActor

OUT = ROOT/'results/assignment4/three_hour_20261122/ppo_native_verification'
SEEDS = tuple(range(1100000, 1100200))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def native_score(ram):
    bcd = lambda v: int(v)//16*10 + int(v)%16
    return bcd(ram[122])*10000 + bcd(ram[121])*100 + bcd(ram[120])

def work(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory=OUT/job['name']
        if (directory/'metrics.json').exists(): return json.loads((directory/'metrics.json').read_text())
        directory.mkdir(parents=True, exist_ok=False)
        path=ROOT/job['model']; assert sha(path)==job['sha256']
        model=PPO.load(path,device='cpu')
        before=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()
        actor=FrozenActor(model) if job['baseline'] else FrozenMotion(model)
        env=ImprovedTask('B'); rows=[]; decisions=0; traces=[]; start=perf_counter()
        try:
            assert abs(env.unwrapped.ale.getFloat('repeat_action_probability')-.25)<1e-8
            for index,seed in enumerate(SEEDS):
                obs,_=env.reset(seed=seed); initial=native_score(env.env.env.ram)
                for step in range(1000):
                    used=obs[:45] if job['baseline'] else obs
                    probabilities=actor.probabilities(used); action=int(probabilities.argmax())
                    if index<2 and step<100:
                        with torch.no_grad(): actual=model.policy.get_distribution(torch.as_tensor(used[None])).distribution.probs.numpy()[0]
                        assert np.allclose(actual,probabilities,atol=1e-5) and actual.argmax()==action
                    if index<2: traces.append((seed,step,env.env.env.ram.copy(),obs.copy(),probabilities.copy(),action))
                    obs,reward,term,trunc,info=env.step(action); decisions+=1
                    assert native_score(env.env.env.ram)-initial==info['score']
                    assert reward==info['raw_reward'] and info['maze']==0
                    if term or trunc:
                        rows.append(dict(evaluation_seed=seed,success=int(info['success']),score=info['score'],
                            length=info['episode_steps'],reason=info['reason'])); break
            frame=pd.DataFrame(rows); assert len(frame)==200
            assert np.array_equal(frame.success.astype(bool),frame.score>=2000)
            frame.to_csv(directory/'evaluation.csv',index=False)
            np.savez_compressed(directory/'first_two_traces.npz',
                seed=np.array([x[0] for x in traces]),step=np.array([x[1] for x in traces]),
                ram=np.array([x[2] for x in traces]),observation=np.array([x[3] for x in traces]),
                probabilities=np.array([x[4] for x in traces]),action=np.array([x[5] for x in traces]))
        finally: env.close()
        assert sha(path)==job['sha256']
        after=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()
        assert before==after
        result=dict(**job,episodes=200,success_rate=float(frame.success.mean()),mean_score=float(frame.score.mean()),
            mean_length=float(frame.length.mean()),decisions=decisions,evaluation_seconds=perf_counter()-start,
            native_score_verified_every_step=True,torch_numpy_verified=True,frozen_parameters_verified=True,sticky_probability=.25)
        (directory/'metrics.json').write_text(json.dumps(result,indent=2))
        return result

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    registry=json.loads((ROOT/'results/assignment4/improvement_confirmation_v1/locked_PPO.json').read_text())
    jobs=[dict(name=f'{"original" if job["baseline"] else "stable"}_{job["seed"]}',
        seed=job['seed'],baseline=job['baseline'],model=job['model'],sha256=job['sha256']) for job in registry['jobs']]
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    sources.update({k:v for k,v in registry['source_hashes'].items()})
    for path,expected in sources.items(): assert sha(ROOT/path)==expected,path
    lock=dict(jobs=jobs,evaluation_seeds=SEEDS,source_hashes=sources,policy='PPO argmax; no training; native score checked every decision.')
    target=OUT/'locked.json'
    if target.exists(): assert json.loads(target.read_text())==json.loads(json.dumps(lock))
    else: target.write_text(json.dumps(lock,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(work,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False)
    print('VERIFIED',flush=True)
