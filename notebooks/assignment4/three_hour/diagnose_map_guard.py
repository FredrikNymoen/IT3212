"""Reproduce the rejected D case without changing its model or task semantics."""
import json,pickle
from pathlib import Path
import numpy as np
import run_study_v2 as study
from fast_evaluation_v2 import FastStudyTask

if __name__=='__main__':
    root=study.ROOT/'results/assignment4/three_hour_20261122/final_v1'
    job=json.loads((root/'D_Q-learning_314/job.json').read_text())
    assert study.sha(study.ROOT/job['model'])==job['model_sha256']
    with (study.ROOT/job['model']).open('rb') as handle:model=pickle.load(handle)
    before=model.fingerprint();env=FastStudyTask('D');found=False
    try:
        for seed in range(1150000,1150200):
            obs,_=env.reset(seed=seed);model.rng=np.random.default_rng(seed+3000000)
            for step in range(1000):
                action=model.action(model.state(obs))
                try:obs,reward,term,trunc,info=env.step(action)
                except RuntimeError as error:
                    ram=env.unwrapped.ale.getRAM()
                    result=dict(seed=seed,decision=step+1,action=int(action),error=str(error),
                        native_maze=int(ram[0]),accumulated_score=env.env.env.score,
                        native_bcd_score=study.native_score(ram),target_reached=env.env.env.score>=2000,
                        frozen_model_verified=model.fingerprint()==before)
                    study.write(root/'map_guard_diagnostic.json',result);print(json.dumps(result),flush=True);found=True;break
                if term or trunc:break
            if found:break
    finally:env.close()
    assert found
