"""Inspect native terminal flags at the exact rejected transition."""
import argparse,json,pickle
import numpy as np
import run_study_v2 as study
from fast_evaluation_v2 import FastStudyTask

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final-run',default='three_hour_20261122/final_v1')
    args=parser.parse_args();root=study.ROOT/'results/assignment4'/args.final_run
    job=json.loads((root/'D_Q-learning_314/job.json').read_text())
    with (study.ROOT/job['model']).open('rb') as handle:model=pickle.load(handle)
    env=FastStudyTask('D');native=env.env.env.env;original_step=native.step;latest={}
    def inspect(action):
        result=original_step(action)
        latest.update(reward=float(result[1]),terminated=bool(result[2]),truncated=bool(result[3]),info=result[4])
        return result
    native.step=inspect
    try:
        seed=1150094;obs,_=env.reset(seed=seed);model.rng=np.random.default_rng(seed+3000000)
        for step in range(1000):
            action=model.action(model.state(obs))
            try:obs,reward,term,trunc,info=env.step(action)
            except RuntimeError as error:
                ram=env.unwrapped.ale.getRAM()
                result=dict(seed=seed,decision=step+1,native_maze=int(ram[0]),
                    native_score=study.native_score(ram),game_over=bool(env.unwrapped.ale.game_over()),
                    **latest)
                study.write(root/'map_guard_native_flags.json',result);print(json.dumps(result),flush=True);break
            if term or trunc:raise AssertionError('Rejected transition not reproduced')
    finally:env.close()
