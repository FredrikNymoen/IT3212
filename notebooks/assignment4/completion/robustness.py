"""Clarification-driven frozen B observation-robustness comparison, no training.

All ten original source-selected B models, clean versus fixed object occlusion,
200 paired episode seeds/model/mode. ROI/model hashes lock before outcomes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'three_hour'),str(HERE.parent/'pilot')]
import q_learning as q
import ppo as p
from observation_shift import ObservationShift,ROI,inside
from navigation_features import ALL_FOOD
from mspacman_env import FOOD_BYTES,FOOD_BITS
ROOT=q.ROOT


def check(output):
    clean=ObservationShift(False);masked=ObservationShift(True);changed=0
    try:
        a,_=clean.reset(seed=1800000);b,_=masked.reset(seed=1800000)
        for i in range(1600):
            action=(i//17)%5
            a,ra,ta,ua,ia=clean.step(action);b,rb,tb,ub,ib=masked.step(action)
            assert np.array_equal(clean.unwrapped.ale.getRAM(),masked.unwrapped.ale.getRAM())
            assert (ra,ta,ua)==(rb,tb,ub) and ia['score']==ib['score']
            changed+=int(not np.array_equal(a,b))
            if ta or ua:a,_=clean.reset(seed=1800001+i);b,_=masked.reset(seed=1800001+i)
        assert changed>0
        ram=clean.unwrapped.ale.getRAM().copy();ram[47]=1;ram[19]=3
        # Mutating hidden ghost coordinates and hidden pellet bits must not leak
        # through nearest-object, density, motion, threat or additional channels.
        altered=ram.copy();ram[6],ram[12]=32,26;altered[6],altered[12]=40,34
        for cell in np.flatnonzero(inside(ALL_FOOD)):
            altered[FOOD_BYTES[cell]]^=FOOD_BITS[cell]
        twins=[ObservationShift(True),ObservationShift(True)]
        try:
            info=dict(lives=3,score=0,episode_steps=10)
            assert np.array_equal(twins[0].encode(ram,info),twins[1].encode(altered,info))
            for obj in twins:
                obj.previous_player=obj.previous_ghosts=obj.food_key=None
                obj.last_player=np.array([80.,100.]);obj.last_orientation=0.
            ram[10],ram[16]=32,26;altered[10],altered[16]=40,34
            assert np.array_equal(twins[0].encode(ram,info),twins[1].encode(altered,info))
        finally:
            for obj in twins:obj.close()
    finally:clean.close();masked.close()
    result=dict(native_parity_steps=1600,observations_changed=changed,ROI=list(ROI),frame_fraction=32*32/(160*210),
        hidden_food_ghost_and_player_noninterference=True,unoccluded_encoder_bitwise_parity=True,
        underlying_RAM_rewards_terminations_unchanged=True,
        scope='Object-feature analogue, not a pixel-CNN experiment; known static geometry and non-spatial status retained.',
        sources={p.relative_to(ROOT).as_posix():q.sha(p) for p in [Path(__file__),HERE/'observation_shift.py']})
    output.mkdir(parents=True,exist_ok=True);q.write_json(output/'engineering.json',result)
    print(json.dumps(result,indent=2))


def load(job):
    file=ROOT/job['model'];assert q.sha(file)==job['model_sha256']
    if job['algorithm']=='Q-learning':
        with file.open('rb') as handle:original=pickle.load(handle)
        model=q.TabularQ(0,alpha=original.alpha,gamma=original.gamma)
        model.table={key:value.copy() for key,value in original.table.items()};model.update_calls=original.update_calls
        assert model.update_calls==job['checkpoint_interactions']
        return model
    agent=p.FrozenPolicy.load(file);assert agent.model.num_timesteps==job['checkpoint_interactions'];return agent


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        q.verify_sources(job['sources']);output=Path(job['directory'])
        if (output/'metrics.json').exists():
            row=json.loads((output/'metrics.json').read_text());assert row['job']==job;return row
        output.mkdir(parents=True,exist_ok=False);q.write_json(output/'job.json',job)
        agent=load(job);factory=q.make_environment
        q.make_environment=lambda setting:ObservationShift(job['occluded'])
        try:
            metrics=q.evaluate(agent,job['algorithm'],'B',job['evaluation_seeds'],output,'evaluation',trace=True)
        finally:q.make_environment=factory
        assert q.sha(ROOT/job['model'])==job['model_sha256']
        result=dict(job=job,**metrics);q.write_json(output/'metrics.json',result);return result


def run(args):
    output=ROOT/'results/assignment4'/args.run_id
    if args.command=='check':check(output);return
    engineering=json.loads((output/'engineering.json').read_text())
    for name,digest in engineering['sources'].items():assert q.sha(ROOT/name)==digest
    sources=q.source_hashes()
    for name in ['robustness.py','observation_shift.py']:
        file=HERE/name;sources[file.relative_to(ROOT).as_posix()]=q.sha(file)
    old_path=ROOT/'results/assignment4/three_hour_20261122/final_v1/locked.json'
    old=json.loads(old_path.read_text())
    for name,digest in old['source_hashes'].items():assert q.sha(ROOT/name)==digest
    jobs=[]
    for source in old['jobs']:
        if source['condition']!='B':continue
        for occluded in [False,True]:
            jobs.append(dict(algorithm=source['algorithm'],seed=source['seed'],model=source['model'],
                model_sha256=source['model_sha256'],checkpoint_interactions=source['checkpoint_interactions'],
                occluded=occluded,evaluation_seeds=list(range(1600000,1600200)),sources=sources,
                directory=str(output/f'{"occluded" if occluded else "clean"}_{source["algorithm"]}_{source["seed"]}')))
    assert len(jobs)==20
    protocol=dict(jobs=jobs,ROI=list(ROI),workers=args.workers,training_interactions=0,source_lock_sha256=q.sha(old_path),
        sources=sources,clarification_sha256=q.sha(ROOT/'clarification-assingment4 (1).pdf'),
        interpretation='Frozen B policies; same game/maze. Centre-based object-feature occlusion in a fixed 32x32 screen region. Not an unseen-maze or literal pixel-encoder experiment.',
        selection='No mask-location/size search; original source-selected B checkpoints; every training seed retained.',
        engineering_sha256=q.sha(output/'engineering.json'))
    locked=output/'protocol.json'
    if locked.exists():assert json.loads(locked.read_text())==protocol
    else:
        q.write_json(locked,protocol)
        for name in sources:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            row=future.result();print(json.dumps(dict(event='evaluated',algorithm=row['job']['algorithm'],seed=row['job']['seed'],occluded=row['job']['occluded'])),flush=True)
    q.write_json(output/'completed.json',dict(cells=20,episodes=4000,training_interactions=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('command',choices=['check','run'])
    cli.add_argument('--run-id',default='clarified_D_v1');cli.add_argument('--workers',type=int,default=2)
    run(cli.parse_args())
