"""Independent score/RAM replay of a frozen exploratory PPO checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from navigation_features import RichTask
from sb3_numpy import FrozenActor


def score_from_ram(ram):
    bcd=lambda n:(int(n)//16)*10+(int(n)%16)
    return bcd(ram[122])*10000+bcd(ram[121])*100+bcd(ram[120])


def main(args):
    path=Path(args.model).resolve();before=hashlib.sha256(path.read_bytes()).hexdigest()
    model=PPO.load(path,device='cpu');actor=FrozenActor(model)
    env=RichTask();obs,_=env.reset(seed=args.seed)
    initial_score=score_from_ram(env.env.ram);rows=[];frames=[]
    directory=ROOT/'results/assignment4/further_exploration/score_replay'/f'{path.parent.name}_{path.stem}_{args.seed}'
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    try:
        while True:
            probs=actor.probabilities(obs);action=int(probs.argmax())
            obs,reward,term,trunc,info=env.step(action)
            native_score=score_from_ram(env.env.ram)-initial_score
            assert native_score==info['score'],(native_score,info)
            row=dict(step=info['episode_steps'],action=action,probabilities=probs.tolist(),
                reward=reward,score=info['score'],ram_score=native_score,
                player=env.env.ram[[10,16]].astype(int).tolist(),power_counter=int(env.env.ram[116]),
                maze=int(env.env.ram[0]),success=bool(info['success']))
            rows.append(row)
            if args.gif and (info['episode_steps']%5==0 or term or trunc):
                frames.append(Image.fromarray(env.unwrapped.ale.getScreenRGB()).copy())
            if reward>=200 or term or trunc:
                Image.fromarray(env.unwrapped.ale.getScreenRGB()).save(directory/f'frame_{info["episode_steps"]}_{int(info["score"])}.png')
            if term or trunc:break
    finally:env.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    (directory/'trace.json').write_text(json.dumps(rows,indent=2))
    summary=dict(seed=args.seed,model=str(path.relative_to(ROOT)),sha256=before,
        all_decisions_match_native_score=True,score=rows[-1]['score'],success=rows[-1]['success'],
        decisions=len(rows),large_rewards=[r for r in rows if r['reward']>=200])
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    if args.gif:
        frames[0].save(directory/'native_replay_4x.gif',save_all=True,append_images=frames[1:],
            duration=80,loop=0)
        summary['animation']='Native RGB every fifth decision, 80 ms/frame: approximately 4x real-time.'
        (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--seed',type=int,default=670001)
    parser.add_argument('--gif',action='store_true')
    torch.set_num_threads(1);main(parser.parse_args())
