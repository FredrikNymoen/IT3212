"""Replay selected recorded episodes against the native Gymnasium transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'pilot')]
from environment import ExperimentTask
from sb3_numpy import FrozenActor


def main(args):
    torch.set_num_threads(1)
    output=ROOT/'results/assignment4'/args.run_id;verified=[]
    # One episode for each A/B/C/D and each algorithm, training seed 42.
    for setting in 'ABCD':
        for algorithm in ('Q-learning','PPO'):
            directory=output/'final'/f'{"B" if setting=="D" else setting}_{algorithm}_42'
            if not (directory/'metrics.json').exists():
                raise RuntimeError(f'Still incomplete: {directory.name}')
            name='transfer' if setting=='D' else 'evaluation'
            episodes=pd.read_csv(directory/f'{name}.csv')
            episode=0;row=episodes.iloc[episode]
            with np.load(directory/f'{name}_traces.npz') as data:
                mask=data['episode']==episode
                saved={key:data[key][mask] for key in ('ram','observation','next_observation','scores','action','reward')}
            path=directory/f'checkpoint_262144{".pkl" if algorithm=="Q-learning" else ".zip"}'
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            if algorithm=='Q-learning':
                with path.open('rb') as handle:agent=pickle.load(handle)
            else:agent=FrozenActor(PPO.load(path,device='cpu'))
            env=ExperimentTask(setting);obs,_=env.reset(seed=int(row.evaluation_seed))
            try:
                for step,action in enumerate(saved['action']):
                    assert np.array_equal(env.env.ram,saved['ram'][step])
                    assert np.array_equal(obs,saved['observation'][step])
                    if step%25==0:
                        values=agent.values(agent.state(obs)) if algorithm=='Q-learning' else agent.probabilities(obs)
                        assert np.allclose(values,saved['scores'][step],atol=1e-6)
                    obs,reward,term,trunc,info=env.step(int(action))
                    assert np.array_equal(obs,saved['next_observation'][step])
                    assert np.isclose(reward,saved['reward'][step],atol=1e-5)
                assert term or trunc
                assert info['score']==row.score and info['episode_steps']==row.length
                assert int(info['success'])==row.success
            finally:env.close()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==before
            verified.append(dict(setting=setting,algorithm=algorithm,training_seed=42,
                evaluation_seed=int(row.evaluation_seed),steps=int(row.length),score=float(row.score)))
    path=output/'episode_replay_validation.json'
    path.write_text(json.dumps(dict(all_native_transitions_and_features_match=True,
        frozen_value_outputs_match=True,episodes=verified),indent=2))
    print('Replayed all eight A--D/algorithm episodes with identical RAM, features and outcomes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='part2_current_v3')
    main(parser.parse_args())
