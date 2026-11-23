"""Check corrected geometry and Gymnasium task equivalence before adoption."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from navigation_features import RichTask,GRID,DISTANCE,GEOMETRY
from mspacman_env import MsPacmanTask,TaskConfig


def main():
    results={'connected_training_nodes':int(GRID.sum()),
        'max_graph_distance':float(DISTANCE.max()),'geometry_source':GEOMETRY['source']}
    original=MsPacmanTask(TaskConfig('B'));rich=RichTask()
    rng=np.random.default_rng(21212);checked=0;episodes=0
    try:
        for seed in (42,7,2026):
            original.reset(seed=seed);obs,_=rich.reset(seed=seed)
            assert rich.observation_space.contains(obs)
            for _ in range(1000):
                action=int(rng.integers(5))
                _,r1,t1,x1,i1=original.step(action)
                obs,r2,t2,x2,i2=rich.step(action)
                assert (r1,t1,x1)==(r2,t2,x2)
                assert np.array_equal(original.ram,rich.env.ram)
                assert rich.observation_space.contains(obs)
                assert i1['score']==i2['score'] and i1['success']==i2['success']
                checked+=1
                if t1 or x1:episodes+=1;break
        results.update(matched_decisions=checked,matched_episodes=episodes,
            reward_and_terminal_equivalence=True,all_observations_in_space=True,
            original_action_count=int(original.action_space.n),rich_action_count=int(rich.action_space.n))
    finally:original.close();rich.close()
    results['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(__file__).with_name('navigation_features.py'),Path(__file__).with_name('maze_geometry.json'))}
    directory=HERE.parents[1]/'results/assignment4/further_exploration/geometry_audit'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'gymnasium_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
