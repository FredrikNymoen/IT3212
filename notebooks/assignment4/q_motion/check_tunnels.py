"""Check repaired feature connectivity against native A/B dynamics only."""
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from motion_q import ImprovedTask
from tunnel_features import TunnelTask


def main():
    checked=0;changed=0
    for setting in 'AB':
        old=ImprovedTask(setting);new=TunnelTask(setting)
        try:
            assert len(new.tunnel_pairs)==2
            for left,right in new.tunnel_pairs:
                assert new.neighbors[left,2]==right and new.neighbors[right,1]==left
                assert new.distance[left,right]==1
            a,_=old.reset(seed=123);b,_=new.reset(seed=123)
            fixed=np.r_[np.arange(15),np.arange(31,49)]
            assert np.array_equal(a[fixed],b[fixed])
            rng=np.random.default_rng(71)
            for step in range(500):
                action=int(rng.integers(5))
                a,ra,ta,xa,ia=old.step(action);b,rb,tb,xb,ib=new.step(action)
                assert np.array_equal(old.env.env.ram,new.env.env.ram)
                assert (ra,ta,xa)==(rb,tb,xb) and ia['score']==ib['score']
                assert np.array_equal(a[fixed],b[fixed]) and new.observation_space.contains(b)
                changed+=not np.array_equal(a[15:31],b[15:31]);checked+=1
                if ta or xa:
                    a,_=old.reset(seed=124);b,_=new.reset(seed=124)
        finally:old.close();new.close()
    # Random actions need not visit a tunnel. Replay an actually recorded wrap.
    root=Path(__file__).resolve().parents[3]
    with np.load(root/'results/assignment4/part2_current_v3/final/B_Q-learning_42/evaluation_traces.npz') as data:
        before=data['observation'][:,:2]*[160,210];after=data['next_observation'][:,:2]*[160,210]
        wrap=(((before[:,0]<20)&(after[:,0]>140))|((before[:,0]>140)&(after[:,0]<20)))&(np.abs(before[:,1]-after[:,1])<8)
        index=np.flatnonzero(wrap)[0];episode=int(data['episode'][index]);stop=int(data['step'][index])
        actions=data['action'][data['episode']==episode][:stop+1]
    old=ImprovedTask('B');new=TunnelTask('B');target_changed=0
    try:
        a,_=old.reset(seed=790000+episode);b,_=new.reset(seed=790000+episode)
        for action in actions:
            target_changed+=not np.array_equal(a[15:31],b[15:31])
            a,ra,ta,xa,_=old.step(int(action));b,rb,tb,xb,_=new.step(int(action))
            assert np.array_equal(old.env.env.ram,new.env.env.ram) and (ra,ta,xa)==(rb,tb,xb)
            assert np.array_equal(a[fixed],b[fixed]);checked+=1
        assert target_changed>0
        changed+=target_changed
    finally:old.close();new.close()
    print(f'PASS: two tunnel pairs; {checked} identical native transitions/rewards; only graph-derived values changed in {changed} observations.')
    # Regress the Windows spawn/import collision before spending a train budget.
    from tunnel_experiment import worker,study
    reference=root/'results/assignment4/q_motion_v1/A_baseline_42/metrics.json'
    job=dict(setting='A',variant='baseline',seed=42,reference_sha256=study.sha(reference),
        directory=str(root/'results/assignment4/q_tunnel_startup_check/A_baseline_42'))
    with ProcessPoolExecutor(max_workers=1) as pool:result=pool.submit(worker,job).result()
    assert result['variant']=='baseline' and len(result['evaluations'])==2
    print('PASS: spawned tunnel-study worker loads the intended study and reuses the matched baseline.')


if __name__=='__main__':main()
