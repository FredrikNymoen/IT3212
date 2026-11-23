"""Independent original-vs-cached observations on A/B/C/D and task timeout."""
import hashlib,json
from pathlib import Path
import numpy as np
from candidates import StudyTask
from fast_evaluation_v2 import FastStudyTask

ROOT=Path(__file__).resolve().parents[3]
if __name__=='__main__':
    rows=[]
    for setting in 'ABCD':
        a=StudyTask(setting);b=FastStudyTask(setting);hits=0;steps=0
        try:
            x,_=a.reset(seed=323);y,_=b.reset(seed=323);assert np.array_equal(x,y)
            rng=np.random.default_rng(177)
            for index in range(700):
                action=0 if index<100 else int(rng.integers(5))
                x,rx,tx,xx,ix=a.step(action);y,ry,ty,xy,iy=b.step(action)
                assert np.array_equal(x,y) and (rx,tx,xx)==(ry,ty,xy)
                assert np.array_equal(a.env.env.ram,b.env.env.ram)
                for field in ['score','success','episode_steps','task_return','reason','lives','emulator_frames']:
                    assert ix[field]==iy[field]
                steps+=1
                if tx or xx:
                    hits+=b.cache_hits;x,_=a.reset(seed=324+index);y,_=b.reset(seed=324+index)
            hits+=b.cache_hits
            a.env.env.steps=b.env.env.steps=999
            x,rx,tx,xx,ix=a.step(0);y,ry,ty,xy,iy=b.step(0)
            assert np.array_equal(x,y) and (rx,tx,xx)==(ry,ty,xy) and ix['reason']==iy['reason']=='time_limit'
        finally:a.close();b.close()
        rows.append(dict(setting=setting,steps=steps+1,cache_hits=hits,bitwise_observations_and_native_dynamics_equal=True))
    assert sum(row['cache_hits'] for row in rows)>0
    result=dict(checks=rows,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(__file__).with_name('fast_evaluation.py'),Path(__file__).with_name('fast_evaluation_v2.py')]})
    (ROOT/'results/assignment4/three_hour_20261122/engineering/fast_evaluation_v2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
