"""Development-only tabular Q-learning on corrected current-state geometry.

All five original actions remain available. No heuristic actions, shaped reward,
demonstrations, simulated future states or held-out map are used. Relative state
aggregation is deliberately shared across maze locations. Optional Watkins
Q(lambda) cuts traces on a nongreedy behavior action, retaining the off-policy
max-Q target. Ordinary one-step Q-learning is the lambda=0 control.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from navigation_features import RichTask
from mspacman_env import coarse_state


def state_key(obs,representation):
    if representation=='corrected_old':return coarse_state(obs[:15])
    available=obs[15:19]>.5
    food=obs[19:23]
    best=(food==food[available].min())&available
    # Quantize current ghost proximity along each corridor; no rollout/lookahead.
    danger=(obs[23:27]>.36)&available
    bits=lambda x:sum(int(v)<<i for i,v in enumerate(x))
    base=(bits(available),bits(best),bits(danger),int(obs[12]>0),int(round(float(obs[13])*3)))
    if representation=='relative':return base
    # Direction of displacement from the nearest junction/pellet-lattice point.
    return (*base,*np.digitize(obs[31:33],[-.2,.2]).tolist())


class NavigationQ:
    def __init__(self,seed,representation='relative',gamma=.99,trace=0.,alpha=.15):
        self.rng=np.random.default_rng(seed)
        self.representation,self.gamma,self.trace,self.alpha=representation,gamma,trace,alpha
        self.table={};self.traces={};self.counts={}

    def state(self,obs):return state_key(obs,self.representation)

    def values(self,state):return self.table.get(state,np.zeros(5,dtype=np.float32))

    def action(self,state,epsilon=0):
        values=self.values(state)
        if self.rng.random()<epsilon:return int(self.rng.integers(5))
        return int(self.rng.choice(np.flatnonzero(values==values.max())))

    def update(self,state,action,reward,nxt,terminal):
        values=self.table.setdefault(state,np.zeros(5,dtype=np.float32))
        target=reward/100+self.gamma*(not terminal)*float(self.values(nxt).max())
        error=target-float(values[action])
        self.counts[(state,action)]=self.counts.get((state,action),0)+1
        if self.trace==0:
            values[action]+=self.alpha*error
            return
        self.traces[(state,action)]=1.
        for key,eligibility in list(self.traces.items()):
            s,a=key
            self.table[s][a]+=self.alpha*error*eligibility
            remaining=eligibility*self.gamma*self.trace
            if remaining<.01:del self.traces[key]
            else:self.traces[key]=remaining


def evaluate(agent,directory,step):
    frozen=lambda:hashlib.sha256(pickle.dumps((agent.table,getattr(agent,'other',None)))).hexdigest()
    fingerprint=frozen()
    rng=agent.rng;agent.rng=np.random.default_rng(987654)
    env=RichTask();rows=[]
    try:
        for seed in range(670000,670020):
            obs,_=env.reset(seed=seed);unseen=0
            for _ in range(1000):
                state=agent.state(obs)
                key=state[1] if agent.representation in ('goal','goal_replay','power_goal') else state
                unseen+=int(key not in agent.table)
                obs,_,term,trunc,info=env.step(agent.action(state))
                if term or trunc:
                    rows.append(dict(development_seed=seed,score=info['score'],success=int(info['success']),
                        length=info['episode_steps'],reason=info['reason'],unseen_fraction=unseen/info['episode_steps']))
                    break
    finally:env.close();agent.rng=rng
    assert frozen()==fingerprint
    pd.DataFrame(rows).to_csv(directory/f'development_{step}.csv',index=False)
    return dict(interactions=step,mean_score=np.mean([r['score'] for r in rows]),
        success_rate=np.mean([r['success'] for r in rows]),episodes=len(rows),states=len(agent.table),
        unseen_fraction=np.mean([r['unseen_fraction'] for r in rows]))


def main(args):
    suffix=f'_{args.tag}' if args.tag else ''
    directory=ROOT/'results/assignment4/further_exploration'/f'q_{args.representation}_lambda{args.trace}_{args.seed}_{args.budget}{suffix}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.csv').exists():
        recorded=json.loads((directory/'protocol.json').read_text())
        for key in ('representation','trace','gamma','seed','budget'):
            if recorded[key]!=getattr(args,key):raise ValueError('Configuration differs; use a new --tag.')
        print('Already complete:',directory,flush=True);return
    if (directory/'protocol.json').exists():raise FileExistsError('Incomplete run exists; preserve it and use a new --tag.')
    protocol=vars(args)|dict(setting='B',goal=2000,horizon=1000,alpha=.15,
        epsilon='Linear 1 to .05 over first 80% of budget',reward_scale=.01,
        development_seeds=list(range(670000,670020)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for source in (Path(__file__),Path(__file__).with_name('navigation_features.py'),Path(__file__).with_name('maze_geometry.json')):
        (directory/source.name).write_bytes(source.read_bytes())
    if args.representation=='goal':
        from canonical_q import CanonicalQ
        agent=CanonicalQ(args.seed,args.representation,args.gamma,args.trace)
        (directory/'canonical_q.py').write_bytes(Path(__file__).with_name('canonical_q.py').read_bytes())
    elif args.representation in ('replay','goal_replay','power_goal'):
        if args.representation in ('goal_replay','power_goal'):
            if args.representation=='power_goal':
                from power_goal_q import PowerGoalQ
                agent=PowerGoalQ(args.seed,args.representation,args.gamma,args.trace)
                (directory/'power_goal_q.py').write_bytes(Path(__file__).with_name('power_goal_q.py').read_bytes())
            else:
                from goal_replay_q import GoalReplayQ
                agent=GoalReplayQ(args.seed,args.representation,args.gamma,args.trace)
            (directory/'goal_replay_q.py').write_bytes(Path(__file__).with_name('goal_replay_q.py').read_bytes())
        else:
            from replay_double_q import ReplayDoubleQ
            agent=ReplayDoubleQ(args.seed,args.representation,args.gamma,args.trace)
        (directory/'replay_double_q.py').write_bytes(Path(__file__).with_name('replay_double_q.py').read_bytes())
        protocol.update(algorithm='Tabular Double Q-learning',updates_per_interaction=5,
            replay_capacity=50000,alpha_schedule='0.15 / sqrt(1 + table/state/action updates / 100)',
            state='Food-relative canonical state with bonus-target routes' if args.representation=='power_goal' else
                  'Food-relative canonical state' if args.representation=='goal_replay' else
                  'phase + actual motion + nearest-food distance bins + 2x2 position')
        (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    else:agent=NavigationQ(args.seed,args.representation,args.gamma,args.trace)
    env=RichTask();episode=0;steps=0;training_seconds=0.;rows=[];episodes=[]
    obs,_=env.reset(seed=args.seed*1000+episode)
    try:
        for stop in range(65536,args.budget+1,65536):
            start=perf_counter()
            while steps<stop:
                state=agent.state(obs)
                epsilon=max(.05,1-.95*steps/(args.budget*.8))
                action=agent.action(state,epsilon)
                # Watkins trace cutting uses pre-update values of the action taken.
                if agent.trace and agent.values(state)[action]<agent.values(state).max():agent.traces.clear()
                nxt,reward,term,trunc,info=env.step(action)
                agent.update(state,action,reward,agent.state(nxt),term or trunc)
                steps+=1;obs=nxt
                if term or trunc:
                    episodes.append(dict(interactions=steps,score=info['score'],success=int(info['success']),length=info['episode_steps']))
                    episode+=1;agent.traces.clear();obs,_=env.reset(seed=args.seed*1000+episode)
            training_seconds+=perf_counter()-start
            with (directory/f'checkpoint_{stop}.pkl').open('wb') as f:pickle.dump(agent,f)
            row=evaluate(agent,directory,stop);row['training_seconds']=training_seconds
            rows.append(row);print(json.dumps(row),flush=True)
            pd.DataFrame(rows).to_csv(directory/'progress.csv',index=False)
            pd.DataFrame(episodes).to_csv(directory/'training.csv',index=False)
        pd.DataFrame(rows).to_csv(directory/'summary.csv',index=False)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--budget',type=int,default=524288)
    parser.add_argument('--representation',choices=('corrected_old','relative','phase','goal','replay','goal_replay','power_goal'),default='relative')
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--trace',type=float,default=0.)
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.budget%65536:raise ValueError('Budget must be a multiple of 65536.')
    main(args)
