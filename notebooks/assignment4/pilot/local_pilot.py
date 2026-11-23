"""Bounded, exploratory B-only pilot; does not change the registered experiment.

Package intervention: relative corridor/food/threat features and a shared
food-lattice action filter (not certified ALE legal-action detection). PPO also
uses the proposed lower learning rate. An untrained filtered control separates
some preprocessing benefit from learning. This is not a one-factor Part 3 study.
The food-lattice graph is only implemented/checked for training layout 0; D is
deliberately not evaluated. All interaction remains through Gymnasium.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from mspacman_env import MsPacmanTask, TaskConfig, FOOD_MASKS, FOOD_XY, FOOD_BYTES, FOOD_BITS
from rl_agents import PPO

OUTPUT=ROOT/'results/assignment4/quick_pilot/local_v1'
DEVELOPMENT=tuple(range(650000,650020))
OBS_SIZE=21


class LocalTask(gym.Wrapper):
    def __init__(self):
        super().__init__(MsPacmanTask(TaskConfig('B')))
        self.observation_space=gym.spaces.Box(-1,1,shape=(OBS_SIZE,),dtype=np.float32)
        cells=np.flatnonzero(FOOD_MASKS[0])
        self.cells=cells
        self.xy=FOOD_XY[cells]+[2,1]
        lookup={int(cell):i for i,cell in enumerate(cells)}
        self.neighbors=np.full((len(cells),4),-1,dtype=int)
        for i,cell in enumerate(cells):
            r,c=divmod(int(cell),18)
            for action,(dr,dc) in enumerate([(-1,0),(0,1),(0,-1),(1,0)]):
                if 0<=r+dr<14 and 0<=c+dc<18:
                    self.neighbors[i,action]=lookup.get((r+dr)*18+c+dc,-1)
        distance=np.full((len(cells),len(cells)),1000,dtype=np.float32)
        np.fill_diagonal(distance,0)
        for i,neighbors in enumerate(self.neighbors):
            distance[i,neighbors[neighbors>=0]]=1
        for k in range(len(cells)):
            distance=np.minimum(distance,distance[:,k,None]+distance[None,k,:])
        assert np.max(distance)<1000
        self.distance=distance
        self.food_key=None

    def observation(self):
        ram=self.env.ram
        assert int(ram[0])==0
        player=np.array([float(ram[10])-9,float(ram[16])+6])
        node=int(np.argmin(np.square((self.xy-player)/[8,12]).sum(axis=1)))
        neighbors=self.neighbors[node]
        available=neighbors>=0
        key=ram[59:101].tobytes()
        if key!=self.food_key:
            present=((ram[FOOD_BYTES]&FOOD_BITS)!=0)[self.cells]
            self.food_distance=self.distance[:,present].min(axis=1) if present.any() else np.full(len(self.cells),32.)
            self.food_key=key
        food=np.full(4,32.)
        food[available]=1+self.food_distance[neighbors[available]]
        regret=np.clip((food-food[available].min())/8,0,1)
        regret[~available]=1
        ghosts=np.column_stack((ram[6:10].astype(float)-9,ram[12:16].astype(float)+6))
        active=np.arange(4)<=int(ram[19])
        if not ram[47]: active[:]=False
        threats=np.zeros(4)
        nearest=0.
        if active.any():
            ghost_nodes=np.argmin(np.square((self.xy[:,None,:]-ghosts[None,:,:])/[8,12]).sum(axis=2),axis=0)[active]
            ghost_distance=self.distance[:,ghost_nodes].min(axis=1)
            threats[available]=np.exp(-ghost_distance[neighbors[available]]/3)
            nearest=float(np.exp(-ghost_distance[node]/3))
        orientation=np.zeros(4)
        orientation[int(ram[56])&3]=1
        # Orientation order in RAM is UP, RIGHT, DOWN, LEFT; action order is
        # UP, RIGHT, LEFT, DOWN. The one-hot is only an observation, not a mask.
        obs=np.array([*np.clip((player-self.xy[node])/[8,12],-1,1),
            *available.astype(float),*regret,*threats,nearest,
            min(int(ram[116]),63)/63,*orientation,
            max(0,1-self.env.steps/1000)],dtype=np.float32)
        assert obs.shape==(OBS_SIZE,)
        return obs

    def reset(self,*,seed=None,options=None):
        _,info=self.env.reset(seed=seed,options=options)
        self.food_key=None
        return self.observation(),info

    def step(self,action):
        _,reward,term,trunc,info=self.env.step(int(action))
        return self.observation(),reward,term,trunc,info


def mask(obs):
    return np.concatenate(([False],obs[2:6]>.5))


class LocalQ:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed)
        self.table={}

    def state(self,obs):
        available=mask(obs)[1:]
        food=obs[6:10]
        best=(food==food[available].min())&available
        bits=lambda x:sum(int(v)<<i for i,v in enumerate(x))
        threat=obs[10:14]
        return (bits(available),bits(best),int(np.argmax(threat)) if obs[14]>0 else 4,
            int(np.searchsorted([.07,.37],obs[14])),
            0 if obs[15]==0 else (1 if obs[15]<.3 else 2),
            int(np.argmax(obs[16:20])),int(obs[20]>.5))

    def scores(self,obs):
        values=self.table.get(self.state(obs),np.zeros(5,dtype=np.float32)).copy()
        values[~mask(obs)]=-1e9
        return values

    def action(self,obs,epsilon=0):
        legal=np.flatnonzero(mask(obs))
        if self.rng.random()<epsilon: return int(self.rng.choice(legal))
        values=self.scores(obs)
        return int(self.rng.choice(np.flatnonzero(values==values.max())))

    def update(self,obs,action,reward,next_obs,term):
        values=self.table.setdefault(self.state(obs),np.zeros(5,dtype=np.float32))
        target=reward/100+.99*(not term)*float(self.scores(next_obs).max())
        values[action]+=.1*(target-values[action])

    def fingerprint(self):
        return hashlib.sha256(pickle.dumps(self.table)).hexdigest()


class LocalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(OBS_SIZE,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor,self.critic=nn.Linear(64,5),nn.Linear(64,1)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,.01)
        nn.init.orthogonal_(self.critic.weight,1)

    def forward(self,obs):
        x=self.body(obs)
        return self.actor(x),self.critic(x).squeeze(-1)


class LocalPPO(PPO):
    def __init__(self,seed):
        super().__init__(seed,learning_rate=.0003)
        torch.manual_seed(seed)
        self.network=LocalNetwork()
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=.0003,eps=1e-5)
        self.refresh()

    def distribution(self,obs):
        probabilities,value=super().distribution(obs)
        probabilities[~mask(obs)]=0
        probabilities/=probabilities.sum()
        return probabilities,value

    def update(self,rollout):
        obs,actions,logp,values,rewards,next_values,terminated,done=map(np.asarray,zip(*rollout))
        advantage=np.zeros(len(rewards),dtype=np.float32)
        last=0.
        for t in range(len(rewards)-1,-1,-1):
            delta=rewards[t]/100+.99*next_values[t]*(1-terminated[t])-values[t]
            last=delta+.99*.95*(1-done[t])*last
            advantage[t]=last
        returns=torch.tensor(advantage+values,dtype=torch.float32)
        advantage=torch.tensor((advantage-advantage.mean())/(advantage.std()+1e-8))
        obs=torch.tensor(obs,dtype=torch.float32)
        actions=torch.tensor(actions,dtype=torch.int64)
        old_logp=torch.tensor(logp,dtype=torch.float32)
        legal=torch.cat((torch.zeros((len(obs),1),dtype=torch.bool),obs[:,2:6]>.5),dim=1)
        for epoch in range(4):
            permutation=self.rng.permutation(len(obs))
            for start in range(0,len(obs),128):
                idx=permutation[start:start+128]
                logits,value=self.network(obs[idx])
                logits=logits.masked_fill(~legal[idx],-1e9)
                distribution=torch.distributions.Categorical(logits=logits)
                ratio=(distribution.log_prob(actions[idx])-old_logp[idx]).exp()
                policy=-torch.minimum(ratio*advantage[idx],ratio.clamp(.8,1.2)*advantage[idx]).mean()
                loss=policy+.25*(returns[idx]-value).square().mean()-.01*distribution.entropy().mean()
                self.optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(),.5)
                self.optimizer.step()
        self.refresh()


def evaluate(agent,mode):
    before=agent.fingerprint()
    state=copy.deepcopy(agent.rng.bit_generator.state)
    env=LocalTask();rows=[]
    try:
        for seed in DEVELOPMENT:
            obs,_=env.reset(seed=seed)
            agent.rng=np.random.default_rng(seed+2000000)
            stationary=0
            for step in range(1000):
                old=env.env.ram[[10,16]].copy()
                action=agent.action(obs,explore=int(mode=='sample')) if isinstance(agent,LocalPPO) else agent.action(obs)
                assert mask(obs)[action]
                obs,reward,term,trunc,info=env.step(action)
                stationary+=int(np.array_equal(old,env.env.ram[[10,16]]))
                if term or trunc:
                    rows.append(dict(seed=seed,score=info['score'],success=int(info['success']),
                        length=info['episode_steps'],stationary_fraction=stationary/info['episode_steps']))
                    break
    finally:
        env.close();agent.rng=np.random.default_rng();agent.rng.bit_generator.state=state
    assert before==agent.fingerprint()
    return rows


def worker(job):
    algorithm,seed=job
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory=OUTPUT/f'{algorithm}_{seed}';directory.mkdir(parents=True,exist_ok=True)
        agent=LocalQ(seed) if algorithm=='Q-learning' else LocalPPO(seed)
        results=[]
        def assess(stage):
            for mode in (('greedy',) if algorithm=='Q-learning' else ('greedy','sample')):
                rows=evaluate(agent,mode)
                pd.DataFrame(rows).to_csv(directory/f'{stage}_{mode}.csv',index=False)
                results.append(dict(algorithm=algorithm,training_seed=seed,stage=stage,mode=mode,
                    mean_score=float(np.mean([r['score'] for r in rows])),
                    success_rate=float(np.mean([r['success'] for r in rows])),
                    stationary_fraction=float(np.mean([r['stationary_fraction'] for r in rows]))))
        if seed==42: assess('untrained')
        env=LocalTask();episode_rng=np.random.default_rng(seed)
        obs,_=env.reset(seed=int(episode_rng.integers(100000)))
        rollout=[];episodes=[];started=perf_counter()
        for step in range(65536):
            if algorithm=='PPO':
                probabilities,value=agent.distribution(obs)
                action=int(agent.rng.choice(5,p=probabilities))
            else:
                action=agent.action(obs,epsilon=max(.05,1-.95*step/(.8*65536)))
            assert mask(obs)[action]
            next_obs,reward,term,trunc,info=env.step(action)
            if algorithm=='PPO':
                rollout.append((obs,action,np.log(max(float(probabilities[action]),1e-12)),value,
                    reward,agent.distribution(next_obs)[1],term,term or trunc))
                if len(rollout)==512:
                    agent.update(rollout);rollout=[]
            else: agent.update(obs,action,reward,next_obs,term)
            obs=next_obs
            if term or trunc:
                episodes.append(dict(interactions=step+1,score=info['score'],success=int(info['success'])))
                obs,_=env.reset(seed=int(episode_rng.integers(100000)))
        env.close()
        seconds=perf_counter()-started
        pd.DataFrame(episodes).to_csv(directory/'training_episodes.csv',index=False)
        if algorithm=='Q-learning':
            with (directory/'q_table.pkl').open('wb') as handle:pickle.dump(agent.table,handle)
        else: np.savez_compressed(directory/'network.npz',**agent.weights)
        assess('trained')
        for row in results:row['training_interactions']=65536 if row['stage']=='trained' else 0;row['train_seconds']=seconds
        (directory/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(results),flush=True)
        return results


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected=list(HERE.glob('*.py'))+[ROOT/'reports/assignment4/Sections/01_predictions.tex']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    protocol=dict(development_only=True,setting='B',training_budget=65536,training_seeds=[42,7],
        evaluation_seeds=DEVELOPMENT,goal=2000,horizon=1000,mask='neighbors of nearest food-lattice node; NOOP excluded',
        limitation='Approximate topology advisory filter, not exact ALE legal moves; training maze only. Package intervention, not one-factor attribution.',
        q_learning_rate=.1,ppo_learning_rate=.0003,main_report_unchanged=True,protected_source_hashes=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(worker,(algorithm,seed)) for seed in (42,7) for algorithm in ('Q-learning','PPO')]):
            results.extend(future.result())
    frame=pd.DataFrame(results);frame.to_csv(OUTPUT/'summary.csv',index=False)
    print(frame.groupby(['algorithm','stage','mode'])[['mean_score','success_rate','stationary_fraction']].mean().to_string(),flush=True)
    for name,value in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name


if __name__=='__main__':main()
