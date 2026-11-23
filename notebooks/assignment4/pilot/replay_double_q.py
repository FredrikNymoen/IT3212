"""Tabular off-policy Double Q-learning with observed-transition replay.

Two independent estimates reduce maximization bias. One online plus four replay
updates reuse each interaction, analogous to PPO's reuse across update epochs.
Replay contains only actual Gymnasium transitions: no environment planning,
reward shaping, demonstration policy or fitted neural Q function is used.
"""
import numpy as np
from navigation_q import NavigationQ,state_key


class ReplayDoubleQ(NavigationQ):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.trace==0
        self.other={};self.buffer=[];self.buffer_index=0

    def state(self,obs):
        phase=state_key(obs,'phase')
        motion=obs[41:43]
        moving=4 if np.linalg.norm(motion)<.01 else int(np.argmax(
            np.array([[0,-1],[1,0],[-1,0],[0,1]])@motion))
        # Include motion and food proximity to distinguish otherwise aliased
        # wall waits / approaches / already-emptied local corridors.
        food=int(np.searchsorted([.01,.04,.12],obs[7]))
        return (*phase[:4],moving,*phase[5:],food,min(int(obs[0]*2),1),min(int(obs[1]*2),1))

    def values(self,state):
        return (self.table.get(state,np.zeros(5,dtype=np.float32))+
                self.other.get(state,np.zeros(5,dtype=np.float32)))/2

    def one_update(self,transition):
        state,action,reward,nxt,terminal=transition
        self.table.setdefault(state,np.zeros(5,dtype=np.float32))
        self.other.setdefault(state,np.zeros(5,dtype=np.float32))
        side=int(self.rng.integers(2))
        current,opposite=(self.table,self.other) if side==0 else (self.other,self.table)
        values=current.get(nxt,np.zeros(5,dtype=np.float32))
        greedy=int(self.rng.choice(np.flatnonzero(values==values.max())))
        target=reward/100+self.gamma*(not terminal)*float(opposite.get(nxt,np.zeros(5,dtype=np.float32))[greedy])
        key=(side,state,action);self.counts[key]=self.counts.get(key,0)+1
        alpha=self.alpha/np.sqrt(1+self.counts[key]/100)
        current[state][action]+=alpha*(target-current[state][action])

    def update(self,state,action,reward,nxt,terminal):
        transition=(state,action,reward,nxt,terminal)
        if len(self.buffer)<50000:self.buffer.append(transition)
        else:self.buffer[self.buffer_index]=transition
        self.buffer_index=(self.buffer_index+1)%50000
        self.one_update(transition)
        for index in self.rng.integers(len(self.buffer),size=4):self.one_update(self.buffer[int(index)])

    def __getstate__(self):
        # Frozen evaluation checkpoints do not need the replay memory.
        state=self.__dict__.copy();state['buffer']=[];state['buffer_index']=0
        return state
