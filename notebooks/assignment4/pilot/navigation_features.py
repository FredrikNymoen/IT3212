"""Corrected ROM coordinates plus retained original information and motion.

No action filtering, reward shaping, teacher actions or look-ahead simulation.
The fixed geometry is extracted from OCAtari and checked against real movement.
All task control still uses the original Gymnasium wrapper. B-only development
is supported here; the held-out D layout is not used in this investigation.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from mspacman_env import MsPacmanTask, TaskConfig, FOOD_BYTES, FOOD_BITS

GEOMETRY=json.loads(Path(__file__).with_name('maze_geometry.json').read_text())
GRID=np.asarray(GEOMETRY['grids'][0],dtype=bool)
ALL_FOOD=np.array([(x,y) for y in GEOMETRY['pellet_y'] for x in GEOMETRY['pellet_x']],dtype=float)+[2,1]
CELLS=np.flatnonzero(GRID.ravel())
XY=ALL_FOOD[CELLS]
LOOKUP={int(cell):i for i,cell in enumerate(CELLS)}
NEIGHBORS=np.full((len(CELLS),4),-1,dtype=int)
for i,cell in enumerate(CELLS):
    r,c=divmod(int(cell),18)
    for action,(dr,dc) in enumerate([(-1,0),(0,1),(0,-1),(1,0)]):
        if 0<=r+dr<14 and 0<=c+dc<18:
            NEIGHBORS[i,action]=LOOKUP.get((r+dr)*18+c+dc,-1)
DISTANCE=np.full((len(CELLS),len(CELLS)),1000,dtype=np.float32)
np.fill_diagonal(DISTANCE,0)
for i,n in enumerate(NEIGHBORS):DISTANCE[i,n[n>=0]]=1
for k in range(len(CELLS)):DISTANCE=np.minimum(DISTANCE,DISTANCE[:,k,None]+DISTANCE[None,k,:])
assert np.max(DISTANCE)<1000
POWER_NODES=np.argmin(np.square(XY[:,None,:]-np.array([[150,150],[150,18],[10,150],[10,18]])[None,:,:]).sum(axis=2),axis=0)
FEATURE_NAMES=(
    'player_x','player_y','ghost_dx','ghost_dy','ghost_distance',
    'pellet_dx','pellet_dy','pellet_distance','pellets_up','pellets_right','pellets_down','pellets_left',
    'power_countdown','orientation','time_left',
    'corridor_up','corridor_right','corridor_left','corridor_down',
    'food_regret_up','food_regret_right','food_regret_left','food_regret_down',
    'threat_up','threat_right','threat_left','threat_down',
    'power_regret_up','power_regret_right','power_regret_left','power_regret_down',
    'node_offset_x','node_offset_y',
    *[f'ghost_{i}_{axis}_motion' for i in range(4) for axis in ('x','y')],
    'player_x_motion','player_y_motion','remaining_food_fraction','corridor_alignment')
assert len(FEATURE_NAMES)==45


class RichTask(gym.Wrapper):
    def __init__(self):
        super().__init__(MsPacmanTask(TaskConfig('B')))
        self.observation_space=gym.spaces.Box(-1,1,shape=(len(FEATURE_NAMES),),dtype=np.float32)
        self.previous_player=None
        self.previous_ghosts=None
        self.food_key=None

    def observation(self):
        ram=self.env.ram
        assert int(ram[0])==0
        player=np.array([float(ram[10])-8,float(ram[16])+6])
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        active=np.arange(4)<=int(ram[19])
        if not ram[47]:active[:]=False
        gd=np.abs(ghosts-player).sum(axis=1)
        gd[~active]=np.inf
        nearest=int(np.argmin(gd))
        ghost_delta=ghosts[nearest]-player if active.any() else np.zeros(2)
        ghost_distance=float(gd[nearest]) if active.any() else 320.
        present=GRID.ravel()&((ram[FOOD_BYTES]&FOOD_BITS)!=0)
        # Power pills occupy the same topological corridors but separate RAM bits.
        power=np.array([bool(int(ram[117])&bit) for bit in (4,8,16,32)])
        food=ALL_FOOD[present]
        if len(food):
            delta=food-player;distance=np.abs(delta).sum(axis=1);j=int(np.argmin(distance))
            pellet_delta,pellet_distance=delta[j],float(distance[j])
            direction=np.where(np.abs(delta[:,0])>np.abs(delta[:,1]),
                np.where(delta[:,0]>0,1,3),np.where(delta[:,1]>0,2,0))
            density=np.bincount(direction,minlength=4)/252
        else:pellet_delta,pellet_distance,density=np.zeros(2),320.,np.zeros(4)
        node=int(np.argmin(np.square((XY-player)/[8,12]).sum(axis=1)))
        neighbors=NEIGHBORS[node];available=neighbors>=0
        key=ram[59:101].tobytes()+bytes([int(ram[117])&60])
        if key!=self.food_key:
            targets=present[CELLS].copy();targets[POWER_NODES[power]]=True
            self.food_dist=DISTANCE[:,targets].min(axis=1) if targets.any() else np.full(len(CELLS),32.)
            self.power_dist=DISTANCE[:,POWER_NODES[power]].min(axis=1) if power.any() else np.full(len(CELLS),32.)
            self.food_key=key
        def regret(dist):
            costs=np.full(4,32.);costs[available]=1+dist[neighbors[available]]
            out=np.clip((costs-costs[available].min())/8,0,1);out[~available]=1
            return out
        threat=np.zeros(4)
        if active.any():
            ghost_nodes=np.argmin(np.square((XY[:,None,:]-ghosts[None,:,:])/[8,12]).sum(axis=2),axis=0)[active]
            path_distance=DISTANCE[:,ghost_nodes].min(axis=1)
            threat[available]=np.exp(-path_distance[neighbors[available]]/3)
        motion=np.zeros((4,2)) if self.previous_ghosts is None else ghosts-self.previous_ghosts
        player_motion=np.zeros(2) if self.previous_player is None else player-self.previous_player
        if np.linalg.norm(player_motion)>20:player_motion[:]=0
        motion[np.abs(motion)>20]=0
        self.previous_ghosts,self.previous_player=ghosts.copy(),player.copy()
        base=[player[0]/160,player[1]/210,ghost_delta[0]/160,ghost_delta[1]/210,ghost_distance/320,
            pellet_delta[0]/160,pellet_delta[1]/210,pellet_distance/320,*density,
            min(int(ram[116]),63)/63,(int(ram[56])&3)/3,max(0,1-self.env.steps/1000)]
        offset=np.clip((player-XY[node])/[8,12],-1,1)
        obs=np.array([*base,*available.astype(float),*regret(self.food_dist),*threat,*regret(self.power_dist),
            *offset,*np.clip(motion.ravel()/8,-1,1),*np.clip(player_motion/8,-1,1),
            present.sum()/154,1/(1+np.linalg.norm(player-XY[node]))],dtype=np.float32)
        self.node=node
        return np.clip(obs,-1,1)

    def reset(self,*,seed=None,options=None):
        _,info=self.env.reset(seed=seed,options=options)
        self.previous_player=self.previous_ghosts=self.food_key=None
        return self.observation(),info

    def step(self,action):
        _,reward,term,trunc,info=self.env.step(int(action))
        return self.observation(),reward,term,trunc,info


def node_from_obs(obs):
    player=np.array([obs[0]*160,obs[1]*210])
    return int(np.argmin(np.square((XY-player)/[8,12]).sum(axis=1)))
