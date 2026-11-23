"""Clarified D: fixed-region OBJECT-feature occlusion, native B dynamics.

The agents do not ingest RGB. A rectangle painted on a video would be an
ineffective intervention. Instead hide object centres in the fixed rectangle
BEFORE computing all object-derived agent features. This is the RAM/object
analogue of the suggested image obstruction, not a pixel-encoder experiment.
Known static maze geometry and non-spatial game-status channels are retained.
An occluded player uses its last visible pose; hidden ghosts/food are absent.
No ground-truth hidden position, food density, motion or threat is forwarded.
"""
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'three_hour'),str(HERE.parent/'pilot')]
from fast_evaluation_v2 import FastStudyTask
from navigation_features import GRID,ALL_FOOD,CELLS,XY,NEIGHBORS,DISTANCE,POWER_NODES
from mspacman_env import FOOD_BYTES,FOOD_BITS

# Locked before any occluded-policy test outcomes: 32x32 pixels, 3.05% of
# the native 160x210 frame, playfield rather than the score/lives display.
ROI=(16,24,48,56)  # left, top, right-exclusive, bottom-exclusive.


def inside(points,roi=ROI):
    points=np.asarray(points)
    return ((points[...,0]>=roi[0])&(points[...,0]<roi[2])&
            (points[...,1]>=roi[1])&(points[...,1]<roi[3]))


class ObservationShift(gym.Wrapper):
    def __init__(self,occluded=True):
        super().__init__(FastStudyTask('B'))
        self.occluded=occluded
        self.last_player=self.last_orientation=self.previous_player=self.previous_ghosts=None
        self.food_key=None

    def encode(self,ram,info):
        actual_player=np.array([float(ram[10])-8,float(ram[16])+6])
        player_hidden=bool(self.occluded and inside(actual_player))
        if not player_hidden:
            self.last_player=actual_player.copy();self.last_orientation=(int(ram[56])&3)/3
        if self.last_player is None:
            raise RuntimeError('The predefined mask must not hide the initial player pose.')
        player=self.last_player.copy()
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        hidden=inside(ghosts) if self.occluded else np.zeros(4,dtype=bool)
        active=(np.arange(4)<=int(ram[19]))&bool(ram[47])&~hidden
        # Hidden coordinates never influence motion, nearest threat or offsets.
        ghosts[hidden]=self.previous_ghosts[hidden] if self.previous_ghosts is not None else 0
        gd=np.abs(ghosts-player).sum(axis=1);gd[~active]=np.inf
        nearest=int(np.argmin(gd));ghost_delta=ghosts[nearest]-player if active.any() else np.zeros(2)
        ghost_distance=float(gd[nearest]) if active.any() else 320.
        present=GRID.ravel()&((ram[FOOD_BYTES]&FOOD_BITS)!=0)
        if self.occluded:present[inside(ALL_FOOD)]=False
        power=np.array([bool(int(ram[117])&bit) for bit in (4,8,16,32)])
        # The four power-pellet centres are all outside this fixed region.
        food=ALL_FOOD[present]
        if len(food):
            delta=food-player;distance=np.abs(delta).sum(axis=1);j=int(np.argmin(distance))
            pellet_delta,pellet_distance=delta[j],float(distance[j])
            direction=np.where(np.abs(delta[:,0])>np.abs(delta[:,1]),np.where(delta[:,0]>0,1,3),np.where(delta[:,1]>0,2,0))
            density=np.bincount(direction,minlength=4)/252
        else:pellet_delta,pellet_distance,density=np.zeros(2),320.,np.zeros(4)
        node=int(np.argmin(np.square((XY-player)/[8,12]).sum(axis=1)))
        neighbors=NEIGHBORS[node];available=neighbors>=0
        key=present.tobytes()+power.tobytes()
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
        if self.occluded:motion[~active]=0
        self.previous_ghosts,self.previous_player=ghosts.copy(),player.copy()
        steps=info['episode_steps'] if 'episode_steps' in info else 0
        base=[player[0]/160,player[1]/210,ghost_delta[0]/160,ghost_delta[1]/210,ghost_distance/320,
              pellet_delta[0]/160,pellet_delta[1]/210,pellet_distance/320,*density,
              min(int(ram[116]),63)/63,self.last_orientation,max(0,1-steps/1000)]
        offset=np.clip((player-XY[node])/[8,12],-1,1)
        raw=np.array([*base,*available.astype(float),*regret(self.food_dist),*threat,*regret(self.power_dist),
                      *offset,*np.clip(motion.ravel()/8,-1,1),*np.clip(player_motion/8,-1,1),present.sum()/154,
                      1/(1+np.linalg.norm(player-XY[node]))],dtype=np.float32)
        raw=np.clip(raw,-1,1)
        nearest_motion=raw[33:41].reshape(4,2)[nearest] if active.any() else np.zeros(2)
        extra=(ghosts-player)/[160,210];extra=np.clip(extra,-1,1);extra[~active]=0
        self.hidden_player,self.hidden_ghosts=player_hidden,int(hidden.sum())
        score=info.get('score',0.)
        return np.r_[raw,nearest_motion,min(score/2000,1),info['lives']/5,extra.ravel(),active].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        original,info=self.env.reset(seed=seed,options=options)
        self.last_player=self.last_orientation=self.previous_player=self.previous_ghosts=self.food_key=None
        obs=self.encode(self.unwrapped.ale.getRAM(),info)
        if not self.occluded:assert np.array_equal(obs,original),'Unoccluded encoder parity'
        return obs,dict(info,observation_occluded=self.occluded)

    def step(self,action):
        original,reward,term,trunc,info=self.env.step(action)
        obs=self.encode(self.unwrapped.ale.getRAM(),info)
        if not self.occluded:assert np.array_equal(obs,original),'Unoccluded encoder parity'
        return obs,reward,term,trunc,dict(info,observation_occluded=self.occluded,
                                       hidden_player=self.hidden_player,hidden_ghosts=self.hidden_ghosts)
