"""Current A--D wrapper: exact pilot observations on maze 0, compatible maze 1.

Atari still executes all dynamics. The graph is observed-feature engineering,
never an action mask. The original pilot files are kept unchanged for audit.
"""
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'pilot')]
from mspacman_env import MsPacmanTask,TaskConfig,FOOD_BYTES,FOOD_BITS
from navigation_features import RichTask,GEOMETRY,ALL_FOOD,FEATURE_NAMES


def geometry(index):
    grid=np.asarray(GEOMETRY['grids'][index],dtype=bool)
    cells=np.flatnonzero(grid.ravel());xy=ALL_FOOD[cells]
    lookup={int(cell):i for i,cell in enumerate(cells)}
    neighbors=np.full((len(cells),4),-1,dtype=int)
    for i,cell in enumerate(cells):
        r,c=divmod(int(cell),18)
        for action,(dr,dc) in enumerate([(-1,0),(0,1),(0,-1),(1,0)]):
            if 0<=r+dr<14 and 0<=c+dc<18:neighbors[i,action]=lookup.get((r+dr)*18+c+dc,-1)
    distance=np.full((len(cells),len(cells)),1000,dtype=np.float32)
    np.fill_diagonal(distance,0)
    for i,n in enumerate(neighbors):distance[i,n[n>=0]]=1
    for k in range(len(cells)):distance=np.minimum(distance,distance[:,k,None]+distance[None,k,:])
    assert distance.max()<1000
    power_nodes=np.argmin(np.square(xy[:,None,:]-np.array([[150,150],[150,18],[10,150],[10,18]])[None,:,:]).sum(axis=2),axis=0)
    return grid,cells,xy,neighbors,distance,power_nodes


TRANSFER_GEOMETRY=geometry(1)


class ExperimentTask(RichTask):
    def __init__(self,setting='B'):
        self.task_config=TaskConfig(setting)
        gym.Wrapper.__init__(self,MsPacmanTask(self.task_config))
        self.observation_space=gym.spaces.Box(-1,1,shape=(45,),dtype=np.float32)
        self.previous_player=self.previous_ghosts=self.food_key=None

    def observation(self):
        if self.task_config.maze==0:
            # Use the identical function that produced the frozen B checkpoints.
            return super().observation()
        ram=self.env.ram
        assert int(ram[0])==1
        grid,cells,xy,neighbors_table,distance_table,power_nodes=TRANSFER_GEOMETRY
        player=np.array([float(ram[10])-8,float(ram[16])+6])
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        active=np.arange(4)<=int(ram[19])
        if not ram[47]:active[:]=False
        gd=np.abs(ghosts-player).sum(axis=1);gd[~active]=np.inf
        nearest=int(np.argmin(gd))
        ghost_delta=ghosts[nearest]-player if active.any() else np.zeros(2)
        ghost_distance=float(gd[nearest]) if active.any() else 320.
        present=grid.ravel()&((ram[FOOD_BYTES]&FOOD_BITS)!=0)
        power=np.array([bool(int(ram[117])&bit) for bit in (4,8,16,32)])
        food=ALL_FOOD[present]
        if len(food):
            delta=food-player;distances=np.abs(delta).sum(axis=1);j=int(np.argmin(distances))
            pellet_delta,pellet_distance=delta[j],float(distances[j])
            direction=np.where(np.abs(delta[:,0])>np.abs(delta[:,1]),
                np.where(delta[:,0]>0,1,3),np.where(delta[:,1]>0,2,0))
            density=np.bincount(direction,minlength=4)/252
        else:pellet_delta,pellet_distance,density=np.zeros(2),320.,np.zeros(4)
        node=int(np.argmin(np.square((xy-player)/[8,12]).sum(axis=1)))
        neighbors=neighbors_table[node];available=neighbors>=0
        key=ram[59:101].tobytes()+bytes([int(ram[117])&60])
        if key!=self.food_key:
            targets=present[cells].copy();targets[power_nodes[power]]=True
            self.food_dist=distance_table[:,targets].min(axis=1) if targets.any() else np.full(len(cells),32.)
            self.power_dist=distance_table[:,power_nodes[power]].min(axis=1) if power.any() else np.full(len(cells),32.)
            self.food_key=key
        def regret(dist):
            costs=np.full(4,32.);costs[available]=1+dist[neighbors[available]]
            out=np.clip((costs-costs[available].min())/8,0,1);out[~available]=1
            return out
        threat=np.zeros(4)
        if active.any():
            ghost_nodes=np.argmin(np.square((xy[:,None,:]-ghosts[None,:,:])/[8,12]).sum(axis=2),axis=0)[active]
            path_distance=distance_table[:,ghost_nodes].min(axis=1)
            threat[available]=np.exp(-path_distance[neighbors[available]]/3)
        motion=np.zeros((4,2)) if self.previous_ghosts is None else ghosts-self.previous_ghosts
        player_motion=np.zeros(2) if self.previous_player is None else player-self.previous_player
        if np.linalg.norm(player_motion)>20:player_motion[:]=0
        motion[np.abs(motion)>20]=0
        self.previous_ghosts,self.previous_player=ghosts.copy(),player.copy()
        base=[player[0]/160,player[1]/210,ghost_delta[0]/160,ghost_delta[1]/210,ghost_distance/320,
            pellet_delta[0]/160,pellet_delta[1]/210,pellet_distance/320,*density,
            min(int(ram[116]),63)/63,(int(ram[56])&3)/3,max(0,1-self.env.steps/1000)]
        offset=np.clip((player-xy[node])/[8,12],-1,1)
        # Keep feature units/scales fixed, including the training-map divisor 154.
        obs=np.array([*base,*available.astype(float),*regret(self.food_dist),*threat,*regret(self.power_dist),
            *offset,*np.clip(motion.ravel()/8,-1,1),*np.clip(player_motion/8,-1,1),
            present.sum()/154,1/(1+np.linalg.norm(player-xy[node]))],dtype=np.float32)
        self.node=node
        return np.clip(obs,-1,1)
