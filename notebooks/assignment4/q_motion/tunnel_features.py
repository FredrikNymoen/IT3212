"""Separate representation repair: add native wrap tunnels to feature graph.

Does not change game transitions, rewards, actions, or any Q-learning update.
The closing-bit experiment never imports this module.
"""
import numpy as np
from motion_q import ImprovedTask
from environment import geometry
from mspacman_env import FOOD_BYTES,FOOD_BITS
from navigation_features import ALL_FOOD


class TunnelTask(ImprovedTask):
    def __init__(self,setting='B'):
        super().__init__(setting)
        self.grid,self.cells,self.xy,self.neighbors,_,self.power_nodes=geometry(self.env.task_config.maze)
        self.lookup={int(cell):i for i,cell in enumerate(self.cells)}
        degree=(self.neighbors>=0).sum(axis=1);self.tunnel_pairs=[]
        for row in range(14):
            left=self.lookup.get(row*18);right=self.lookup.get(row*18+17)
            if left is not None and right is not None and degree[left]==degree[right]==1:
                self.neighbors[left,2]=right;self.neighbors[right,1]=left
                self.tunnel_pairs.append((left,right))
        self.distance=np.full((len(self.cells),len(self.cells)),1000,dtype=np.float32)
        np.fill_diagonal(self.distance,0)
        for i,neighbors in enumerate(self.neighbors):self.distance[i,neighbors[neighbors>=0]]=1
        for k in range(len(self.cells)):
            self.distance=np.minimum(self.distance,self.distance[:,k,None]+self.distance[None,k,:])
        assert self.distance.max()<1000
        self.tunnel_food_key=None

    def reset(self,*,seed=None,options=None):
        self.tunnel_food_key=None
        return super().reset(seed=seed,options=options)

    def extend(self,obs,info):
        result=super().extend(obs,info)
        ram=self.env.env.ram;node=self.env.node
        player=np.array([float(ram[10])-8,float(ram[16])+6])
        neighbors=self.neighbors[node];available=neighbors>=0
        present=self.grid.ravel()&((ram[FOOD_BYTES]&FOOD_BITS)!=0)
        power=np.array([bool(int(ram[117])&bit) for bit in (4,8,16,32)])
        key=ram[59:101].tobytes()+bytes([int(ram[117])&60])
        if key!=self.tunnel_food_key:
            targets=present[self.cells].copy();targets[self.power_nodes[power]]=True
            self.tunnel_food_distance=self.distance[:,targets].min(axis=1) if targets.any() else np.full(len(self.cells),32.)
            self.tunnel_power_distance=self.distance[:,self.power_nodes[power]].min(axis=1) if power.any() else np.full(len(self.cells),32.)
            self.tunnel_food_key=key
        def regret(dist):
            costs=np.full(4,32.);costs[available]=1+dist[neighbors[available]]
            value=np.clip((costs-costs[available].min())/8,0,1);value[~available]=1
            return value
        ghosts=np.column_stack((ram[6:10].astype(float)-8,ram[12:16].astype(float)+6))
        active=np.arange(4)<=int(ram[19])
        if not ram[47]:active[:]=False
        threat=np.zeros(4)
        if active.any():
            ghost_nodes=np.argmin(np.square((self.xy[:,None,:]-ghosts[None,:,:])/[8,12]).sum(axis=2),axis=0)[active]
            distance=self.distance[:,ghost_nodes].min(axis=1)
            threat[available]=np.exp(-distance[neighbors[available]]/3)
        result[15:19]=available
        result[19:23]=regret(self.tunnel_food_distance)
        result[23:27]=threat
        result[27:31]=regret(self.tunnel_power_distance)
        return result
