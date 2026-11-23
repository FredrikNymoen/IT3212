"""Numerical checks for the ordinary Bellman target and one-update contract."""
import json
from pathlib import Path

import numpy as np
from tabular_q import TabularQ


def main():
    agent=TabularQ(42,alpha=.5,gamma=.9)
    state=(0,('current',));nxt=(1,('next',))
    agent.table[state[1]]=np.array([1,0,0,0,0],dtype=np.float32)
    agent.table[nxt[1]]=np.array([0,4,1,2,3],dtype=np.float32)
    agent.update(state,0,200,nxt,False)
    # Target is 2 + .9*4 = 5.6; current 1 moves halfway to 3.3.
    assert np.isclose(agent.table[state[1]][0],3.3)
    assert agent.update_calls==1
    agent.table[state[1]][0]=3
    agent.update(state,0,100,nxt,True)
    assert np.isclose(agent.table[state[1]][0],2),'Terminal target must ignore next-state values'
    assert agent.update_calls==2
    # Every frame maps all five learned actions bijectively to native actions.
    agent.table[('mapping',)]=np.arange(5,dtype=np.float32)
    mapping_checks=0
    for anchor in range(4):
        frame=(anchor,('mapping',))
        for direction,native in enumerate(agent.ACTIONS):
            expected=1+(direction-anchor)%4
            assert agent.values(frame)[native]==expected
            mapping_checks+=1
        assert agent.values(frame)[0]==0
    before=agent.fingerprint()
    for _ in range(100):
        assert 0<=agent.action((0,('unseen',)))<5
    assert agent.fingerprint()==before,'Read-only behavior must not create or update entries'
    assert not any(hasattr(agent,name) for name in ('other','buffer','traces'))
    result=dict(ordinary_nonterminal_target=True,terminal_target_ignores_bootstrap=True,
        one_table=True,one_update_per_call=True,replay=False,eligibility_traces=False,
        native_action_mapping_checks=mapping_checks,read_only_action_selection=True)
    directory=Path(__file__).resolve().parents[3]/'results/assignment4/single_q_v1'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'algorithm_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
