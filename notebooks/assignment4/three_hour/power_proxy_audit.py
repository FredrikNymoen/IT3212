"""Describe observed RAM116 bonus jumps; do not claim a verified exact timer."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def main():
    path=ROOT/'results/assignment4/three_hour_20261122/ppo_native_verification/stable_42/first_two_traces.npz'
    with np.load(path) as z:ram=z['ram'];steps=z['step'];seeds=z['seed']
    bcd=lambda v:(v.astype(int)//16)*10+v.astype(int)%16
    score=bcd(ram[:,122])*10000+bcd(ram[:,121])*100+bcd(ram[:,120])
    values=ram[:,116].astype(int);events=[]
    for index in range(1,len(ram)):
        if seeds[index]!=seeds[index-1]:continue
        if values[index]-values[index-1]!=64:continue
        if (int(ram[index,117])&60)!=(int(ram[index-1,117])&60):continue
        events.append(dict(evaluation_seed=int(seeds[index]),step=int(steps[index]),
            score_increment=int(score[index]-score[index-1]),ram_before=int(values[index-1]),ram_after=int(values[index]),
            old_capped_before=min(int(values[index-1]),63)/63,old_capped_after=min(int(values[index]),63)/63,
            low_bits_before=int(values[index-1]&63)/63,low_bits_after=int(values[index]&63)/63))
    assert events and all(event['low_bits_before']==event['low_bits_after'] for event in events)
    result=dict(source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        events=events,conclusion='Observed +64-byte changes coincide with point gains without new power-pellet consumption. Capping the full byte saturates; low six bits retain variation. This does not validate an exact timer or prove a learning-performance mechanism.')
    out=ROOT/'results/assignment4/three_hour_20261122/engineering/power_proxy_audit.json'
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
