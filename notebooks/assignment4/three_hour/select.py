"""Select from source A/B development only; no final or D records are read."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    frames=[];registry={}
    for name in args.runs:
        directory=ROOT/'results/assignment4'/name
        assert (directory/'completed.json').exists(),f'Incomplete development run: {directory}'
        protocol=json.loads((directory/'protocol.json').read_text())
        for path,expected in protocol['source_hashes'].items():assert sha(ROOT/path)==expected,path
        table=pd.read_csv(directory/'comparison.csv');assert set(table.setting)<=set('AB')
        assert set(table.episodes)=={20} and table.frozen_parameters_verified.all()
        frames.append(table);registry[str((directory/'comparison.csv').relative_to(ROOT))]=sha(directory/'comparison.csv')
    frame=pd.concat(frames,ignore_index=True)
    assert not frame.duplicated(['algorithm','variant','seed','setting','interactions']).any()
    count=frame[frame.interactions>0].groupby(['algorithm','variant','interactions']).size()
    required=4 if args.stage=='lock' else 2
    complete=count[count==required].index
    grouped=frame[frame.interactions>0].groupby(['algorithm','variant','interactions'])[['success_rate','mean_score']].mean()
    ranking=grouped.loc[complete].reset_index().sort_values(['algorithm','success_rate','mean_score','interactions','variant'],
        ascending=[True,False,False,True,True])
    rows=ranking.drop_duplicates(['algorithm','variant'])
    result={}
    for algorithm in ['Q-learning','PPO']:
        subset=rows[rows.algorithm==algorithm].head(1 if args.stage=='lock' else 2)
        assert len(subset)==(1 if args.stage=='lock' else 2)
        result[algorithm]=subset.to_dict('records')
    output=ROOT/'results/assignment4'/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(stage=args.stage,algorithms=result,development_sources=registry,
        rule='Mean A/B success at one common trained checkpoint, then mean score, then earlier checkpoint; no D/final selection.',
        selection_source_sha256=sha(__file__))
    if output.exists():assert json.loads(output.read_text())==record,'Preserve previous selection.'
    else:output.write_text(json.dumps(record,indent=2))
    ranking.to_csv(output.with_suffix('.csv'),index=False)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['shortlist','lock'])
    parser.add_argument('--runs',nargs='+',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
