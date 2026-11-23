"""Choose a fixed operating policy from source development only; no test reads."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4/three_hour_20261122'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    training=BASE/'main_v1';protocol=json.loads((training/'protocol.json').read_text())
    plan=dict(main_protocol_sha256=sha(training/'protocol.json'),source_sha256=sha(__file__),
        rule='Among the five fresh main seeds, select source-condition development success, then game score, then earlier checkpoint, then lower seed. Same rule for Q/PPO. D inherits chosen B model. No test files are read.',
        timing='Recorded while main training/streamed testing runs, before inspecting any final summaries; this is not the optional uniform50-candidate operating study.',
        scope='n=1 fixed chosen policy per condition. Retain all five trained agents in the primary benchmark.')
    path=BASE/'main_operating_rule.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    if args.plan_only:print('Source-only fixed-policy rule recorded; no test files read.');return
    assert (training/'completed.json').exists()
    candidates=[]
    for job in protocol['jobs']:
        metrics=json.loads((Path(job['directory'])/'metrics.json').read_text())
        for row in metrics['checkpoints']:
            if row['interactions']==0:continue
            candidates.append(dict(algorithm=job['algorithm'],condition=job['setting'],seed=job['seed'],variant=job['variant'],
                checkpoint_interactions=row['interactions'],model=row['model'],model_sha256=row['model_sha256'],
                development_success=row['success_rate'],development_score=row['mean_score']))
    chosen=[]
    for condition in 'ABC':
        for algorithm in ['Q-learning','PPO']:
            pool=[row for row in candidates if row['condition']==condition and row['algorithm']==algorithm]
            selected=sorted(pool,key=lambda row:(-row['development_success'],-row['development_score'],row['checkpoint_interactions'],row['seed']))[0]
            assert sha(ROOT/selected['model'])==selected['model_sha256']
            chosen.append(selected)
            if condition=='B':chosen.append(dict(selected,condition='D',source_condition='B'))
    result=dict(plan_sha256=sha(path),jobs=chosen,selection_uses_test_data=False)
    (BASE/'main_operating_selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');main(parser.parse_args())
