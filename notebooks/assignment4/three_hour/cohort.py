"""Train a locked cohort, including the separately defined fifth candidates."""
import argparse
from pathlib import Path
import sys
import run_study_v2 as implementation
sys.modules['run_study']=implementation
import additions_runner as extension
import extra_registry
import compact_registry

study=extension.study
ORIGINAL_SOURCES=study.sources

def sources():
    result=ORIGINAL_SOURCES();path=Path(__file__)
    result[str(path.relative_to(study.ROOT))]=study.sha(path)
    return result

study.sources=sources

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--budget',type=int,default=262144);parser.add_argument('--settings',default='AB')
    parser.add_argument('--seeds',default='7');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--dev-start',type=int,default=1120000)
    parser.add_argument('--checkpoints',default='65536,131072,196608,262144')
    parser.add_argument('--q-variants',required=True);parser.add_argument('--ppo-variants',required=True)
    study.main(parser.parse_args())
