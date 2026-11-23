"""Run one additional matched candidate per algorithm, with separate sources."""
import argparse
from pathlib import Path
import additions_runner
import extra_registry

study=extra_registry.study
OLD_SOURCES=study.sources
def sources():
    result=OLD_SOURCES();path=Path(__file__)
    result[str(path.relative_to(study.ROOT))]=study.sha(path)
    return result
study.sources=sources

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    study.main(argparse.Namespace(run_id='three_hour_20261122/extra_v1',workers=args.workers,seeds='42',budget=262144,
        settings='AB',episodes=20,dev_start=1110000,checkpoints='65536,131072,196608,262144',
        q_variants='gamma999',ppo_variants='explore'))
