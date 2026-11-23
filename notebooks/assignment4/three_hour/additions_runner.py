"""Balanced extra candidates registered separately from the frozen screen."""
import argparse
from pathlib import Path
import run_study as study
from additions import AlignmentQ,LowBitsTask

ORIGINAL_MAKE_Q=study.make_q
ORIGINAL_SETTINGS=study.ppo_settings
ORIGINAL_SOURCES=study.sources
ORIGINAL_WORKER=study.worker
ORIGINAL_TASK=study.StudyTask

def make_q(variant,seed):
    return AlignmentQ(seed) if variant=='alignment' else ORIGINAL_MAKE_Q(variant,seed)

def settings(variant):return ORIGINAL_SETTINGS('stable' if variant=='powerclock' else variant)

def sources():
    result=ORIGINAL_SOURCES()
    for path in [Path(__file__),Path(__file__).with_name('additions.py')]:
        result[str(path.relative_to(study.ROOT))]=study.sha(path)
    return result

def worker(job):
    study.StudyTask=LowBitsTask if job['variant']=='powerclock' else ORIGINAL_TASK
    return ORIGINAL_WORKER(job)

study.make_q=make_q;study.ppo_settings=settings;study.sources=sources;study.worker=worker
study.Q_VARIANTS=(*study.Q_VARIANTS,'alignment');study.PPO_VARIANTS=(*study.PPO_VARIANTS,'powerclock')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='three_hour_20261122/additions_v1')
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--seeds',default='42')
    args=parser.parse_args()
    study.main(argparse.Namespace(run_id=args.run_id,workers=args.workers,seeds=args.seeds,budget=262144,
        settings='AB',episodes=20,dev_start=1110000,checkpoints='65536,131072,196608,262144',
        q_variants='alignment',ppo_variants='powerclock'))
