"""Register seventh candidates without touching any frozen earlier source."""
from pathlib import Path
import additions_runner
import extra_registry
from compact_candidates import CompactQ,SmallPolicy

study=extra_registry.study
OLD_MAKE=study.make_q;OLD_SETTINGS=study.ppo_settings;OLD_SOURCES=study.sources
OLD_WORKER=study.worker;OLD_POLICY=study.StudyPolicy

def make_q(variant,seed):return CompactQ(seed) if variant=='compact' else OLD_MAKE(variant,seed)
def settings(variant):return OLD_SETTINGS('stable' if variant=='small' else variant)
def sources():
    result=OLD_SOURCES()
    for path in [Path(__file__),Path(__file__).with_name('compact_candidates.py')]:
        result[str(path.relative_to(study.ROOT))]=study.sha(path)
    return result
def worker(job):
    study.StudyPolicy=SmallPolicy if job['variant']=='small' else OLD_POLICY
    return OLD_WORKER(job)

study.make_q=make_q;study.ppo_settings=settings;study.sources=sources;study.worker=worker
study.Q_VARIANTS=(*study.Q_VARIANTS,'compact');study.PPO_VARIANTS=(*study.PPO_VARIANTS,'small')
