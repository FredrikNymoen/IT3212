"""Sixth candidates: isolate longer Q horizon and sustained PPO entropy."""
from pathlib import Path
import run_study as study
from candidates import TabularQ

OLD_MAKE=study.make_q
OLD_SETTINGS=study.ppo_settings
OLD_SOURCES=study.sources
OLD_CALLBACK=study.Checkpoints

def make_q(variant,seed):
    return TabularQ(seed,alpha=.15,gamma=.999) if variant=='gamma999' else OLD_MAKE(variant,seed)

def settings(variant):return OLD_SETTINGS('stable' if variant=='explore' else variant)

def sources():
    result=OLD_SOURCES();path=Path(__file__)
    result[str(path.relative_to(study.ROOT))]=study.sha(path)
    return result

class EntropyCheckpoints(OLD_CALLBACK):
    def _on_step(self):
        if self.model.policy.actor_variant=='explore':
            self.model.ent_coef=.01+.03*max(0,1-self.num_timesteps/self.budget)
            return True
        return super()._on_step()

study.make_q=make_q;study.ppo_settings=settings;study.sources=sources;study.Checkpoints=EntropyCheckpoints
study.Q_VARIANTS=(*study.Q_VARIANTS,'gamma999');study.PPO_VARIANTS=(*study.PPO_VARIANTS,'explore')
