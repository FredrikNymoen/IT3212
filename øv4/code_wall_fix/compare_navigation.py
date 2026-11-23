"""Paired PPO comparison on B: wall fix with and without navigation features."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import part2_run as runner
from mspacman import GHOST_X, GHOST_Y, MAZE_TOP, SCREEN_DX, SCREEN_DY, MsPacManTask
from reference_policy import Maze


class NavigationTask(MsPacManTask):
    """Add reachable food distances, recent motion, and previous action to v8."""

    def _read_layout(self):
        super()._read_layout()
        key = (self.walls.tobytes(), self.pellets.tobytes())
        if key != getattr(self, '_maze_key', None):
            self._maze_key = key
            self.graph = Maze(self.walls, self.pellets)
            self._pellet_nodes = dict(zip(map(tuple, self.pellets),
                                          self.graph.nearest(self.pellets)))
        self.last_action = None
        self.previous = None

    def step(self, action):
        self.previous = self._actors(self.env.unwrapped.ale.getRAM())
        self.previous_lives = self.lives
        self.last_action = action
        return super().step(action)

    def _actors(self, ram):
        player = self._position(ram)
        ghosts = [(float(ram[y])+SCREEN_DY-MAZE_TOP, float(ram[x])+SCREEN_DX)
                  for x, y in zip(GHOST_X, GHOST_Y)]
        return np.array([player, *ghosts])+4.5

    def features(self, ram):
        base = super().features(ram)
        actors = self._actors(ram)
        position, ghosts = actors[0], actors[1:]
        exits = self.graph.exits(position)
        food = np.fromiter((self._pellet_nodes[tuple(p)] for p in self.pellets), dtype=int)
        ghost_distance = np.abs(ghosts-position).sum(axis=1)
        nearest = int(ghost_distance.argmin())
        ghost_nodes = self.graph.nearest(ghosts)
        food_distance = np.ones(4)
        ghost_path = np.ones(4)
        for action, ((_, distance), node) in exits.items():
            food_distance[action] = min(1., (distance+self.graph.distance[node, food].min(initial=np.inf))/160)
            ghost_path[action] = min(1., (distance+self.graph.distance[node, ghost_nodes].min())/160)
        velocities = np.zeros((2, 2))
        closing = 0.
        action_onehot = np.zeros(self.n_actions)
        if self.previous is not None and self.previous_lives == self.lives:
            previous_ghost = self.previous[nearest+1]
            velocities = np.clip((actors[[0, nearest+1]]-self.previous[[0, nearest+1]])/8, -1, 1)
            closing = np.clip((np.abs(previous_ghost-position).sum()-ghost_distance[nearest])/8, -1, 1)
            action_onehot[self.last_action] = 1.
        return np.concatenate((base, food_distance, ghost_path, velocities.ravel(),
                               [closing], action_onehot)).astype(np.float32)


def train_one(variant, seed, budget, episodes, directory):
    if variant == 'navigation':
        runner.MsPacManTask = NavigationTask
    else:
        runner.MsPacManTask = MsPacManTask
    path = directory/f'{variant}_{seed}'
    path.mkdir()
    agent = runner.train('PPO', 'B', seed, budget, lr=.001, log_dir=path)
    runner.save_model(agent, path/'model.npz')
    result = runner.evaluate(agent, 'B', episodes, log_dir=path, trace_count=2)
    with (path/'final_B_episodes.csv').open(encoding='utf-8') as stream:
        results = list(csv.DictReader(stream))
    result.update(variant=variant, seed=seed, budget=agent.environment_interactions,
                  episodes=episodes, reached_1500=sum(float(r['raw_score']) >= 1500 for r in results))
    (path/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=100_000)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.budget <= runner.START_DECISIONS or args.episodes < 1:
        parser.error('Use a positive worker and episode count and a budget above startup length.')
    seeds = [42] if args.smoke else runner.TRAIN_SEEDS
    if args.smoke:
        args.budget, args.episodes = 256, 2
    elif args.episodes < 200:
        parser.error('Use at least 200 final episodes in the full comparison.')
    directory = args.results.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    manifest = dict(protocol=runner.PROTOCOL, baseline_protocol_id=runner.PROTOCOL_ID,
                    setting='B', algorithm='PPO', lr=.001, policy='sampled',
                    training_seeds=seeds, budget=args.budget, episodes=args.episodes,
                    eval_seed_start=runner.EVAL_SEED_START, smoke=args.smoke,
                    code_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ('compare_navigation.py','reference_policy.py',
                                              'mspacman.py','part2_run.py')})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    collected = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(train_one, variant, seed, args.budget, args.episodes, directory)
                for seed in seeds for variant in ('baseline', 'navigation')]
        for job in as_completed(jobs):
            result = job.result()
            collected.append(result)
            print(f'{result["variant"]} {result["seed"]}: '
                  f'{result["success_rate"]:.1%} success, score {result["mean_score"]:.0f}', flush=True)
    paired = []
    for seed in seeds:
        by_variant = {r['variant']: r for r in collected if r['seed'] == seed}
        paired.append(dict(seed=seed, **{key: by_variant['navigation'][key]-by_variant['baseline'][key]
                                        for key in ('success_rate','mean_score','median_score')}))
    report = dict(runs=sorted(collected, key=lambda r: (r['variant'], r['seed'])), paired_deltas=paired,
                  mean_success_delta=mean(r['success_rate'] for r in paired),
                  sd_success_delta=stdev(r['success_rate'] for r in paired) if len(paired)>1 else None)
    (directory/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
