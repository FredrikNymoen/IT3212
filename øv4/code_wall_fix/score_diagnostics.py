"""Test action temperature, collision alternatives, and life-loss reward shaping in B."""
import argparse
import csv
import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from statistics import mean

from imitate_reference import OriginalPPO, SEEDS, np, runner
from compare_navigation import NavigationTask
from extend_dagger_ppo import ENTROPY, LEARNING_RATE, digest, write_json


TEMPERATURES = (1., .75, .5)
VALIDATION_START = 993000
TEST_START = 994000
LIFE_PENALTY = 25.
HORIZON = 24
LOOKBACK = (1, 4, 8)


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class TemperaturePolicy(OriginalPPO):
    def __init__(self, source, temperature):
        super().__init__(source.n, source.p['W1'].shape[0], 0, hidden=source.p['W1'].shape[1])
        self.p = source.p
        self.temperature = temperature

    def policy_value(self, obs):
        _, _, logits, value = self.forward(obs[None, :])
        return runner.softmax(logits/self.temperature)[0], float(value[0])


def evaluate(seed, arm, model, out, count, validation=False, temperature=1.):
    path = out/arm/f'seed_{seed}'
    path.mkdir(parents=True, exist_ok=True)
    runner.MsPacManTask = NavigationTask
    agent = runner.load_model(model)
    if temperature != 1.:
        agent = TemperaturePolicy(agent, temperature)
    label = 'validation' if validation else 'test'
    result = runner.evaluate(agent, 'B', count,
                             seed_start=VALIDATION_START if validation else TEST_START,
                             log_dir=path, label=label, trace_count=int(not validation))
    rows = read_rows(path/f'{label}_B_episodes.csv')
    reached = sum(float(row['raw_score']) >= 1500 for row in rows)
    result.update(seed=seed, arm=arm, temperature=temperature, label=label, episodes=count,
                  reached_1500=reached, conversion_after_1500=(
                      sum(row['outcome']=='success' for row in rows)/reached if reached else None))
    write_json(path/f'{label}.json', result)
    print(f'{label} {arm}, seed {seed}: {result["success_rate"]:.1%} success', flush=True)
    return result


class LifePenaltyTask(NavigationTask):
    def step(self, action):
        old_lives = self.lives
        obs, reward, done, info = super().step(action)
        return obs, reward-LIFE_PENALTY*max(0, old_lives-self.lives), done, info


def train_reward(seed, arm, model, out, budget):
    path = out/arm/f'seed_{seed}'
    path.mkdir(parents=True)
    source = runner.load_model(model)

    class WarmStartPPO(OriginalPPO):
        def __init__(self, n_actions, n_features, train_seed, **config):
            super().__init__(n_actions, n_features, train_seed, hidden=source.p['W1'].shape[1],
                             **config)
            self.p = {name: value.copy() for name, value in source.p.items()}

    runner.MsPacManTask = LifePenaltyTask if arm == 'life_penalty' else NavigationTask
    runner.PPOAgent = WarmStartPPO
    try:
        student = runner.train('PPO', 'B', seed, budget, lr=LEARNING_RATE, entropy=ENTROPY,
                               log_dir=path)
        assert student.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
        runner.MsPacManTask = NavigationTask
    runner.save_model(student, path/'model.npz')
    print(f'{arm}, seed {seed}: {budget:,} training interactions complete', flush=True)
    return path/'model.npz'


def set_episode_rng(agent, seed):
    agent.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))


def replay_collisions(seed, model, source, out, limit):
    rows = read_rows(source/'targeted'/f'seed_{seed}'/'test_B_episodes.csv')
    cases = [row for row in rows if row['outcome']=='failure' and
             1800 <= float(row['raw_score']) < 2000][:limit]
    agent = runner.load_model(model)
    task = NavigationTask(runner.PROTOCOL, 'B')
    ale = task.env.unwrapped.ale
    branches = []
    try:
        for case in cases:
            episode_seed = int(case['seed'])
            obs = task.reset(episode_seed)
            set_episode_rng(agent, episode_seed)
            recent = deque(maxlen=max(LOOKBACK))
            while True:
                # These attributes are assigned fresh by step(); the graph is read-only.
                recent.append((ale.cloneSystemState(), vars(task).copy(), obs.copy(),
                               deepcopy(agent.rng.bit_generator.state)))
                obs, _, done, info = task.step(agent.act(obs))
                if done:
                    break
            assert task.t == int(case['length'])
            assert task.score == float(case['raw_score']) and info['outcome']=='failure'
            for lookback in LOOKBACK:
                if len(recent) < lookback:
                    continue
                emulator, attributes, observation, rng_state = recent[-lookback]
                probs, _ = agent.policy_value(observation)
                agent.rng.bit_generator.state = deepcopy(rng_state)
                nominal = agent.act(observation)
                for action in range(task.n_actions):
                    ale.restoreSystemState(emulator)
                    vars(task).update(attributes)
                    agent.rng.bit_generator.state = deepcopy(rng_state)
                    # Consume the original first draw in every branch.
                    assert agent.act(observation) == nominal
                    start_lives, start_score = task.lives, task.score
                    obs, _, done, info = task.step(action)
                    steps = 1
                    while steps < HORIZON and not done and task.lives == start_lives:
                        obs, _, done, info = task.step(agent.act(obs))
                        steps += 1
                    if action == nominal:
                        # Close enough to death that the original first input must
                        # reproduce it, including sticky-action randomness.
                        assert task.lives < start_lives or info['outcome']=='failure'
                    branches.append(dict(training_seed=seed, episode_seed=episode_seed,
                                         decision=attributes['t']+1, lookback=lookback,
                                         action=task.actions[action], nominal=action==nominal,
                                         probability=float(probs[action]), steps=steps,
                                         no_life_lost=(task.lives==start_lives and info['outcome']!='failure'),
                                         outcome=info['outcome'], start_score=start_score,
                                         end_score=task.score))
    finally:
        task.close()
    if branches:
        write_rows(out/f'collision_seed_{seed}.csv', branches)
    print(f'Collision replay seed {seed}: {len(cases)} reproduced episodes, '
          f'{len(branches)} branches', flush=True)
    return branches


def summarize(rows):
    metrics = ('success_rate', 'mean_score', 'failure_rate', 'timeout_rate',
               'reached_1500', 'conversion_after_1500')
    return {key: mean(values) if (values := [row[key] for row in rows if row[key] is not None])
            else None for key in metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--source', type=Path, default=here.parent/'results'/'assignment4_targeted_dagger_B')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.results.resolve()
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != runner.PROTOCOL or manifest['setting'] != 'B':
        parser.error('The source must be the matching Setting B targeted DAgger run.')
    for name, checksum in manifest['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source study.')
    val_count, test_count, budget, collision_limit = (2, 2, 256, 1) if args.smoke else (60, 200, 30_000, 4)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    inputs = ('manifest.json', *(f'targeted/seed_{seed}/{file}' for seed in seeds
                                 for file in ('model.npz','test_B_episodes.csv')))
    source_sha256 = {name: digest(source/name) for name in inputs}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'manifest.json', dict(
        source=str(source), source_sha256=source_sha256,
        code_sha256={name: digest(here/name) for name in ('score_diagnostics.py', *manifest['code_sha256'])},
        protocol=runner.PROTOCOL, setting='B', seeds=seeds, smoke=args.smoke,
        temperatures=TEMPERATURES, validation_start=VALIDATION_START, validation_episodes=val_count,
        test_start=TEST_START, test_episodes=test_count, budget=budget, life_penalty=LIFE_PENALTY,
        learning_rate=LEARNING_RATE, entropy=ENTROPY, collision_range=[1800, 1999],
        collision_cases_per_seed=collision_limit, lookback=LOOKBACK, horizon=HORIZON))
    validation, tests, collisions = [], [], []
    chosen = None
    with ProcessPoolExecutor(max_workers=3 if args.smoke else 9) as pool:
        jobs = {}
        models = {seed: source/'targeted'/f'seed_{seed}'/'model.npz' for seed in seeds}
        for seed, model in models.items():
            jobs[pool.submit(replay_collisions, seed, model, source, out, collision_limit)] = ('collision', seed, None)
        for seed, model in models.items():
            for arm in ('reward_control', 'life_penalty'):
                jobs[pool.submit(train_reward, seed, arm, model, out, budget)] = ('train', seed, arm)
        for temp in TEMPERATURES:
            for seed, model in models.items():
                arm = f'temp_{temp:g}'
                jobs[pool.submit(evaluate, seed, arm, model, out, val_count, True, temp)] = ('val', seed, arm)
        while jobs:
            done, _ = wait(jobs, return_when=FIRST_COMPLETED)
            for job in done:
                kind, seed, arm = jobs.pop(job)
                result = job.result()
                if kind == 'collision':
                    collisions.extend(result)
                elif kind == 'train':
                    jobs[pool.submit(evaluate, seed, arm, result, out, test_count)] = ('test', seed, arm)
                elif kind == 'val':
                    validation.append(result)
                else:
                    tests.append(result)
            if chosen is None and len(validation)==len(seeds)*len(TEMPERATURES):
                scores = {temp: summarize([r for r in validation if r['temperature']==temp])
                          for temp in TEMPERATURES}
                chosen = max(TEMPERATURES, key=lambda temp: (scores[temp]['success_rate'],
                                                              scores[temp]['mean_score'], temp))
                write_json(out/'temperature_selection.json', dict(chosen=chosen, means=scores,
                                                                   runs=validation))
                print(f'Validation selected temperature {chosen:g}', flush=True)
                for temp in sorted({1., chosen}):
                    for seed, model in models.items():
                        arm = f'temp_{temp:g}'
                        jobs[pool.submit(evaluate, seed, arm, model, out, test_count, False, temp)] = ('test', seed, arm)
    nominal = [row for row in collisions if row['nominal']]
    rescued = {(r['training_seed'], r['episode_seed'], r['lookback']) for r in collisions
               if not r['nominal'] and r['no_life_lost']}
    episode_keys = {(r['training_seed'], r['episode_seed']) for r in nominal}
    collision_summary = dict(episodes=len(episode_keys), positions=len(nominal),
                             positions_with_surviving_alternative=len(rescued),
                             episodes_with_surviving_alternative=len({key[:2] for key in rescued}),
                             episodes_with_success_alternative=len({
                                 (r['training_seed'], r['episode_seed']) for r in collisions
                                 if not r['nominal'] and r['outcome']=='success'}))
    means = {arm: summarize([r for r in tests if r['arm']==arm])
             for arm in sorted({r['arm'] for r in tests})}
    result = dict(chosen_temperature=chosen, test_means=means, collision=collision_summary,
                  test_runs=sorted(tests, key=lambda row: (row['arm'], row['seed'])))
    write_json(out/'comparison.json', result)
    print(json.dumps(dict(test_means=means, collision=collision_summary), indent=2))


if __name__ == '__main__':
    main()
