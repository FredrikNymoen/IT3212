"""Branch near-miss episodes before the terminal animation, not during it."""
import argparse
import csv
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from score_diagnostics import NavigationTask, digest, np, runner, set_episode_rng, write_json, write_rows


LOOKBACK = (1, 4, 8, 16)
HORIZON = 64


def replay(seed, source, diagnostics, out):
    with (diagnostics/f'collision_seed_{seed}.csv').open(encoding='utf-8') as stream:
        cases = {int(row['episode_seed']): (int(row['decision'])+int(row['lookback'])-1,
                                            float(row['end_score']))
                 for row in csv.DictReader(stream) if row['nominal']=='True'}
    agent = runner.load_model(source/'targeted'/f'seed_{seed}'/'model.npz')
    task = NavigationTask(runner.PROTOCOL, 'B')
    ale = task.env.unwrapped.ale
    branches = []
    try:
        for episode, (length, score) in cases.items():
            obs = task.reset(episode)
            set_episode_rng(agent, episode)
            recent = deque(maxlen=64)
            original_actions = []
            while True:
                ram = ale.getRAM().copy()
                snapshot = (ram, vars(task).copy(), obs.copy(),
                            deepcopy(agent.rng.bit_generator.state))
                original_actions.append(agent.act(obs))
                obs, _, done, info = task.step(original_actions[-1])
                recent.append((snapshot, np.array_equal(ram[6:17], ale.getRAM()[6:17])))
                if done:
                    break
            assert task.t==length and task.score==score and info['outcome']=='failure'
            # First six consecutive frozen decisions in the last 27. The ROM
            # normally freezes actors for roughly 20 decisions on a final hit.
            frozen = [entry[1] for entry in recent]
            starts = [i for i in range(len(frozen)-27, len(frozen)-5) if all(frozen[i:i+6])
                      and sum(not flag for flag in frozen[i:]) <= 2]
            if not starts:
                raise RuntimeError(f'Could not locate the terminal animation in episode {episode}.')
            freeze = starts[0]
            freeze_decision = recent[freeze][0][1]['t']+1
            for lookback in LOOKBACK:
                ram, attrs, obs, rng = recent[freeze-lookback][0]
                probabilities, _ = agent.policy_value(obs)
                agent.rng.bit_generator.state = deepcopy(rng)
                original_action = agent.act(obs)
                for action in range(task.n_actions):
                    # Replay emulator inputs from reset to recover all sticky-action
                    # state. The saved task attributes supply the observation history.
                    task.reset(episode)
                    for earlier in original_actions[:attrs['t']]:
                        _, _, terminal, truncated, _ = task.env.step(earlier)
                        assert not terminal and not truncated
                    assert np.array_equal(ram, ale.getRAM())
                    vars(task).update(attrs)
                    agent.rng.bit_generator.state = deepcopy(rng)
                    assert agent.act(obs)==original_action
                    lives = task.lives
                    next_obs, _, done, info = task.step(action)
                    steps = 1
                    while steps < HORIZON and not done and (action==original_action or task.lives==lives):
                        next_obs, _, done, info = task.step(agent.act(next_obs))
                        steps += 1
                    if action == original_action:
                        if task.t != length or task.score != score or info['outcome'] != 'failure':
                            raise RuntimeError(
                                f'Seed {seed}, episode {episode}, lookback {lookback}: '
                                f'expected failure at {length} with {score}, '
                                f'got {info["outcome"]} at {task.t} with {task.score}.')
                    branches.append(dict(training_seed=seed, episode_seed=episode,
                                         decision=attrs['t']+1, freeze_decision=freeze_decision,
                                         lookback=lookback, action=task.actions[action],
                                         nominal=action==original_action, probability=float(probabilities[action]),
                                         steps=steps, no_life_lost=task.lives==lives and info['outcome']!='failure',
                                         outcome=info['outcome'], start_score=attrs['score'], end_score=task.score))
    finally:
        task.close()
    write_rows(out/f'pre_freeze_seed_{seed}.csv', branches)
    return branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    diagnostics = args.diagnostics.resolve()
    manifest = json.loads((diagnostics/'manifest.json').read_text(encoding='utf-8'))
    source = Path(manifest['source'])
    out = args.output.resolve() if args.output else diagnostics/'pre_freeze'
    out.mkdir()
    with ProcessPoolExecutor(max_workers=len(manifest['seeds'])) as pool:
        jobs = [pool.submit(replay, seed, source, diagnostics, out) for seed in manifest['seeds']]
        rows = [row for job in jobs for row in job.result()]
    key = lambda row: (row['training_seed'], row['episode_seed'])
    alternatives = [row for row in rows if not row['nominal'] and row['no_life_lost']]
    finish = [row for row in alternatives if row['outcome']=='success']
    summary = dict(episodes=len({key(row) for row in rows}),
                   replayed_positions=sum(row['nominal'] for row in rows),
                   episodes_with_surviving_alternative=len({key(row) for row in alternatives}),
                   episodes_with_success_alternative=len({key(row) for row in finish}),
                   positions_with_surviving_alternative=len({(*key(row),row['decision']) for row in alternatives}),
                   horizon=HORIZON, freeze_lookbacks=LOOKBACK,
                   replay_code_sha256=digest(Path(__file__)))
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
