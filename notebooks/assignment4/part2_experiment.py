"""Reproduce Assignment 4 Part 2 using the Gymnasium API.

From this directory: python part2_experiment.py run
Small engineering check: python part2_experiment.py audit
Report refresh from existing measurements: python part2_experiment.py report
No report PDF is compiled. Existing completed runs can be resumed; protocol
and source hashes must match. Use --run-id to start a separate experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from mspacman_env import MsPacmanTask, TaskConfig, FEATURES, task_description, coarse_state
from rl_agents import make_agent, PPO

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = 'part2_controlled_v2'
ALGORITHMS = ('Q-learning', 'PPO')
TRAIN_SEEDS = (42, 7, 2026)
FINAL_SEEDS = tuple(range(900_000, 900_200))
CURVE_SEEDS = tuple(range(300_000, 300_020))
TUNING_SEEDS = tuple(range(200_000, 200_050))
SOURCES = {
    'gymnasium': 'https://gymnasium.farama.org/api/env/',
    'ale': 'https://ale.farama.org/environments/ms_pacman/',
    'features': 'https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py',
    'map_intervention': 'https://github.com/k4ntz/HackAtari/blob/master/hackatari/games/mspacman.py',
    'ppo': 'https://arxiv.org/abs/1707.06347',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    files = list(HERE.glob('*.py')) + [ROOT / 'reports/assignment4/Sections/01_predictions.tex',
                                      ROOT / 'IT3212 - Assignment 4.pdf']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def protocol(budget, horizon):
    return dict(run_id=DEFAULT_RUN, environment='ALE/MsPacman-v5',
        main_budget=budget, tuning_budget=16384, horizon=horizon,
        training_seeds=TRAIN_SEEDS, tuning_training_seed=31415,
        final_evaluation_seeds=FINAL_SEEDS, tuning_evaluation_seeds=TUNING_SEEDS,
        curve_evaluation_seeds=CURVE_SEEDS, curve_checkpoints=[0, budget//4, budget//2, 3*budget//4, budget],
        candidates={'Q-learning': [{'learning_rate': .1}, {'learning_rate': .3}],
                    'PPO': [{'learning_rate': .0003}, {'learning_rate': .001}]},
        controlled_settings=True,
        selection='mean development success across A/B/C, then mean game score, then candidate index; one common setting per algorithm',
        revision='Corrected per-condition tuning after partial v1 evaluation inspection. Uses development data only; reused evaluation seeds are not a new independent test.',
        final_policy={'Q-learning': 'greedy, uniformly random ties with fixed episode seed',
                      'PPO': 'highest-probability action'},
        reward_approval='Team confirmed score rewards in A/B were agreed with instructor (chat, 2026-11-18).',
        tasks={setting: task_description(TaskConfig(setting, horizon=horizon)) for setting in 'ABCD'},
        q_learning={'gamma': .99, 'epsilon_start': 1., 'epsilon_end': .05, 'decay_budget_fraction': .8},
        ppo={'gamma': .99, 'gae_lambda': .95, 'rollout': 512, 'minibatch': 128,
             'epochs': 4, 'clip': .2, 'entropy': .01, 'value_coefficient': .5,
             'max_grad_norm': .5, 'hidden_layers': [64, 64], 'activation': 'tanh'},
        reward_scale_for_both_agents=.01,
        sources=SOURCES, source_hashes=source_hashes())


def environment_record(workers):
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    packages = {name: metadata.version(name) for name in
                ('numpy', 'pandas', 'torch', 'gymnasium', 'ale-py', 'matplotlib', 'threadpoolctl')}
    import ale_py
    roms = list((Path(ale_py.__file__).parent / 'roms').glob('*pac*'))
    return dict(cpu=cpu, platform=platform.platform(), python=sys.version, packages=packages,
                max_concurrent_workers=workers, torch_threads=1, blas_threads=1,
                rom_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in roms if p.is_file()})


def evaluate(agent, task, seeds, trace_path=None):
    """Frozen parameters, read-only Q lookup, independent and restored policy RNG."""
    fingerprint = agent.fingerprint()
    old_rng = copy.deepcopy(agent.rng.bit_generator.state)
    env = MsPacmanTask(task)
    rows, traces = [], []
    trace = {key: [] for key in ('ram', 'scores', 'action', 'episode', 'step', 'reward')}
    started = perf_counter()
    try:
        for episode, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            agent.rng = np.random.default_rng(int(seed) + 1_000_000)
            for step in range(task.horizon):
                if trace_path:
                    trace['ram'].append(env.ram.copy())
                    trace['scores'].append(agent.scores(obs))
                    trace['episode'].append(episode)
                    trace['step'].append(step)
                action = agent.action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if trace_path:
                    trace['action'].append(action)
                    trace['reward'].append(reward)
                if terminated or truncated:
                    rows.append(dict(episode=episode, evaluation_seed=int(seed),
                        success=int(info['success']), failure=int(not info['success']),
                        score=info['score'], task_return=info['task_return'],
                        length=info['episode_steps'], reason=info['reason']))
                    break
        seconds = perf_counter()-started
    finally:
        env.close()
        agent.rng = np.random.default_rng()
        agent.rng.bit_generator.state = old_rng
    assert fingerprint == agent.fingerprint(), 'Evaluation changed trained parameters.'
    if trace_path:
        arrays = {key: np.asarray(values, dtype={'ram': np.uint8, 'scores': np.float32,
                  'action': np.uint8, 'episode': np.uint16, 'step': np.uint16, 'reward': np.float32}[key])
                  for key, values in trace.items()}
        np.savez_compressed(trace_path, **arrays)
    return rows, seconds


def summary(rows):
    frame = pd.DataFrame(rows)
    return dict(success_rate=float(frame.success.mean()), failure_rate=float(frame.failure.mean()),
        mean_score=float(frame.score.mean()), mean_return=float(frame.task_return.mean()),
        mean_length=float(frame.length.mean()),
        success_length=float(frame.loc[frame.success.eq(1), 'length'].mean()) if frame.success.any() else None,
        failure_length=float(frame.loc[frame.failure.eq(1), 'length'].mean()) if frame.failure.any() else None,
        evaluation_episodes=len(frame), time_limit_failures=int(frame.reason.eq('time_limit').sum()))


def train_agent(agent, task, budget, seed, checkpoints=()):
    env = MsPacmanTask(task)
    episode_rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(episode_rng.integers(100_000)))
    rollout, episodes, curves, updates = [], [], [], []
    started, evaluation_seconds, frames = perf_counter(), 0.0, 0
    for interaction in range(budget+1):
        if interaction in checkpoints:
            rows, seconds = evaluate(agent, task, CURVE_SEEDS)
            evaluation_seconds += seconds
            curves.append(dict(interactions=interaction, **summary(rows)))
        if interaction == budget:
            break
        if isinstance(agent, PPO):
            probabilities, value = agent.distribution(obs)
            action = int(agent.rng.choice(5, p=probabilities))
        else:
            epsilon = max(.05, 1-.95*interaction/(.8*budget))
            action = agent.action(obs, explore=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        frames += info['emulator_frames']
        done = terminated or truncated
        if isinstance(agent, PPO):
            next_value = agent.distribution(next_obs)[1]
            rollout.append((obs, action, np.log(max(float(probabilities[action]), 1e-12)),
                            value, reward, next_value, terminated, done))
            if len(rollout) == 512 or interaction+1 == budget:
                metrics = agent.update(rollout)
                updates.append(dict(interactions=interaction+1, policy_loss=metrics[0],
                                    value_loss=metrics[1], entropy=metrics[2]))
                rollout.clear()
        else:
            agent.update(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if done:
            episodes.append(dict(interactions=interaction+1, score=info['score'],
                task_return=info['task_return'], length=info['episode_steps'],
                success=int(info['success']), reason=info['reason']))
            obs, _ = env.reset(seed=int(episode_rng.integers(100_000)))
    train_seconds = perf_counter()-started-evaluation_seconds
    env.close()
    return dict(train_seconds=train_seconds, curve_evaluation_seconds=evaluation_seconds,
                training_interactions=budget, emulator_frames=frames,
                completed_training_episodes=len(episodes), q_states=len(agent.table) if hasattr(agent, 'table') else None), episodes, curves, updates


def worker(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory = Path(job['directory'])
        complete = directory / 'complete.json'
        if complete.is_file():
            return json.loads(complete.read_text())
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'configuration.json', job)
        task = TaskConfig(job['setting'], horizon=job['horizon'])
        agent = make_agent(job['algorithm'], job['seed'], job['parameters'])
        checkpoints = job.get('checkpoints', [])
        timing, episodes, curves, updates = train_agent(agent, task, job['budget'], job['seed'], checkpoints)
        pd.DataFrame(episodes).to_csv(directory / 'training_episodes.csv', index=False)
        if curves:
            pd.DataFrame(curves).to_csv(directory / 'learning_curve.csv', index=False)
        if updates:
            pd.DataFrame(updates).to_csv(directory / 'ppo_updates.csv', index=False)
        with (directory / 'agent.pkl').open('wb') as handle:
            pickle.dump(agent, handle)
        rows, seconds = evaluate(agent, task,
            TUNING_SEEDS if job['stage'] == 'tuning' else FINAL_SEEDS,
            directory / 'evaluation_traces.npz' if job['stage'] == 'final' else None)
        pd.DataFrame(rows).to_csv(directory / 'evaluation_episodes.csv', index=False)
        output = {**job, **timing, **summary(rows), 'evaluation_seconds': seconds,
                  'parameter_fingerprint': agent.fingerprint()}
        if job['stage'] == 'final' and job['setting'] == 'B':
            drows, dseconds = evaluate(agent, TaskConfig('D', horizon=job['horizon']), FINAL_SEEDS,
                                      directory / 'transfer_traces.npz')
            pd.DataFrame(drows).to_csv(directory / 'transfer_episodes.csv', index=False)
            output['transfer'] = dict(**summary(drows), evaluation_seconds=dseconds,
                additional_training_interactions=0, parameter_fingerprint=agent.fingerprint())
        write_json(complete, output)
        return output


def run_jobs(jobs, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(stage=row['stage'], algorithm=row['algorithm'],
                setting=row['setting'], seed=row['seed'], success=row['success_rate'],
                score=row['mean_score'], completed=len(rows), total=len(jobs))), flush=True)
    return rows


def reuse_compatible(source, output, stage, current, selected=None):
    """Reuse identical measured jobs with explicit origin; never relabel changed jobs."""
    previous = json.loads((source / 'protocol.json').read_text())
    for filename in ('mspacman_env.py', 'rl_agents.py', '01_predictions.tex'):
        old = next(v for k, v in previous['source_hashes'].items() if Path(k).name == filename)
        new = next(v for k, v in current['source_hashes'].items() if Path(k).name == filename)
        if old != new:
            raise ValueError('Cannot reuse changed numerical definitions or predictions: '+filename)
    for key in ('tasks', 'q_learning', 'ppo', 'reward_scale_for_both_agents',
                'training_seeds', 'tuning_training_seed', 'final_evaluation_seeds',
                'tuning_evaluation_seeds', 'curve_evaluation_seeds', 'curve_checkpoints'):
        if previous[key] != current[key]:
            raise ValueError('Cannot reuse changed protocol: '+key)
    manifest_path = output / 'reuse_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for complete in sorted((source / stage).glob('*/complete.json')):
        row = json.loads(complete.read_text())
        if row['horizon'] != current['horizon']:
            continue
        if stage == 'final':
            if row['budget'] != current['main_budget'] or row['parameters'] != selected[f"{row['setting']}_{row['algorithm']}"]:
                continue
        elif row['budget'] != current['tuning_budget'] or row['parameters'] not in current['candidates'][row['algorithm']]:
            continue
        destination = output / stage / complete.parent.name
        if destination.exists():
            continue
        shutil.copytree(complete.parent, destination)
        origin = dict(source_run=source.name, original_directory=str(complete.parent),
            original_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest())
        row['directory'], row['reused_from'] = str(destination), origin
        config = json.loads((destination/'configuration.json').read_text())
        config['directory'], config['reused_from'] = str(destination), origin
        write_json(destination/'configuration.json', config)
        write_json(destination/'complete.json', row)
        manifest.append(dict(stage=stage, destination=str(destination), **origin))
    write_json(manifest_path, manifest)


def run(run_id=DEFAULT_RUN, budget=65536, horizon=1000, workers=3, reuse_from=None):
    if budget < 2048 or budget % 2048:
        raise ValueError('Budget must be a positive multiple of 2048 (at least 2048).')
    output = ROOT / 'results/assignment4' / run_id
    output.mkdir(parents=True, exist_ok=True)
    current = json.loads(json.dumps(protocol(budget, horizon)))
    current['run_id'] = run_id
    path = output / 'protocol.json'
    if path.is_file():
        if json.loads(path.read_text()) != current:
            raise ValueError('Protocol/source changed. Use a new --run-id; do not overwrite runs.')
    else:
        write_json(path, current)
        write_json(output / 'environment.json', environment_record(workers))
    source = ROOT / 'results/assignment4' / reuse_from if reuse_from else None
    if source is not None:
        reuse_compatible(source, output, 'tuning', current)
    tuning = []
    for setting in 'ABC':
        for algorithm in ALGORITHMS:
            for index, parameters in enumerate(current['candidates'][algorithm]):
                tuning.append(dict(stage='tuning', algorithm=algorithm, setting=setting,
                    candidate=index, seed=31415, parameters=parameters, horizon=horizon,
                    budget=current['tuning_budget'],
                    directory=str(output / 'tuning' / f'{setting}_{algorithm}_{index}')))
    tuned = run_jobs(tuning, workers)
    selected = {}
    for algorithm in ALGORITHMS:
        choices = []
        for index, parameters in enumerate(current['candidates'][algorithm]):
            group = [r for r in tuned if r['algorithm'] == algorithm and r['candidate'] == index]
            assert len(group) == 3
            choices.append((np.mean([r['success_rate'] for r in group]),
                            np.mean([r['mean_score'] for r in group]), -index, parameters))
        common = max(choices, key=lambda r: r[:3])[3]
        for setting in 'ABC':
            selected[f'{setting}_{algorithm}'] = common
    # Written before any final held-out evaluation starts; final data never tune.
    lock = output / 'selected_parameters.json'
    if lock.is_file() and json.loads(lock.read_text()) != selected:
        raise ValueError('Selected parameters changed on resume.')
    write_json(lock, selected)
    if source is not None:
        reuse_compatible(source, output, 'final', current, selected)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'transfer'} for r in tuned]).to_csv(output / 'tuning_summary.csv', index=False)
    final = [dict(stage='final', algorithm=algorithm, setting=setting, seed=seed,
        parameters=selected[f'{setting}_{algorithm}'], horizon=horizon, budget=budget,
        checkpoints=current['curve_checkpoints'],
        directory=str(output / 'final' / f'{setting}_{algorithm}_{seed}'))
        for setting in 'ABC' for algorithm in ALGORITHMS for seed in TRAIN_SEEDS]
    rows = run_jobs(final, workers)
    write_json(output / 'final_summary.json', rows)
    report(run_id)
    return output


def audit():
    """Meaningful integration checks; no assessed training or tuning."""
    from gymnasium.utils.env_checker import check_env
    from PIL import Image
    directory = ROOT / 'results/assignment4/engineering_audit'
    directory.mkdir(parents=True, exist_ok=True)
    fingerprints = []
    for setting in 'ABCD':
        task = TaskConfig(setting, horizon=16, target_score=2000)
        env = MsPacmanTask(task, render_mode='rgb_array')
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=123)
        assert env.observation_space.contains(first)
        transitions, rewards, frames = [], [], 0
        for action in [0, 1, 2, 3]*4:
            obs, reward, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert info['maze'] == task.maze
            transitions.append(obs)
            rewards.append(reward)
            frames += info['emulator_frames']
        assert term and not trunc and info['reason'] == 'time_limit'
        assert len(transitions) == 16 and frames == 64
        Image.fromarray(env.render()).save(directory / f'maze_{setting}.png')
        fingerprints.append(hashlib.sha256(env.render().tobytes()).hexdigest())
        env.close()
    assert fingerprints[1] != fingerprints[3], 'D must actually change the rendered maze.'
    # B/C must differ only in reward when seed and actions are the same.
    b, c = [MsPacmanTask(TaskConfig(s, horizon=64)) for s in 'BC']
    bo, _ = b.reset(seed=512)
    co, _ = c.reset(seed=512)
    np.testing.assert_array_equal(bo, co)
    for action in [1, 2, 3, 4]*16:
        bo, br, bt, bx, bi = b.step(action)
        co, cr, ct, cx, ci = c.step(action)
        np.testing.assert_array_equal(bo, co)
        assert bt == ct and bx == cx and bi['score'] == ci['score']
        np.testing.assert_allclose(br-cr, .1, atol=1e-12)
    b.close()
    c.close()
    success_env = MsPacmanTask(TaskConfig('A', horizon=200, target_score=10))
    success_env.reset(seed=42)
    for _ in range(200):
        _, _, term, trunc, info = success_env.step(3)
        if term or trunc:
            break
    assert term and info['success'] and info['reason'] == 'success'
    success_env.close()
    # Verify frozen evaluation does not grow a Q table, mutate weights or
    # disturb the training policy's RNG. Also check NumPy/Torch PPO agreement.
    for algorithm in ALGORITHMS:
        agent = make_agent(algorithm, 42, {})
        before = copy.deepcopy(agent.rng.bit_generator.state)
        rows, _ = evaluate(agent, TaskConfig('B', horizon=16), (900000, 900001))
        assert len(rows) == 2 and agent.rng.bit_generator.state == before
        if isinstance(agent, PPO):
            obs = np.linspace(-.5, .5, len(FEATURES), dtype=np.float32)
            with torch.no_grad():
                logits, value = agent.network(torch.tensor(obs))
            probabilities, numpy_value = agent.distribution(obs)
            np.testing.assert_allclose(probabilities, logits.softmax(-1).numpy(), atol=1e-6)
            np.testing.assert_allclose(numpy_value, value.item(), atol=1e-6)
        old_fingerprint = agent.fingerprint()
        timing, episodes, curves, updates = train_agent(agent, TaskConfig('B', horizon=64), 512, 42)
        assert timing['training_interactions'] == 512
        assert agent.fingerprint() != old_fingerprint
        if isinstance(agent, PPO):
            assert np.isfinite(np.array(list(agent.weights.values()), dtype=object)[0]).all()
            assert all(np.isfinite(weight).all() for weight in agent.weights.values())
        else:
            assert len(agent.table) > 0
            start, following = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
            agent.table = {coarse_state(following): np.full(5, 10., dtype=np.float32)}
            agent.update(start, 0, 100, following, True)
            np.testing.assert_allclose(agent.scores(start)[0], agent.learning_rate)
    write_json(directory / 'audit.json', dict(gymnasium_api=True,
        finite_horizon_checked=True, frozen_evaluation_checked=True,
        ppo_inference_checked=True, reward_isolation_checked=True,
        success_threshold_checked=True, training_updates_checked=True,
        map_image_hashes=fingerprints,
        source_hashes=source_hashes()))
    print('AUDIT PASSED: Gymnasium contract, horizon, map change, frozen evaluation and PPO inference.', flush=True)


def report(run_id=DEFAULT_RUN):
    from part2_report import render
    return render(ROOT / 'results/assignment4' / run_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'run', 'report'))
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--budget', type=int, default=65536)
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reuse-from', default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'audit':
        audit()
    elif args.command == 'run':
        run(args.run_id, args.budget, args.horizon, args.workers, args.reuse_from)
    else:
        report(args.run_id)
