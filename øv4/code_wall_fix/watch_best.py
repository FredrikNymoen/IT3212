"""Watch a saved MsPacMan policy and inspect every action choice.

    python watch_best.py
    python watch_best.py --paused --seed 300001
    python watch_best.py --condition B
    python watch_best.py --headless --steps 25 --output ../results/viewer.png

Best means highest recorded primary success rate among trained agents, then
highest raw score. This is retrospective demo selection, not a new unbiased
performance estimate. No model parameters or experiment records are changed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np

from mspacman import MsPacManTask, SETTINGS
from part2_run import PROTOCOL_ID, PPOAgent, load_model
from compare_navigation import NavigationTask
from score_diagnostics import TemperaturePolicy

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent/'results'
FRONTIER_STUDY = RESULTS/'assignment4_ppo_hour_frontier_B'
TRANSFER_STUDY = RESULTS/'assignment4_transfer_final'
LATE_TRANSFER_STUDY = RESULTS/'assignment4_late_confirmation'
AVERAGED_TRANSFER_STUDY = RESULTS/'assignment4_late_averaged_confirmation'
DEFAULT_STUDY = (FRONTIER_STUDY if (FRONTIER_STUDY/'comparison.json').is_file()
                 else RESULTS/'assignment4_cool_ppo_B')


def resolve_model_path(recorded):
    """Locate an archived checkpoint relative to this project's results folder."""
    saved = Path(recorded)
    parts = PureWindowsPath(str(recorded)).parts
    indexes = [i for i, part in enumerate(parts) if part.lower() == 'results']
    if indexes:
        local = RESULTS.joinpath(*parts[indexes[-1] + 1:])
        if local.is_file():
            return local.resolve()
    if saved.is_file():
        return saved.resolve()
    raise FileNotFoundError(
        f'Checkpoint missing: {recorded}. Expected the same path under {RESULTS}. '
        'Extract the complete results folder from the archive.')


def select_best(results, condition=None):
    if (results/'comparison.json').is_file() and (results/'manifest.json').is_file():
        config = read_navigation_manifest(results)
        if condition and condition != config['setting']:
            raise ValueError(f'This study evaluated Setting {config["setting"]}.')
        report = json.loads((results/'comparison.json').read_text(encoding='utf-8'))
        candidates = report['runs']
        if 'means' in report:
            # Select the best group across training seeds before showing one model.
            means = report['means']
            arm = max(means, key=lambda name: (means[name]['success_rate'], means[name]['mean_score']))
            candidates = [row for row in candidates if row['name'] == arm]
        best = max(candidates, key=lambda r: (r['success_rate'], r['mean_score']))
        return resolve_model_path(best['model']), dict(best, condition=config['setting'],
                                        final_evaluation_episodes=config['eval_episodes'])
    rows = [row for path in sorted(results.glob('seed_results*.csv'))
            for row in csv.DictReader(path.open(encoding='utf-8'))
            if row['stage'] == 'main' and
            (condition is None or row['condition'] == condition)]
    if not rows:
        raise ValueError(f'No trained agents found in {results} for {condition or "A/B/C"}.')
    best = max(rows, key=lambda r: (float(r['success_rate']),
                                    float(r['mean_score']), r['run_id']))
    return results / best['run_id'] / 'model.npz', best


def read_navigation_manifest(directory):
    config = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if (config.get('smoke') or 'compare_navigation.py' not in config.get('code_sha256', {})
            or ('eval_temperature' not in config and 'test_episodes' not in config)):
        raise ValueError('This study does not record a compatible navigation policy for playback.')
    if 'eval_temperature' not in config:
        config['eval_seed_start'] = config['test_start']
        config['eval_episodes'] = config['test_episodes']
        if 'temperature' in config:
            config['eval_temperature'] = config['temperature']
    for name in ('compare_navigation.py', 'mspacman.py', 'part2_run.py', 'score_diagnostics.py'):
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != config['code_sha256'][name]:
            raise ValueError(f'{name} differs from the recorded navigation study.')
    return config


def read_protocol(model):
    paths = sorted(model.parent.parent.glob('part2_config*.json'))
    if not paths:
        raise ValueError('No experiment manifest found alongside the model directory.')
    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    config = manifests[0]
    if any(m['protocol_id'] != config['protocol_id'] or
           m['protocol'] != config['protocol'] or
           m['code_sha256'] != config['code_sha256'] for m in manifests):
        raise ValueError('The experiment directory contains inconsistent manifests.')
    if config['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Model protocol differs from the current code. Use its original code.')
    for name, expected in config['code_sha256'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name} differs from the recorded training source.')
    return config['protocol']


def transfer_complete(results):
    return (results/'frozen.json').is_file() and (results/'comparison.json').is_file()


def preferred_transfer():
    if not transfer_complete(LATE_TRANSFER_STUDY):
        return TRANSFER_STUDY
    try:
        path = LATE_TRANSFER_STUDY/'comparison.json'
        means = json.loads(path.read_text(encoding='utf-8'))['means']
        baseline = means['anchor']['1']['success_rate']
        candidates = [(settings['1']['success_rate'], settings['1']['mean_score'], LATE_TRANSFER_STUDY)
                      for name, settings in means.items() if name != 'anchor']
    except (OSError, ValueError, KeyError):
        return TRANSFER_STUDY
    if transfer_complete(AVERAGED_TRANSFER_STUDY):
        try:
            plans = [json.loads((p/'frozen.json').read_text(encoding='utf-8'))
                     for p in (LATE_TRANSFER_STUDY, AVERAGED_TRANSFER_STUDY)]
            keys = ('seed_start', 'episodes', 'protocol', 'temperature', 'primary_test')
            if all(plans[0][key] == plans[1][key] for key in keys):
                path = AVERAGED_TRANSFER_STUDY/'comparison.json'
                metrics = json.loads(path.read_text(encoding='utf-8'))['means']['averaged']['1']
                candidates.append((metrics['success_rate'], metrics['mean_score'], AVERAGED_TRANSFER_STUDY))
        except (OSError, ValueError, KeyError):
            pass
    if candidates:
        success, _, directory = max(candidates, key=lambda candidate: candidate[:2])
        if success > baseline:
            return directory
    return TRANSFER_STUDY


def read_transfer(results, condition, model):
    from cached_navigation import CACHED_TASKS

    plan = json.loads((results/'frozen.json').read_text(encoding='utf-8'))
    comparison = json.loads((results/'comparison.json').read_text(encoding='utf-8'))
    condition = condition or 'D'
    layout = {'B': 0, 'D': 1}.get(condition)
    if layout not in plan['layouts']:
        raise ValueError(f'The transfer study did not evaluate Setting {condition}.')
    candidates = [row for row in comparison['runs'] if row['layout'] == layout]
    if model:
        candidates = [row for row in candidates if resolve_model_path(row['model']) == model.resolve()]
    else:
        means = {arm: rows[str(layout)] for arm, rows in comparison['means'].items()}
        best_arm = max(means, key=lambda arm: (means[arm]['success_rate'], means[arm]['mean_score']))
        candidates = [row for row in candidates if row['arm'] == best_arm]
    if not candidates:
        raise ValueError(f'No evaluated transfer checkpoint found for Setting {condition}.')
    best = max(candidates, key=lambda row: (row['success_rate'], row['mean_score']))
    entry = next((row for row in plan['runs']
                  if (row['arm'], row['seed']) == (best['arm'], best['seed'])), None)
    if entry is None:
        raise ValueError('The selected checkpoint has no frozen entry.')
    saved = resolve_model_path(entry['model'])
    if saved != resolve_model_path(best['model']) or hashlib.sha256(saved.read_bytes()).hexdigest() != entry['model_sha256']:
        raise ValueError('The checkpoint differs from the frozen transfer plan.')
    selected = dict(best, final_evaluation_episodes=best['episodes'])
    if 'opening_model' in entry:
        opening = resolve_model_path(entry['opening_model'])
        if hashlib.sha256(opening.read_bytes()).hexdigest() != entry['opening_sha256']:
            raise ValueError('The opening checkpoint differs from the frozen transfer plan.')
        selected.update(opening_model=opening, switch_score=entry['switch_score'])
    if entry['feature_type'] == 'wrapped':
        from wrapped_transfer import WrappedNavigationTask
        task_type = WrappedNavigationTask
    else:
        task_type = CACHED_TASKS[entry['feature_type']]
    return saved, selected, condition, plan, task_type


class Playback:
    def __init__(self, model, protocol, condition, seed, greedy=False,
                 navigation=False, temperature=1., task_type=None, opening_model=None,
                 switch_score=None):
        self.agent = load_model(model)
        if opening_model is not None:
            from late_specialist import PhasePolicy
            self.agent = PhasePolicy(load_model(opening_model), self.agent, temperature, switch_score)
        elif isinstance(self.agent, PPOAgent) and temperature != 1.:
            self.agent = TemperaturePolicy(self.agent, temperature)
        task_type = task_type or (NavigationTask if navigation else MsPacManTask)
        self.task = task_type(protocol, condition)
        self.condition, self.seed, self.greedy = condition, seed, greedy
        self.records = []
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed
        self.obs = self.task.reset(seed)
        self.agent.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.done, self.total = False, 0.
        self.last = None

    def advance(self):
        if self.done:
            return self.last
        obs = self.obs.copy()
        frame = self.task.env.unwrapped.ale.getScreenRGB().copy()
        if isinstance(self.agent, PPOAgent):
            values, _ = self.agent.policy_value(obs)
            action = self.agent.act(obs, sampled=not self.greedy)
            kind = 'probabilities'
        else:
            values = self.agent.scores(obs).copy()
            action = self.agent.act(obs)
            kind = 'q_values'
        self.obs, reward, self.done, info = self.task.step(action)
        self.total += reward
        record = dict(seed=self.seed, setting=self.condition, step=self.task.t,
                      state=obs.tolist(), next_state=self.obs.tolist(),
                      action=self.task.actions[action], action_index=action,
                      actions=self.task.actions, **{kind: values.tolist()},
                      reward=reward, episode_return=self.total, **info)
        self.records.append(record)
        # This screenshot precedes the action, so bars and image describe the same state.
        self.last = frame, values, action, obs, record
        return self.last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, help='study directory (defaults to the latest completed model comparison)')
    parser.add_argument('--model', type=Path, help='inspect a specific model.npz instead of selecting the best')
    parser.add_argument('--condition', choices=list(SETTINGS), help='filter best selection, or evaluate an explicit model here')
    parser.add_argument('--seed', type=int, help='episode seed; default matches a recorded evaluation episode')
    parser.add_argument('--temperature', type=float, help='override the recorded PPO sampling temperature')
    parser.add_argument('--greedy', action='store_true', help='use highest-probability PPO actions instead of the primary sampled mode')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--fps', type=float, default=15, help='decision updates per second')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--headless', action='store_true', help='save a preview without opening a window')
    parser.add_argument('--steps', type=int, default=25, help='decisions to inspect in headless mode')
    parser.add_argument('--output', type=Path, help='PNG preview path (required with --headless)')
    parser.add_argument('--trace', type=Path, help='optional JSON of viewed decisions')
    args = parser.parse_args()
    if (args.fps <= 0 or args.episodes < 1 or args.steps < 1
            or (args.seed is not None and args.seed < 0)):
        parser.error('--fps, --episodes and --steps must be positive; --seed must be nonnegative.')
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error('--temperature must be positive and finite.')
    if args.headless and not args.output:
        parser.error('--headless requires --output PATH.png')
    latest = DEFAULT_STUDY
    if args.condition == 'D' and not args.model:
        candidate = preferred_transfer()
        if transfer_complete(candidate):
            latest = candidate
    results = (args.results or latest).resolve()
    transfer = transfer_complete(results)
    if args.condition == 'D' and not args.model and not transfer:
        parser.error('D has no trained models; use --model PATH/TO/B/model.npz --condition D')
    try:
        if transfer:
            model, selected, condition, plan, task_type = read_transfer(results, args.condition, args.model)
            seed = args.seed if args.seed is not None else selected['seed_start']
            temperature = args.temperature if args.temperature is not None else plan['temperature']
            playback = Playback(model, plan['protocol'], condition, seed, args.greedy,
                                temperature=temperature, task_type=task_type,
                                opening_model=selected.get('opening_model'),
                                switch_score=selected.get('switch_score'))
        else:
            model, selected, condition, seed, temperature, playback = read_legacy(args, results)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    display(args, model, selected, condition, seed, temperature, playback)


def read_legacy(args, results):
    model, selected = ((args.model.resolve(), None) if args.model else
                       select_best(results, args.condition))
    if args.model and not args.results:
        frontier_path = FRONTIER_STUDY/'manifest.json'
        frontier = json.loads(frontier_path.read_text(encoding='utf-8')) if frontier_path.is_file() else {}
        if (FRONTIER_STUDY/'comparison.json').is_file() and any(
                resolve_model_path(row['model']) == model for row in frontier.get('candidates', ())):
            results = FRONTIER_STUDY
        else:
            results = next((parent for parent in model.parents
                            if (parent/'manifest.json').is_file()), model.parent)
    navigation = (results/'manifest.json').is_file() and (results/'comparison.json').is_file()
    config = read_navigation_manifest(results) if navigation else {}
    protocol = config['protocol'] if navigation else read_protocol(model)
    name_parts = model.parent.name.rsplit('_', 2)
    inferred = (config['setting'] if navigation else selected['condition'] if selected
                else name_parts[-2] if len(name_parts) == 3 else None)
    condition = args.condition or inferred
    if condition not in SETTINGS:
        raise ValueError('Specify --condition for a model with a nonstandard directory name.')
    seed = args.seed if args.seed is not None else config.get('eval_seed_start', 300000)
    recorded_temperature = selected.get('temperature') if selected else None
    if recorded_temperature is None and navigation and 'eval_temperature' not in config:
        matches = [row['temperature'] for row in config.get('candidates', ())
                   if resolve_model_path(row['model']) == model.resolve()]
        if len(set(matches)) > 1 and args.temperature is None:
            raise ValueError('This checkpoint was tested at multiple temperatures. Specify --temperature.')
        recorded_temperature = matches[0] if matches else None
    temperature = (args.temperature if args.temperature is not None else
                   recorded_temperature if recorded_temperature is not None else config.get('eval_temperature', 1.))
    playback = Playback(model, protocol, condition, seed, args.greedy, navigation, temperature)
    return model, selected, condition, seed, temperature, playback


def display(args, model, selected, condition, seed, temperature, playback):
    import matplotlib
    matplotlib.use('Agg' if args.headless else 'TkAgg')
    import matplotlib.pyplot as plt

    is_ppo = isinstance(playback.agent, PPOAgent)
    mode = ('greedy PPO' if args.greedy else f'sampled PPO, T={temperature:g}') if is_ppo else 'greedy Q, uniform ties'
    print(f'Model: {model}\nSetting: {condition}; policy: {mode}; episode seed: {seed}', flush=True)
    selection = 'Explicit model'
    if selected:
        selection = (f'Best recorded success: {float(selected["success_rate"]):.1%}; '
                     f'mean score {float(selected["mean_score"]):.0f} over '
                     f'{selected["final_evaluation_episodes"]} evaluation episodes')
        print(selection + '\nSelection is retrospective; these are not fresh performance estimates.', flush=True)

    fig = plt.figure(figsize=(12, 8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.55), height_ratios=(2.2, 1))
    game = fig.add_subplot(grid[:, 0])
    chart = fig.add_subplot(grid[0, 1])
    details = fig.add_subplot(grid[1, 1])
    details.axis('off')
    game.axis('off')
    screen = game.imshow(playback.task.env.unwrapped.ale.getScreenRGB(), interpolation='nearest')
    game.set_title('State BEFORE the selected action')
    bars = chart.barh(playback.task.actions, np.zeros(playback.task.n_actions), color='#477bb5')
    labels = [chart.text(0, i, '', va='center') for i in range(playback.task.n_actions)]
    chart.invert_yaxis()
    chart.set_xlabel('Action probability' if is_ppo else 'Q-value')
    text = details.text(0, 1, '', va='top', family='monospace', fontsize=10)
    fig.suptitle(f'{model.parent.name} | {mode}\n{selection}', fontsize=12)
    fig.supxlabel('Space: pause/play   Right: one decision   N: next episode   R: replay   Esc: close')
    state = dict(paused=args.paused, episodes=1)

    def draw(result):
        frame, values, action, obs, record = result
        screen.set_data(frame)
        low = min(0., float(values.min()))
        high = max(float(values.max()), .01)
        span = max(high-low, .01)
        chart.set_xlim(low-.05*span, high+.35*span)
        for i, (bar, value, label) in enumerate(zip(bars, values, labels)):
            bar.set_width(value)
            bar.set_color('#2c9b69' if i == action else '#477bb5')
            label.set_position((float(value)+.02*span, i))
            label.set_text(f'{value:.1%}' if is_ppo else f'{value:.3f}')
        chart.set_title(f'Chosen: {record["action"]} (green) | largest value: '
                        f'{playback.task.actions[int(np.argmax(values))]}')
        outcome = record['outcome'] or 'in progress'
        probes = ', '.join(f'{name}={int(value)}' for name, value in
                           zip(('U','D','L','R'), obs[10:14]))
        frightened_timer = obs[44] if len(obs) == 55 else obs[6]
        text.set_text(f'Seed {record["seed"]} | decision {record["step"]} | {outcome}\n'
                      f'AFTER action: score {record["raw_score"]:.0f}, lives {record["lives"]}\n'
                      f'Reward {record["reward"]:+.1f} | return {record["episode_return"]:.1f}\n'
                      f'BEFORE action: ghost distance {obs[4]*160:.1f}px\n'
                      f'Frightened timer {frightened_timer:.2f} | open probes {probes}\n'
                      f'Pellet offset ({obs[14]*80:+.1f}, {obs[15]*80:+.1f})px\n'
                      f'Chosen action is a request; sticky actions may repeat prior input.\n'
                      f'{"PAUSED" if state["paused"] else "PLAYING"} | no parameter updates')
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(1, round(1000/args.fps)))

    def next_episode():
        if state['episodes'] >= args.episodes:
            state['paused'] = True
            return False
        state['episodes'] += 1
        playback.reset(playback.seed+1)
        return True

    def tick(force=False):
        if state['paused'] and not force:
            return
        if playback.done and not next_episode():
            if playback.last:
                draw(playback.last)
            return
        draw(playback.advance())

    def key(event):
        if event.key in (' ', 'space'):
            state['paused'] = not state['paused']
            if playback.last: draw(playback.last)
        elif event.key == 'right':
            state['paused'] = True
            tick(force=True)
        elif event.key == 'n':
            if next_episode(): tick(force=True)
        elif event.key == 'r':
            playback.reset(playback.seed)
            tick(force=True)
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', key)
    timer.add_callback(tick)
    try:
        if args.headless:
            state['paused'] = True
            for _ in range(args.steps):
                draw(playback.advance())
                if playback.done: break
        else:
            tick(force=True)
            timer.start()
            plt.show()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output, dpi=150)
            print(f'Preview: {args.output.resolve()}', flush=True)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(dict(model=str(model), mode=mode,
                                                 decisions=playback.records), indent=2), encoding='utf-8')
            print(f'Trace: {args.trace.resolve()}', flush=True)
    finally:
        timer.stop()
        playback.task.close()
        plt.close(fig)


if __name__ == '__main__':
    main()
