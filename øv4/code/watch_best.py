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
from pathlib import Path

import numpy as np

from mspacman import MsPacManTask, SETTINGS
from part2_run import PROTOCOL_ID, PPOAgent, load_model

HERE = Path(__file__).resolve().parent


def select_best(results, condition=None):
    rows = [row for path in sorted(results.glob('seed_results*.csv'))
            for row in csv.DictReader(path.open(encoding='utf-8'))
            if row['stage'] == 'main' and
            (condition is None or row['condition'] == condition)]
    if not rows:
        raise ValueError(f'No trained agents found in {results} for {condition or "A/B/C"}.')
    best = max(rows, key=lambda r: (float(r['success_rate']),
                                    float(r['mean_score']), r['run_id']))
    return results / best['run_id'] / 'model.npz', best


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


class Playback:
    def __init__(self, model, protocol, condition, seed, greedy=False):
        self.agent = load_model(model)
        self.task = MsPacManTask(protocol, condition)
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
    parser.add_argument('--results', type=Path, default=HERE.parent/'results'/'assignment4_v8')
    parser.add_argument('--model', type=Path, help='inspect a specific model.npz instead of selecting the best')
    parser.add_argument('--condition', choices=list(SETTINGS), help='filter best selection, or evaluate an explicit model here')
    parser.add_argument('--seed', type=int, default=300000, help='episode seed; default matches a recorded evaluation episode')
    parser.add_argument('--greedy', action='store_true', help='use highest-probability PPO actions instead of the primary sampled mode')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--fps', type=float, default=15, help='decision updates per second')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--headless', action='store_true', help='save a preview without opening a window')
    parser.add_argument('--steps', type=int, default=25, help='decisions to inspect in headless mode')
    parser.add_argument('--output', type=Path, help='PNG preview path (required with --headless)')
    parser.add_argument('--trace', type=Path, help='optional JSON of viewed decisions')
    args = parser.parse_args()
    if args.fps <= 0 or args.episodes < 1 or args.steps < 1 or args.seed < 0:
        parser.error('--fps, --episodes and --steps must be positive; --seed must be nonnegative.')
    if args.headless and not args.output:
        parser.error('--headless requires --output PATH.png')
    if args.condition == 'D' and not args.model:
        parser.error('D has no trained models; use --model PATH/TO/B/model.npz --condition D')
    try:
        model, selected = ((args.model.resolve(), None) if args.model else
                           select_best(args.results.resolve(), args.condition))
        protocol = read_protocol(model)
        name_parts = model.parent.name.rsplit('_', 2)
        inferred = selected['condition'] if selected else (name_parts[-2] if len(name_parts) == 3 else None)
        condition = args.condition or inferred
        if condition not in SETTINGS:
            raise ValueError('Specify --condition for a model with a nonstandard directory name.')
        playback = Playback(model, protocol, condition, args.seed, args.greedy)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    import matplotlib
    matplotlib.use('Agg' if args.headless else 'TkAgg')
    import matplotlib.pyplot as plt

    is_ppo = isinstance(playback.agent, PPOAgent)
    mode = ('greedy PPO (additional mode)' if args.greedy else 'sampled PPO') if is_ppo else 'greedy Q, uniform ties'
    print(f'Model: {model}\nSetting: {condition}; policy: {mode}; episode seed: {args.seed}', flush=True)
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
        text.set_text(f'Seed {record["seed"]} | decision {record["step"]} | {outcome}\n'
                      f'AFTER action: score {record["raw_score"]:.0f}, lives {record["lives"]}\n'
                      f'Reward {record["reward"]:+.1f} | return {record["episode_return"]:.1f}\n'
                      f'BEFORE action: ghost distance {obs[4]*160:.1f}px\n'
                      f'Frightened timer {obs[6]:.2f} | open probes {probes}\n'
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
    import subprocess
    import sys

    cli = sys.argv[1:]
    routing = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    routing.add_argument('--results', type=Path)
    routing.add_argument('--model', type=Path)
    routing.add_argument('--condition')
    route, _ = routing.parse_known_args(cli)
    latest = HERE.parent/'results'/'assignment4_cool_ppo_B'/'comparison.json'
    transfer = HERE.parent/'results'/'assignment4_transfer_final'
    use_navigation = (route.results is None and route.model is None
                      and (route.condition in (None, 'B') and latest.is_file()
                           or route.condition == 'D' and (transfer/'comparison.json').is_file()))
    if route.results:
        use_navigation = ((route.results/'comparison.json').is_file()
                          and any((route.results/name).is_file() for name in ('manifest.json', 'frozen.json')))
    elif route.model:
        use_navigation = any((parent/'manifest.json').is_file() and (parent/'comparison.json').is_file()
                             for parent in route.model.resolve().parents)
    # Navigation viewer resolves archived model paths under this project's results.
    if use_navigation:
        viewer = HERE.parent/'code_wall_fix'/'watch_best.py'
        raise SystemExit(subprocess.call([sys.executable, '-X', 'utf8', str(viewer), *cli]))
    main()
