"""Part 2: run Settings A/B/C/D on MsPacMan for Q-learning and PPO.

    python part2_run.py --selfcheck
    python part2_run.py
    python part2_tables.py

Default output is results/assignment4_v8 under the project. Each trained
policy has a model.npz, raw episode CSVs, and step traces for the first two
primary evaluation episodes. To reconstruct another logged primary episode:

    python part2_run.py --replay PATH/TO/model.npz --condition D --eval-seed 300017

The protocol sets a +100 success bonus (5% of the 2,000-point goal),
applied once. Raw score always comes from the ROM. The common interaction
budget includes forced startup actions. D sets RAM[0] before startup. This
is a modified start state; both wall landmarks are checked before training.

Writes seed_results.csv there, one row per trained agent per setting, in the schema
part2_tables.py reads. The task, the success definition and the RAM map live in
mspacman.py; this file is only the two agents and the experiment loop.

Settings change one thing at a time (mspacman.SETTINGS):
    A  deterministic      sticky 0.00, maze mode 0
    B  stochastic         sticky 0.25, maze mode 0
    C  B plus step cost   sticky 0.25, maze mode 0, -0.5 per decision
    D  transfer           sticky 0.25, held-out maze 1, same mode as B

Both agents see the same 17 features. Q discretises them into a compact table;
PPO keeps them continuous and learns a small tanh network over them, which is the
distinction the submitted Part 1 predicts on.

Evaluation policy, declared before the main study as the brief's rule 5 requires:
greedy for Q, sampled for PPO. The choice was made from separate pilot runs on
Setting A at 25k and 60k decisions, not from any result reported below. Greedy
PPO is degenerate in those pilots, scoring
exactly 60.0 with zero spread at both 25k and 60k decisions, because the argmax of
a lightly trained near-uniform policy is one fixed action, which in a maze means
walking into a wall. Sampled PPO scores 571 then 657 over the same budgets, so it
is the mode that reflects what was learned. Greedy PPO is still recorded, in the
greedy_* columns, as the separately reported additional mode.
"""
import argparse
import csv
import gzip
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mspacman import MsPacManTask, SETTINGS, START_DECISIONS

PROTOCOL = dict(frameskip=4, success_score=2000, max_decisions=1500, success_bonus=100.)
PROTOCOL_ID = 'mspacman_v8_power_pellet_wall_fix'
TRAIN_SEEDS = [42, 7, 2026, 123, 999]
EVAL_SEED_START = 300_000          # disjoint from every training seed
CURVE_SEED_START = 200_000         # disjoint from both training and final evaluation
CURVE_EPISODES = 20                # per checkpoint; the final measurement uses 200
TUNE_SEED_START = 100_000
TUNE_EPISODES = 20
# Rule 6: the same allowance for both approaches. Two candidates each, the same
# seeds, the same reduced budget. The ranking key is fixed before the search runs
# and reads only the tuning seed range, never the held-out evaluation episodes.
CANDIDATES = {
    'Q':   [dict(epsilon_fraction=.6), dict(epsilon_fraction=.9),
            dict(position_bins=4), dict(lr=.3)],
    'PPO': [dict(lr=3e-4), dict(lr=1e-3),
            dict(entropy=.003), dict(rollout=1024)],
}


# ----------------------------------------------------------------- Q-learning

class QAgent:
    """Tabular Q-learning over a deliberately coarse discretisation."""

    def __init__(self, n_actions, seed, position_bins=6, distance_bins=4,
                 lr=.1, gamma=.99, epsilon_fraction=.6, min_epsilon=.05):
        self.n, self.table = n_actions, {}
        self.rng = np.random.default_rng(seed)
        self.position_bins, self.distance_bins = position_bins, distance_bins
        self.lr, self.gamma = lr, gamma
        self.epsilon_fraction, self.min_epsilon = epsilon_fraction, min_epsilon

    def key(self, obs):
        """Every observation PPO receives, coarsened.

        An earlier key dropped the second ghost, the clock and the lives entirely,
        so "PPO beats Q" partly measured "more information beats less". Q now sees
        all of it; the resolution is lower because a table has to pay for each
        distinction, and that cost is the algorithmic difference under study.
        """
        b, d = self.position_bins, self.distance_bins
        return (min(b-1, int(obs[0]*b)), min(b-1, int(obs[1]*b)),       # position
                int(np.sign(obs[2])), int(np.sign(obs[3])),             # ghost direction
                min(d-1, int(obs[4]*d)),                                # nearest ghost
                int(obs[5] < .25),                                      # second ghost close
                int(obs[6] > 0),                                        # frightened
                min(3, int(obs[7]*4)),                                  # score progress
                int(obs[8] < .5),                                       # past half time
                int(obs[9] < .5),                                       # lives running out
                int(obs[10]), int(obs[11]), int(obs[12]), int(obs[13]), # open directions
                int(np.sign(obs[14])), int(np.sign(obs[15])),           # pellet direction
                min(2, int(obs[16]*3)))                                 # pellet distance

    def seen(self, obs):
        return self.key(obs) in self.table

    def scores(self, obs):
        # Evaluation must never insert unseen states into the learned table.
        return self.table.get(self.key(obs), np.zeros(self.n, dtype=np.float32))

    def act(self, obs, progress=1., training=False):
        epsilon = 0.
        if training:
            decayed = 1 - progress/self.epsilon_fraction
            epsilon = max(self.min_epsilon, min(1., decayed))
        if training and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n))
        q = self.scores(obs)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, obs, action, reward, nxt, terminal):
        q = self.table.setdefault(self.key(obs), np.zeros(self.n, dtype=np.float32))
        target = reward + (0. if terminal else self.gamma*self.scores(nxt).max())
        q[action] += self.lr*(target - q[action])


# ------------------------------------------------------------------- PPO


class Adam:
    """The course notebook's optimiser, unchanged."""

    def __init__(self, shape):
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, params, grad, lr):
        self.t += 1
        self.m = .9*self.m + .1*grad
        self.v = .999*self.v + .001*grad**2
        params -= lr*(self.m/(1-.9**self.t))/(np.sqrt(self.v/(1-.999**self.t))+1e-8)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


class PPOAgent:
    """Clipped-objective PPO over a two-layer tanh network, written out by hand.

    Policy and value share the body, as in the lecture material. Backpropagation
    is explicit, so selfcheck() compares it against finite differences.
    """

    def __init__(self, n_actions, n_features, seed, lr=3e-4, gamma=.99,
                 gae_lambda=.95, clip=.2, epochs=4, entropy=.01, rollout=512, hidden=64,
                 reward_scale=.01):
        # Game scores are hundreds, so unscaled value targets give gradients ~65x the
        # policy's. Because both heads share the body, the global 0.5 clip then fired
        # on every update and throttled the policy too. Scaling is applied to the
        # learner's own copy of the reward only: the task, the logged return and the
        # reported score keep the brief's definition.
        rng = np.random.default_rng(seed)
        self.rng, self.n = rng, n_actions
        scale = lambda a, b: rng.normal(0, np.sqrt(1/a), (a, b)).astype(np.float64)
        self.p = dict(W1=scale(n_features, hidden), b1=np.zeros(hidden),
                      W2=scale(hidden, hidden), b2=np.zeros(hidden),
                      Wp=scale(hidden, n_actions)*.01, bp=np.zeros(n_actions),
                      Wv=scale(hidden, 1)*.01, bv=np.zeros(1))
        self.opt = {k: Adam(v.shape) for k, v in self.p.items()}
        self.lr, self.gamma, self.gae_lambda = lr, gamma, gae_lambda
        self.clip, self.epochs, self.entropy, self.rollout = clip, epochs, entropy, rollout
        self.reward_scale = reward_scale

    def forward(self, x):
        p = self.p
        h1 = np.tanh(x @ p['W1'] + p['b1'])
        h2 = np.tanh(h1 @ p['W2'] + p['b2'])
        return h1, h2, h2 @ p['Wp'] + p['bp'], (h2 @ p['Wv'] + p['bv']).squeeze(-1)

    def policy_value(self, obs):
        _, _, logits, value = self.forward(obs[None, :])
        return softmax(logits)[0], float(value[0])

    def act(self, obs, progress=1., training=False, sampled=True):
        probabilities, _ = self.policy_value(obs)
        if training or sampled:
            return int(self.rng.choice(self.n, p=probabilities))
        return int(probabilities.argmax())

    def backward(self, x, dlogits, dvalue):
        """Gradients of a scalar loss, given its derivative at the two heads."""
        p = self.p
        h1, h2, _, _ = self.forward(x)
        g = {'Wp': h2.T @ dlogits, 'bp': dlogits.sum(0),
             'Wv': h2.T @ dvalue[:, None], 'bv': dvalue.sum(0, keepdims=True)}
        dh2 = dlogits @ p['Wp'].T + dvalue[:, None] @ p['Wv'].T
        dz2 = dh2*(1 - h2**2)
        g['W2'], g['b2'] = h1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ p['W2'].T)*(1 - h1**2)
        g['W1'], g['b1'] = x.T @ dz1, dz1.sum(0)
        return g

    def update(self, batch):
        x = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch])
        advantages = np.array([b[4] for b in batch])
        returns = np.array([b[5] for b in batch])
        old_logp = np.array([b[6] for b in batch])
        normalised = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        for _ in range(self.epochs):
            for ids in np.array_split(self.rng.permutation(len(batch)),
                                      max(1, len(batch)//64)):
                xb, a, adv = x[ids], actions[ids], normalised[ids]
                _, _, logits, value = self.forward(xb)
                probabilities = softmax(logits)
                logp = np.log(probabilities[np.arange(len(ids)), a] + 1e-12)
                ratio = np.exp(logp - old_logp[ids])
                # Clipped surrogate: outside the trust region the gradient is zero.
                unclipped = ratio*adv
                clipped = np.clip(ratio, 1-self.clip, 1+self.clip)*adv
                active = (unclipped <= clipped).astype(np.float64)
                onehot = np.zeros_like(probabilities)
                onehot[np.arange(len(ids)), a] = 1.
                # d(-ratio*adv)/dlogits, plus an entropy bonus that is maximised.
                coefficient = -(active*ratio*adv)/len(ids)
                dlogits = coefficient[:, None]*(onehot - probabilities)
                logprob = np.log(probabilities + 1e-12)
                entropy_term = probabilities*(logprob + (probabilities*logprob).sum(1, keepdims=True)*-1)
                dlogits += self.entropy*entropy_term/len(ids)
                dvalue = (value - returns[ids])/len(ids)
                grads = self.backward(xb, dlogits, dvalue)
                total = sum(float((g**2).sum()) for g in grads.values())
                norm = max(1., np.sqrt(total)/.5)      # gradient clipping at 0.5
                for name, grad in grads.items():
                    self.opt[name].step(self.p[name], grad/norm, self.lr)


# --------------------------------------------------------------- experiment


def save_model(agent, path):
    if isinstance(agent, QAgent):
        # The key width follows QAgent.key; hardcoding it silently breaks saving
        # the moment a feature is added, after training and evaluation are done.
        keys = np.array(list(agent.table), dtype=np.int32).reshape(len(agent.table), -1)
        values = np.array(list(agent.table.values()), dtype=np.float32).reshape(-1, agent.n)
        np.savez_compressed(path, algorithm='Q', n=agent.n, keys=keys, values=values,
                            position_bins=agent.position_bins, distance_bins=agent.distance_bins)
    else:
        np.savez_compressed(path, algorithm='PPO', n=agent.n, **agent.p)


def load_model(path):
    """Restore a final evaluation policy from its numeric weights."""
    with np.load(path, allow_pickle=False) as data:
        if data['algorithm'].item() == 'Q':
            agent = QAgent(int(data['n']), 0, int(data['position_bins']), int(data['distance_bins']))
            agent.table = dict(zip(map(tuple, data['keys']), data['values']))
        else:
            inputs, hidden = data['W1'].shape
            agent = PPOAgent(int(data['n']), inputs, 0, hidden=hidden)
            agent.p = {key: data[key] for key in agent.p}
    return agent


def write_rows(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def evaluate(agent, setting, episodes, protocol=PROTOCOL, sampled=True,
             seed_start=EVAL_SEED_START, log_dir=None, label='final', trace_count=0):
    # Both agents draw actions from agent.rng, so a mid-training checkpoint would
    # otherwise consume draws and change the training trajectory it is measuring.
    training_rng = agent.rng
    start = time.perf_counter()
    task = MsPacManTask(protocol, setting)
    rows = []
    records = []
    try:
        for episode in range(episodes):
            seed = seed_start + episode
            obs = task.reset(seed)
            agent.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            total, length = 0., 0
            unseen = 0
            trace = []
            while True:
                action = (agent.act(obs, sampled=sampled)
                          if isinstance(agent, PPOAgent) else agent.act(obs))
                previous = obs
                # A tabular key that was never trained returns all zeros, so the
                # action is an arbitrary tie-break. Counting these says how much of
                # a changed map the table simply has no entry for.
                if isinstance(agent, QAgent) and not agent.seen(previous):
                    unseen += 1
                obs, reward, done, info = task.step(action)
                total += reward; length += 1
                if log_dir is not None and episode < trace_count:
                    is_q = isinstance(agent, QAgent)
                    values = agent.scores(previous) if is_q else agent.policy_value(previous)[0]
                    trace.append(dict(step=length, state=previous.tolist(), action=task.actions[action],
                                      actions=task.actions, next_state=obs.tolist(), reward=reward,
                                      **{'q_values' if is_q else 'probabilities': values.tolist()}, **info))
                if done:
                    break
            rows.append((info['outcome'], length, total, info['raw_score'],
                         unseen/max(length, 1)))
            if log_dir is not None:
                records.append(dict(seed=seed, setting=setting, length=length, outcome=info['outcome'],
                                    raw_score=info['raw_score'], episode_return=total))
                if trace:
                    with gzip.open(log_dir / f'{label}_{setting}_{seed}.json.gz', 'wt', encoding='utf-8') as f:
                        json.dump(trace, f)
    finally:
        task.close()
        agent.rng = training_rng
    if log_dir is not None:
        write_rows(log_dir / f'{label}_{setting}_episodes.csv', records)
    summary = {}
    for outcome in ('success', 'cleared', 'failure', 'timeout'):
        lengths = [n for o, n, _, _, _ in rows if o == outcome]
        summary[outcome + '_rate'] = len(lengths)/episodes
        # No episodes of this kind means no length, which is not a length of zero.
        summary[outcome + '_length'] = float(np.mean(lengths)) if lengths else None
    summary['mean_return'] = float(np.mean([r for _, _, r, _, _ in rows]))
    # Only meaningful for Q; zero for PPO, which has no table to miss from.
    summary['unseen_state_fraction'] = float(np.mean([u for _, _, _, _, u in rows]))
    # B and C define the reward differently, so return cannot compare them. The raw
    # game score can, and it is the only non-degenerate axis while successes are rare.
    scores = np.array([s for _, _, _, s, _ in rows])
    summary['mean_score'] = float(scores.mean())
    summary['median_score'] = float(np.median(scores))
    summary['max_score'] = float(scores.max())
    summary['evaluation_seconds'] = time.perf_counter() - start
    summary['evaluation_decisions'] = task.decisions
    return summary


def train(algorithm, setting, seed, budget, protocol=PROTOCOL, eval_every=0, log_dir=None,
          curve_episodes=CURVE_EPISODES, **config):
    task = MsPacManTask(protocol, setting)
    agent = (QAgent(task.n_actions, seed, **config) if algorithm == 'Q'
             else PPOAgent(task.n_actions, len(task.reset(seed)), seed, **config))
    rng = np.random.default_rng(seed)
    agent.curve = []
    agent.curve_seconds = 0.

    def checkpoint():
        spent = task.decisions
        if not eval_every or (spent % eval_every and spent != budget):
            return
        started = time.perf_counter()
        point = evaluate(agent, setting, curve_episodes, protocol,
                         seed_start=CURVE_SEED_START, log_dir=log_dir, label=f'curve_{spent}')
        # Checkpoint cost is not training cost; keep it out of training_seconds.
        agent.curve_seconds += time.perf_counter() - started
        agent.curve.append(dict(environment_interactions=spent, decisions=spent-task.startup,
                                success_rate=point['success_rate'],
                                mean_return=point['mean_return'],
                                # C subtracts a step cost, so only score compares to B.
                                mean_score=point['mean_score']))

    checkpoint()
    episode_rows = []
    done = True
    batch, pending = [], []
    try:
        while task.decisions < budget:
            if done:
                episode_seed = int(rng.integers(10**6, 2**31-1))
                remaining = min(START_DECISIONS, budget - task.decisions)
                obs = task.reset(episode_seed, remaining, checkpoint)
                episode_return = 0.
                if task.decisions == budget:
                    break
            # Startup steps are not decisions, so they must not consume exploration.
            progress = (task.decisions - task.startup)/max(1, budget - task.startup)
            if algorithm == 'PPO':
                probabilities, value = agent.policy_value(obs)
                action = int(agent.rng.choice(agent.n, p=probabilities))
                logp = float(np.log(probabilities[action] + 1e-12))
            else:
                action = agent.act(obs, progress, training=True)
            nxt, reward, done, info = task.step(action)
            episode_return += reward
            # A horizon timeout cuts the episode; that state still has a future,
            # so only a real ending zeroes the bootstrap. Inert while agents die
            # early, but it bites as soon as they survive towards the cap.
            terminal = done and info['outcome'] != 'timeout'
            # A remaining budget shorter than the next startup yields no more samples.
            last_sample = task.decisions == budget or (done and budget-task.decisions <= START_DECISIONS)
            if algorithm == 'Q':
                agent.update(obs, action, reward, nxt, terminal)
            else:
                pending.append([obs, action, reward*agent.reward_scale, terminal,
                                0., 0., logp, value])
                if len(batch) + len(pending) >= agent.rollout or last_sample or done:
                    tail = 0. if terminal else agent.policy_value(nxt)[1]
                    advantage = 0.
                    for i in reversed(range(len(pending))):
                        row = pending[i]
                        next_v = tail if i == len(pending)-1 else pending[i+1][7]
                        keep = 0. if row[3] else 1.
                        delta = row[2] + agent.gamma*next_v*keep - row[7]
                        advantage = delta + agent.gamma*agent.gae_lambda*keep*advantage
                        row[4] = advantage
                        row[5] = advantage + row[7]
                    batch.extend(pending); pending = []
                if len(batch) >= agent.rollout or (last_sample and batch):
                    agent.update(batch); batch = []
            if log_dir is not None and (done or last_sample):
                episode_rows.append(dict(seed=episode_seed, setting=setting, length=task.t,
                                         outcome=info['outcome'] or 'budget_cut', raw_score=info['raw_score'],
                                         episode_return=episode_return, environment_interactions=task.decisions))
            obs = nxt
            checkpoint()
    finally:
        # The brief counts interactions, not decisions: reset() spends START_DECISIONS
        # steps per episode skipping the opening animation, and those are real steps.
        agent.environment_interactions = task.decisions
        agent.startup_interactions = task.startup
        task.close()
    if log_dir is not None:
        write_rows(log_dir / 'training_episodes.csv', episode_rows)
    return agent


def tune(budget, seeds, protocol, results, suffix):
    """Pick one configuration per algorithm under an identical search allowance."""
    chosen, log = {}, []
    for algorithm, options in CANDIDATES.items():
        scored = []
        for index, config in enumerate(options):
            summaries = []
            for seed in seeds:
                agent = train(algorithm, 'B', seed, budget, protocol, **config)
                summaries.append(evaluate(agent, 'B', TUNE_EPISODES, protocol,
                                          seed_start=TUNE_SEED_START))
            # Predeclared: mean score first, then success rate as the tie-break.
            key = (float(np.mean([s['mean_score'] for s in summaries])),
                   float(np.mean([s['success_rate'] for s in summaries])))
            scored.append((key, index, config))
            log.append(dict(algorithm=algorithm, candidate=index, config=config,
                            mean_score=key[0], success_rate=key[1], seeds=len(seeds),
                            budget=budget, evaluation_episodes=TUNE_EPISODES))
            print(f'  tune {algorithm:>3} {config}  score {key[0]:.0f}', flush=True)
        best = max(scored, key=lambda item: item[0])
        if sum(item[0] == best[0] for item in scored) > 1:
            print(f'WARNING: {algorithm} tuning did not separate its candidates; '
                  'the first is reported as selected by order, not by evidence.')
        chosen[algorithm] = best[2]
    (results/f'selected{suffix}.json').write_text(
        json.dumps(dict(selected=chosen, search=log, seed_range=TUNE_SEED_START,
                        note='Chosen on tuning seeds only; held-out episodes were not read.'),
                   indent=2), encoding='utf-8')
    return chosen


def selfcheck():
    """Hand-written backprop is the one thing here that can be silently wrong."""
    agent = PPOAgent(9, 17, seed=0)
    x = np.random.default_rng(1).normal(size=(5, 17))
    target = np.random.default_rng(2).normal(size=(5, 9))

    def loss(p=None):
        _, _, logits, value = agent.forward(x)
        return float(((logits - target)**2).sum()/2 + (value**2).sum()/2)

    _, _, logits, value = agent.forward(x)
    grads = agent.backward(x, logits - target, value)
    worst = 0.
    for name in ('W1', 'b1', 'W2', 'b2', 'Wp', 'bp', 'Wv', 'bv'):
        flat, g = agent.p[name].ravel(), grads[name].ravel()
        for j in range(0, flat.size, max(1, flat.size//6)):
            original, eps = flat[j], 1e-6
            flat[j] = original + eps; up = loss()
            flat[j] = original - eps; down = loss()
            flat[j] = original
            worst = max(worst, abs((up - down)/(2*eps) - g[j]))
    print(f'  backprop vs finite differences: worst absolute error {worst:.2e}')
    assert worst < 1e-5, 'PPO gradients are wrong'

    probabilities, _ = agent.policy_value(np.zeros(17))
    assert abs(probabilities.sum() - 1) < 1e-9 and len(probabilities) == 9
    print('  policy is a distribution over all 9 actions')

    for algorithm in ('Q', 'PPO'):
        trained = train(algorithm, 'A', 42, 3000)
        summary = evaluate(trained, 'A', 3)
        rates = sum(summary[o + '_rate']
                    for o in ('success', 'cleared', 'failure', 'timeout'))
        assert abs(rates - 1) < 1e-9, 'outcomes must partition the episodes'
        assert trained.environment_interactions == 3000
        print(f'  {algorithm:>3}: 3000 interactions, mean return {summary["mean_return"]:.0f}')
    verify_mazes()
    print('selfcheck passed')


def verify_mazes():
    """Check two wall/corridor locations well away from the spawning sprites."""
    for setting, maze in [('B', 0), ('D', 1)]:
        task = MsPacManTask(PROTOCOL, setting)
        try:
            obs = task.reset(42)
            assert obs.shape == (17,) and task.n_actions == 9
            assert task.walls.any() and len(task.pellets) > 100
            row, column = task._position(task.env.unwrapped.ale.getRAM())
            assert not task.walls[int(row), int(column)], (
                'the RAM-to-screen offset puts the player inside a wall')
            ale = task.env.unwrapped.ale
            screen = ale.getScreenRGB()
            wall = (228, 111, 111)
            corridor = (0, 28, 136)
            landmarks = tuple(screen[4, 40]), tuple(screen[16, 32])
            expected = (corridor, wall) if maze else (wall, corridor)
            if int(ale.getRAM()[0]) != maze or landmarks != expected:
                raise RuntimeError(f'Maze {maze} failed layout verification in {setting}')
        finally:
            task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--budget', type=int, default=100_000,
                        help='environment interactions per trained agent')
    parser.add_argument('--episodes', type=int, default=200,
                        help='evaluation episodes per trained agent (brief requires >= 200)')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='checkpoint every N interactions for the learning curve '
                             '(0 = a quarter of the budget, giving five points)')
    parser.add_argument('--only', default='',
                        help='run one shard, e.g. Q:A, PPO:B or just Q. Shards are '
                             'independent, write their own CSV and can run at the '
                             'same time; part2_tables.py reads them all. A B shard '
                             'also produces its own D transfer rows.')
    parser.add_argument('--tune', action='store_true',
                        help='run the rule 6 hyperparameter search first and use its result')
    parser.add_argument('--tune-budget', type=int, default=25_000)
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='short technical test in its own directory')
    parser.add_argument('--replay', type=Path, metavar='MODEL', help='replay one logged evaluation episode')
    parser.add_argument('--condition', choices=list('ABCD'))
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--greedy', action='store_true', help='replay the additional greedy PPO evaluation')
    args = parser.parse_args()
    if args.replay:
        if args.condition is None or args.eval_seed is None:
            parser.error('Replay needs --condition and --eval-seed')
        run_dir = args.replay.resolve().parent
        manifests = sorted(run_dir.parent.glob('part2_config*.json'))
        if not manifests:
            parser.error('The experiment manifest must be alongside the model directory')
        config = json.loads(manifests[0].read_text(encoding='utf-8'))
        if config.get('protocol_id') != PROTOCOL_ID:
            parser.error('Model protocol is incompatible with this version')
        label = 'greedy' if args.greedy else 'final'
        destination = args.results or run_dir / f'replay_{args.condition}_{args.eval_seed}_{label}'
        destination.mkdir(parents=True, exist_ok=False)
        summary = evaluate(load_model(args.replay), args.condition, 1, config['protocol'],
                           sampled=not args.greedy, seed_start=args.eval_seed,
                           log_dir=destination, label=label, trace_count=1)
        print(f"Score {summary['mean_score']:.0f}. Trace saved in {destination}")
        return
    if args.selfcheck:
        return selfcheck()
    protocol = PROTOCOL.copy()
    seeds = TRAIN_SEEDS
    curve_episodes = CURVE_EPISODES
    if args.smoke:
        args.budget, args.episodes, seeds = 256, 2, [42]
        curve_episodes = 2
        protocol['max_decisions'] = 64
    if args.budget <= START_DECISIONS or args.eval_every < 0:
        parser.error('Budget must exceed startup length and checkpoint spacing must be nonnegative')
    if not args.smoke and args.episodes < 200:
        raise SystemExit('The brief requires at least 200 evaluation episodes.')
    if args.results is None:
        name = 'assignment4_wall_fix' if not args.smoke else time.strftime('smoke_wall_fix_%Y%m%d_%H%M%S')
        args.results = Path(__file__).resolve().parent.parent / 'results' / name
    args.results.mkdir(parents=True, exist_ok=True)
    eval_every = args.eval_every or max(1, args.budget//4)
    wanted = [part for part in args.only.upper().split(':') if part]
    algorithms = [a for a in wanted if a in ('Q', 'PPO')] or ['Q', 'PPO']
    settings = [c for c in wanted if c in ('A', 'B', 'C')] or ['A', 'B', 'C']
    if wanted and args.only.upper() not in {'Q','PPO','A','B','C', *(f'{a}:{s}' for a in ('Q','PPO') for s in 'ABC')}:
        raise SystemExit(f'--only {args.only!r} selects nothing; use Q:A, PPO:B, Q or A.')
    # Each shard owns its own files, so concurrent shards never write the same path.
    suffix = ('_' + '_'.join(wanted)) if wanted else ''
    verify_mazes()
    manifest = dict(protocol_id=PROTOCOL_ID, environment='ALE/MsPacman-v5', settings=SETTINGS,
                    protocol=protocol, smoke=args.smoke, budget=args.budget, train_seeds=seeds,
                    evaluation_episodes=args.episodes, eval_seed_start=EVAL_SEED_START,
                    curve_seed_start=CURVE_SEED_START, curve_episodes=curve_episodes,
                    every_environment_interactions=eval_every, smoothing='none',
                    startup_steps=START_DECISIONS, action_rng='SeedSequence([episode_seed, 1])',
                    python=platform.python_version(), os=platform.platform(), cpu=platform.processor(),
                    packages={p:version(p) for p in ('ale-py','gymnasium','numpy','pandas','scipy')},
                    code_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                 for p in ('part2_run.py','mspacman.py')})
    configs = list(args.results.glob('part2_config*.json'))
    if any(json.loads(p.read_text(encoding='utf-8')) != manifest for p in configs) or (
            not configs and any(args.results.glob('seed_results*.csv'))):
        parser.error('This directory contains results from another protocol. Choose a new --results directory.')
    config_path = args.results / f'part2_config{suffix}.json'
    with config_path.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    selected = {}
    if args.tune:
        selected = tune(args.tune_budget, seeds, protocol, args.results, suffix)
        print('selected:', selected, flush=True)
    else:
        # Tuning writes selected_<shard>.json, so a later shard must look wider
        # than the bare name or it silently falls back to untuned defaults.
        found = sorted(args.results.glob('selected*.json'))
        if found:
            choices = [json.loads(p.read_text(encoding='utf-8'))['selected'] for p in found]
            if any(choice != choices[0] for choice in choices[1:]):
                parser.error('Tuning files disagree. Use a fresh experiment with one fixed selection.')
            selected = choices[0]
            print(f'using {found[0].name}: {selected}', flush=True)

    rows, curves = [], []
    for algorithm in algorithms:
        for setting in settings:
            for seed in seeds:
                run_id = f'main_{algorithm}_{setting}_{seed}'
                run_dir = args.results / run_id
                run_dir.mkdir()
                start = time.perf_counter()
                agent = train(algorithm, setting, seed, args.budget, protocol,
                              eval_every=eval_every, log_dir=run_dir,
                              curve_episodes=curve_episodes, **selected.get(algorithm, {}))
                trained = time.perf_counter() - start
                model_path = run_dir / 'model.npz'
                save_model(agent, model_path)
                row = dict(run_id=run_id, algorithm=algorithm, protocol_id=PROTOCOL_ID,
                           condition=setting, seed=seed, stage='main',
                           config=json.dumps(selected.get(algorithm, {}), sort_keys=True),
                           training_interactions=args.budget,
                           agent_decisions=agent.environment_interactions-agent.startup_interactions,
                           training_seconds=trained - agent.curve_seconds,
                           curve_seconds=agent.curve_seconds,
                           environment_interactions=agent.environment_interactions,
                           startup_interactions=agent.startup_interactions,
                           final_evaluation_episodes=args.episodes,
                           **evaluate(agent, setting, args.episodes, protocol, log_dir=run_dir, trace_count=2))
                row['final_eval_seconds'] = row['evaluation_seconds']
                row['greedy_eval_seconds'] = 0.
                if algorithm == 'PPO':
                    # Rule 5: the additional evaluation mode, reported separately.
                    # Its cost is its own; it must not make PPO look slower to train.
                    greedy = evaluate(agent, setting, args.episodes, protocol, sampled=False,
                                      log_dir=run_dir, label='greedy')
                    row.update(greedy_success_rate=greedy['success_rate'],
                               greedy_mean_return=greedy['mean_return'],
                               greedy_mean_score=greedy['mean_score'],
                               greedy_eval_seconds=greedy['evaluation_seconds'])
                row['evaluation_policy'] = ('sampled' if algorithm == 'PPO'
                                            else 'greedy, ties uniform')
                row['wall_seconds'] = time.perf_counter() - start
                rows.append(row)
                print(f"{row['run_id']:>20}  success {row['success_rate']:.3f} "
                      f" return {row['mean_return']:8.1f}  {row['wall_seconds']:5.0f}s", flush=True)
                if setting == 'B':
                    # Use the persisted B policy, with the same action meanings and reward.
                    transfer = dict(run_id=f'transfer_{algorithm}_D_{seed}',
                                    algorithm=algorithm, condition='D', seed=seed, protocol_id=PROTOCOL_ID,
                                    stage='transfer', training_interactions=0,
                                    agent_decisions=0, environment_interactions=0, training_seconds=0.,
                                    source_run=row['run_id'],
                                    final_evaluation_episodes=args.episodes,
                                    **evaluate(load_model(model_path), 'D', args.episodes, protocol,
                                               log_dir=run_dir, trace_count=2))
                    transfer['evaluation_policy'] = row['evaluation_policy']
                    transfer['final_eval_seconds'] = transfer['evaluation_seconds']
                    rows.append(transfer)
                    print(f"{transfer['run_id']:>20}  success "
                          f"{transfer['success_rate']:.3f}", flush=True)
                curves += [dict(point, run_id=row['run_id'], algorithm=algorithm,
                                condition=setting, seed=seed) for point in agent.curve]
                import pandas as pd
                for prefix, data in [('seed_results', rows), ('curves', curves)]:
                    path = args.results / f'{prefix}{suffix}.csv'
                    tmp = path.with_suffix('.tmp')
                    pd.DataFrame(data).to_csv(tmp, index=False)
                    tmp.replace(path)

    print(f'\n{len(rows)} runs written to '
          f'{args.results/f"seed_results{suffix}.csv"}')


if __name__ == '__main__':
    main()
