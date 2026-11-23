"""Compare PPO training temperatures on fresh stochastic Setting B episodes."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from score_diagnostics import (ENTROPY, LEARNING_RATE, NavigationTask, OriginalPPO,
                               TemperaturePolicy, digest, np, read_rows, runner,
                               summarize, write_json)


TEMPERATURE = .5
TEST_START = 996000
ARMS = ('starting_checkpoint', 'trained_at_1', 'trained_at_0.5')


class CoolPPO(OriginalPPO):
    """Apply temperature to rollout probabilities and the PPO loss."""

    def forward(self, obs):
        h1, h2, logits, values = super().forward(obs)
        return h1, h2, logits/TEMPERATURE, values

    def backward(self, obs, dlogits, dvalue):
        return super().backward(obs, dlogits/TEMPERATURE, dvalue)


def verify_temperature_gradient():
    agent = CoolPPO(9, 39, 42)
    x = np.random.default_rng(42).normal(size=(4, 39))
    y = np.array([1, 3, 7, 2])

    def loss():
        probabilities = runner.softmax(agent.forward(x)[2])
        return -np.log(probabilities[np.arange(len(x)), y]).mean()

    probs = runner.softmax(agent.forward(x)[2])
    probs[np.arange(len(x)), y] -= 1.
    grads = agent.backward(x, probs/len(x), np.zeros(len(x)))
    for name, direction in grads.items():
        params = agent.p[name]
        direction = direction/max(float(np.linalg.norm(direction)), 1.)
        original = params.copy()
        params[:] = original + 1e-5*direction
        upper = loss()
        params[:] = original - 1e-5*direction
        lower = loss()
        params[:] = original
        assert abs((upper-lower)/2e-5 - np.sum(grads[name]*direction)) < 1e-7, name


def fit(start_path, output, seed, budget):
    anchor = runner.load_model(start_path)

    class WarmStartPPO(CoolPPO):
        def __init__(self, n_actions, n_features, training_seed, **config):
            super().__init__(n_actions, n_features, training_seed,
                             hidden=anchor.p['W1'].shape[1], **config)
            self.p = {key: value.copy() for key, value in anchor.p.items()}

    runner.MsPacManTask = NavigationTask
    runner.PPOAgent = WarmStartPPO
    try:
        agent = runner.train('PPO', 'B', seed, budget, lr=LEARNING_RATE, entropy=ENTROPY,
                             log_dir=output)
        assert agent.environment_interactions == budget
    finally:
        runner.PPOAgent = OriginalPPO
    model = output/'model.npz'
    runner.save_model(agent, model)
    return model


def run(arm, seed, start_path, control_path, directory, budget, episodes):
    runner.MsPacManTask = NavigationTask
    folder = directory/arm/f'seed_{seed}'
    folder.mkdir(parents=True)
    if arm == 'trained_at_0.5':
        model = fit(start_path, folder, seed, budget)
        print(f'Cold PPO seed {seed}: {budget:,} training interactions complete', flush=True)
    else:
        model = start_path if arm == 'starting_checkpoint' else control_path
    agent = TemperaturePolicy(runner.load_model(model), TEMPERATURE)
    result = runner.evaluate(agent, 'B', episodes, seed_start=TEST_START, log_dir=folder,
                             label='test', trace_count=1)
    rows = read_rows(folder/'test_B_episodes.csv')
    reached = sum(float(row['raw_score']) >= 1500 for row in rows)
    successes = sum(row['outcome'] == 'success' for row in rows)
    result.update(seed=seed, arm=arm, model=str(model), reached_1500=reached,
                  conversion_after_1500=successes/reached if reached else None)
    write_json(folder/'test.json', result)
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=here.parent/'results'/'assignment4_score_diagnostics_B')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, directory = args.source.resolve(), args.results.resolve()
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['protocol'] != runner.PROTOCOL or manifest['setting'] != 'B'
            or manifest['learning_rate'] != LEARNING_RATE or manifest['entropy'] != ENTROPY
            or manifest['budget'] != 30_000 or manifest['smoke']):
        parser.error('Use the matching full 30k score diagnostics study.')
    for name, checksum in manifest['code_sha256'].items():
        if digest(here/name) != checksum:
            parser.error(f'{name} differs from the source study.')
    starts_root = Path(manifest['source'])
    seeds = manifest['seeds'][:1] if args.smoke else manifest['seeds']
    starts = {s: starts_root/'targeted'/f'seed_{s}'/'model.npz' for s in seeds}
    controls = {s: source/'reward_control'/f'seed_{s}'/'model.npz' for s in seeds}
    for path in starts.values():
        if digest(path) != manifest['source_sha256'][path.relative_to(starts_root).as_posix()]:
            parser.error(f'Starting checkpoint has changed: {path}')
    verify_temperature_gradient()
    budget, episodes = (256, 2) if args.smoke else (manifest['budget'], 200)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'manifest.json', dict(
        code_sha256={name: digest(here/name)
                     for name in ('train_cool_ppo.py', *manifest['code_sha256'])},
        inputs_sha256={str(p): digest(p) for p in (manifest_path, *starts.values(), *controls.values())},
        protocol=runner.PROTOCOL, setting='B', seeds=seeds, smoke=args.smoke,
        new_training_budget=budget, control_training_budget=manifest['budget'],
        learning_rate=LEARNING_RATE, entropy=ENTROPY, eval_temperature=TEMPERATURE,
        eval_seed_start=TEST_START, eval_episodes=episodes))
    results = []
    with ProcessPoolExecutor(max_workers=3 if args.smoke else 9) as pool:
        jobs = [pool.submit(run, arm, seed, starts[seed], controls[seed],
                            directory, budget, episodes) for arm in ARMS for seed in seeds]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(f'{result["arm"]}, seed {result["seed"]}: {result["success_rate"]:.1%}', flush=True)
    means = {arm: summarize([r for r in results if r['arm']==arm]) for arm in ARMS}
    write_json(directory/'comparison.json', dict(arm_means=means,
                                                runs=sorted(results, key=lambda r: (r['arm'], r['seed']))))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
