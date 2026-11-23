"""Compare near-greedy and greedy playback with the selected 0.2 policy."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from improve_ppo_session import finish_jobs
from score_diagnostics import NavigationTask, TemperaturePolicy, digest, read_rows, runner, summarize, write_json


TEMPERATURES = (.2, .15, .1, .05, 0.)
VALIDATION_START = 997100
TEST_START = 997200


def evaluate(candidate, root, label, start, count, temperature):
    runner.MsPacManTask = NavigationTask
    agent = runner.load_model(Path(candidate['model']))
    if temperature:
        agent = TemperaturePolicy(agent, temperature)
    mode = f'temp_{temperature:g}' if temperature else 'greedy'
    folder = root/label/mode/f"seed_{candidate['seed']}"
    folder.mkdir(parents=True)
    result = runner.evaluate(agent, 'B', count, sampled=bool(temperature),
                             seed_start=start, log_dir=folder, label=label,
                             trace_count=int(label=='test'))
    rows = read_rows(folder/f'{label}_B_episodes.csv')
    reached = sum(float(row['raw_score']) >= 1500 for row in rows)
    result.update(candidate, name=mode, temperature=temperature, episodes=count,
                  episode_seed_start=start, reached_1500=reached, conversion_after_1500=(
                      sum(row['outcome']=='success' for row in rows)/reached if reached else None))
    write_json(folder/'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    sweep, root = args.sweep.resolve(), args.results.resolve()
    setup = json.loads((sweep/'manifest.json').read_text(encoding='utf-8'))
    study = json.loads((sweep/'comparison.json').read_text(encoding='utf-8'))
    if study['selected_temperature'] != .2:
        parser.error('Expected the completed comparison that selected temperature 0.2.')
    for path, checksum in setup['checkpoint_sha256'].items():
        if digest(Path(path)) != checksum:
            parser.error(f'Checkpoint changed: {path}')
    source = json.loads((Path(setup['source'])/'manifest.json').read_text(encoding='utf-8'))
    candidates = setup['checkpoints']
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(source=str(sweep), checkpoints=candidates,
        checkpoint_sha256=setup['checkpoint_sha256'],
        code_sha256={**source['code_sha256'], Path(__file__).name:digest(Path(__file__))},
        setting='B', temperatures=TEMPERATURES, validation_start=VALIDATION_START,
        validation_episodes=60, test_start=TEST_START, test_episodes=200))
    with ProcessPoolExecutor(max_workers=6) as pool:
        validation = finish_jobs([pool.submit(evaluate, row, root, 'validation', VALIDATION_START,
                                              60, temp) for temp in TEMPERATURES for row in candidates],
                                 'Greedy limit validation')
        scores = {t:summarize([row for row in validation if row['temperature']==t])
                  for t in TEMPERATURES}
        chosen = max(TEMPERATURES, key=lambda t:(scores[t]['success_rate'], scores[t]['mean_score'], t))
        write_json(root/'selection.json', dict(chosen=chosen, validation_means=scores, runs=validation))
        print(f'Greedy limit selected {"greedy" if chosen==0 else chosen}', flush=True)
        temperatures = sorted({.2, chosen})
        tests = finish_jobs([pool.submit(evaluate, row, root, 'test', TEST_START, 200, temp)
                            for temp in temperatures for row in candidates], 'Greedy limit test')
    means = {t:summarize([row for row in tests if row['temperature']==t]) for t in temperatures}
    write_json(root/'comparison.json', dict(chosen=chosen, validation_means=scores, test=tests,
                                            test_means=means))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
