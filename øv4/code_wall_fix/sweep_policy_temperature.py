"""Select one shared temperature for a frozen set of screened PPO checkpoints."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from improve_ppo_session import compact, evaluate, finish_jobs, rank
from score_diagnostics import digest, summarize, write_json


TEMPERATURES = (.2, .275, .35, .425, .5)
VALIDATION_START = 999400
TEST_START = 999500


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    study, root = args.study.resolve(), args.results.resolve()
    progress = json.loads((study/'progress.json').read_text(encoding='utf-8'))
    screen = progress.get('screen')
    if not screen:
        screen = json.loads((study/'comparison.json').read_text(encoding='utf-8'))['screen']
    seeds = sorted({row['seed'] for row in screen})
    parents = [compact(rank(screen, seed)[0]) for seed in seeds]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(
        source=str(study), checkpoints=parents, source_trial=progress['trial'],
        checkpoint_sha256={r['model']: digest(Path(r['model'])) for r in parents},
        code_sha256=digest(Path(__file__)),
        temperatures=TEMPERATURES, validation_start=VALIDATION_START, validation_episodes=60,
        test_start=TEST_START, test_episodes=200))

    def candidate(parent, temp):
        return dict(parent, name=f'temp_{temp:g}', temperature=temp)

    with ProcessPoolExecutor(max_workers=6) as pool:
        estimates = finish_jobs([pool.submit(evaluate, candidate(r, temperature), root,
                                             'validation', VALIDATION_START, 60)
                                 for temperature in TEMPERATURES for r in parents], 'Fine sweep')
        scores = {temp: summarize([r for r in estimates if r['temperature']==temp])
                  for temp in TEMPERATURES}
        selected = max(TEMPERATURES, key=lambda temp: (scores[temp]['success_rate'],
                                                       scores[temp]['mean_score'], temp))
        write_json(root/'selection.json', dict(selected_temperature=selected,
                                                validation_means=scores, validation=estimates))
        print(f'Fine sweep selected {selected:g}', flush=True)
        test_temperatures = sorted({.5, selected})
        tests = finish_jobs([pool.submit(evaluate, candidate(r, temp), root, 'test', TEST_START, 200)
                             for temp in test_temperatures for r in parents], 'Fine sweep test')
    results = dict(selected_temperature=selected, validation_means=scores, test=tests,
                   test_means={temp: summarize([r for r in tests if r['temperature']==temp])
                               for temp in test_temperatures})
    write_json(root/'comparison.json', results)
    print(json.dumps(results['test_means'], indent=2), flush=True)


if __name__ == '__main__':
    main()
