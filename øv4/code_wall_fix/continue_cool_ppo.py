"""Check whether training at the selected lower temperature improves the strong policies."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from improve_ppo_session import evaluate, finish_jobs, fit
from score_diagnostics import digest, summarize, write_json


BUDGETS = (0, 30_000, 60_000)
VALIDATION_START = 999800
TEST_START = 999200


def validate(candidate, budget, root):
    result = dict(candidate, name=f'budget_{budget}', budget=budget)
    if budget:
        path = root/'training'/result['name']/f"seed_{candidate['seed']}"
        path.mkdir(parents=True)
        result['model'] = str(fit(Path(candidate['model']), candidate['seed'], path, budget,
                                  2e-5, candidate['temperature'], .08, 512))
    return evaluate(result, root, 'validation', VALIDATION_START, 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    sweep, root = args.sweep.resolve(), args.results.resolve()
    config = json.loads((sweep/'manifest.json').read_text(encoding='utf-8'))
    comparison = json.loads((sweep/'comparison.json').read_text(encoding='utf-8'))
    temperature = comparison['selected_temperature']
    parents = [dict(candidate, temperature=temperature) for candidate in config['checkpoints']]
    for file, checksum in config['checkpoint_sha256'].items():
        if digest(Path(file))!=checksum:
            parser.error(f'Checkpoint changed: {file}')
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    previous = json.loads((Path(config['source'])/'manifest.json').read_text(encoding='utf-8'))
    scripts = ('continue_cool_ppo.py', *previous['code_sha256'])
    write_json(root/'manifest.json', dict(
        setting='B', sweep=str(sweep), checkpoints=parents, budgets=BUDGETS,
        code_sha256={name: digest(here/name) for name in scripts},
        parent_sha256=config['checkpoint_sha256'], temperature=temperature,
        lr=2e-5, kl_weight=.08, rollout=512, validation_start=VALIDATION_START,
        validation_episodes=60, test_start=TEST_START, test_episodes=200))
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = finish_jobs([pool.submit(validate, candidate, budget, root)
                            for budget in BUDGETS for candidate in parents], 'Cold continuation')
        scores = {b: summarize([row for row in rows if row['budget']==b]) for b in BUDGETS}
        selected = max(BUDGETS, key=lambda budget: (scores[budget]['success_rate'],
                                                    scores[budget]['mean_score'], -budget))
        write_json(root/'selection.json', dict(selected_budget=selected, means=scores, runs=rows))
        print(f'Cold continuation selected {selected:,} additional interactions', flush=True)
        plan = [{name: row[name] for name in ('name','model','temperature','seed','budget')}
                for row in rows if row['budget'] in {0, selected}]
        tests = finish_jobs([pool.submit(evaluate, row, root, 'test', TEST_START, 200)
                             for row in plan], 'Cold continuation test')
    result = dict(selected_budget=selected, validation_means=scores, test=tests,
                  test_means={b:summarize([row for row in tests if row['budget']==b])
                              for b in sorted({0, selected})})
    write_json(root/'comparison.json', result)
    print(json.dumps(result['test_means'], indent=2), flush=True)


if __name__ == '__main__':
    main()
