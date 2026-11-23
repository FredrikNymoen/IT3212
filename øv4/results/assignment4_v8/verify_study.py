"""Validate the completed standard five-seed v8 study from its raw episode logs."""
from pathlib import Path
import csv, hashlib, json, math, statistics

root = Path(__file__).resolve().parent
code = root.parents[1] / 'code'
seeds = {42, 7, 2026, 123, 999}
rows = [row for path in sorted(root.glob('seed_results*.csv'))
        for row in csv.DictReader(path.open(encoding='utf-8'))]
assert len(rows) == 40, f'Expected 40 rows, got {len(rows)}'
assert len({row['run_id'] for row in rows}) == 40
assert sum(row['stage'] == 'main' for row in rows) == 30
assert sum(row['stage'] == 'transfer' for row in rows) == 10
for condition in 'ABCD':
    for algorithm in ('Q', 'PPO'):
        group = [r for r in rows if r['condition'] == condition and r['algorithm'] == algorithm]
        assert {int(r['seed']) for r in group} == seeds
for path in root.glob('part2_config*.json'):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert manifest['protocol_id'] == 'mspacman_maze_pellets_walls_v8'
    assert manifest['evaluation_episodes'] == 200
    for name, digest in manifest['code_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest() == digest
for row in rows:
    source = row['source_run'] if row['stage'] == 'transfer' else row['run_id']
    directory = root/source
    assert (directory/'model.npz').is_file()
    path = directory/f'final_{row["condition"]}_episodes.csv'
    episodes = list(csv.DictReader(path.open(encoding='utf-8')))
    assert len(episodes) == 200
    assert {int(e['seed']) for e in episodes} == set(range(300000, 300200))
    assert int(row['environment_interactions']) == (0 if row['stage'] == 'transfer' else 100000)
    assert int(row['final_evaluation_episodes']) == 200
    for outcome in ('success', 'cleared', 'failure', 'timeout'):
        rate = sum(e['outcome'] == outcome for e in episodes)/200
        assert math.isclose(rate, float(row[outcome+'_rate']), abs_tol=1e-10)
    for field, summary in (('raw_score', 'mean_score'), ('episode_return', 'mean_return')):
        mean = statistics.mean(float(e[field]) for e in episodes)
        assert math.isclose(mean, float(row[summary]), abs_tol=1e-8)
    for episode in episodes:
        expected = float(episode['raw_score']) + (100 if episode['outcome']=='success' else 0)
        if row['condition']=='C': expected -= .5*int(episode['length'])
        assert math.isclose(expected, float(episode['episode_return']), abs_tol=1e-8)
    for seed in (300000, 300001):
        assert (directory/f'final_{row["condition"]}_{seed}.json.gz').is_file()
assert (root/'part2_tables.md').is_file()
for condition in 'ABC': assert (root/f'learning_curve_{condition}.png').is_file()
print('Verified 30 trained agents, 10 frozen transfers, 8,000 primary evaluation episodes, source hashes, metrics, reward accounting, model files and traces.')
