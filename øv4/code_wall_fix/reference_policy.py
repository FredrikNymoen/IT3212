"""Evaluate a maze-aware, hand-written reference in stochastic Setting B."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from mspacman import GHOST_X, GHOST_Y, MAZE_TOP, SCREEN_DX, SCREEN_DY, MsPacManTask
from part2_run import PROTOCOL, PROTOCOL_ID


MOVES = ('UP', 'DOWN', 'LEFT', 'RIGHT')
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
POWER = np.array([(18-MAZE_TOP, 9.5), (150-MAZE_TOP, 9.5),
                  (17-MAZE_TOP, 149.5), (149-MAZE_TOP, 149.5)])


class Maze:
    """Corridor graph through the visible dot grid. Coordinates are (row, column)."""

    def __init__(self, walls, pellets):
        rows = np.unique(pellets[:, 0])
        cols = np.unique(pellets[:, 1])
        lattice = np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)
        self.nodes = lattice.reshape(-1, 2)
        self.edges = []
        links = np.zeros((len(self.nodes), len(self.nodes)))
        for row, col in np.ndindex(lattice.shape[:2]):
            start = row*len(cols)+col
            for end_row, end_col, action in ((row+1, col, 1), (row, col+1, 3)):
                if end_row >= len(rows) or end_col >= len(cols):
                    continue
                end = end_row*len(cols)+end_col
                a, b = self.nodes[[start, end]]
                segment = np.linspace(a, b, 1+int(np.abs(a-b).sum())).astype(int)
                if not walls[segment[:, 0], segment[:, 1]].any():
                    links[start, end] = links[end, start] = np.abs(a-b).sum()
                    self.edges.append((start, end, action))
        self.distance = shortest_path(csr_matrix(links), directed=False)
        self.start, self.end, self.forward = np.array(self.edges).T
        self.a = self.nodes[self.start]
        self.b = self.nodes[self.end]
        self.axis = (self.forward == 3).astype(int)
        self.length = np.abs(self.b-self.a).sum(axis=1)
        self.connected = np.flatnonzero(links.any(axis=1))

    def nearest(self, positions):
        candidates = self.nodes[self.connected]
        return self.connected[np.abs(positions[:, None]-candidates).sum(axis=2).argmin(axis=1)]

    def exits(self, position):
        """Give the path node ahead of each locally usable cardinal action."""
        lateral = np.abs(position[1-self.axis]-self.a[np.arange(len(self.a)), 1-self.axis])
        offset = position[self.axis]-self.a[np.arange(len(self.a)), self.axis]
        usable = (lateral <= 4) & (offset >= -4) & (offset <= self.length+4)
        exits = {}
        for edge in np.flatnonzero(usable):
            for action, node, distance in (
                (self.forward[edge], self.end[edge], self.length[edge]-offset[edge]),
                (self.forward[edge]-1, self.start[edge], offset[edge]),
            ):
                if distance < 1:
                    continue
                rank = (lateral[edge], distance)
                if action not in exits or rank < exits[action][0]:
                    exits[action] = (rank, node)
        return exits


class Reference:
    """Route to food while avoiding nearby ghosts and exploiting power pellets."""

    def __init__(self, task):
        self.task = task
        self.maze = Maze(task.walls, task.pellets)
        self.power = POWER.copy()
        self.previous = None
        self.previous_ghosts = None
        self.last_action = None
        self.visits = Counter()
        self.stalled = 0
        self.rng = np.random.default_rng(task.seed if hasattr(task, 'seed') else 0)

    def choose(self):
        t, maze = self.task, self.maze
        ram = t.env.unwrapped.ale.getRAM()
        position = np.array(t._position(ram))+4.5
        ghosts = np.array([(int(ram[y])+SCREEN_DY-MAZE_TOP+4.5,
                            int(ram[x])+SCREEN_DX+4.5) for x, y in zip(GHOST_X, GHOST_Y)])
        exits = maze.exits(position)
        self.stalled = self.stalled+1 if self.previous is not None and np.array_equal(position, self.previous) else 0
        ghost_distance = np.abs(ghosts-position).sum(axis=1)
        closing = (np.abs(self.previous_ghosts-position).sum(axis=1)-ghost_distance
                   if self.previous_ghosts is not None else np.zeros(4))
        self.previous, self.previous_ghosts = position, ghosts
        if not exits or self.stalled >= 4:
            return t.actions.index(MOVES[int(self.rng.integers(4))])

        # The RAM timer is global. Near the end of the window or after eating
        # a ghost, treat ambiguous ghost positions conservatively.
        ghost_nodes = maze.nearest(ghosts)
        food = maze.nearest(t.pellets) if len(t.pellets) else np.empty(0, dtype=int)
        power = maze.nearest(self.power) if len(self.power) else np.empty(0, dtype=int)
        current = int(maze.nearest(position[None, :])[0])
        self.visits[current] += 1
        choices = {}
        for action, ((_, distance), node) in exits.items():
            direction = DIRECTIONS[action]
            segment_food = (np.abs(t.pellets-position)[:, 1-(action >= 2)] < 4)
            along = (t.pellets-position) @ direction
            direct = along[segment_food & (along > 0) & (along <= distance+3)]
            food_dist = min(distance+maze.distance[node, food].min(initial=np.inf),
                            direct.min(initial=np.inf))
            power_dist = distance+maze.distance[node, power].min(initial=np.inf)
            goal = min(food_dist, power_dist-12)
            danger = 0.
            if t.frightened >= 24:
                chase = distance+maze.distance[node, ghost_nodes].min()
                if chase < 50 and chase < t.frightened:
                    goal = min(goal, chase-24)
            else:
                route = maze.distance[node, ghost_nodes]
                ahead = (ghosts-position) @ direction
                lateral = np.abs(ghosts-position)[:, 1-(action >= 2)]
                in_lane = (ahead >= 0) & (lateral < 7) & (ahead <= distance+10)
                near = np.minimum(route, np.where(in_lane, ghost_distance, np.inf))
                danger = float(np.maximum(0, 34-near+2*np.maximum(closing, 0)).max())*2
                if power_dist < 14:
                    danger *= .3
            choices[action] = (min(goal, 300) + danger + min(self.visits[node], 30)*.4
                               - (1.5 if action == self.last_action else 0))
        self.last_action = min(choices, key=choices.get)
        return t.actions.index(MOVES[self.last_action])

    def observe(self, info, old_lives, old_timer):
        t = self.task
        if t.lives < old_lives:
            self.previous = self.previous_ghosts = self.last_action = None
            self.stalled = 0
        if len(self.power) and (info['raw_reward'] in (50, 60, 70) or
                                (0 < old_timer < t.frightened <= 62 and info['raw_reward'] >= 50)):
            position = np.array(t._position(t.env.unwrapped.ale.getRAM()))+4.5
            distances = np.abs(self.power-position).sum(axis=1)
            if distances.min() < 15:
                self.power = np.delete(self.power, distances.argmin(), axis=0)


def evaluate(episodes, seed_start):
    task = MsPacManTask(PROTOCOL, 'B')
    rows = []
    try:
        for seed in range(seed_start, seed_start+episodes):
            task.reset(seed)
            policy = Reference(task)
            policy.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            total = 0.
            lost_lives = 0
            while True:
                action = policy.choose()
                old_lives, old_timer = task.lives, task.frightened
                _, reward, done, info = task.step(action)
                policy.observe(info, old_lives, old_timer)
                total += reward
                lost_lives += max(0, old_lives-task.lives)
                if done:
                    break
            rows.append(dict(seed=seed, outcome=info['outcome'], raw_score=info['raw_score'],
                             episode_return=total, length=task.t, lost_lives=lost_lives))
    finally:
        task.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=300000)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error('Use a positive episode count and a nonnegative seed.')
    if args.results:
        args.results.mkdir(parents=True, exist_ok=False)
    rows = evaluate(args.episodes, args.seed_start)
    scores = [row['raw_score'] for row in rows]
    counts = Counter(row['outcome'] for row in rows)
    summary = dict(protocol=PROTOCOL_ID, setting='B', policy='maze-aware reference',
                   episodes=len(rows), seed_start=args.seed_start, outcome_counts=dict(counts),
                   success_rate=counts['success']/len(rows), mean_score=mean(scores),
                   median_score=median(scores), max_score=max(scores),
                   reached_1500=sum(score >= 1500 for score in scores),
                   mean_lost_lives=mean(r['lost_lives'] for r in rows))
    if args.results:
        with (args.results/'episodes.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (args.results/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
