"""Cache repeated corridor queries without changing the navigation observations."""
import time

from mspacman import MsPacManTask
from score_diagnostics import NavigationTask, np, runner
from transfer_ghost_ppo import GhostTask
from transfer_power_ppo import PowerTask


class CachedQueries:
    def __init__(self, graph):
        self.graph = graph
        self.exit_cache = {}
        self.node_cache = {}

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def exits(self, position):
        key = tuple(position)
        if key not in self.exit_cache:
            self.exit_cache[key] = self.graph.exits(position)
        return self.exit_cache[key]

    def nearest(self, positions):
        return np.fromiter((self.nearest_node(point) for point in positions), dtype=int,
                           count=len(positions))

    def nearest_node(self, position):
        key = tuple(position)
        if key not in self.node_cache:
            self.node_cache[key] = self.graph.nearest(position[None, :])[0]
        return self.node_cache[key]


class CachedNavigationTask(NavigationTask):
    def __init__(self, protocol, setting):
        super().__init__(protocol, setting)
        self.layout_graphs = {}
        self.cached_pellets = None

    def _read_layout(self):
        maze = self.s["maze"]
        if maze in self.layout_graphs:
            self._maze_key, self.graph, self._pellet_nodes = self.layout_graphs[maze]
        super()._read_layout()
        if not isinstance(self.graph, CachedQueries):
            self.graph = CachedQueries(self.graph)
        self.layout_graphs[maze] = self._maze_key, self.graph, self._pellet_nodes
        self.cached_pellets = None

    def features(self, ram):
        base = MsPacManTask.features(self, ram)
        actors = self._actors(ram)
        position, ghosts = actors[0], actors[1:]
        if self.cached_pellets is not self.pellets:
            self.cached_pellets = self.pellets
            self.food_nodes = self.graph.nearest(self.pellets)
        ghost_distance = np.abs(ghosts - position).sum(axis=1)
        nearest = int(ghost_distance.argmin())
        ghost_nodes = self.graph.nearest(ghosts)
        food_distance = np.ones(4)
        ghost_path = np.ones(4)
        for action, ((_, distance), node) in self.graph.exits(position).items():
            row = self.graph.distance[node]
            food_distance[action] = min(
                1., (distance + row[self.food_nodes].min(initial=np.inf)) / 160.)
            ghost_path[action] = min(1., (distance + row[ghost_nodes].min()) / 160.)
        velocities = np.zeros((2, 2))
        closing = 0.
        action_onehot = np.zeros(self.n_actions)
        if self.previous is not None and self.previous_lives == self.lives:
            previous_ghost = self.previous[nearest + 1]
            ids = [0, nearest + 1]
            velocities = np.clip((actors[ids] - self.previous[ids]) / 8, -1, 1)
            closing = np.clip((np.abs(previous_ghost - position).sum()
                               - ghost_distance[nearest]) / 8, -1, 1)
            action_onehot[self.last_action] = 1.
        return np.concatenate((base, food_distance, ghost_path, velocities.ravel(),
                               [closing], action_onehot)).astype(np.float32)


class CachedPowerTask(PowerTask, CachedNavigationTask):
    pass


class CachedGhostTask(GhostTask, CachedNavigationTask):
    pass


CACHED_TASKS = {
    "navigation": CachedNavigationTask,
    "power": CachedPowerTask,
    "ghost": CachedGhostTask,
}


def verify():
    rng = np.random.default_rng(64138)
    for name, original in (("navigation", NavigationTask), ("power", PowerTask), ("ghost", GhostTask)):
        tasks = [cls(runner.PROTOCOL, "B") for cls in (original, CACHED_TASKS[name])]
        durations = [0., 0.]
        decisions = 0
        try:
            for i, layout in enumerate((0, 2, 3, 0, 3, 2)):
                for task in tasks:
                    task.s = dict(task.s, maze=layout)
                first, second = tasks
                observations = [task.reset(981000 + i) for task in tasks]
                np.testing.assert_array_equal(*observations)
                for action in rng.integers(first.n_actions, size=300):
                    results = []
                    for index, task in enumerate(tasks):
                        since = time.perf_counter()
                        results.append(task.step(action))
                        durations[index] += time.perf_counter() - since
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    assert results[0][1:] == results[1][1:]
                    decisions += 1
                    if results[0][2]:
                        break
        finally:
            for task in tasks:
                task.close()
        print(f"{name}: {decisions} identical step results, "
              f"{durations[0]:.2f}s original, {durations[1]:.2f}s cached", flush=True)


if __name__ == "__main__":
    verify()

