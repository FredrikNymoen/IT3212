"""Regression checks for observation corruption and incompatible saved Q keys."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mspacman import MsPacManTask, PELLET_ROW_OFFSET
from part2_run import QAgent, save_model, load_model


class ObservationTests(unittest.TestCase):
    def task(self):
        task = MsPacManTask.__new__(MsPacManTask)
        task.walls = np.zeros((30, 30), dtype=bool)
        task.pellets = np.array([[15 + PELLET_ROW_OFFSET, 15],
                                 [15 + PELLET_ROW_OFFSET, 20],
                                 [15 + PELLET_ROW_OFFSET, 25]], dtype=np.float32)
        return task

    def test_intervening_wall_blocks_even_when_endpoint_is_clear(self):
        task = self.task()
        task.walls[15, 16] = True
        self.assertEqual(task._open(15, 15)[3], 0.)
        self.assertEqual(task._open(15, 15)[2], 1.)

    def test_vertical_exits_block_and_side_tunnel_remains_open(self):
        task = self.task()
        self.assertEqual(task._open(1, 15)[0], 0.)
        self.assertEqual(task._open(28, 15)[1], 0.)
        self.assertEqual(task._open(15, 1)[2], 1.)
        self.assertEqual(task._open(15, 28)[3], 1.)

    def test_ghost_fruit_power_and_mixed_rewards_do_not_delete_dots(self):
        for reward in (0., 50., 100., 200., 210., 500., 1600.):
            with self.subTest(reward=reward):
                task = self.task()
                before = task.pellets.copy()
                task._eat((15, 15), reward)
                np.testing.assert_array_equal(task.pellets, before)

    def test_one_or_two_dot_rewards_remove_only_the_requested_count(self):
        for reward, remaining in ((10., 2), (20., 1)):
            with self.subTest(reward=reward):
                task = self.task()
                task._eat((15, 15), reward)
                self.assertEqual(len(task.pellets), remaining)
                self.assertTrue(np.all(task.pellets[:, 1] > 15))

    def test_empty_pellet_estimate_is_safe(self):
        task = self.task()
        task.pellets = np.empty((0, 2), dtype=np.float32)
        task._eat((15, 15), 20.)
        self.assertEqual(task._nearest_pellet(15, 15), (0., 0., 1.))

    def test_pellet_distance_changes_q_key(self):
        agent = QAgent(9, 0)
        near = np.zeros(17, dtype=np.float32)
        far = near.copy()
        far[16] = .9
        self.assertNotEqual(agent.key(near), agent.key(far))

    def test_new_q_key_survives_save_and_load(self):
        agent = QAgent(9, 0, position_bins=4)
        obs = np.zeros(17, dtype=np.float32)
        agent.update(obs, 2, 10., obs, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            save_model(agent, path)
            restored = load_model(path)
        self.assertEqual(agent.key(obs), restored.key(obs))
        np.testing.assert_array_equal(agent.scores(obs), restored.scores(obs))


if __name__ == '__main__':
    unittest.main()
