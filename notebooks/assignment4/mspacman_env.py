"""Gymnasium MsPacMan task and object features for Assignment 4 Part 2.

All interaction uses Gymnasium reset/step. ALE is only used by the wrapper for
RAM observations and the controlled map intervention, never as an agent API.
RAM position/food locations were checked against OCAtari's MsPacman extractor:
https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py
The maze selector (RAM 0) is also used by HackAtari's set_level modifications:
https://github.com/k4ntz/HackAtari/blob/master/hackatari/games/mspacman.py
The food-lattice masks below describe the ROM's two layouts, not fitted data.
RAM 116's power countdown was checked against a consumed power pellet and RGB
frames in an engineering diagnostic; its value is used as a counter, not seconds.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

LAYOUTS = (
    ('111101111111101111', '100101000000101001', '111111111111111111',
     '010100100001010010', '110111111111111011', '010000100001000010',
     '011111100001111110', '010000100001000010', '110111111111111011',
     '010101010010101010', '111101011110101111', '100101101101100101',
     '100100010010001001', '111111111111111111'),
    ('111111111111111111', '100001000000100001', '110111011110111011',
     '010101010010101010', '110101111111101011', '100101000000101001',
     '110101101101101011', '010100100001001010', '110111111111111011',
     '111111000000111111', '001001011110100100', '111101111111101111',
     '100100000000001001', '111101111111101111'),
)
# A cell's RAM byte/bit is determined by its column and row.
FOOD_BYTES = np.array([59, 60, 60, 60, 60, 61, 61, 61, 61,
                      60, 60, 60, 60, 61, 61, 61, 61, 59])
FOOD_BITS = np.array([64, 128, 32, 8, 2, 1, 4, 16, 64,
                     64, 16, 4, 1, 2, 8, 32, 128, 16], dtype=np.uint8)
FOOD_BYTES = (FOOD_BYTES[None, :] + 3 * np.arange(14)[:, None]).ravel()
FOOD_BITS = np.tile(FOOD_BITS, 14)
FOOD_XY = np.array([(8 + 8*c, 7 + 12*r) for r in range(14) for c in range(18)], dtype=float)
FOOD_MASKS = np.array([[int(bit) for line in layout for bit in line] for layout in LAYOUTS], dtype=bool)
FEATURES = ('player_x', 'player_y', 'ghost_dx', 'ghost_dy', 'ghost_distance',
            'pellet_dx', 'pellet_dy', 'pellet_distance', 'pellets_up',
            'pellets_right', 'pellets_down', 'pellets_left', 'power_countdown',
            'orientation', 'time_left')


@dataclass(frozen=True)
class TaskConfig:
    setting: str = 'B'
    horizon: int = 1000
    target_score: int = 2000
    frameskip: int = 4
    step_penalty: float = 0.1

    @property
    def maze(self):
        return int(self.setting == 'D')

    @property
    def sticky(self):
        return 0.0 if self.setting == 'A' else 0.25


def features_from_ram(ram, maze, time_left):
    """Fixed feature engineering; no fitting and no policy-dependent defaults."""
    player = np.array([int(ram[10]) - 13, int(ram[16]) + 1], dtype=float)
    ghosts = np.column_stack((ram[6:10].astype(float)-13, ram[12:16].astype(float)+1))
    active = np.arange(4) <= int(ram[19])
    if not ram[47]:
        active[:] = False
    delta = ghosts - player
    distances = np.abs(delta).sum(axis=1)
    distances[~active] = np.inf
    nearest = int(np.argmin(distances))
    ghost = delta[nearest] if active.any() else np.zeros(2)
    gd = float(distances[nearest]) if active.any() else 320.0
    food = FOOD_XY[FOOD_MASKS[maze] & ((ram[FOOD_BYTES] & FOOD_BITS) != 0)]
    density = np.zeros(4)
    if len(food):
        fd = food - player
        distance = np.abs(fd).sum(axis=1)
        nearest = int(np.argmin(distance))
        pellet, pd = fd[nearest], float(distance[nearest])
        direction = np.where(np.abs(fd[:, 0]) > np.abs(fd[:, 1]),
                             np.where(fd[:, 0] > 0, 1, 3), np.where(fd[:, 1] > 0, 2, 0))
        density = np.bincount(direction, minlength=4) / 252.0
    else:
        pellet, pd = np.zeros(2), 320.0
    values = [player[0]/160, player[1]/210, ghost[0]/160, ghost[1]/210, gd/320,
              pellet[0]/160, pellet[1]/210, pd/320, *density,
              min(int(ram[116]), 63)/63, (int(ram[56]) & 3)/3, time_left]
    return np.clip(values, -1, 1).astype(np.float32)


def coarse_state(obs):
    """Coarsen the same information supplied more finely to PPO."""
    def bearing(dx, dy, absent):
        if absent:
            return 4
        return (1 if dx > 0 else 3) if abs(dx) > abs(dy) else (2 if dy > 0 else 0)
    return (min(int(obs[0]*4), 3), min(int(obs[1]*4), 3),
            bearing(obs[2], obs[3], obs[4] >= 1),
            int(np.searchsorted([.08, .25], obs[4])),
            bearing(obs[5], obs[6], obs[7] >= 1),
            int(np.searchsorted([.05, .2], obs[7])),
            int(np.argmax(obs[8:12])),
            0 if obs[12] == 0 else (1 if obs[12] < .3 else 2),
            int(round(float(obs[13])*3)), min(int(obs[14]*3), 2))


class MsPacmanTask(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Score-based finite-horizon task, as authorised by the team.

    A/B reward = game score increment; C subtracts 0.1 points per decision.
    A decision repeats one cardinal action for four emulator frames. Success
    is >=2,000 newly earned points. A 1,000-decision limit is a task failure,
    therefore terminal (not an external continuing-task truncation); time left
    is observable. No life-loss pseudo-episodes or hidden warm-up actions.
    """
    def __init__(self, config=TaskConfig(), render_mode=None, env=None):
        if config.setting not in 'ABCD' or len(config.setting) != 1:
            raise ValueError('Setting must be A, B, C or D.')
        self.config = config
        gym.utils.RecordConstructorArgs.__init__(self, config=config, render_mode=render_mode)
        super().__init__(env if env is not None else gym.make('ALE/MsPacman-v5', obs_type='ram',
            frameskip=config.frameskip, repeat_action_probability=config.sticky,
            mode=0, render_mode=render_mode))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(len(FEATURES),), dtype=np.float32)
        # NOOP, UP, RIGHT, LEFT, DOWN. Same mapping in all conditions.
        self.action_space = gym.spaces.Discrete(5)
        self.ram = None
        self.steps = 0
        self.last_frame = 0
        self.score = 0.0
        self.episode_return = 0.0
        self.finished = True

    def reset(self, *, seed=None, options=None):
        ram, info = self.env.reset(seed=seed, options=options)
        # Set the ROM layout selector before the first decision/frame. This
        # intervention changes the map; mode=1 alone does not change the map.
        self.unwrapped.ale.setRAM(0, self.config.maze)
        self.ram = self.unwrapped.ale.getRAM().copy()
        self.steps, self.score, self.episode_return = 0, 0.0, 0.0
        self.last_frame = int(info['episode_frame_number'])
        self.finished = False
        return features_from_ram(self.ram, self.config.maze, 1), dict(info)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset() after a terminal episode.')
        ram, raw_reward, terminal, truncated, info = self.env.step(int(action))
        self.steps += 1
        self.score += float(raw_reward)
        success = self.score >= self.config.target_score
        timeout = self.steps >= self.config.horizon and not success
        # Fail loudly rather than silently allowing the held-out map in B.
        if int(ram[0]) != self.config.maze:
            raise RuntimeError('Natural level transition crossed the map holdout. Revisit task design.')
        self.ram = ram.copy()
        reward = float(raw_reward) - (self.config.step_penalty if self.config.setting == 'C' else 0)
        self.episode_return += reward
        self.finished = bool(terminal or truncated or success or timeout)
        reason = 'success' if success else ('time_limit' if timeout else ('game_over' if terminal else 'external_limit'))
        info = dict(info, score=self.score, raw_reward=float(raw_reward),
                    success=bool(success), episode_steps=self.steps, reason=reason,
                    task_return=self.episode_return, maze=int(ram[0]))
        info['emulator_frames'] = int(info['episode_frame_number']) - self.last_frame
        self.last_frame = int(info['episode_frame_number'])
        obs = features_from_ram(ram, self.config.maze, max(0, 1-self.steps/self.config.horizon))
        return obs, reward, bool(terminal or success or timeout), bool(truncated), info


def task_description(config):
    return dict(asdict(config), maze=config.maze, sticky_probability=config.sticky,
                feature_names=FEATURES, action_meanings=('NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN'),
                reward='score increment minus step penalty in C only',
                finite_horizon_is_terminal=True, power_counter='RAM 116 / 63 (not seconds)')
