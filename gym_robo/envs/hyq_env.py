import os
from datetime import datetime
from typing import Tuple, Dict

import gym
from gym.spaces import Box

import numpy

from gym_robo.robots import HyQSim
from gym_robo.utils import HyQLogger, hyq_obs_to_numpy
# from gym_robo.tasks import

from HyQPy import HyQObservation


class HyQEnv(gym.Env):
    """OpenAI Gym environment for HyQ, utilises continuous action space."""

    def __init__(self, robot_cls: type, task_cls: type, robot_kwargs: Dict = None, task_kwargs: Dict = None):
        if task_kwargs is None:
            task_kwargs = {}
        if robot_kwargs is None:
            robot_kwargs = {}
        self.__robot: HyQSim = robot_cls(**robot_kwargs)
        self.robot = self.__robot
        self.__task = task_cls(self.__robot, **task_kwargs)
        self.action_space = self.__robot.get_action_space()
        self.observation_space = self.__task.get_observation_space()
        self.__episode_num = 0
        self.__cumulated_episode_reward = 0
        self.__cumulated_norm_reward = 0
        self.__cumulated_unshaped_reward = 0
        self.__cumulated_exp_reward = 0
        self.__step_num = 0
        self.__last_done_info = None
        now = datetime.now()
        table_name = f'run_{now.strftime("%d_%m_%Y__%H_%M_%S")}'
        self.__logger = HyQLogger(table_name, "hyq_log.db")
        # self.reset()

    def step(self, action: numpy.ndarray) -> Tuple[numpy.ndarray, float, bool, dict]:
        self.__robot.set_action(action)
        obsDataStruct: HyQObservation = self.__robot.get_observations()
        done, done_info = self.__task.is_done(obsDataStruct, self.observation_space, self.__step_num)
        state = done_info['state']
        reward, reward_info = self.__task.compute_reward(obsDataStruct, state, self.__step_num)
        info: dict = {**reward_info, **done_info}
        self.__cumulated_episode_reward += reward
        self.__cumulated_norm_reward += reward_info['normalised_reward']
        self.__cumulated_unshaped_reward += reward_info['normal_reward']
        self.__cumulated_exp_reward += reward_info['exp_reward']

        self.__step_num += 1
        self.__last_done_info = done_info
        log_kwargs = {
                      'episode_num': self.__episode_num,
                      'step_num': self.__step_num,
                      'state': state,
                      'dist_to_goal': reward_info['distance_to_goal'],
                      'target_coords': reward_info['target_coords'],
                      'current_coords': reward_info['current_coords'],
                      'joint_pos': numpy.array(obsDataStruct.joint_positions),
                      'joint_vel': numpy.array(obsDataStruct.joint_velocities),
                      'reward': reward,
                      'normalised_reward': reward_info['normalised_reward'],
                      'exp_reward': reward_info['exp_reward'],
                      'yaw_penalty_factor': reward_info['yaw_penalty_factor'],
                      'pitch_penalty_factor': reward_info['pitch_penalty_factor'],
                      'height_penalty_factor': reward_info['height_penalty_factor'],
                      'cum_unshaped_reward': self.__cumulated_unshaped_reward,
                      'cum_normalised_reward': self.__cumulated_norm_reward,
                      'cum_exp_reward': self.__cumulated_exp_reward,
                      'cum_reward': self.__cumulated_episode_reward,
                      'action': action
                    }
        self.__logger.store(**log_kwargs)
        obs = self.__task.get_observations(obsDataStruct, self.__step_num)

        # print(f"Reward for step {self.__step_num}: {reward}, \t cumulated reward: {self.__cumulated_episode_reward}")
        return obs, reward, done, info

    def reset(self):
        if self.__last_done_info is not None:
            print(f'Episode {self.__episode_num: <6}     Reward: {self.__cumulated_episode_reward:.9f}     '
                  f'Reason: {self.__last_done_info["state"]:<35}      Timesteps: {self.__step_num:<4}')
        else:
            print(f'Episode {self.__episode_num: <6}     Reward: {self.__cumulated_episode_reward:.9f}     '
                  f'total timesteps: {self.__step_num:<4}')
        self.__robot.reset()
        obs = self.__robot.get_observations()
        self.__task.reset()
        self.__step_num = 0
        self.__last_done_info = None
        self.__episode_num += 1
        self.__cumulated_episode_reward = 0
        self.__cumulated_norm_reward = 0
        self.__cumulated_unshaped_reward = 0
        self.__cumulated_exp_reward = 0

        return self.__task.get_observations(obs, self.__step_num)

    def close(self):
        print('Closing ' + self.__class__.__name__ + ' environment.')

    def render(self, mode='human'):
        self.robot.impl.Gui()
        pass
