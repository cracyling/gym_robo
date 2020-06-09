import random
from collections import deque
from typing import Dict, Tuple
from enum import Enum, auto
import numpy
from gym_robo.robots import HyQSim
from gym.spaces import Box
import os
import math
from HyQPy import HyQObservation, Pose
import pickle


class HyQState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Fallen = auto()
    Timeout = auto()
    Undefined = auto()


def quaternion_to_euler(w, x, y, z) -> Tuple[float, float, float]:
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(numpy.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class HyQTask1:
    def __init__(self, robot, max_time_step: int = 1000, accepted_dist_to_bounds=0.01,
                 accepted_error=0.01, reach_target_bonus_reward=0.0, reach_bounds_penalty=0.0, fall_penalty=0.0,
                 episodes_per_goal=1, goal_buffer_size=20, goal_from_buffer_prob=0.0, num_adjacent_goals=0, is_validation=False,
                 random_goal_seed=None, random_goal_file=None, normalise_reward=False, continuous_run=False, exp_rew_scaling=None):
        self.robot = robot
        self._max_time_step = max_time_step
        self.accepted_dist_to_bounds = accepted_dist_to_bounds
        self.accepted_error = accepted_error
        self.reach_target_bonus_reward = reach_target_bonus_reward
        self.reach_bounds_penalty = reach_bounds_penalty
        self.fall_penalty = fall_penalty
        self.episodes_per_goal = episodes_per_goal
        self.goal_buffer_size = goal_buffer_size
        self.goal_from_buffer_prob = goal_from_buffer_prob
        self.num_adjacent_goals = num_adjacent_goals
        self.is_validation = is_validation
        self.random_goal_seed = random_goal_seed
        self.random_goal_file = random_goal_file
        self.normalise_reward = normalise_reward
        self.continuous_run = continuous_run
        self.exp_rew_scaling = exp_rew_scaling
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print('accepted_dist_to_bounds: %8.7f    # Allowable distance to joint limits (radians)' % self.accepted_dist_to_bounds)
        print('accepted_error: %8.7f             # Allowable distance from target coordinates (metres)' % self.accepted_error)
        print('reach_target_bonus_reward: %8.7f # Bonus reward upon reaching target' % self.reach_target_bonus_reward)
        print('reach_bounds_penalty: %8.7f      # Reward penalty when reaching joint limit' % self.reach_bounds_penalty)
        print('fall_penalty: %8.7f           # Reward penalty for collision' % self.fall_penalty)
        print('episodes_per_goal: %8d           # Number of episodes before generating another random goal' % self.episodes_per_goal)
        print('goal_buffer_size: %8d            # Number goals to store in buffer to be reused later' % self.goal_buffer_size)
        print(
            'goal_from_buffer_prob: %8.7f      # Probability of selecting a random goal from the goal buffer, value between 0 and 1' % self.goal_from_buffer_prob)
        print('num_adjacent_goals: %8d          # Number of nearby goals to be generated for each randomly generated goal ' % self.num_adjacent_goals)
        print(f'random_goal_seed: {str(self.random_goal_seed):8}            # Seed used to generate the random goals')
        print(f'random_goal_file: {self.random_goal_file}       # Path to the numpy save file containing the random goals')
        print(
            'is_validation: %8r               # Whether this is a validation run, if true will print which points failed and how many reached' % self.is_validation)
        print(
            'normalise_reward: %8r            # Perform reward normalisation, this happens before reward bonus and penalties' % self.normalise_reward)
        print('continuous_run: %8r              # Continuously run the simulation, even after it reaches the destination' % self.continuous_run)
        print(
            f'exp_rew_scaling: {self.exp_rew_scaling}            # Constant for exponential reward scaling (None by default, recommended 5.0, cumulative exp_reward = 29.48)')
        print(f'-------------------------------------------------------------------------------------')

        assert self.accepted_dist_to_bounds >= 0.0, 'Allowable distance to joint limits should be positive'
        assert self.accepted_error >= 0.0, 'Accepted error to end coordinates should be positive'
        assert self.reach_target_bonus_reward >= 0.0, 'Reach target bonus reward should be positive'
        assert self.fall_penalty >= 0.0, 'Contact penalty should be positive'
        assert self.reach_bounds_penalty >= 0.0, 'Reach bounds penalty should be positive'
        assert isinstance(self.episodes_per_goal, int), f'Episodes per goal should be an integer, current type: {type(self.episodes_per_goal)}'
        assert self.episodes_per_goal >= 1, 'Episodes per goal be greater than or equal to 1, i.e. episodes_per_goal >= 1'
        assert isinstance(self.goal_buffer_size, int), f'Goal buffer size should be an integer, current type: {type(self.goal_buffer_size)}'
        assert self.goal_buffer_size > 0, 'Goal buffer size should be greater than or equal to 1, i.e. episodes_per_goal >= 1'
        assert 0 <= self.goal_from_buffer_prob <= 1, 'Probability of selecting goal from buffer should be between 0 and 1'
        assert isinstance(self.num_adjacent_goals,
                          int), f'Number of adjacent goals should be an integer, current type: {type(self.num_adjacent_goals)}'
        assert self.num_adjacent_goals >= 0, f'Number of adjacent goals should be positive, current value: {self.num_adjacent_goals}'
        if self.random_goal_seed is not None:
            assert isinstance(self.random_goal_seed, int), f'Random goal seed should be an integer, current type: {type(self.random_goal_seed)}'
        if self.random_goal_file is not None:
            assert os.path.exists(self.random_goal_file), f'Random goal file does not exist, path: {self.random_goal_file}'
        if exp_rew_scaling is not None:
            assert isinstance(self.exp_rew_scaling,
                              float), f'Exponential reward scaling factor should be a float, current type: {type(self.exp_rew_scaling)}'

        self._max_time_step = max_time_step

        self.coords_buffer = deque(maxlen=self.goal_buffer_size)
        self.target_coords = self.__get_target_coords()

        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])

        self.__reset_count: int = 0
        self.__reach_count: int = 0

    def is_done(self, obs: HyQObservation, observation_space: Box, time_step: int = -1) -> Tuple[bool, Dict]:
        failed, state = self.__is_failed(obs, observation_space, time_step)
        info_dict = {'state': state}
        if failed:
            # self.fail_points.append((self.target_coords, self.target_coords_ik))
            if self.is_validation:
                print(f'Failed to reach {self.target_coords}')
            return True, info_dict

        current_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        # As long as any coordinate is out of acceptance range, we are not done
        for i in range(3):
            if abs(self.target_coords[i] - current_coords[i]) > self.accepted_error:
                info_dict['state'] = HyQState.InProgress
                return False, info_dict
        # If all coordinates within acceptance range AND time step within limits, we are done
        info_dict['state'] = HyQState.Reached
        info_dict['step_count'] = time_step
        print(f'Reached destination, target coords: {self.target_coords}, current coords: {current_coords}, time step: {time_step}')
        self.__reach_count += 1
        if self.is_validation:
            print(f'Reach count: {self.__reach_count}')

        if self.continuous_run:
            return False, info_dict
        else:
            return True, info_dict

    def compute_reward(self, obs: HyQObservation, state: HyQState) -> Tuple[float, Dict]:

        current_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])

        assert state != HyQState.Undefined, f'State cannot be undefined, please check logic'

        reward = self.__calc_dist_change(self.previous_coords, current_coords)

        # normalise rewards
        mag_target = numpy.linalg.norm(self.initial_coords - self.target_coords)
        normalised_reward = reward / mag_target

        # Scale up normalised reward slightly such that the total reward is between 0 and 10 instead of between 0 and 1
        normalised_reward *= 10

        # Scale up reward so that it is not so small if not normalised
        normal_scaled_reward = reward * 100

        # Calculate current distance to goal (for information purposes only)
        dist = numpy.linalg.norm(current_coords - self.target_coords)

        reward_info = {'normalised_reward': normalised_reward,
                       'normal_reward': normal_scaled_reward,
                       'distance_to_goal': dist,
                       'target_coords': self.target_coords,
                       'current_coords': current_coords}

        if self.normalise_reward:
            reward = normalised_reward
        else:
            reward = normal_scaled_reward

        # Calculate exponential reward component
        if self.exp_rew_scaling is not None:
            exp_reward = self.__calc_exponential_reward(self.previous_coords, current_coords)
            reward_info['exp_reward'] = exp_reward
            reward += exp_reward
        else:
            reward_info['exp_reward'] = 0.0

        self.previous_coords = current_coords

        # Reward shaping logic

        # Check if it has reached target destination
        if state == HyQState.Reached:
            # if reached target destination and is continuous run, we generate another set of coordinates
            # This has to be after the __calc_dist_change function because that uses self.target_coords to calculate
            if self.continuous_run:
                self.target_coords = self.__get_target_coords()
                print(f'Moving to [{self.target_coords[0]:.6f}, {self.target_coords[1]:.6f}, {self.target_coords[2]:.6f}]')
            reward += self.reach_target_bonus_reward

        # Scaling reward penalties
        total_penalty_factor = self.__calc_rew_penalty_scale(obs.pose, reward_info)
        reward *= total_penalty_factor

        # Check if it has approached any joint limits
        if state == HyQState.ApproachJointLimits:
            reward -= self.reach_bounds_penalty

        # Check for collision
        if state == HyQState.Fallen:
            reward -= self.fall_penalty
        return reward, reward_info

    def reset(self):
        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.__reset_count += 1
        if self.__reset_count % self.episodes_per_goal == 0:
            self.target_coords = self.__get_target_coords()
            print(f'Moving to [{self.target_coords[0]:.6f}, {self.target_coords[1]:.6f}, {self.target_coords[2]:.6f}]')
            self.robot.spawn_marker(self.target_coords[0], self.target_coords[1], self.target_coords[2], 0.2, 1)

    def __is_failed(self, obs: HyQObservation, observation_space: Box, time_step: int = -1) -> Tuple[bool, HyQState]:
        info_dict = {'state': HyQState.Undefined}

        # Check if time step exceeds limits, i.e. timed out
        # Time step starts from 0, that means if we only want to run 2 steps time_step will be 0,1 and we need to stop at 1
        if time_step + 1 >= self._max_time_step:
            return True, HyQState.Timeout

        # Check that joint values are not approaching limits
        joint_angles = numpy.array(obs.joint_positions)
        upper_bound = observation_space.high[:12]  # First 12 values are the joint angles
        lower_bound = observation_space.low[:12]
        min_dist_to_upper_bound = min(abs(joint_angles - upper_bound))
        min_dist_to_lower_bound = min(abs(joint_angles - lower_bound))
        # self.accepted_dist_to_bounds is basically how close to the joint limits can the joints go,
        # i.e. limit of 1.57 with accepted dist of 0.1, then the joint can only go until 1.47
        if min_dist_to_lower_bound < self.accepted_dist_to_bounds or min_dist_to_upper_bound < self.accepted_dist_to_bounds:
            info_dict['state'] = HyQState.ApproachJointLimits
            return True, HyQState.ApproachJointLimits

        # Didn't fail
        return False, HyQState.Undefined

    def __calc_dist_change(self, coords_init: numpy.ndarray,
                           coords_next: numpy.ndarray) -> float:
        # Efficient euclidean distance calculation by numpy, most likely uses vector instructions
        diff_abs_init = numpy.linalg.norm(coords_init - self.target_coords)
        diff_abs_next = numpy.linalg.norm(coords_next - self.target_coords)

        return diff_abs_init - diff_abs_next

    def __calc_exponential_reward(self, coords_init: numpy.ndarray, coords_next: numpy.ndarray) -> float:
        def calc_cum_reward(dist: float, scaling=5.0):
            # Since dist scales from 1 and ends with 0, which is the opposite of the intended curve, we change x to y where y = 1-x
            # Now y scales from 0 to 1, and then we use y as the "normalised distance"
            y = 1 - dist  # Change the variable such that max reward is when dist is = 0, and reward = 0 when dist is 1
            if y < 0:
                # Linear in negative region, if y = -1 reward is -5, y = 0 reward is 0
                if y < -8:
                    cum_neg_rew = (y + 8) * 0.5 + (-40)
                else:
                    cum_neg_rew = y * 5

                # cum_neg_rew = -1 / scaling * (math.exp(scaling * -y) - 1)
                return cum_neg_rew
            else:
                cum_positive_rew = 1 / scaling * (math.exp(scaling * y) - 1)
                return cum_positive_rew

        # compute exponential scaling normalised reward
        # formula = integral(e^0.4x) from x_init to x_final, x is normalised distance from goal
        # total cumulative reward = 1/scaling * (e^0.4 x_final - 1)
        mag_target = numpy.linalg.norm(self.initial_coords - self.target_coords)
        diff_abs_init_scaled = numpy.linalg.norm(coords_init - self.target_coords) / mag_target
        diff_abs_next_scaled = numpy.linalg.norm(coords_next - self.target_coords) / mag_target

        prev_cum_rew = calc_cum_reward(diff_abs_init_scaled, self.exp_rew_scaling)
        current_cum_rew = calc_cum_reward(diff_abs_next_scaled, self.exp_rew_scaling)
        cum_rew_change = current_cum_rew - prev_cum_rew
        return cum_rew_change

    """
    Calculates the reward penalties based on orientation and height of the robot

    :param pose: The pose of the robot, expects a Pose object of HyQPy.Pose
    :param reward_info: Dictionary of the reward info, since python accepts dictionaries by reference we just directly modify this
    :returns: Total penalty scaling, a multiple of both the orientation and height penalty
    """
    @staticmethod
    def __calc_rew_penalty_scale(pose, reward_info: Dict) -> float:
        q = pose.rotation
        [roll, pitch, yaw] = quaternion_to_euler(q.w, q.x, q.y, q.z)

        # ----- Orientation penalty -----
        # For orientation, we allow 2.5 degrees each side for allowance, then start penalising after
        allowable_yaw_deg = 2.5
        allowable_yaw_rad = allowable_yaw_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(yaw) > allowable_yaw_rad:
            orientation_penalty_factor = math.cos(yaw)
        else:
            orientation_penalty_factor = 1.0
        reward_info['orientation_penalty_factor'] = orientation_penalty_factor

        # ----- Height Penalty -----
        # For height, we do not penalise for height between 0.42 and 0.47 (spawn height is 0.47 then dropped to 0.445 at steady state during ep start)
        penalty_scale_height = 1.0
        current_height = pose.position.z
        if 0.52 < current_height < 0.72:
            height_penalty_factor = 1.0
        else:
            height_diff = abs(0.62-current_height)
            height_penalty_factor = math.exp(height_diff * -2)
        reward_info['height_penalty_factor'] = height_penalty_factor
        return orientation_penalty_factor * height_penalty_factor

    def __get_target_coords(self) -> numpy.ndarray:
        return numpy.array([5.0, 0.0, 0.5])
