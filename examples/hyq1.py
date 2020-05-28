#!/usr/bin/env python3
import gym
import numpy
from gym_robo.envs import HyQEnv
from gym.spaces import Box
from typing import Type


def main(args=None):
    env: HyQEnv = gym.make('HyQ-v0')
    action_space: Type[Box] = env.action_space
    done = False
    for _ in range(10):
        print("-------------Starting----------------")
        count = 0
        while not done:
            # action = numpy.array([1.00, -1.01, 1.01])
            action = action_space.sample()
            action_do_nothing = numpy.zeros((12,))
            observation, reward, done, info = env.step(action)
            # Type hints
            observation: numpy.ndarray
            reward: float
            done: bool
            info: dict
            if done:
                arm_state = info['state']
                print(arm_state)
        # time.sleep(1.0)
        env.reset()
        done = False
        print("-------------Reset finished---------------")
    print("Finished")


if __name__ == '__main__':
    main()
