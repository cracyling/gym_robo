#!/usr/bin/env python3
import json
import os
import gym
import numpy
from gym_robo.envs import HyQEnv
from gym.spaces import Box
from typing import Type

# joint_lower_limits = numpy.array([-1.22173, -0.872665, -2.44346,
#                       -1.22173, -1.22173, 0.349066,
#                       -1.22173, -0.872665, -2.44346,
#                       -1.22173, -1.22173, 0.349066])
# joint_upper_limits = numpy.array([0.436332, 1.22173, -0.349066,
#                       0.436332, 0.872665, 2.44346,
#                       0.436332, 1.22173, -0.349066,
#                       0.436332, 0.872665, 2.44346])
scriptDir = os.path.dirname(os.path.realpath(__file__))
filePath = scriptDir + "/ref_trajectory.json"
print(f"trying to open ref traj from {filePath}")
fp = open(filePath)
data = json.load(fp)

time = [x['time_from_start']['nsecs']/1000000000 + x['time_from_start']['secs'] for x in data]
joint_angles = []
num_joints = len(data[0]['joint_positions'])
for i in range(num_joints):
    angles = [x['joint_positions'][i] for x in data]
    joint_angles.append(angles)
    print(i)

# swap index 3,4,5 with 6,7,8
joint_angles[3], joint_angles[6] = joint_angles[6], joint_angles[3]
joint_angles[4], joint_angles[7] = joint_angles[7], joint_angles[4]
joint_angles[5], joint_angles[8] = joint_angles[8], joint_angles[5]
action_list = [*zip(*joint_angles)]
def main(args=None):
    env: HyQEnv = gym.make('HyQ-v0')
    action_space: Type[Box] = env.action_space
    done = False
    index = 0
    print("-------------Starting----------------")
    while not done and index < len(action_list):
        # action = numpy.array([1.00, -1.01, 1.01])
        action = action_space.sample()
        action_do_nothing = numpy.zeros((12,))
        action_ref = numpy.array(action_list[index])
        observation, reward, done, info = env.step(action_ref)
        index += 1
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
