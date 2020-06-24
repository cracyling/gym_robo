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
filePath = scriptDir + "/stable_traj3.json"
print(f"trying to open ref traj from {filePath}")
with open(filePath, 'r') as fp:
    data = json.load(fp)
time = [x['time_from_start']['nsecs'] / 1000000000 + x['time_from_start']['secs'] for x in data]
joint_angles = []
num_joints = len(data[0]['joint_positions'])
for i in range(num_joints):
    angles = [x['joint_positions'][i] for x in data]
    joint_angles.append(angles)

action_list = [*zip(*joint_angles)]
joint_values_real = []
deltas = []
initial_joint_angles = action_list[0]

def main(args=None):
    env: HyQEnv = gym.make('HyQ-v0', robot_kwargs={
        'use_gui': True,
        'rtf': 0.2,
        'control_mode': "Absolute",
        'initial_joint_pos': initial_joint_angles,
        'spawn_position': numpy.array([0.0,0.0,0.65]),
        'start_steps': 1000
    })
    action_space: Type[Box] = env.action_space
    done = False
    index = 0
    env.reset()
    while not done and index < len(action_list) + 500:
        # action = numpy.array([1.00, -1.01, 1.01])
        action = action_space.sample()
        action_do_nothing = numpy.zeros((12,))
        if (index >= len(action_list)):
            action_ref = numpy.array(action_list[-1])
        else:
            action_ref = numpy.array(action_list[index])
        observation, reward, done, info = env.step(action_ref)
        joint_values_real.append(observation[:12])
        deltas.append(action_ref - observation[:12])
        index += 1
        if done:
            arm_state = info['state']
            print(arm_state)
    # time.sleep(1.0)
    index = 0
    env.reset()
    done = False
    with open('joint_vals_real.npy', 'wb') as f:
        numpy.save(f, numpy.array(joint_values_real))
    with open('deltas.npy', 'wb') as f2:
        numpy.save(f2, numpy.array(deltas))
    print("Finished")


if __name__ == '__main__':
    main()
