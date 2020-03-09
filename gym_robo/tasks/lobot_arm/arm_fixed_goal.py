import numpy
from typing import Dict

from gym_robo.robots import LobotArmSim
from .arm_random_goal import LobotArmRandomGoal


class LobotArmFixedGoal(LobotArmRandomGoal):
    def __init__(self, robot, **kwargs):
        super().__init__(robot, **kwargs)

        # The target coords is currently arbitrarily set to some point achievable
        # This is the target for grip_end_point when target joint values are: [1.00, -1.00, 1.00]
        target_x = 0.10175
        target_y = -0.05533
        target_z = 0.1223
        self.target_coords = numpy.array([target_x, target_y, target_z])
        if isinstance(self.robot, LobotArmSim):  # Check if is gazebo
            # Spawn the target marker if it is gazebo
            self.robot.spawn_marker(target_x, target_y, target_z, 0.002, 1)

        self.previous_coords = numpy.array([0.0, 0.0, 0.0])

    def reset(self):
        self.previous_coords = numpy.array([0.0, 0.0, 0.0])
