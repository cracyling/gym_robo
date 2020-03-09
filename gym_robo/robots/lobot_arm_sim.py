import LobotArmPy as Arm
from LobotArmPy import LobotArmObservation, LobotArmConfig

from gym.spaces import Box

import numpy
import time

class LobotArmSim:
    """Simulated Lobot Arm"""

    '''-------------PUBLIC METHODS START-------------'''

    def __init__(self, use_gui=False):
        config = LobotArmConfig()
        config.sim_step_size = 0.001
        config.real_time_factor = 0.0
        self.impl = Arm.LobotArm(config)
        self.impl.SetVerbosity(4)
        if use_gui:
            self.impl.Gui()
            self.impl.Run(100)
        # time.sleep(2)

    def set_action(self, action: numpy.ndarray) -> None:
        """
        Sets the action and runs the simulation
        :param action:
        :return: None
        """
        assert len(action) == 3, f'{len(action)} actions passed to LobotArmSim, expected: 3'
        assert action.shape == (3,), f'Expected action shape of {(3,)}, actual shape: {action.shape}'

        self.impl.SetAction(action)
        self.impl.Run(10)

    def reset(self) -> None:
        self.impl.Reset()
        self.impl.Run(100)

    def get_action_space(self):
        return Box(numpy.array([-2.356, -1.570796, -1.570796]), numpy.array([2.356, 0.5, 1.570796]))

    def get_observations(self) -> LobotArmObservation:
        return self.impl.GetObservations()

    def spawn_marker(self, x: float, y: float, z: float, diameter: float, id: int):
        return self.impl.SpawnMarker(x, y, z, diameter, id)
