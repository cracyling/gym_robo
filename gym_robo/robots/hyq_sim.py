import HyQPy
from HyQPy import HyQObservation

from gym.spaces import Box

import numpy
import time

import subprocess, threading

class HyQSim:
    """Simulated HyQ robot"""

    '''-------------PUBLIC METHODS START-------------'''

    def __init__(self, use_gui=False, rtf=1.0, sim_step_size=0.001, control_mode="Relative", initial_joint_pos=None, spawn_position=None,
                 spawn_orientation=None, start_steps=None):
        # HyQPy.HyQ.SetVerbosity(4)
        self.impl = HyQPy.HyQ(rtf=rtf, step_size=sim_step_size, control_mode=control_mode,
                              initial_joint_pos=initial_joint_pos, spawn_position=spawn_position,
                              spawn_orientation=spawn_orientation, start_steps=start_steps)
        self.control_mode = control_mode
        self.proc = None
        self.t = None
        if use_gui:
            self.impl.Gui()
            time.sleep(3.5)

    def set_action(self, action: numpy.ndarray) -> None:
        """
        Sets the action and runs the simulation
        :param action:
        :return: None
        """
        assert len(action) == 12, f'{len(action)} actions passed to HyQ, expected: 12'
        assert action.shape == (12,), f'Expected action shape of {(12,)}, actual shape: {action.shape}'

        self.impl.SetAction(action)
        self.impl.Run(10)

    def reset(self) -> None:
        self.impl.Reset()

    def get_action_space(self):
        if self.control_mode.lower() == "relative":
            return Box(-1, 1, (12,))
        elif self.control_mode.lower() == "absolute":
            joint_lower_limits = [-1.22173, -0.872665, -2.44346,
                                  -1.22173, -1.22173, 0.349066,
                                  -1.22173, -0.872665, -2.44346,
                                  -1.22173, -1.22173, 0.349066]
            joint_upper_limits = [0.436332, 1.22173, -0.349066,
                                  0.436332, 0.872665, 2.44346,
                                  0.436332, 1.22173, -0.349066,
                                  0.436332, 0.872665, 2.44346]
            return Box(numpy.array(joint_lower_limits), numpy.array(joint_upper_limits))

    @staticmethod
    def get_observation_space():
        obs_min = numpy.array([-1.22173, -0.872665, -2.44346,
                               -1.22173, -1.22173, 0.349066,
                               -1.22173, -0.872665, -2.44346,
                               -1.22173, -1.22173, 0.349066,
                               -12, -12, -12,
                               -12, -12, -12,
                               -12, -12, -12,
                               -12, -12, -12,
                               0, 0, 0, 0, 0,
                               -1, -1, -1, -1,
                               -50, -50, -50,
                               -10, -10, 0])
        obs_max = numpy.array([0.436332, 1.22173, -0.349066,
                               0.436332, 0.872665, 2.44346,
                               0.436332, 1.22173, -0.349066,
                               0.436332, 0.872665, 2.44346,
                               12, 12, 12,
                               12, 12, 12,
                               12, 12, 12,
                               12, 12, 12,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1,
                               50, 50, 50,
                               10, 10, 1])
        return Box(obs_min, obs_max)

    def get_observations(self) -> HyQObservation:
        return self.impl.GetObservations()

    def spawn_marker(self, x: float, y: float, z: float, diameter: float, id: int):
        return self.impl.SpawnMarker(x, y, z, diameter, id)


