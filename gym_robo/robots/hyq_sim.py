import HyQPy
from HyQPy import HyQObservation

from gym.spaces import Box

import numpy
import time

import subprocess, threading

class HyQSim:
    """Simulated HyQ robot"""

    '''-------------PUBLIC METHODS START-------------'''

    def __init__(self, use_gui=False, rtf=1.0, sim_step_size=0.001, control_mode="Relative"):
        HyQPy.HyQ.SetVerbosity(4)
        self.impl = HyQPy.HyQ(rtf=rtf, step_size=sim_step_size, control_mode=control_mode)
        self.control_mode = control_mode
        self.proc = None
        self.t = None
        if use_gui:
            self.impl.Gui()
            time.sleep(2)

    def gui(self):
        def output_reader(proc):
            for line in iter(proc.stdout.readline, b''):
                print('GUI output: {0}'.format(line.decode('utf-8')), end='')

        self.proc = subprocess.Popen(['ign', 'gazebo', '-g'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        self.t = threading.Thread(target=output_reader, args=(self.proc,))
        self.t.start()

    def close_gui(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=0.2)
                print('== subprocess exited with code =', self.proc.returncode)
            except subprocess.TimeoutExpired:
                print('subprocess did not terminate in time')
            finally:
                self.proc = None
        if self.t is not None:
            self.t.join()
            self.t = None

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
                               -10, -10, 0,
                               0.0])
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
                               10, 10, 1,
                               5.0])
        return Box(obs_min, obs_max)

    def get_observations(self) -> HyQObservation:
        return self.impl.GetObservations()

    def spawn_marker(self, x: float, y: float, z: float, diameter: float, id: int):
        return self.impl.SpawnMarker(x, y, z, diameter, id)


