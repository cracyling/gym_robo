import numpy

from HyQPy import HyQObservation


def hyq_obs_to_numpy(obs: HyQObservation):
    values = numpy.zeros((39,))
    values[:12] = obs.joint_positions
    values[12:24] = obs.joint_velocities
    values[24] = obs.lf_foot_contact
    values[25] = obs.lh_foot_contact
    values[26] = obs.rf_foot_contact
    values[27] = obs.rh_foot_contact
    values[28] = obs.trunk_contact
    values[29] = obs.pose.rotation.w
    values[30] = obs.pose.rotation.x
    values[31] = obs.pose.rotation.y
    values[32] = obs.pose.rotation.z
    values[33] = obs.linear_acceleration.x
    values[34] = obs.linear_acceleration.y
    values[35] = obs.linear_acceleration.z
    values[36] = obs.pose.position.x
    values[37] = obs.pose.position.y
    values[38] = obs.pose.position.z

    return values
