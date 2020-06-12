import numpy

from HyQPy import HyQObservation


def hyq_obs_to_numpy(obs: HyQObservation, initial_time_sec: float):
    values = numpy.zeros((40,))
    values[:12] = obs.joint_positions
    values[12:24] = obs.joint_velocities
    for pair in obs.contact_pairs:
        if "lf_foot" in pair[0] and "ground_collision" in pair[1]:
            values[24] = 1
            continue
        if "lh_foot" in pair[0] and "ground_collision" in pair[1]:
            values[25] = 1
            continue
        if "rf_foot" in pair[0] and "ground_collision" in pair[1]:
            values[26] = 1
            continue
        if "rh_foot" in pair[0] and "ground_collision" in pair[1]:
            values[27] = 1
    if obs.trunk_contact:
        values[28] = 1
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
    values[39] = obs.sec + obs.nanosec/1000000000 - initial_time_sec

    return values
