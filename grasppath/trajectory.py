"""Turn a joint-space waypoint path into a timed trajectory.

Each segment uses a trapezoidal velocity profile: speed up for the first quarter,
cruise for the middle half, slow down for the last quarter. All joints in a segment
start and stop together, timed by the joint that needs the longest, so the arm moves
in a straight line in joint space between waypoints.
"""
import numpy as np

# Panda limits from Franka's documentation, scaled to 50% for safety in simulation
PANDA_VMAX = 0.5 * np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
PANDA_AMAX = 0.5 * np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


def segment_duration(dq, vmax, amax):
    """Shortest T such that no joint breaks vmax or amax with the quarter-trapezoid profile.

    For distance d over time T: peak speed = d / (0.75 T), peak accel = 16 d / (3 T^2).
    """
    d = np.abs(np.asarray(dq, float))
    t_v = d / (0.75 * vmax)
    t_a = np.sqrt(16.0 * d / (3.0 * amax))
    return float(np.max(np.maximum(t_v, t_a)))


def profile(T, t):
    """Fraction of the segment covered at time t (0 to 1)."""
    if T <= 0:
        return 1.0
    ta = T / 4.0
    v = 1.0 / (0.75 * T)
    a = v / ta
    if t <= ta:
        return 0.5 * a * t * t
    if t <= T - ta:
        return 0.5 * a * ta * ta + v * (t - ta)
    tr = T - t
    return 1.0 - 0.5 * a * tr * tr


def time_parameterize(path, vmax=PANDA_VMAX, amax=PANDA_AMAX, dt=1.0 / 240.0):
    """Return (times, positions) sampled every dt seconds (240 Hz matches PyBullet)."""
    vmax, amax = np.asarray(vmax, float), np.asarray(amax, float)
    times, qs = [0.0], [np.asarray(path[0], float)]
    t_offset = 0.0
    for q_a, q_b in zip(path[:-1], path[1:]):
        q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
        dq = q_b - q_a
        T = segment_duration(dq, vmax, amax)
        if T == 0:
            continue
        for t in np.arange(dt, T, dt):
            times.append(t_offset + t)
            qs.append(q_a + profile(T, t) * dq)
        times.append(t_offset + T)
        qs.append(q_b)
        t_offset += T
    return np.array(times), np.array(qs)
