"""Forward and inverse kinematics for a serial arm described by modified DH parameters.

Written from scratch with NumPy so every step can be explained:
- forward kinematics: chain one 4x4 transform per joint
- geometric Jacobian: built from the joint axes and origins
- inverse kinematics: damped least squares with joint limits and random restarts
"""
from dataclasses import dataclass, field
import numpy as np


def mdh_transform(a, alpha, d, theta):
    """Modified DH (Craig) transform from frame i-1 to frame i."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def trans(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


@dataclass
class SerialArm:
    """A revolute serial arm. Each row of `dh` is (a, alpha, d)."""
    dh: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    tool: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def n(self):
        return len(self.dh)

    def frames(self, q):
        """World transform of every joint frame, then the tool frame (n + 1 transforms)."""
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        out = []
        for (a, alpha, d), theta in zip(self.dh, q):
            T = T @ mdh_transform(a, alpha, d, theta)
            out.append(T.copy())
        out.append(T @ self.tool)
        return out

    def fk(self, q):
        """Tool pose as a 4x4 homogeneous transform."""
        return self.frames(q)[-1]

    def jacobian(self, q):
        """6 x n geometric Jacobian (linear rows first, then angular) at the tool point."""
        fr = self.frames(q)
        p_tool = fr[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            z = fr[i][:3, 2]  # in modified DH, joint i rotates about z of frame i
            p = fr[i][:3, 3]
            J[:3, i] = np.cross(z, p_tool - p)
            J[3:, i] = z
        return J

    def clip(self, q):
        return np.clip(q, self.lower, self.upper)

    def random_q(self, rng):
        return rng.uniform(self.lower, self.upper)


def pose_error(T_now, T_goal):
    """6-vector error: position error, then orientation error as an axis-angle vector."""
    dp = T_goal[:3, 3] - T_now[:3, 3]
    R_err = T_goal[:3, :3] @ T_now[:3, :3].T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        dr = np.zeros(3)
    elif np.pi - angle < 1e-6:
        # near 180 degrees: take the axis from the largest diagonal entry
        k = np.argmax(np.diag(R_err))
        axis = R_err[:, k] + np.eye(3)[k]
        dr = angle * axis / np.linalg.norm(axis)
    else:
        axis = np.array([R_err[2, 1] - R_err[1, 2],
                         R_err[0, 2] - R_err[2, 0],
                         R_err[1, 0] - R_err[0, 1]]) / (2.0 * np.sin(angle))
        dr = angle * axis
    return np.concatenate([dp, dr])


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    pos_err: float   # metres
    rot_err: float   # radians


def solve_ik(arm, T_goal, q0, rng=None, pos_tol=1e-4, rot_tol=1e-3,
             max_iters=300, damping=0.05, max_step=0.2, restarts=10):
    """Damped least squares IK. Tries q0 first, then random restarts inside the joint limits."""
    rng = rng or np.random.default_rng(0)
    total_iters = 0
    best = None
    starts = [np.asarray(q0, dtype=float)] + [arm.random_q(rng) for _ in range(restarts)]
    for start in starts:
        q = arm.clip(start)
        for _ in range(max_iters):
            total_iters += 1
            e = pose_error(arm.fk(q), T_goal)
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < pos_tol and re < rot_tol:
                return IKResult(q, True, total_iters, pe, re)
            J = arm.jacobian(q)
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            step = np.max(np.abs(dq))
            if step > max_step:
                dq *= max_step / step
            q = arm.clip(q + dq)
        e = pose_error(arm.fk(q), T_goal)
        pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if best is None or pe + re < best.pos_err + best.rot_err:
            best = IKResult(q, False, total_iters, pe, re)
    best.iterations = total_iters
    return best


def panda(tcp_offset=0.1034):
    """Franka Emika Panda, modified DH parameters from Franka's public documentation.

    Tool frame = the point between the fingertips: flange (d=0.107), then the hand's
    -45 degree rotation about z, then tcp_offset to the fingertip centre.
    Franka's docs give 0.1034 m; the panda.urdf shipped with PyBullet uses 0.105 m,
    so the simulator passes tcp_offset=0.105.
    """
    dh = np.array([
        [0.0, 0.0, 0.333],
        [0.0, -np.pi / 2, 0.0],
        [0.0, np.pi / 2, 0.316],
        [0.0825, np.pi / 2, 0.0],
        [-0.0825, -np.pi / 2, 0.384],
        [0.0, np.pi / 2, 0.0],
        [0.088, np.pi / 2, 0.0],
    ])
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    tool = trans(z=0.107) @ rot_z(-np.pi / 4) @ trans(z=tcp_offset)
    return SerialArm(dh, lower, upper, tool)


PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def top_down_pose(x, y, z, yaw=0.0):
    """Gripper pointing straight down at (x, y, z), turned by yaw about the vertical axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [x, y, z]
    return T
