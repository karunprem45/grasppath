"""Compare my forward kinematics with PyBullet's on random joint angles.

Run this first. If the errors are tiny, my DH model matches the robot in the simulator.
"""
import numpy as np
import pybullet as p
import pybullet_data

from grasppath.kinematics import panda
from grasppath.sim import ARM_JOINTS, TCP_LINK


def main(n=200, seed=0):
    arm = panda(tcp_offset=0.105)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    rng = np.random.default_rng(seed)
    pos_err, rot_err = [], []
    for _ in range(n):
        q = arm.random_q(rng)
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(robot, j, v)
        st = p.getLinkState(robot, TCP_LINK, computeForwardKinematics=True)
        pos_sim = np.array(st[4])
        R_sim = np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        T = arm.fk(q)
        pos_err.append(np.linalg.norm(T[:3, 3] - pos_sim))
        cos = np.clip((np.trace(T[:3, :3].T @ R_sim) - 1) / 2, -1, 1)
        rot_err.append(np.degrees(np.arccos(cos)))
    p.disconnect()
    pos_err, rot_err = np.array(pos_err) * 1000, np.array(rot_err)
    print(f"{n} random configurations")
    print(f"position error (mm): mean {pos_err.mean():.4f}, max {pos_err.max():.4f}")
    print(f"rotation error (deg): mean {rot_err.mean():.4f}, max {rot_err.max():.4f}")


if __name__ == "__main__":
    main()
