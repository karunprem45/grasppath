"""Checks my DH model against PyBullet's Panda. Skipped if PyBullet isn't installed."""
import numpy as np
import pytest

p = pytest.importorskip("pybullet")
import pybullet_data  # noqa: E402

from grasppath.kinematics import panda  # noqa: E402
from grasppath.sim import ARM_JOINTS, TCP_LINK  # noqa: E402


def test_my_fk_matches_pybullet_within_1mm():
    arm = panda(tcp_offset=0.105)
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = arm.random_q(rng)
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(robot, j, v)
        pos = np.array(p.getLinkState(robot, TCP_LINK, computeForwardKinematics=True)[4])
        assert np.linalg.norm(arm.fk(q)[:3, 3] - pos) < 1e-3
    p.disconnect(cid)
