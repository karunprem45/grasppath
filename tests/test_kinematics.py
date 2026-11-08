import numpy as np
from grasppath.kinematics import panda, PANDA_HOME, solve_ik, pose_error, top_down_pose

ARM = panda()


def test_home_pose_is_in_front_of_base_pointing_down():
    T = ARM.fk(PANDA_HOME)
    x, y, z = T[:3, 3]
    assert 0.25 < x < 0.45 and abs(y) < 1e-6 and 0.3 < z < 0.6
    assert T[2, 2] < -0.99  # tool z axis points down


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = ARM.random_q(rng)
        J = ARM.jacobian(q)
        eps = 1e-6
        for i in range(ARM.n):
            dq = np.zeros(ARM.n)
            dq[i] = eps
            num = pose_error(ARM.fk(q), ARM.fk(q + dq)) / eps
            assert np.allclose(J[:, i], num, atol=1e-4)


def test_ik_round_trip_on_random_reachable_poses():
    rng = np.random.default_rng(2)
    solved = 0
    for _ in range(20):
        q_true = ARM.random_q(rng)
        res = solve_ik(ARM, ARM.fk(q_true), PANDA_HOME, rng=rng)
        if res.success:
            solved += 1
            assert res.pos_err < 1e-4 and res.rot_err < 1e-3
    assert solved >= 18


def test_ik_top_down_grasp_poses_on_the_table():
    rng = np.random.default_rng(3)
    for x, y in [(0.5, 0.0), (0.45, 0.25), (0.45, -0.25), (0.6, 0.1)]:
        res = solve_ik(ARM, top_down_pose(x, y, 0.05), PANDA_HOME, rng=rng)
        assert res.success, (x, y, res)
        assert np.all(res.q >= ARM.lower) and np.all(res.q <= ARM.upper)


def test_ik_reports_failure_for_unreachable_pose():
    res = solve_ik(ARM, top_down_pose(2.0, 0.0, 0.2), PANDA_HOME, restarts=2)
    assert not res.success and res.pos_err > 0.5
