import numpy as np
from grasppath.kinematics import panda, PANDA_HOME, solve_ik, top_down_pose
from grasppath.collision import Box, CollisionChecker
from grasppath.planner import rrt_connect, shortcut, path_length

ARM = panda()
# a 0.5 m tall wall between the pick side (y > 0) and the place side (y < 0)
WALL = Box(center=np.array([0.5, 0.0, 0.25]), half=np.array([0.1, 0.02, 0.25]))


def _goal(x, y, z):
    res = solve_ik(ARM, top_down_pose(x, y, z), PANDA_HOME, rng=np.random.default_rng(0))
    assert res.success
    return res.q


def test_box_distance():
    b = Box(np.zeros(3), np.ones(3))
    assert b.distance([0.5, 0.5, 0.5]) == 0.0
    assert np.isclose(b.distance([3.0, 0.0, 0.0]), 2.0)


def test_plan_around_wall_is_collision_free():
    checker = CollisionChecker(ARM, [WALL])
    q_a, q_b = _goal(0.5, 0.25, 0.2), _goal(0.5, -0.25, 0.2)
    assert not checker.in_collision(q_a) and not checker.in_collision(q_b)
    assert not checker.edge_free(q_a, q_b)  # the straight path really is blocked
    res = rrt_connect(q_a, q_b, checker, rng=np.random.default_rng(4))
    assert res.success
    assert np.allclose(res.path[0], q_a) and np.allclose(res.path[-1], q_b)
    for p, n in zip(res.path[:-1], res.path[1:]):
        assert checker.edge_free(p, n)


def test_shortcut_keeps_path_valid_and_not_longer():
    checker = CollisionChecker(ARM, [WALL])
    q_a, q_b = _goal(0.5, 0.25, 0.2), _goal(0.5, -0.25, 0.2)
    res = rrt_connect(q_a, q_b, checker, rng=np.random.default_rng(5))
    short = shortcut(res.path, checker, rng=np.random.default_rng(5))
    assert path_length(short) <= path_length(res.path) + 1e-9
    for p, n in zip(short[:-1], short[1:]):
        assert checker.edge_free(p, n)


def test_start_in_collision_fails_fast():
    blocker = Box(ARM.fk(PANDA_HOME)[:3, 3], np.array([0.05, 0.05, 0.05]))
    checker = CollisionChecker(ARM, [blocker])
    res = rrt_connect(PANDA_HOME, PANDA_HOME + 0.1, checker)
    assert not res.success
