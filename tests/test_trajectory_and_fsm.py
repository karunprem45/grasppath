import numpy as np
from grasppath.trajectory import time_parameterize, PANDA_VMAX, PANDA_AMAX
from grasppath.state_machine import PickPlaceFSM, State


def test_trajectory_hits_waypoints_and_respects_limits():
    path = [np.zeros(7), np.full(7, 0.5), np.array([0.5, -0.3, 0.2, 0.0, 0.1, 0.4, -0.2])]
    dt = 1.0 / 240.0
    t, q = time_parameterize(path, dt=dt)
    assert np.allclose(q[0], path[0]) and np.allclose(q[-1], path[-1])
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    assert np.all(np.abs(v) <= PANDA_VMAX * 1.01)
    assert np.all(np.diff(t) > 0)


class FakeController:
    def __init__(self, fail_at=None, slip_times=0):
        self.fail_at, self.slips = fail_at, slip_times
        self.calls = []

    def run(self, state):
        self.calls.append(state)
        if state == State.LIFT and self.slips > 0:
            self.slips -= 1
            return False
        return state != self.fail_at


def test_fsm_happy_path():
    fsm = PickPlaceFSM(FakeController())
    assert fsm.run() == State.DONE
    assert fsm.history[-1] == State.DONE and fsm.failure is None


def test_fsm_retries_once_after_slip():
    fsm = PickPlaceFSM(FakeController(slip_times=1))
    assert fsm.run() == State.DONE and fsm.retries == 1


def test_fsm_fails_after_second_slip():
    fsm = PickPlaceFSM(FakeController(slip_times=2))
    assert fsm.run() == State.FAILED and fsm.failure == State.LIFT


def test_fsm_reports_where_it_failed():
    fsm = PickPlaceFSM(FakeController(fail_at=State.TRANSFER))
    assert fsm.run() == State.FAILED and fsm.failure == State.TRANSFER
