"""Pick-and-place state machine.

The machine only decides what happens next. A controller (the PyBullet sim, or a fake
one in the tests) does the work for each state and reports success or failure.
"""
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    APPROACH = auto()      # move above the object
    DESCEND = auto()       # straight down to grasp height
    GRASP = auto()         # close the gripper
    LIFT = auto()          # straight up, then check the object came with us
    TRANSFER = auto()      # plan around obstacles to above the place spot
    PLACE = auto()         # straight down to place height
    RELEASE = auto()       # open the gripper
    RETREAT = auto()       # straight up
    DONE = auto()
    FAILED = auto()


ORDER = [State.APPROACH, State.DESCEND, State.GRASP, State.LIFT, State.TRANSFER,
         State.PLACE, State.RELEASE, State.RETREAT, State.DONE]


class PickPlaceFSM:
    def __init__(self, controller, max_grasp_retries=1):
        self.ctl = controller
        self.state = State.IDLE
        self.history = [State.IDLE]
        self.failure = None
        self.max_retries = max_grasp_retries
        self.retries = 0

    def _go(self, s):
        self.state = s
        self.history.append(s)

    def step(self):
        if self.state in (State.DONE, State.FAILED):
            return self.state
        if self.state == State.IDLE:
            self._go(State.APPROACH)
            return self.state
        if self.ctl.run(self.state):
            self._go(ORDER[ORDER.index(self.state) + 1])
        elif self.state == State.LIFT and self.retries < self.max_retries:
            # the object slipped: open, go back above it and grasp again
            self.retries += 1
            self.ctl.run(State.RELEASE)
            self._go(State.APPROACH)
        else:
            self.failure = self.state
            self._go(State.FAILED)
        return self.state

    def run(self, max_steps=50):
        for _ in range(max_steps):
            if self.step() in (State.DONE, State.FAILED):
                break
        return self.state
