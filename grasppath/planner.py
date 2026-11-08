"""RRT-Connect in joint space, then shortcut smoothing."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class PlanResult:
    path: list
    success: bool
    plan_time_s: float
    nodes: int


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, dtype=float)]
        self.parent = [-1]

    def nearest(self, q):
        d = np.linalg.norm(np.array(self.nodes) - q, axis=1)
        return int(np.argmin(d))

    def add(self, q, parent):
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, i):
        out = []
        while i != -1:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


def _extend(tree, q_target, checker, step):
    i = tree.nearest(q_target)
    q_near = tree.nodes[i]
    diff = q_target - q_near
    dist = np.linalg.norm(diff)
    q_new = q_target if dist <= step else q_near + diff * (step / dist)
    if not checker.edge_free(q_near, q_new):
        return "trapped", None
    j = tree.add(q_new, i)
    return ("reached" if dist <= step else "advanced"), j


def _connect(tree, q_target, checker, step):
    while True:
        status, j = _extend(tree, q_target, checker, step)
        if status != "advanced":
            return status, j


def rrt_connect(q_start, q_goal, checker, rng=None, step=0.3, max_iters=5000, timeout_s=10.0):
    """Grow one tree from the start and one from the goal until they meet."""
    rng = rng or np.random.default_rng(0)
    t0 = time.perf_counter()
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    if checker.in_collision(q_start) or checker.in_collision(q_goal):
        return PlanResult([], False, time.perf_counter() - t0, 0)
    if checker.edge_free(q_start, q_goal):
        return PlanResult([q_start, q_goal], True, time.perf_counter() - t0, 2)
    a, b = _Tree(q_start), _Tree(q_goal)
    a_is_start = True
    for _ in range(max_iters):
        if time.perf_counter() - t0 > timeout_s:
            break
        status, i = _extend(a, checker.arm.random_q(rng), checker, step)
        if status != "trapped":
            status_b, j = _connect(b, a.nodes[i], checker, step)
            if status_b == "reached":
                path = a.path_to_root(i)[::-1] + b.path_to_root(j)[1:]
                if not a_is_start:
                    path = path[::-1]
                return PlanResult(path, True, time.perf_counter() - t0,
                                  len(a.nodes) + len(b.nodes))
        a, b = b, a
        a_is_start = not a_is_start
    return PlanResult([], False, time.perf_counter() - t0, len(a.nodes) + len(b.nodes))


def shortcut(path, checker, rng=None, tries=100):
    """Randomly replace stretches of the path with straight collision-free segments."""
    rng = rng or np.random.default_rng(0)
    path = [np.asarray(p, float) for p in path]
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if checker.edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def path_length(path):
    """Total joint-space distance along the path (radians)."""
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))
