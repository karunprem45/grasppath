"""Plan one move around the wall and plot joint position and velocity over time.

Runs without PyBullet. Saves results/trajectory.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grasppath.kinematics import panda, PANDA_HOME, solve_ik, top_down_pose
from grasppath.collision import Box, CollisionChecker
from grasppath.planner import rrt_connect, shortcut, path_length
from grasppath.trajectory import time_parameterize, PANDA_VMAX


def main(seed=0):
    rng = np.random.default_rng(seed)
    arm = panda()
    checker = CollisionChecker(arm, [Box(np.array([0.5, 0.0, 0.25]), np.array([0.1, 0.02, 0.25]))])
    q_a = solve_ik(arm, top_down_pose(0.5, 0.25, 0.2), PANDA_HOME, rng=rng).q
    q_b = solve_ik(arm, top_down_pose(0.5, -0.25, 0.2), PANDA_HOME, rng=rng).q
    plan = rrt_connect(q_a, q_b, checker, rng=rng)
    short = shortcut(plan.path, checker, rng=rng)
    t, q = time_parameterize(short)
    v = np.gradient(q, t, axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for j in range(7):
        ax[0].plot(t, q[:, j], label=f"joint {j + 1}")
        ax[1].plot(t, v[:, j] / PANDA_VMAX[j])
    ax[0].set_ylabel("position (rad)")
    ax[0].legend(ncol=4, fontsize=8)
    ax[1].set_ylabel("velocity / limit")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].axhline(-1, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("time (s)")
    ax[0].set_title(f"RRT-Connect + shortcut: {len(plan.path)} -> {len(short)} waypoints, "
                    f"path {path_length(plan.path):.2f} -> {path_length(short):.2f} rad")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/trajectory.png", dpi=120)
    print("saved results/trajectory.png")


if __name__ == "__main__":
    main()
