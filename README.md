# GraspPath

Pick-and-place for a 7-joint robot arm (Franka Panda) in PyBullet. The kinematics, motion planning, trajectory timing and state machine are written from scratch in NumPy. PyBullet is only used for physics.

A cube sits at a random spot and angle on one side of a wall. The arm has to pick it up, plan a path around the wall, and place it on a target on the other side.

## What's in it

| File | What it does |
|---|---|
| `grasppath/kinematics.py` | Forward kinematics from modified DH parameters, geometric Jacobian, inverse kinematics by damped least squares with joint limits and random restarts |
| `grasppath/collision.py` | Arm links as chains of spheres, obstacles as boxes |
| `grasppath/planner.py` | RRT-Connect in joint space, plus shortcut smoothing |
| `grasppath/trajectory.py` | Trapezoidal velocity profiles that respect the Panda's joint speed and acceleration limits |
| `grasppath/state_machine.py` | Pick-and-place state machine: approach, descend, grasp, lift, transfer, place, release, retreat, with one grasp retry if the cube slips |
| `grasppath/sim.py` | PyBullet scene and the controller that runs each state |
| `scripts/check_fk_vs_sim.py` | Compares my forward kinematics with PyBullet's |
| `scripts/run_trials.py` | Randomized trials, saved to `results/trials.csv` |
| `scripts/plot_trajectory.py` | Plots joint position and velocity for one planned move |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyBullet sometimes fails to install on the newest Python. If `pip install pybullet` fails, use Python 3.10 or 3.11, or `conda install -c conda-forge pybullet`. On a Mac it may need Xcode command line tools (`xcode-select --install`).

## Run it, in this order

```bash
# 1. Unit tests (the PyBullet test is skipped if PyBullet isn't installed)
pytest -v

# 2. Check my FK model against the simulator. Errors should be well under 1 mm.
python -m scripts.check_fk_vs_sim

# 3. Watch one trial
python -m scripts.run_trials --trials 3 --gui

# 4. The real run
python -m scripts.run_trials --trials 50 --seed 0

# 5. The plot
python -m scripts.plot_trajectory
```

Write your numbers in `RESULTS.md` as you go.

## If something fails

- **FK check shows big errors:** the DH model doesn't match the URDF. Check the TCP offset first (`panda(tcp_offset=0.105)`), then the joint limits and signs.
- **Many failures in DESCEND or PLACE:** the straight-line IK is jumping. Try `n=40` in `_straight_line`.
- **Failures in LIFT:** the cube slips. Try raising `lateralFriction` or `FINGER_FORCE` in `sim.py`, or lowering `GRASP_Z` a little.
- **Failures in TRANSFER:** the planner timed out. Raise `timeout_s` in `rrt_connect`, or lower the wall.

Fixing one of these is a good interview story. Write down what broke and what you changed.

## How the main pieces work

**Forward kinematics.** Each joint has a 4x4 transform built from its DH parameters (a, alpha, d) and its angle. Multiplying them in order gives each joint's pose, and the tool pose at the end.

**Jacobian.** For a revolute joint i with axis z_i at point p_i, column i is `[z_i x (p_tool - p_i), z_i]`. A test checks it against finite differences.

**Inverse kinematics.** Repeat: get the 6D pose error, then solve `dq = J^T (J J^T + lambda^2 I)^-1 e`. The damping term keeps steps small near singularities. Steps are capped, joints are clipped to their limits, and if it gets stuck it restarts from a random configuration.

**RRT-Connect.** One tree grows from the start and one from the goal. Each round, one tree steps toward a random sample, then the other tree tries to reach that new node directly. Every edge is collision-checked in small steps.

**Shortcutting.** Pick two random waypoints. If the straight segment between them is collision-free, drop everything in between.

**Trajectory.** Each segment speeds up for its first quarter, cruises, then slows down for its last quarter. All joints start and stop together, and the segment time is set by whichever joint would otherwise break its speed or acceleration limit.

## Limits (be honest about these in interviews)

- The collision model uses spheres along each link and boxes for obstacles, which is rough. It doesn't check the held cube or the arm against itself.
- Grasps are always top-down. There is no grasp planning from camera data.
- Everything runs in simulation only.
