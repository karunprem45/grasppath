"""PyBullet scene and controller that runs the pick-and-place state machine on a simulated Panda.

My own code does the kinematics, planning and timing. PyBullet only does the physics:
it moves the joints toward the targets I send and tells me where the cube ends up.
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

from .kinematics import panda, PANDA_HOME, solve_ik, top_down_pose
from .collision import Box, CollisionChecker
from .planner import rrt_connect, shortcut, path_length
from .trajectory import time_parameterize
from .state_machine import State

ARM_JOINTS = list(range(7))
FINGER_JOINTS = [9, 10]
TCP_LINK = 11                      # panda_grasptarget in pybullet_data's panda.urdf
ARM_FORCES = [87.0] * 4 + [12.0] * 3
FINGER_OPEN, FINGER_CLOSED, FINGER_FORCE = 0.04, 0.0, 40.0
DT = 1.0 / 240.0

CUBE_HALF = 0.025                  # cube_small.urdf is a 5 cm cube
HOVER = 0.15                       # height above the cube for approach and retreat
GRASP_Z = CUBE_HALF                # fingertip centre at the cube centre
PLACE_Z = CUBE_HALF + 0.005
LIFT_CHECK = 0.08                  # cube must rise this much or the grasp counts as slipped


class PandaScene:
    """Ground plane, a fixed Panda at the origin, one cube and box obstacles."""

    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.gui = gui
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        for j in FINGER_JOINTS:
            p.changeDynamics(self.robot, j, lateralFriction=1.5)
        self.cube = None
        self.obstacles = []
        self.obstacle_ids = []
        self.finger_target = FINGER_OPEN
        self.reset_arm(PANDA_HOME)

    def close(self):
        p.disconnect(self.client)

    # ---------- scene setup ----------
    def reset_arm(self, q):
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, v)
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot, j, FINGER_OPEN)
        self.finger_target = FINGER_OPEN

    def add_box(self, center, half, rgba=(0.8, 0.3, 0.3, 1.0)):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half))
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(rgba))
        self.obstacle_ids.append(p.createMultiBody(0, col, vis, basePosition=list(center)))
        self.obstacles.append(Box(np.array(center, float), np.array(half, float)))

    def place_cube(self, x, y, yaw):
        if self.cube is not None:
            p.removeBody(self.cube)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        self.cube = p.loadURDF("cube_small.urdf", [x, y, CUBE_HALF], orn)
        p.changeDynamics(self.cube, -1, lateralFriction=1.5)
        self.settle(60)

    # ---------- state reading ----------
    def arm_q(self):
        return np.array([p.getJointState(self.robot, j)[0] for j in ARM_JOINTS])

    def cube_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.cube)
        return np.array(pos), p.getEulerFromQuaternion(orn)[2]

    def tcp_position(self):
        return np.array(p.getLinkState(self.robot, TCP_LINK, computeForwardKinematics=True)[4])

    # ---------- stepping ----------
    def _command(self, q):
        p.setJointMotorControlArray(self.robot, ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=list(q), forces=ARM_FORCES)
        p.setJointMotorControlArray(self.robot, FINGER_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=[self.finger_target] * 2,
                                    forces=[FINGER_FORCE] * 2)

    def step(self):
        p.stepSimulation()
        if self.gui:
            time.sleep(DT)

    def settle(self, steps, q=None):
        q = self.arm_q() if q is None else q
        for _ in range(steps):
            self._command(q)
            self.step()

    def execute(self, times, qs):
        """Send one trajectory sample per physics step. Returns max joint tracking error (rad)."""
        worst = 0.0
        for q in qs:
            self._command(q)
            self.step()
            worst = max(worst, float(np.max(np.abs(self.arm_q() - q))))
        self.settle(60, qs[-1])
        return worst


class SimController:
    """Does the work for each state of PickPlaceFSM and records metrics."""

    def __init__(self, scene, place_xy, rng):
        self.s = scene
        self.arm = panda(tcp_offset=0.105)
        self.checker = CollisionChecker(self.arm, scene.obstacles)
        self.place_xy = np.array(place_xy, float)
        self.rng = rng
        self.q_cmd = scene.arm_q()
        self.cube_start_z = None
        self.m = {"plan_time_s": 0.0, "rrt_nodes": 0, "ik_calls": 0, "ik_fail": 0,
                  "path_len_raw": 0.0, "path_len_short": 0.0, "max_track_err": 0.0,
                  "sim_time_s": 0.0}

    # ---------- helpers ----------
    def _grasp_yaw(self):
        _, yaw = self.s.cube_pose()
        # a cube looks the same every 90 degrees, so pick the yaw closest to 0
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _ik(self, T, seed, restarts=10):
        self.m["ik_calls"] += 1
        res = solve_ik(self.arm, T, seed, rng=self.rng, restarts=restarts)
        if not res.success:
            self.m["ik_fail"] += 1
        return res

    def _run_path(self, path):
        t, qs = time_parameterize(path)
        self.m["sim_time_s"] += float(t[-1])
        err = self.s.execute(t, qs)
        self.m["max_track_err"] = max(self.m["max_track_err"], err)
        self.q_cmd = np.asarray(path[-1], float)
        return True

    def _plan_to(self, T_goal):
        """Collision-free joint-space move to a pose (used for big moves)."""
        res = self._ik(T_goal, self.q_cmd)
        if not res.success or self.checker.in_collision(res.q):
            return False
        plan = rrt_connect(self.q_cmd, res.q, self.checker, rng=self.rng)
        self.m["plan_time_s"] += plan.plan_time_s
        self.m["rrt_nodes"] += plan.nodes
        if not plan.success:
            return False
        short = shortcut(plan.path, self.checker, rng=self.rng)
        self.m["path_len_raw"] += path_length(plan.path)
        self.m["path_len_short"] += path_length(short)
        return self._run_path(short)

    def _straight_line(self, T_goal, n=20, max_jump=0.2):
        """Straight-line Cartesian move: IK at n points, each seeded from the last, no restarts."""
        T_now = self.arm.fk(self.q_cmd)
        path, q = [self.q_cmd], self.q_cmd
        for s in np.linspace(0, 1, n + 1)[1:]:
            T = T_goal.copy()
            T[:3, 3] = (1 - s) * T_now[:3, 3] + s * T_goal[:3, 3]
            res = self._ik(T, q, restarts=0)
            if not res.success or np.max(np.abs(res.q - q)) > max_jump:
                return False
            path.append(res.q)
            q = res.q
        return self._run_path(path)

    def _set_fingers(self, target, steps=120):
        self.s.finger_target = target
        self.s.settle(steps, self.q_cmd)
        self.m["sim_time_s"] += steps * DT
        return True

    # ---------- one call per state ----------
    def run(self, state):
        cube_pos, _ = self.s.cube_pose()
        yaw = self._grasp_yaw()
        px, py = self.place_xy
        if state == State.APPROACH:
            self.cube_start_z = cube_pos[2]
            return self._plan_to(top_down_pose(cube_pos[0], cube_pos[1], GRASP_Z + HOVER, yaw))
        if state == State.DESCEND:
            return self._straight_line(top_down_pose(cube_pos[0], cube_pos[1], GRASP_Z, yaw))
        if state == State.GRASP:
            return self._set_fingers(FINGER_CLOSED)
        if state == State.LIFT:
            T = self.arm.fk(self.q_cmd)
            T[2, 3] += HOVER
            if not self._straight_line(T):
                return False
            return self.s.cube_pose()[0][2] > self.cube_start_z + LIFT_CHECK
        if state == State.TRANSFER:
            return self._plan_to(top_down_pose(px, py, PLACE_Z + HOVER, 0.0))
        if state == State.PLACE:
            return self._straight_line(top_down_pose(px, py, PLACE_Z, 0.0))
        if state == State.RELEASE:
            return self._set_fingers(FINGER_OPEN)
        if state == State.RETREAT:
            T = self.arm.fk(self.q_cmd)
            T[2, 3] += HOVER
            return self._straight_line(T)
        return False
